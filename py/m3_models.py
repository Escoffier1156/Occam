"""Measure each model on its own before any pipeline is built.

The order experiment is only meaningful if its two parts work. Assembling
them without checking first is what produced a binding stage that ranked
worse than chance.
"""
import json, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import folds as folds_mod
from m3_order import write_parts, stats_from, sh, score_with, top_m

ENGINE = {"desc": ROOT/"engine"/"target"/"release"/"engine",
          "fp": ROOT/"engine"/"target-fp"/"release"/"engine"}
WORK = ROOT / "data" / "work"


def auc(pairs):
    rows = sorted(pairs)
    pos = sum(1 for _, y in rows if y == 1); neg = len(rows) - pos
    if not pos or not neg: return None
    rs, i = 0.0, 0
    while i < len(rows):
        j = i
        while j+1 < len(rows) and rows[j+1][0] == rows[i][0]: j += 1
        ar = (i+j)/2 + 1
        rs += sum(ar for k in range(i, j+1) if rows[k][1] == 1)
        i = j+1
    return (rs - pos*(pos+1)/2) / (pos*neg)


def evaluate(features_path, engine, epochs, lr, tag):
    table = pq.read_table(features_path)
    feats = [n for n in table.column_names if n not in ("label","scaffold","smiles")]
    groups = table.column("scaffold").to_pylist()
    assign, _ = folds_mod.assign_folds(Counter(groups), 2, 0)
    is_eval = [assign[g] == 1 for g in groups]
    splits = [folds_mod.TEST if e else folds_mod.TRAIN for e in is_eval]
    tag = tag.replace(" ", "_").replace("(", "").replace(")", "")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); d = tmp/"parts"
        write_parts(table, feats, splits, d)
        (d/"stats.json").write_text(json.dumps(stats_from(table, feats, [not e for e in is_eval]))+"\n")
        w = tmp/"w.txt"
        sh(engine, d, "--stats", d/"stats.json", "--evaluate", "test",
           "--epochs", epochs, "--learning-rate", lr, "--save-weights", w)
        scores = score_with(d, d/"stats.json", w, tmp, tag, splits, engine=engine)
    labels = table.column("label").to_pylist()
    pairs = [(score, labels[i]) for i, score in scores.items()]
    pos = sum(1 for _, y in pairs if y == 1)
    return auc(pairs), len(pairs), pos


if __name__ == "__main__":
    print(f"{'model':34} {'features':>9} {'AUC':>7} {'test n':>8} {'pos':>7}")
    for name, path, eng, ep, lr in [
        ("binding (9 physchem, quadratic)", WORK/"imi_feat_bind"/"features.parquet", "desc", 10, 0.5),
        ("binding (ECFP8 2048 bits)",       WORK/"imi_fp"/"features.parquet",        "fp",   10, 0.5),
    ]:
        if not path.exists():
            print(f"{name:34} -- missing {path}"); continue
        a, n, p = evaluate(path, ENGINE[eng], ep, lr, name)
        print(f"{name:34} {'2048' if eng=='fp' else '54':>9} {a:>7.4f} {n:>8} {p:>7}")
