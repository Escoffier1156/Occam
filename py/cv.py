"""Repeated k-fold cross-validation, pooling predictions across folds.

Each repeat tests every molecule exactly once, so one AUC is computed from the
whole dataset rather than averaged over small per-fold sets. Repeats with
different seeds then show how much the grouping itself moves the result.

The learning rate is chosen on each fold's validation fold, never on its test
fold, so no test row influences the model that scores it.

Baselines run on the same folds with the same rows, which makes the comparison
paired: per-repeat differences cancel the variation that comes from the split
rather than from the method.

    nix-shell --run 'python3 py/cv.py data/work/champ_feat/features.parquet \\
        --grouping butina --folds 5 --repeats 5'
"""

import argparse
import json
import math
import random
import shutil
import os
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import folds as folds_mod  # noqa: E402
import m3_order  # noqa: E402

ENGINE = Path(os.environ.get("OCCAM_ENGINE", ROOT / "engine" / "target" / "release" / "engine"))
LEARNING_RATES = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


def auc_from_scores(scored):
    """Exact ROC AUC from (score, label) pairs, averaging ranks over ties."""
    rows = sorted(scored)
    pos = sum(1 for _, y in rows if y == 1)
    neg = len(rows) - pos
    if not pos or not neg:
        return None
    rank_sum, i = 0.0, 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1][0] == rows[i][0]:
            j += 1
        average_rank = (i + j) / 2 + 1
        rank_sum += sum(average_rank for k in range(i, j + 1) if rows[k][1] == 1)
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def run_engine(part_dir, evaluate, learning_rate, dump=None, epochs=1):
    """Runs the engine once; returns its key=value report."""
    cmd = [
        str(ENGINE), str(part_dir),
        "--stats", str(part_dir / "stats.json"),
        "--evaluate", evaluate,
        "--learning-rate", str(learning_rate),
        "--epochs", str(epochs),
    ]
    if dump:
        cmd += ["--dump-scores", str(dump)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"engine failed on {part_dir}:\n{result.stderr}")
    return dict(line.split("\t", 1) for line in result.stdout.splitlines() if "\t" in line)


def read_scores(path, tag="scores"):
    """Reads a --dump-scores file as (logit, label) pairs.

    Ranking uses the logit, not the probability: in f32 the logistic function
    saturates to exactly 1.0 away from the training distribution, and tied
    scores leave the sort to fall back on file order.
    """
    out = []
    with open(path) as fh:
        next(fh)
        for line in fh:
            label, score, logit = line.split("\t")
            out.append((float(logit), int(label)))
    m3_order.assert_ranking_is_usable([s for s, _ in out], tag)
    return out


def occam_fold(part_dir, tmp, epochs=1):
    """Picks a learning rate on validation, then scores the test fold.

    More than one epoch is not a change of regime: each epoch streams the file
    again and holds one bulk at a time, so the memory invariants are untouched.
    Peak RSS is flat from 1 to 20 epochs. Boosting lets the baseline revisit
    its training data many times, so holding Occam to a single look would be a
    handicap this project never asked for.
    """
    best = max(
        (float(run_engine(part_dir, "valid", lr, epochs=epochs).get("auc", "nan") or "nan"), lr)
        for lr in LEARNING_RATES
    )
    rate = best[1]
    dump = tmp / f"{part_dir.name}.tsv"
    run_engine(part_dir, "test", rate, dump=dump, epochs=epochs)
    return read_scores(dump, part_dir.name), rate


def gbt_fold(part_dir, seed):
    """Gradient-boosted trees on the same fold, as a batch-learning baseline.

    CHAMP has no published baseline, so one is trained here under matched
    conditions: same rows, same nine descriptors, same test fold.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Name the parts explicitly: stats.json sits in the same directory and
    # pyarrow would try to parse it as parquet.
    table = pa.concat_tables(
        [pq.read_table(f) for f in sorted(part_dir.glob("part-*.parquet"))]
    )
    names = [n for n in table.column_names if n not in ("label", "split")]
    columns = [table.column(n).to_pylist() for n in names]
    labels = table.column("label").to_pylist()
    splits = table.column("split").to_pylist()

    rows = list(zip(*columns))
    # Train on everything that is not the test fold: the baseline is batch, so
    # it has no use for a separate validation fold, and withholding it would
    # hand Occam more data than the baseline.
    train_x = [r for r, s in zip(rows, splits) if s != folds_mod.TEST]
    train_y = [y for y, s in zip(labels, splits) if s != folds_mod.TEST]
    test_x = [r for r, s in zip(rows, splits) if s == folds_mod.TEST]
    test_y = [y for y, s in zip(labels, splits) if s == folds_mod.TEST]

    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(train_x, train_y)
    scores = model.predict_proba(test_x)[:, 1]
    return list(zip((float(s) for s in scores), test_y))


def recall_at(scored, fraction):
    """Share of all positives kept when only the top `fraction` is kept.

    This is what a filter is actually judged on. AUC scores every rank at
    once; a filter only ever runs at one operating point, and what matters
    there is how many true positives survive the cut.
    """
    ordered = sorted(scored, key=lambda sl: -sl[0])
    total_positive = sum(1 for _, y in ordered if y == 1)
    if not total_positive:
        return None
    cut = max(1, round(len(ordered) * fraction))
    return sum(1 for _, y in ordered[:cut] if y == 1) / total_positive


def bootstrap_ci(values, rounds=20000, seed=0):
    rng = random.Random(seed)
    boot = sorted(st.fmean(rng.choices(values, k=len(values))) for _ in range(rounds))
    return boot[int(0.025 * rounds)], boot[int(0.975 * rounds)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("features", type=Path)
    parser.add_argument("--grouping", choices=["scaffold", "butina"], default="butina")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1, help="passes over the training rows")
    parser.add_argument("--out", type=Path, default=None, help="write the result as json")
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args()

    operating_points = (0.10, 0.25)
    results = {"occam": [], "gbt": [], "difference": [], "rates": []}
    for f in operating_points:
        results[f"occam_recall@{f:.0%}"] = []
        results[f"gbt_recall@{f:.0%}"] = []
        results[f"difference_recall@{f:.0%}"] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for repeat in range(args.repeats):
            root = tmp / f"rep{repeat}"
            folds_mod.build(args.features, root, "kfold", args.grouping, args.folds, repeat)

            occam_scores, gbt_scores = [], []
            for fold in range(args.folds):
                part_dir = root / f"r{repeat}f{fold}"
                scores, rate = occam_fold(part_dir, tmp, epochs=args.epochs)
                occam_scores += scores
                results["rates"].append(rate)
                gbt_scores += gbt_fold(part_dir, repeat)

            occam_auc = auc_from_scores(occam_scores)
            gbt_auc = auc_from_scores(gbt_scores)
            results["occam"].append(occam_auc)
            results["gbt"].append(gbt_auc)
            results["difference"].append(occam_auc - gbt_auc)
            for f in operating_points:
                m, g = recall_at(occam_scores, f), recall_at(gbt_scores, f)
                results[f"occam_recall@{f:.0%}"].append(m)
                results[f"gbt_recall@{f:.0%}"].append(g)
                results[f"difference_recall@{f:.0%}"].append(m - g)
            print(
                f"repeat {repeat}: pooled over {len(occam_scores)} rows  "
                f"Occam {occam_auc:.4f}  GBT {gbt_auc:.4f}  diff {occam_auc - gbt_auc:+.4f}",
                flush=True,
            )
            if not args.keep_parts:
                shutil.rmtree(root, ignore_errors=True)

    summary = {
        "grouping": args.grouping, "folds": args.folds,
        "repeats": args.repeats, "epochs": args.epochs,
    }
    reported = ["occam", "gbt", "difference"]
    for f in operating_points:
        reported += [f"occam_recall@{f:.0%}", f"gbt_recall@{f:.0%}", f"difference_recall@{f:.0%}"]
    for key in reported:
        v = results[key]
        sd = st.stdev(v) if len(v) > 1 else 0.0
        lo, hi = bootstrap_ci(v) if len(v) > 1 else (v[0], v[0])
        summary[key] = {"mean": st.fmean(v), "sd": sd, "ci95": [lo, hi], "values": v}
        print(f"{key:26} {st.fmean(v):.4f} +- {sd:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    summary["learning_rates"] = sorted(set(results["rates"]))
    print(f"\nlearning rates chosen: {summary['learning_rates']}")

    if args.out:
        args.out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
