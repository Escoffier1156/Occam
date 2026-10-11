"""Does the permeability model survive a change of chemical space?

CHAMP is 1,132 azide probes: every molecule carries an azide, and two
Bemis-Murcko "scaffolds" (a lone benzene ring, and no ring at all) cover 52.7%
of them. An AUC measured inside that library says nothing about whether the
model learned what makes a molecule cross a membrane or what makes an azide
probe cross one. The only open dataset that can answer it is GramNegAccum, the
eNTRy rules data: LC-MS/MS whole-cell accumulation for amine-bearing
antibiotics, a different assay on a different chemical space.

Both directions are run, because they answer different questions. CHAMP ->
GramNegAccum asks whether what was learned on probes carries to drugs;
GramNegAccum -> CHAMP asks the reverse on a much smaller training set.

CRITERIA, FIXED BEFORE THE RESULTS WERE SEEN
--------------------------------------------
  transfer holds      lower bound of the AUC's 95% CI is above 0.5
  transfer fails      the interval contains 0.5
  not interpretable   the z distribution on the target is far outside the
                      source's, whatever the AUC: an ordering produced by
                      extrapolation is not a prediction. The threshold is the
                      target's median against the source's 99th percentile,
                      the comparison that caught the CHAMP -> IMI case, where
                      the median landed at 32.9 against a p99 of 17.1.

The AUC interval is bootstrapped over the evaluation rows, since 151 to 267
molecules leave it wide.
"""

import argparse
import json
import random
import statistics as st
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import folds as folds_mod  # noqa: E402
from m3_order import assert_ranking_is_usable, sh, stats_from, write_parts  # noqa: E402
ENGINE = ROOT / "engine" / "target" / "release" / "engine"

WORK = ROOT / "data" / "work"
BOOTSTRAP_ROUNDS = 20000


def auc(pairs):
    """ROC AUC from (score, label), averaging ranks over ties."""
    rows = sorted(pairs)
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


def auc_ci(pairs, rounds=None, seed=0):
    """Percentile interval for the AUC, resampling the evaluation rows.

    `rounds` defaults to BOOTSTRAP_ROUNDS at call time rather than at import,
    so a caller sweeping many pairs can lower it. The interval narrows with the
    evaluation set, so thousands of rows do not need the full count that 129 do.
    """
    rounds = BOOTSTRAP_ROUNDS if rounds is None else rounds
    rng = random.Random(seed)
    boot = []
    for _ in range(rounds):
        sample = rng.choices(pairs, k=len(pairs))
        a = auc(sample)
        if a is not None:
            boot.append(a)
    boot.sort()
    return boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]


def train_and_apply(source, target, epochs, learning_rate, tmp, tag):
    """Fits on every row of `source`, scores every row of `target`."""
    src = pq.read_table(source)
    tgt = pq.read_table(target)
    features = [n for n in src.column_names if n not in ("label", "scaffold", "smiles")]
    if [n for n in tgt.column_names if n not in ("label", "scaffold", "smiles")] != features:
        sys.exit(f"{tag}: the two datasets describe molecules differently")

    train_dir = tmp / f"{tag}_train"
    write_parts(src, features, [folds_mod.TRAIN] * src.num_rows, train_dir)
    stats = train_dir / "stats.json"
    stats.write_text(json.dumps(stats_from(src, features, [True] * src.num_rows)) + "\n")
    weights = tmp / f"{tag}.txt"
    sh(ENGINE, train_dir, "--stats", stats, "--evaluate", "train",
       "--epochs", epochs, "--learning-rate", learning_rate, "--save-weights", weights)

    def logits_of(table, name):
        part = tmp / f"{tag}_{name}"
        write_parts(table, features, [folds_mod.TEST] * table.num_rows, part)
        dump = tmp / f"{tag}_{name}.tsv"
        sh(ENGINE, part, "--stats", stats, "--evaluate", "test",
           "--load-weights", weights, "--dump-scores", dump)
        rows = []
        with open(dump) as fh:
            next(fh)
            for line in fh:
                label, score, logit = line.split("\t")
                rows.append((float(logit), int(label)))
        return rows

    return logits_of(src, "source"), logits_of(tgt, "target")


def extrapolation(source_z, target_z):
    """How far the target lands outside the range the model was fitted on."""
    s = sorted(source_z)
    t = sorted(target_z)
    p99 = s[int(0.99 * len(s))]
    median = t[len(t) // 2]
    return {
        "source_median": s[len(s) // 2],
        "source_p99": p99,
        "target_median": median,
        "target_p99": t[int(0.99 * len(t))],
        "target_median_over_source_p99": median > p99,
        "target_share_beyond_source_p99": sum(1 for z in t if z > p99) / len(t),
    }


def run(name, source, target, epochs, learning_rate, tmp):
    src_rows, tgt_rows = train_and_apply(source, target, epochs, learning_rate, tmp, name)
    diag = assert_ranking_is_usable([z for z, _ in tgt_rows], name)
    a = auc(tgt_rows)
    lo, hi = auc_ci(tgt_rows)
    ex = extrapolation([z for z, _ in src_rows], [z for z, _ in tgt_rows])

    if ex["target_median_over_source_p99"]:
        verdict = "not interpretable (extrapolation)"
    elif lo > 0.5:
        verdict = "transfer holds"
    else:
        verdict = "transfer fails"

    print(f"\n=== {name}")
    print(f"  evaluated on {len(tgt_rows)} molecules, "
          f"{sum(1 for _, y in tgt_rows if y == 1)} positive")
    print(f"  AUC {a:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  source logit: median {ex['source_median']:+.2f}, p99 {ex['source_p99']:+.2f}")
    print(f"  target logit: median {ex['target_median']:+.2f}, p99 {ex['target_p99']:+.2f}")
    print(f"  target rows beyond the source's p99: "
          f"{ex['target_share_beyond_source_p99']:.1%}")
    print(f"  ties {diag['tie_share']:.2%}, distinct {diag['distinct_share']:.1%}")
    print(f"  --> {verdict}")
    return {
        "direction": name, "auc": a, "ci95": [lo, hi], "n": len(tgt_rows),
        "positives": sum(1 for _, y in tgt_rows if y == 1),
        "extrapolation": ex, "ranking": diag, "verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--champ", type=Path, default=WORK / "champ_feat" / "features.parquet")
    parser.add_argument("--gna", type=Path, default=WORK / "gna_feat" / "features.parquet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--engine", type=Path, default=None)
    args = parser.parse_args()
    if args.engine:
        globals()["ENGINE"] = args.engine

    print(__doc__[__doc__.index("CRITERIA"):].strip())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        results = [
            run("CHAMP -> GramNegAccum", args.champ, args.gna,
                args.epochs, args.learning_rate, tmp),
            run("GramNegAccum -> CHAMP", args.gna, args.champ,
                args.epochs, args.learning_rate, tmp),
        ]
    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
