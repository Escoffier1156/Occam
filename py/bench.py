"""AUC over several scaffold splits. Numbers only; no model lives here.

Descriptors are computed once; each seed then only redraws the 8:1:1 split via
py/folds.py, and the engine runs twice per seed (learn, then score). A single
scaffold split is one draw, so the summary is mean +- standard deviation over
the seeds, matching how the published comparison values are reported.

This 8:1:1 protocol exists to match published values that were produced that
way. Where there is no such value to match, py/cv.py's repeated k-fold is the
better design, because it tests every molecule once instead of a tenth of them.

    nix-shell --run 'python3 py/bench.py data/work/B3DB_classification.parquet \\
        --label-column "BBB+/BBB-" --positive-label BBB+ --seeds 21'
"""

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = ROOT / "data" / "work"
ENGINE = ROOT / "engine" / "target" / "release" / "engine"


def sh(*cmd):
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def engine_report(desc_dir, evaluate, epochs, bulk_size, dump_scores=None, learning_rate=None):
    """Runs the engine and returns its tab-separated output as a dict."""
    cmd = [
        ENGINE, desc_dir,
        "--stats", desc_dir / "stats.json",
        "--evaluate", evaluate,
        "--epochs", str(epochs),
        "--bulk-size", str(bulk_size),
    ]
    if learning_rate is not None:
        cmd += ["--learning-rate", str(learning_rate)]
    if dump_scores:
        cmd += ["--dump-scores", dump_scores]
    out = sh(*cmd)
    return dict(line.split("\t", 1) for line in out.splitlines() if "\t" in line)


def exact_auc(path):
    """AUC straight from per-row scores, for checking the binned one.

    This reads every score into memory, which is why it is a check and not the
    engine's own output: the result is O(D), the engine's carry is not.
    """
    rows = []
    with open(path) as fh:
        next(fh)
        for line in fh:
            label, score = line.split("\t")
            rows.append((float(score), int(label)))
    rows.sort()
    pos = sum(1 for _, y in rows if y == 1)
    neg = len(rows) - pos
    if not pos or not neg:
        return None
    # Mann-Whitney with average ranks for ties.
    rank_sum, i = 0.0, 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1][0] == rows[i][0]:
            j += 1
        average_rank = (i + j) / 2 + 1
        rank_sum += sum(average_rank for k in range(i, j + 1) if rows[k][1] == 1)
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def select_learning_rate(desc_dir, candidates, epochs, bulk_size):
    """Picks the learning rate with the best AUC on the validation split.

    The test split is never read here. Doing this on test would make the
    reported number a best-of-N over the data it is supposed to be measured on.
    """
    scored = []
    for rate in candidates:
        report = engine_report(desc_dir, "valid", epochs, bulk_size, learning_rate=rate)
        auc = report["auc"]
        scored.append((float(auc) if auc != "undefined" else 0.0, rate))
    best_auc, best_rate = max(scored)
    return best_rate, best_auc, scored


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="parquet file produced by ingest.py")
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--positive-label", required=True)
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument(
        "--keep-labels", nargs="+", default=None, metavar="VALUE",
        help="passed to descriptors.py: only these label values are used",
    )
    parser.add_argument("--seeds", type=int, default=21)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bulk-size", type=int, default=1024)
    parser.add_argument("--evaluate", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--grouping", choices=["scaffold", "butina"], default="scaffold")
    parser.add_argument("--keep-splits", action="store_true", help="keep each seed's parts")
    parser.add_argument("--check-exact", action="store_true", help="verify the binned AUC on seed 0")
    parser.add_argument(
        "--learning-rates", type=float, nargs="+",
        default=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
        help="candidates; the one best on the validation split is used for test",
    )
    parser.add_argument(
        "--fixed-learning-rate", type=float, default=None,
        help="skip validation-based selection and use this rate",
    )
    args = parser.parse_args()

    # Descriptors do not depend on the split, so compute them once.
    feature_dir = WORK_DIR / f"{args.source.stem}_feat"
    sh(
        sys.executable, ROOT / "py" / "descriptors.py", args.source,
        "--label-column", args.label_column,
        "--positive-label", args.positive_label,
        "--smiles-column", args.smiles_column,
        "--out-dir", feature_dir,
        *(["--keep-labels", *args.keep_labels] if args.keep_labels else []),
    )

    aucs, prequential, rates = [], [], []
    print(f"seed\t{args.evaluate}_auc\tprequential_auc\tlearning_rate\tvalid_auc")
    for seed in range(args.seeds):
        # Descriptors are already computed; this only redraws the split.
        desc_dir = feature_dir / f"s{seed}"
        sh(
            sys.executable, ROOT / "py" / "folds.py", feature_dir / "features.parquet",
            "--scheme", "holdout", "--grouping", args.grouping,
            "--repeat", seed, "--out-root", feature_dir,
        )
        if args.fixed_learning_rate is not None:
            rate, valid_auc = args.fixed_learning_rate, float("nan")
        else:
            rate, valid_auc, _ = select_learning_rate(
                desc_dir, args.learning_rates, args.epochs, args.bulk_size
            )
        report = engine_report(
            desc_dir, args.evaluate, args.epochs, args.bulk_size, learning_rate=rate
        )
        auc = float(report["auc"])
        preq = float(report["prequential_auc"])
        aucs.append(auc)
        prequential.append(preq)
        rates.append(rate)
        print(f"{seed}\t{auc:.4f}\t{preq:.4f}\t{rate}\t{valid_auc:.4f}", flush=True)

        if args.check_exact and seed == 0:
            with tempfile.NamedTemporaryFile(suffix=".tsv") as scores:
                engine_report(
                    desc_dir, args.evaluate, args.epochs, args.bulk_size, scores.name, rate
                )
                exact = exact_auc(scores.name)
            print(f"  binned={auc:.6f} exact={exact:.6f} diff={abs(auc - exact):.2e}")

        if not args.keep_splits:
            shutil.rmtree(desc_dir, ignore_errors=True)

    mean = statistics.fmean(aucs)
    sd = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
    print(f"\n{args.evaluate} AUC: {mean:.4f} +- {sd:.4f}  (n={len(aucs)})")
    print(f"prequential AUC: {statistics.fmean(prequential):.4f}")
    print(f"learning rates chosen: {sorted(set(rates))}")
    print(json.dumps({"mean": mean, "sd": sd, "seeds": len(aucs), "aucs": aucs, "rates": rates}))


if __name__ == "__main__":
    main()
