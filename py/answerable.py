"""Does falling outside the applicability domain actually make a molecule
harder to predict?

py/coverage.py reports what share of a library sits inside a training set's
domain -- 0.6% for CHAMP against the IMI pool, 22.7% for GramNegAccum. Those
are distances, measured without a label or a model. The tempting reading is
that the rest cannot be answered. That reading is not established, and one
result argues against it: in the B3DB check, three pairs were cut on a
descriptor the box is built from, so 100% of the evaluation set fell outside
the box by construction -- and the transfer worked anyway (AUC 0.72 to 0.85).

The box is a conjunction over nine descriptors. A molecule leaves it by being
out of range on any one of them, including an axis that carries no weight in
the model. So "outside the box" and "unpredictable" are different claims.

This measures the second one directly: for one training set and one evaluation
set, split the evaluation rows by whether they are inside the training domain
and compute the AUC on each side separately. Same model, same weights, same
assay -- only the membership differs. If the two AUCs match, membership does
not predict answerability, and a coverage share must not be reported as a
share of molecules the model can speak about.

CRITERIA, FIXED BEFORE THE RESULTS WERE SEEN

  membership predicts answerability   the 95% interval of (AUC on covered -
                                      AUC on uncovered) lies above 0
  membership predicts the opposite    that interval lies below 0
  membership says nothing             the interval contains 0; its width is
                                      what the measurement can resolve
  not measurable                      either side has fewer than 50 rows or
                                      only one class

The difference is bootstrapped directly, resampling the two sides
independently, because they are disjoint. Comparing two separate intervals and
calling an overlap "no difference" is the mistake taboo 44 in HANDOFF.md
warns about, and the first version of this module made it.

WHAT THIS CAN AND CANNOT ESTABLISH

An AUC is a rank statistic inside the set it was computed on. The covered and
uncovered rows are two different discrimination problems with their own label
spread, so their AUCs do not share a scale. A null result therefore supports

    the inside/outside split does not pick out a subset where the model's
    ranking degrades

which is enough to refuse reading a coverage share as a share of molecules the
model cannot speak about -- that reading needs the outside to be worse. It does
NOT support "the model is fine outside its domain".

To show whether a difference comes from the problems being of unequal
difficulty rather than from membership, two diagnostics are printed per side:
the label ratio, and the best AUC any single descriptor achieves on that side.
A side where one descriptor alone separates the classes well is an easier
problem, whatever the model does.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import coverage as coverage_mod  # noqa: E402
import transfer as transfer_mod  # noqa: E402

WORK = ROOT / "data" / "work"
MIN_SIDE = 50


def difference_ci(covered_rows, uncovered_rows, rounds=None, seed=0):
    """Percentile interval for AUC(covered) - AUC(uncovered).

    The two sides are disjoint, so each is resampled on its own. This is the
    quantity the verdict is about; two separate intervals cannot answer it.
    """
    import random as _random

    rounds = transfer_mod.BOOTSTRAP_ROUNDS if rounds is None else rounds
    rng = _random.Random(seed)
    boot = []
    for _ in range(rounds):
        a = transfer_mod.auc(rng.choices(covered_rows, k=len(covered_rows)))
        b = transfer_mod.auc(rng.choices(uncovered_rows, k=len(uncovered_rows)))
        if a is not None and b is not None:
            boot.append(a - b)
    boot.sort()
    return boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]


def best_single_descriptor_auc(table, names, mask):
    """How well the best one of the nine descriptors separates the classes.

    A difficulty proxy that does not involve the fitted model: if one side's
    classes are pulled apart by a single descriptor, ranking it is easier
    there, and a higher AUC on that side says nothing about membership.
    """
    labels = np.asarray(table.column("label").to_pylist())[mask]
    raw = coverage_mod.matrix(table, names)[mask]
    best = 0.0
    for column in range(raw.shape[1]):
        pairs = list(zip(raw[:, column].tolist(), labels.tolist()))
        a = transfer_mod.auc(pairs)
        if a is None:
            continue
        best = max(best, a, 1.0 - a)
    return best

def one(name, source, target, epochs, learning_rate, tmp, percentile):
    """Trains on `source`, scores `target`, splits the scores by membership."""
    src_rows, tgt_rows = transfer_mod.train_and_apply(
        source, target, epochs, learning_rate, tmp, name.replace(" ", "_").replace("/", "_")
    )
    target_table = pq.read_table(target)
    names = coverage_mod.feature_names(target_table)
    labels = target_table.column("label").to_pylist()

    # Taboo 10: the dumped scores are only trustworthy row-by-row if the
    # labels line up. Every row here was written as TEST, so the dump should be
    # the whole table in order; check it rather than assume it.
    if len(tgt_rows) != len(labels) or [y for _, y in tgt_rows] != list(labels):
        sys.exit(f"{name}: dumped scores do not line up with the target rows")

    domain = coverage_mod.Domain("source", pq.read_table(source), names, percentile)
    raw = coverage_mod.matrix(target_table, names)
    in_box, near, _, distance = domain.test(raw)
    covered = in_box & near

    sides, rows_of = {}, {}
    for label, mask in (("covered", covered), ("uncovered", ~covered)):
        rows = [tgt_rows[i] for i in range(len(tgt_rows)) if mask[i]]
        rows_of[label] = rows
        positives = sum(1 for _, y in rows if y == 1)
        entry = {"n": len(rows), "positives": positives,
                 "positive_share": positives / len(rows) if rows else None,
                 "auc": None, "ci95": None, "best_single_descriptor": None}
        if len(rows) >= MIN_SIDE and positives not in (0, len(rows)):
            entry["auc"] = transfer_mod.auc(rows)
            entry["ci95"] = list(transfer_mod.auc_ci(rows))
            entry["best_single_descriptor"] = best_single_descriptor_auc(
                target_table, names, mask
            )
        sides[label] = entry

    c, u = sides["covered"], sides["uncovered"]
    diff = diff_ci = None
    if c["auc"] is None or u["auc"] is None:
        verdict = "not measurable"
    else:
        diff = c["auc"] - u["auc"]
        diff_ci = list(difference_ci(rows_of["covered"], rows_of["uncovered"]))
        if diff_ci[0] > 0:
            verdict = "membership predicts answerability"
        elif diff_ci[1] < 0:
            verdict = "membership predicts the OPPOSITE"
        else:
            verdict = "membership says nothing"

    print(f"\n=== {name}")
    print(f"  covered share {covered.mean():>6.1%}   "
          f"median distance to nearest training point {np.median(distance):.2f} "
          f"(threshold {domain.threshold:.2f})")
    for label in ("covered", "uncovered"):
        s = sides[label]
        if s["auc"] is None:
            print(f"  {label:10} n={s['n']:>5} positives={s['positives']:>5}   "
                  f"too small or single-class")
        else:
            print(f"  {label:10} n={s['n']:>5} positive {s['positive_share']:>5.1%}   "
                  f"AUC {s['auc']:.4f}  95% CI [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]"
                  f"   best single descriptor {s['best_single_descriptor']:.4f}")
    if diff is not None:
        print(f"  difference covered - uncovered: {diff:+.4f}  "
              f"95% CI [{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}]  "
              f"(width {diff_ci[1] - diff_ci[0]:.4f})")
        print(f"  difficulty gap (best single descriptor, uncovered - covered): "
              f"{u['best_single_descriptor'] - c['best_single_descriptor']:+.4f}")
    print(f"  --> {verdict}")
    return {"direction": name, "covered_share": float(covered.mean()),
            "sides": sides, "difference": diff, "difference_ci95": diff_ci,
            "verdict": verdict}


def pairs_from_predictions(path):
    """Both directions of every pair py/inclusion.py built."""
    out = []
    for p in json.loads(Path(path).read_text()):
        a, b = Path(p["a"]["path"]), Path(p["b"]["path"])
        out.append((f"{p['pair']}: {p['a']['name']} -> {p['b']['name']}", a, b))
        out.append((f"{p['pair']}: {p['b']['name']} -> {p['a']['name']}", b, a))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path,
                        default=WORK / "inclusion" / "predictions.json")
    parser.add_argument("--champ", type=Path, default=WORK / "champ_feat" / "features.parquet")
    parser.add_argument("--gna", type=Path, default=WORK / "gna_feat" / "features.parquet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--neighbour-percentile", type=float,
                        default=coverage_mod.NEIGHBOUR_PERCENTILE)
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument("--skip-b3db", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.engine:
        transfer_mod.ENGINE = args.engine
    transfer_mod.BOOTSTRAP_ROUNDS = args.bootstrap

    print(__doc__[__doc__.index("CRITERIA"):].strip())

    work = [("CHAMP -> GramNegAccum", args.champ, args.gna),
            ("GramNegAccum -> CHAMP", args.gna, args.champ)]
    if not args.skip_b3db and args.predictions.exists():
        work += pairs_from_predictions(args.predictions)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, source, target in work:
            results.append(one(name, source, target, args.epochs, args.learning_rate,
                               tmp, args.neighbour_percentile))

    print("\n\n================ SUMMARY ================\n")
    print(f"{'direction':44} {'cov':>6} {'diff':>8} {'95% CI of the diff':>22} "
          f"{'easier side':>12}  verdict")
    tally = {}
    for r in results:
        c, u = r["sides"]["covered"], r["sides"]["uncovered"]
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
        if r["difference"] is None:
            print(f"{r['direction']:44} {r['covered_share']:>5.1%} {'-':>8} {'-':>22} "
                  f"{'-':>12}  {r['verdict']}")
            continue
        gap = u["best_single_descriptor"] - c["best_single_descriptor"]
        easier = "uncovered" if gap > 0.02 else ("covered" if gap < -0.02 else "neither")
        print(f"{r['direction']:44} {r['covered_share']:>5.1%} {r['difference']:>+8.4f} "
              f"[{r['difference_ci95'][0]:+.4f}, {r['difference_ci95'][1]:+.4f}]".ljust(0)
              + f"  {easier:>12}  {r['verdict']}")
    print()
    for verdict, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>2} directions: {verdict}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
