"""How much of a library can a permeability dataset actually speak about?

Coverage is a property of the descriptor ranges, not of any label, so nothing
is trained here. Two measures are reported side by side because one of them is
only a necessary condition.

CRITERIA, FIXED BEFORE THE RESULTS WERE SEEN

  box        every descriptor inside the training set's 1st-99th percentile
             range. This is necessary but not sufficient: 197 points placed in
             nine dimensions leave the interior almost empty, so a molecule
             can sit inside the box with no training point anywhere near it.

  neighbour  distance, in training-standardised descriptor space, to the
             nearest training point, at or below the 95th percentile of the
             training set's own leave-one-out nearest-neighbour distances.
             The reference is the training set's internal spacing, so the
             question is "is this molecule no further from the training data
             than the training data is from itself", and the threshold does
             not depend on how many descriptors there are.

  covered    inside the box and within neighbour distance. The headline number.

The neighbour measure is expected to be the smaller of the two. Reporting the
box alone was the mistake this module exists to correct.

Standardisation uses the training set's mean and standard deviation only, as
py/folds.py does: a library row must never influence the frame it is judged in.

Per-descriptor exceedances are printed alongside, because a share on its own
does not say what is out of range (see the taboo list in docs/HANDOFF.md).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import folds as folds_mod  # noqa: E402

WORK = ROOT / "data" / "work"
NON_FEATURE_COLUMNS = ("label", "scaffold", "smiles", "inchikey", "skeleton", "tables")
# Fixed before running; see CRITERIA above.
BOX_LOW_PERCENTILE = 1
BOX_HIGH_PERCENTILE = 99
NEIGHBOUR_PERCENTILE = 95


def feature_names(table):
    return [n for n in table.column_names if n not in NON_FEATURE_COLUMNS]


def matrix(table, names):
    return np.column_stack([np.asarray(table.column(n).to_pylist(), dtype=np.float64)
                            for n in names])


def nearest_distances(points, reference):
    """For each row of `points`, the distance to the closest row of `reference`.

    Chunked over `points` so the pairwise block never grows with the library:
    the engine's memory invariants do not bind preprocessing, but a 22k x 22k
    block is pointless when 4096 rows at a time give the same answer.
    """
    out = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), 4096):
        block = points[start:start + 4096]
        d = np.linalg.norm(block[:, None, :] - reference[None, :, :], axis=2)
        out[start:start + len(block)] = d.min(axis=1)
    return out


def internal_neighbour_distances(reference):
    """Leave-one-out nearest-neighbour distance within the training set."""
    d = np.linalg.norm(reference[:, None, :] - reference[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


class Domain:
    """One training set's applicability domain, ready to test a library against."""

    def __init__(self, name, table, names, neighbour_percentile=NEIGHBOUR_PERCENTILE):
        self.name = name
        self.names = names
        self.neighbour_percentile = neighbour_percentile
        raw = matrix(table, names)
        self.n = len(raw)
        self.mean = raw.mean(axis=0)
        # A descriptor that is constant in training carries no scale; leave it
        # at 1 so standardising cannot divide by zero.
        self.sd = np.where(raw.std(axis=0) > 0, raw.std(axis=0), 1.0)
        self.low = np.percentile(raw, BOX_LOW_PERCENTILE, axis=0)
        self.high = np.percentile(raw, BOX_HIGH_PERCENTILE, axis=0)
        self.points = (raw - self.mean) / self.sd
        internal = internal_neighbour_distances(self.points)
        self.threshold = float(np.percentile(internal, neighbour_percentile))
        self.internal_median = float(np.median(internal))

    def test(self, raw_library):
        """Returns (in_box, within_neighbour) boolean arrays and exceedance counts."""
        in_range = (raw_library >= self.low) & (raw_library <= self.high)
        in_box = in_range.all(axis=1)
        standardised = (raw_library - self.mean) / self.sd
        distance = nearest_distances(standardised, self.points)
        return in_box, distance <= self.threshold, (~in_range).sum(axis=0), distance


def evaluation_pool(table, folds, fold):
    """The same scaffold-grouped half of the library py/cv.py evaluates on."""
    groups = table.column("scaffold").to_pylist()
    assignment, _ = folds_mod.assign_folds(Counter(groups), folds, 0)
    return [i for i in range(len(groups)) if assignment[groups[i]] == fold]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", type=Path,
                        default=WORK / "imi_feat_bind" / "features.parquet")
    parser.add_argument("--training", type=Path, nargs="+",
                        default=[WORK / "champ_feat" / "features.parquet",
                                 WORK / "gna_feat" / "features.parquet"])
    parser.add_argument("--names", nargs="+", default=["CHAMP", "GramNegAccum"],
                        help="one label per --training path")
    parser.add_argument("--truth", type=Path, default=WORK / "imi_clean.parquet")
    parser.add_argument("--truth-column", default="true_hit")
    parser.add_argument("--truth-positive", default="active")
    parser.add_argument("--eval-pool-folds", type=int, default=2)
    parser.add_argument("--eval-pool-fold", type=int, default=1)
    parser.add_argument("--whole-library", action="store_true",
                        help="measure the full library instead of the evaluation pool")
    parser.add_argument("--neighbour-percentile", type=float, default=NEIGHBOUR_PERCENTILE,
                        help="the declared default is 95; other values are a "
                             "sensitivity check, not a new definition")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if len(args.names) != len(args.training):
        sys.exit(f"{len(args.names)} names for {len(args.training)} training sets")

    print(__doc__[__doc__.index("CRITERIA"):__doc__.index("Standardisation")].strip())
    print()

    library = pq.read_table(args.library)
    names = feature_names(library)
    domains = [Domain(n, pq.read_table(p), names, args.neighbour_percentile)
               for n, p in zip(args.names, args.training)]

    if args.whole_library:
        rows = list(range(library.num_rows))
    else:
        rows = evaluation_pool(library, args.eval_pool_folds, args.eval_pool_fold)
    raw = matrix(library, names)[rows]

    truth = None
    if args.truth and args.truth.exists():
        t = pq.read_table(args.truth, columns=["smiles", args.truth_column])
        label_of = dict(zip(t.column(0).to_pylist(),
                            [1 if v == args.truth_positive else 0
                             for v in t.column(1).to_pylist()]))
        smiles = library.column("smiles").to_pylist()
        truth = np.asarray([label_of[smiles[i]] for i in rows])

    scope = "whole library" if args.whole_library else "evaluation pool"
    header = f"{scope}: {len(rows)} molecules"
    if truth is not None:
        header += f", {int(truth.sum())} true hits ({truth.mean():.2%})"
    print(header)
    for d in domains:
        print(f"  {d.name}: {d.n} training molecules, internal nearest-neighbour "
              f"distance median {d.internal_median:.2f}, "
              f"p{d.neighbour_percentile:g} {d.threshold:.2f}")
    print()

    results = {"scope": scope, "library_rows": len(rows), "measures": {}}
    if truth is not None:
        results["true_hits"] = int(truth.sum())

    tested = {}
    for d in domains:
        in_box, near, exceed, distance = d.test(raw)
        tested[d.name] = (in_box, near, exceed, distance)

    def report(label, mask):
        n = int(mask.sum())
        line = f"{label:34} {n:>6} {n/len(rows):>7.1%}"
        entry = {"n": n, "share": n / len(rows)}
        if truth is not None and n:
            hits = int(truth[mask].sum())
            line += f" {hits/n:>9.2%} {hits:>6} {hits/truth.sum():>11.1%}"
            entry.update(hit_rate=hits / n, hits=hits,
                         share_of_all_hits=float(hits / truth.sum()))
        elif truth is not None:
            line += f" {'-':>9} {0:>6} {0.0:>11.1%}"
            entry.update(hit_rate=None, hits=0, share_of_all_hits=0.0)
        print(line)
        results["measures"][label] = entry

    head = f"{'applicability domain':34} {'n':>6} {'share':>7}"
    if truth is not None:
        head += f" {'hit rate':>9} {'hits':>6} {'of all hits':>12}"
    print(head)
    for d in domains:
        in_box, near, _, _ = tested[d.name]
        report(f"{d.name}: box", in_box)
        report(f"{d.name}: neighbour", near)
        report(f"{d.name}: covered", in_box & near)
    if len(domains) > 1:
        any_box = np.logical_or.reduce([tested[d.name][0] for d in domains])
        any_cov = np.logical_or.reduce([tested[d.name][0] & tested[d.name][1]
                                        for d in domains])
        report("union: box", any_box)
        report("union: covered", any_cov)

    # A share says how much is out of range; this says what is out of range.
    print("\nper-descriptor rows outside the box (count and share of the scope):")
    for d in domains:
        _, _, exceed, distance = tested[d.name]
        worst = sorted(zip(names, exceed), key=lambda kv: -kv[1])
        shown = " ".join(f"{n}={c} ({c/len(rows):.1%})" for n, c in worst if c)
        print(f"  {d.name}: {shown or 'none'}")
        results["measures"].setdefault(f"{d.name}: exceedances", {}) \
            .update({n: int(c) for n, c in worst})
        results["measures"][f"{d.name}: distance"] = {
            "median": float(np.median(distance)),
            "p90": float(np.percentile(distance, 90)),
            "threshold": d.threshold,
        }

    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
