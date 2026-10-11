"""Scaffold-based k-fold splits over a features.parquet from py/descriptors.py.

Why folds and not a single 8:1:1 split: with a small dataset one held-out
split is tiny, and its class balance swings far enough that an AUC computed on
it says more about the draw than about the model. Under k folds every molecule
is tested exactly once per repeat, so the reported number rests on the whole
dataset. Repeats with different seeds then measure how much the assignment
itself matters.

A scaffold never straddles folds. Scaffolds larger than half a fold go first,
so they cannot dominate whichever fold they land in; the rest are shuffled by
the seed and packed into whichever fold is emptiest.

For fold f the test set is fold f, the validation set is fold (f+1) % k, and
the remaining folds train. Validation is only ever used to choose the learning
rate, never to measure, so the rotation introduces no leakage into the test
fold.

Standardization statistics are recomputed per fold from its training rows
alone, by Welford's method, and written next to the parts the engine reads.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
PART_ROWS = 8192
# Split codes, matching SPLITS in the engine (engine/src/bulk.rs).
TRAIN, VALID, TEST = 0, 1, 2


class Welford:
    """Streaming mean and variance, one instance per feature."""

    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def add(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def variance(self):
        return self.m2 / self.n if self.n else 0.0


# Fingerprint settings for Butina grouping. Changing either changes the
# clusters, so both belong in any description of a result produced this way.
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
BUTINA_CUTOFF = 0.4
BUTINA_MAX_MOLECULES = 6000


def butina_groups(smiles, cutoff=BUTINA_CUTOFF):
    """Groups molecules by Tanimoto distance; returns one group id per row.

    Bemis-Murcko scaffolds are too coarse for a small, mostly monocyclic
    library: on CHAMP two "scaffolds" (a lone benzene ring, and no ring at all)
    cover 52.7% of the compounds, which cannot be divided across folds without
    either splitting them or unbalancing the folds badly. Clustering on
    fingerprints puts a ceiling on group size while still keeping similar
    molecules together.
    """
    # The lower triangle is O(n^2) floats held at once. At 22k molecules that
    # is 240M entries and the process is killed, so refuse early and say why.
    if len(smiles) > BUTINA_MAX_MOLECULES:
        raise ValueError(
            f"butina grouping needs the full distance matrix: {len(smiles)} molecules "
            f"would hold {len(smiles) * (len(smiles) - 1) // 2:,} distances. "
            f"Above {BUTINA_MAX_MOLECULES} use --grouping scaffold, which is only "
            f"degenerate on narrow libraries."
        )
    import sys

    from rdkit import DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina

    # Imported here, not at module scope: the rest of this module runs without
    # rdkit and only the Butina path needs it.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import smiles as smiles_mod

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
    )
    fingerprints = [generator.GetFingerprint(smiles_mod.parse(s)) for s in smiles]

    # Butina wants the lower triangle of the distance matrix, row by row.
    distances = []
    for i in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        distances.extend(1.0 - x for x in similarities)

    clusters = Butina.ClusterData(distances, len(fingerprints), cutoff, isDistData=True)
    group_of_row = [None] * len(fingerprints)
    for cluster_id, members in enumerate(clusters):
        for i in members:
            group_of_row[i] = f"c{cluster_id}"
    return group_of_row


def assign_holdout(group_sizes, seed, fractions=(0.8, 0.1, 0.1)):
    """Chemprop's scaffold_balanced, generalised to any grouping.

    Groups larger than half a held-out split go first so they cannot dominate
    one, then the rest are shuffled and packed greedily.
    """
    total = sum(group_sizes.values())
    sizes = [f * total for f in fractions]

    big, small = [], []
    for group, size in group_sizes.items():
        (big if size > sizes[1] / 2 or size > sizes[2] / 2 else small).append(group)
    big.sort()
    small.sort()
    rng = random.Random(seed)
    rng.shuffle(big)
    rng.shuffle(small)

    filled = [0] * 3
    assignment = {}
    for group in big + small:
        size = group_sizes[group]
        if filled[0] + size <= sizes[0]:
            i = 0
        elif filled[1] + size <= sizes[1]:
            i = 1
        else:
            i = 2
        assignment[group] = i
        filled[i] += size
    return assignment, filled


def assign_folds(scaffold_sizes, k, seed):
    """Maps each scaffold to a fold index; returns (assignment, fold sizes)."""
    total = sum(scaffold_sizes.values())
    fold_target = total / k

    big, small = [], []
    for scaffold, size in scaffold_sizes.items():
        (big if size > fold_target / 2 else small).append(scaffold)
    # Sort before shuffling so the order depends on the seed, not on dict order.
    big.sort()
    small.sort()
    rng = random.Random(seed)
    rng.shuffle(big)
    rng.shuffle(small)

    filled = [0] * k
    assignment = {}
    for scaffold in big + small:
        i = min(range(k), key=lambda j: (filled[j], j))
        assignment[scaffold] = i
        filled[i] += scaffold_sizes[scaffold]
    return assignment, filled


def write_holdout(table, features, split_of_row, out_dir):
    """Writes one 8:1:1 directory, for comparing against published values."""
    return _write(table, features, split_of_row, out_dir, {"scheme": "holdout"})


def _write(table, features, split_of_row, out_dir, extra):
    """Writes engine-readable parts plus stats.json computed on train rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("part-*.parquet"):
        stale.unlink()

    schema = pa.schema(
        [(name, pa.float32()) for name in features]
        + [("label", pa.int8()), ("split", pa.int8())]
    )
    columns = {name: table.column(name).to_pylist() for name in features}
    labels = table.column("label").to_pylist()

    accumulators = [Welford() for _ in features]
    counts = Counter()
    rows = [[] for _ in schema]
    parts = 0

    def flush():
        nonlocal parts, rows
        if rows[0]:
            pq.write_table(
                pa.table(rows, schema=schema),
                out_dir / f"part-{parts:08d}.parquet",
                row_group_size=PART_ROWS,
            )
            parts += 1
            rows = [[] for _ in schema]

    for i, split in enumerate(split_of_row):
        values = [columns[name][i] for name in features]
        if split == TRAIN:
            for acc, v in zip(accumulators, values):
                acc.add(v)
        counts[split] += 1
        counts[(split, labels[i])] += 1
        for col, v in zip(rows, values):
            col.append(v)
        rows[-2].append(labels[i])
        rows[-1].append(split)
        if len(rows[0]) >= PART_ROWS:
            flush()
    flush()

    stats = {
        "features": list(features),
        "mean": [a.mean for a in accumulators],
        # A constant feature would divide by zero when standardizing; the
        # engine treats a variance of 0 as 1 and the feature stays at 0.
        "variance": [a.variance for a in accumulators],
        "train_rows": accumulators[0].n,
        "counts": {
            "train": counts[TRAIN], "valid": counts[VALID], "test": counts[TEST],
            "test_positive": counts[(TEST, 1)], "test_negative": counts[(TEST, 0)],
            "train_positive": counts[(TRAIN, 1)], "train_negative": counts[(TRAIN, 0)],
        },
        **extra,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def groups_for(table, grouping):
    """Returns one group id per row, under the chosen grouping."""
    if grouping == "scaffold":
        return table.column("scaffold").to_pylist()
    if grouping == "butina":
        return butina_groups(table.column("smiles").to_pylist())
    raise ValueError(f"unknown grouping: {grouping}")


def build(features_path, out_root, scheme, grouping, k, repeat):
    """Writes the split directories for one repeat; returns (sizes, stats)."""
    table = pq.read_table(features_path)
    features = [n for n in table.column_names if n not in ("label", "scaffold", "smiles")]
    groups = groups_for(table, grouping)
    sizes = Counter(groups)

    if scheme == "holdout":
        assignment, filled = assign_holdout(sizes, repeat)
        split_of_row = [assignment[g] for g in groups]
        stats = write_holdout(table, features, split_of_row, out_root / f"s{repeat}")
        stats["grouping"] = grouping
        return filled, [stats]

    assignment, filled = assign_folds(sizes, k, repeat)
    fold_of_row = [assignment[g] for g in groups]
    out = []
    for test_fold in range(k):
        valid_fold = (test_fold + 1) % k
        split_of_row = [
            TEST if f == test_fold else (VALID if f == valid_fold else TRAIN)
            for f in fold_of_row
        ]
        stats = _write(
            table, features, split_of_row, out_root / f"r{repeat}f{test_fold}",
            {"scheme": "kfold", "grouping": grouping, "folds": k,
             "test_fold": test_fold, "valid_fold": valid_fold},
        )
        out.append(stats)
    return filled, out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("features", type=Path, help="features.parquet from descriptors.py")
    parser.add_argument("--scheme", choices=["kfold", "holdout"], default="kfold")
    parser.add_argument("--grouping", choices=["scaffold", "butina"], default="scaffold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=0, help="seed for the assignment")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args()

    out_root = args.out_root or args.features.parent
    filled, stats = build(
        args.features, out_root, args.scheme, args.grouping, args.folds, args.repeat
    )
    total = sum(filled)
    print(f"repeat {args.repeat} [{args.scheme}/{args.grouping}]: "
          f"sizes {filled} ({[round(f/total, 3) for f in filled]})")
    for s in stats:
        c = s["counts"]
        tag = f"fold {s['test_fold']}" if "test_fold" in s else "holdout"
        print(f"  {tag}: train={c['train']:5} valid={c['valid']:4} "
              f"test={c['test']:4} (pos {c['test_positive']}, neg {c['test_negative']})")


if __name__ == "__main__":
    main()
