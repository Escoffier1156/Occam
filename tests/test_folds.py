"""py/folds.py must keep a group whole and compute statistics on train rows only.

    nix-shell --run 'python3 -m unittest -v tests/test_folds.py'
"""

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import folds  # noqa: E402

FEATURES = ["f0", "f1"]
GROUPS = [f"g{i}" for i in range(20)]


def make_features(path, per_group=5):
    rows = []
    for gi, g in enumerate(GROUPS):
        for j in range(per_group):
            rows.append((float(gi), float(j), gi % 2, g, f"C{'C' * (gi + 1)}"))
    table = pa.table(
        {
            "f0": pa.array([r[0] for r in rows], pa.float32()),
            "f1": pa.array([r[1] for r in rows], pa.float32()),
            "label": pa.array([r[2] for r in rows], pa.int8()),
            "scaffold": pa.array([r[3] for r in rows]),
            "smiles": pa.array([r[4] for r in rows]),
        }
    )
    pq.write_table(table, path, row_group_size=8192)
    return table


class Folds(unittest.TestCase):
    def test_group_never_straddles_folds(self):
        sizes = Counter({f"g{i}": (50 if i < 2 else 5) for i in range(20)})
        assignment, filled = folds.assign_folds(sizes, 5, 0)
        self.assertEqual(sum(filled), sum(sizes.values()))
        # One fold per group, and every group placed.
        self.assertEqual(set(assignment), set(sizes))
        self.assertTrue(all(0 <= f < 5 for f in assignment.values()))

    def test_holdout_reduces_to_three_splits(self):
        sizes = Counter({f"g{i}": 5 for i in range(20)})
        assignment, filled = folds.assign_holdout(sizes, 0)
        self.assertEqual(len(filled), 3)
        self.assertEqual(sum(filled), 100)
        self.assertAlmostEqual(filled[0] / 100, 0.8, delta=0.1)

    def test_stats_use_train_rows_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            table = make_features(tmp / "features.parquet")
            _, stats = folds.build(tmp / "features.parquet", tmp, "kfold", "scaffold", 5, 0)
            self.assertEqual(len(stats), 5)
            for s in stats:
                out = tmp / f"r0f{s['test_fold']}"
                written = pa.concat_tables(
                    [pq.read_table(f) for f in sorted(out.glob("part-*.parquet"))]
                )
                train = written.filter(pa.compute.equal(written.column("split"), folds.TRAIN))
                self.assertEqual(s["train_rows"], train.num_rows)
                for i, name in enumerate(FEATURES):
                    values = train.column(name).to_pylist()
                    self.assertAlmostEqual(
                        s["mean"][i], sum(values) / len(values), places=4, msg=name
                    )
                # Validation and test are disjoint and cover the fold exactly.
                self.assertNotEqual(s["test_fold"], s["valid_fold"])
                self.assertEqual(
                    s["counts"]["train"] + s["counts"]["valid"] + s["counts"]["test"],
                    written.num_rows,
                )

    def test_every_row_is_tested_once_per_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_features(tmp / "features.parquet")
            _, stats = folds.build(tmp / "features.parquet", tmp, "kfold", "scaffold", 5, 0)
            self.assertEqual(sum(s["counts"]["test"] for s in stats), 100)

    def test_butina_refuses_an_unaffordable_matrix(self):
        with self.assertRaises(ValueError) as ctx:
            folds.butina_groups(["CCO"] * (folds.BUTINA_MAX_MOLECULES + 1))
        self.assertIn("scaffold", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
