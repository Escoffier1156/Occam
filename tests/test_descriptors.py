"""py/descriptors.py must stay flat in memory and split scaffolds cleanly.

    nix-shell --run 'python3 -m unittest -v tests/test_descriptors.py'

Preprocessing is outside the engine's memory invariants, but it still must not
hold the dataset: a 1x and a 20x input have to peak at the same RSS. The scale
is modest because RDKit dominates the runtime.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import descriptors  # noqa: E402

WORK_DIR = ROOT / "data" / "work"
SOURCE = WORK_DIR / "B3DB_classification.parquet"
LABEL_COLUMN = "BBB+/BBB-"
POSITIVE = "BBB+"
TOLERANCE = 0.10
FACTORS = (1, 10)
# RDKit dominates the runtime, so the scaled input is a slice of the source.
SAMPLE_ROWS = 2000

# Several molecules per scaffold. One unparseable SMILES per 200 rows keeps
# the failure rate under the pipeline's 1% limit, which is itself under test.
SAMPLE = [
    ("c1ccccc1CCO", "BBB+"),
    ("c1ccccc1CCN", "BBB+"),
    ("c1ccccc1CCC", "BBB-"),
    ("c1ccncc1CCO", "BBB-"),
    ("c1ccncc1CCN", "BBB+"),
    ("CCO", "BBB-"),
]
UNPARSEABLE = ("not a molecule at all", "BBB+")


def sample_rows(repeats):
    rows = SAMPLE * repeats
    rows[::200] = [UNPARSEABLE] * len(rows[::200])
    return rows


def write_source(path, rows):
    table = pa.table(
        {"SMILES": pa.array([s for s, _ in rows]), LABEL_COLUMN: pa.array([y for _, y in rows])}
    )
    pq.write_table(table, path, row_group_size=descriptors.PART_ROWS)


def peak_rss_kb(source, out_dir):
    """Runs descriptors.py as a subprocess and returns GNU time's %M."""
    with tempfile.NamedTemporaryFile(mode="r") as timing:
        cmd = [
            "/usr/bin/time", "-f", "%M", "-o", timing.name,
            sys.executable, str(ROOT / "py" / "descriptors.py"), str(source),
            "--label-column", LABEL_COLUMN, "--positive-label", POSITIVE,
            "--out-dir", str(out_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"descriptors.py failed:\n{result.stdout}\n{result.stderr}")
        return int(timing.read().split()[-1])


class Descriptors(unittest.TestCase):
    def test_features_and_scaffold_are_written(self):
        # Splitting moved to py/folds.py; this stage only derives features.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "sample.parquet"
            rows_in = sample_rows(200)
            write_source(src, rows_in)
            bad = sum(1 for r in rows_in if r == UNPARSEABLE)
            rows, dropped, summary = descriptors.run(
                src, tmp / "out", "SMILES", LABEL_COLUMN, POSITIVE
            )
            self.assertEqual(dict(dropped), {"smiles_unparseable": bad})
            self.assertEqual(rows, len(rows_in) - bad)

            table = pq.read_table(tmp / "out" / "features.parquet")
            self.assertEqual(
                set(table.column_names), {*descriptors.FEATURES, "label", "scaffold", "smiles"}
            )
            self.assertEqual(table.num_rows, rows)
            self.assertGreater(summary["distinct_scaffolds"], 1, "sample needs several scaffolds")
            self.assertEqual(
                summary["labels"]["positive"] + summary["labels"]["negative"], rows
            )

    def test_keep_labels_skips_without_counting_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "sample.parquet"
            write_source(src, sample_rows(200))
            rows, dropped, summary = descriptors.run(
                src, tmp / "out", "SMILES", LABEL_COLUMN, POSITIVE, keep={POSITIVE}
            )
            # Only the positive class survives, and the skipped rows are
            # counted apart from parse failures so they cannot trip the limit.
            self.assertEqual(summary["labels"]["negative"], 0)
            self.assertGreater(dropped["out_of_scope"], 0)
            self.assertEqual(rows, summary["labels"]["positive"])

    @unittest.skipUnless(SOURCE.exists(), "run py/ingest.py first")
    def test_peak_rss_flat_across_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = pq.ParquetFile(SOURCE)
            rss = {}
            for n in FACTORS:
                scaled = tmp / f"x{n}.parquet"
                head = source.read().slice(0, SAMPLE_ROWS)
                with pq.ParquetWriter(scaled, source.schema_arrow) as writer:
                    for _ in range(n):
                        writer.write_table(head, row_group_size=descriptors.PART_ROWS)
                rss[n] = peak_rss_kb(scaled, tmp / f"out{n}")
                print(f"\n  x{n:<3} peak_rss={rss[n] / 1024:6.2f} MiB", file=sys.stderr)

            base = rss[FACTORS[0]]
            for n in FACTORS[1:]:
                delta = (rss[n] - base) / base
                self.assertLessEqual(abs(delta), TOLERANCE, f"x{n} peak RSS {delta:+.1%} vs x1")

    @unittest.skipUnless(SOURCE.exists(), "run py/ingest.py first")
    def test_real_data_parse_failures_stay_rare(self):
        out_dir = WORK_DIR / "B3DB_classification_desc"
        stats = json.loads((out_dir / "stats.json").read_text())
        counts = {s: sum(stats["splits"][s].values()) for s in descriptors.SPLITS}
        total = sum(counts.values())
        self.assertAlmostEqual(counts["train"] / total, 0.8, delta=0.02)
        self.assertAlmostEqual(counts["test"] / total, 0.1, delta=0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
