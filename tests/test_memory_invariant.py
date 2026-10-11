"""M0 acceptance test: working memory must not depend on data volume.

Builds the engine, ingests data/raw/, computes descriptors, scales the
descriptor part to 1x / 10x / 100x and runs the engine on each. The scaled
input is in the pipeline's own format, so the numbers describe the engine as
it actually runs rather than a format kept only for this test. Every scale must stay within +-10% of the
1x peak RSS.

Peak RSS is the larger of two kernel readings: VmHWM reported by the engine
(engine/src/metrics.rs) and GNU time's %M. wait4(2) called from this process
is not usable: subprocess spawns through vfork, so the child's high-water mark
already includes this Python interpreter before exec.

    nix-shell --run 'python3 -m unittest -v tests/test_memory_invariant.py'

OCCAM_SOURCE selects the ingested table (default: B3DB_classification).
OCCAM_FACTORS overrides the scales, e.g. OCCAM_FACTORS="1 10 100".
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from hypothesis import given, settings, strategies as st

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
WORK_DIR = ROOT / "data" / "work"
ENGINE_DIR = ROOT / "engine"
ENGINE = ENGINE_DIR / "target" / "release" / "engine"
GNU_TIME = shutil.which("time") or ("/usr/bin/time" if os.path.exists("/usr/bin/time") else None)

FACTORS = tuple(sorted({1, *map(int, os.environ.get("OCCAM_FACTORS", "1 10 100 1000").split())}))
SOURCE_STEM = os.environ.get("OCCAM_SOURCE", "B3DB_classification")
LABEL_COLUMN = "BBB+/BBB-"
POSITIVE = "BBB+"
DESC_DIR = WORK_DIR / f"{SOURCE_STEM}_desc"
STATS = DESC_DIR / "stats.json"
TOLERANCE = 0.10
BULK_SIZE = 1024
# The 100x input must dwarf the peak RSS, otherwise a flat RSS proves nothing.
MIN_DATA_TO_RSS_RATIO = 10

SCALED = {}


def sh(*cmd, cwd=ROOT):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def raw_digest():
    """Fingerprint of every file under data/raw/, which must never change.

    Walks recursively: data/raw/ holds one directory per dataset, so reading
    only its top level would try to read a directory as bytes.
    """
    h = hashlib.sha256()
    for path in sorted(p for p in RAW_DIR.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(RAW_DIR)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def run_engine(path, bulk_size):
    """Runs the engine once; returns its key=value report plus peak RSS in KiB."""
    with tempfile.NamedTemporaryFile(mode="r") as timing:
        cmd = [ENGINE, path, "--bulk-size", str(bulk_size), "--stats", str(STATS)]
        if GNU_TIME:
            cmd = [GNU_TIME, "-f", "%M", "-o", timing.name, *cmd]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"engine failed on {path}:\n{result.stderr}")
        time_kb = int(timing.read().split()[-1]) if GNU_TIME else 0
    report = dict(kv.split("=", 1) for kv in result.stderr.split() if "=" in kv)
    hwm_kb = int(report["peak_rss_kb"])
    report["time_kb"] = time_kb
    report["maxrss_kb"] = max(hwm_kb, time_kb)
    return report


def setUpModule():
    before = raw_digest()
    sh("cargo", "build", "--release", "--quiet", cwd=ENGINE_DIR)
    sh(sys.executable, "py/ingest.py")
    if raw_digest() != before:
        raise RuntimeError("data/raw/ changed during ingest")

    # Descriptors, then scale the single descriptor part. The engine's input
    # is always a directory of one-row-group parts.
    if not STATS.exists():
        sh(
            sys.executable, "py/descriptors.py", WORK_DIR / f"{SOURCE_STEM}.parquet",
            "--label-column", LABEL_COLUMN, "--positive-label", POSITIVE,
            "--out-dir", DESC_DIR,
        )
    part = DESC_DIR / "part-00000000.parquet"
    sh(sys.executable, "py/gen_scale.py", part, "--factors", *map(str, FACTORS))
    for n in FACTORS:
        SCALED[n] = WORK_DIR / f"{part.stem}_x{n}"


class MemoryInvariant(unittest.TestCase):
    def test_engine_unit_tests(self):
        # Carry size is a compile-time constant; two live bulks panic.
        sh("cargo", "test", "--release", "--quiet", cwd=ENGINE_DIR)

    def test_peak_rss_flat_across_scales(self):
        runs = {n: run_engine(SCALED[n], BULK_SIZE) for n in FACTORS}
        base = runs[1]["maxrss_kb"]

        print(f"\n  input: {SCALED[1].name[:-3]}  bulk_size={BULK_SIZE}", file=sys.stderr)
        width = len(str(FACTORS[-1]))
        for n, r in runs.items():
            delta = (r["maxrss_kb"] - base) / base
            print(
                f"  x{n:<{width}} rows={r['rows']:>9} bulks={r['bulks']:>5} "
                f"payload={int(r['payload_bytes']) / 2**20:8.1f} MiB  "
                f"peak_rss={r['maxrss_kb'] / 1024:6.2f} MiB ({delta:+.1%})  "
                f"[VmHWM={r['peak_rss_kb']} KiB, time={r['time_kb'] or '-'} KiB]",
                file=sys.stderr,
            )

        # The engine really read n times the data.
        for n in FACTORS:
            self.assertEqual(int(runs[n]["rows"]), n * int(runs[1]["rows"]))
            self.assertEqual(int(runs[n]["carry_bytes"]), int(runs[1]["carry_bytes"]))

        largest = runs[FACTORS[-1]]
        ratio = int(largest["payload_bytes"]) / (largest["maxrss_kb"] * 1024)
        print(f"  x{FACTORS[-1]} data / peak RSS = {ratio:.0f}x", file=sys.stderr)
        self.assertGreaterEqual(
            int(largest["payload_bytes"]),
            MIN_DATA_TO_RSS_RATIO * largest["maxrss_kb"] * 1024,
            "100x input is too small to tell flat RSS from proportional RSS",
        )

        for n in FACTORS[1:]:
            delta = (runs[n]["maxrss_kb"] - base) / base
            self.assertLessEqual(abs(delta), TOLERANCE, f"x{n} peak RSS {delta:+.1%} vs x1")

    @settings(max_examples=8, deadline=None, database=None)
    @given(bulk_size=st.integers(min_value=1, max_value=8192))
    def test_property_flat_for_any_bulk_size(self, bulk_size):
        # Bulk size moves M (D / N term) but must never couple it to D.
        small = run_engine(SCALED[1], bulk_size)["maxrss_kb"]
        large = run_engine(SCALED[10], bulk_size)["maxrss_kb"]
        delta = (large - small) / small
        self.assertLessEqual(abs(delta), TOLERANCE, f"bulk_size={bulk_size}: x10 {delta:+.1%} vs x1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
