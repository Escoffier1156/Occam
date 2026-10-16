"""M0 acceptance test: working memory must not depend on data volume.

Builds the engine, ingests data/raw/, computes descriptors, scales the
descriptor part to 1x / 10x / 100x and runs the engine on each. The scaled
input is in the pipeline's own format, so the numbers describe the engine as
it actually runs rather than a format kept only for this test. Every scale must stay within +-10% of the
1x peak RSS.

Peak RSS is VmHWM, read by the engine from /proc/self/status
(engine/src/metrics.rs). GNU time's %M is read too and printed beside it, as a
cross-check on that self-report, but it does not set the verdict. Which
instrument decides is argued at the assignment in run_engine().

wait4(2) called from this process is not usable at all: subprocess spawns
through vfork, so the child's high-water mark already includes this Python
interpreter before exec (contraindication 2).

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
# How far GNU time's %M may sit from the engine's own VmHWM before the
# self-report is the suspect rather than the noise. /usr/bin/time -f %M on
# `true` reports 1,428-1,540 KiB, which is the wrapper's own footprint and the
# floor under every %M reading; 2 MiB covers it. The two readings were never
# observed more than 180 KiB apart, so this fires on a self-report that is
# wrong by a factor and not on measurement noise.
CROSS_CHECK_SLACK_KB = 2048
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
    # ONE INSTRUMENT DECIDES, AND WHY IT IS THIS ONE
    #
    # This was max(hwm_kb, time_kb) -- "the larger of two kernel readings",
    # stated in the module docstring. That was conservatism, not a fallback:
    # VmHWM is already mandatory (int() on "unknown" would raise) and %M is
    # already optional (0 without GNU time), so %M was never standing in for
    # a missing VmHWM.
    #
    # Conservatism is right for an absolute bound -- how big did it get, at
    # most. This test asserts a RATIO between two runs, and max() of two
    # readings that straddle each other is biased upward independently in the
    # numerator and the denominator, so the ratio carries both inflations.
    #
    # Measured, 10 runs at each scale: max() picked VmHWM 5 times and %M 5
    # times at BOTH scales, and %M - VmHWM ran -180 to +136 KiB. VmHWM spread
    # 140-200 KiB, %M spread 284-384 KiB. Worst ratio the instrument alone
    # can produce: with max(), 4,332/3,956 = +9.5%, and +11.4% was observed
    # against a 10% tolerance; with VmHWM alone, 4,208/3,964 = +6.2%.
    #
    # So the tolerance was never the thing to move. Contraindication 109.
    report["maxrss_kb"] = hwm_kb
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
        """Carry size is a compile-time constant; two live bulks panic.

        ONE TEST THREAD, AND WHY THE GUARD IS NOT THE THING TO CHANGE

        LIVE_BULKS is a process-wide AtomicUsize, which is the right grain:
        the production path is one loop, and a counter that every thread can
        see is what makes "two bulks alive at once" detectable at all.

        cargo runs tests in parallel by default, and bulk.rs's own
        `one_bulk_at_a_time` test deliberately holds a Bulk while trying to
        build a second. Any other test constructing a Bulk on another thread
        during that window trips the assert -- a false positive from the test
        runner, not a violation. It fired once here, as
        `process::tests::learning_and_scoring` panicking inside
        `bulk.rs:49`, and did not reproduce in ten further runs.

        Making the counter thread-local would silence it by weakening the
        guard in production, where the threads do not exist. Serialising the
        tests costs nothing and leaves the guard at full strength.

        The failure was only diagnosable because the run's output was in a
        file (contraindication 48). Standard output alone would have lost the
        traceback, and ten clean reruns would have closed it as a fluke.
        """
        sh("cargo", "test", "--release", "--quiet", "--",
           "--test-threads=1", cwd=ENGINE_DIR)

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

        # The engine's self-report is the instrument now, so something has to
        # check the instrument. %M cannot set the verdict on a ratio, but a
        # VmHWM that has drifted free of it is not noise.
        if GNU_TIME:
            for n, r in runs.items():
                self.assertLessEqual(
                    abs(r["time_kb"] - int(r["peak_rss_kb"])), CROSS_CHECK_SLACK_KB,
                    f"x{n}: VmHWM {r['peak_rss_kb']} KiB and GNU time's %M "
                    f"{r['time_kb']} KiB disagree by more than the wrapper's "
                    f"own footprint -- the self-report is the suspect",
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
