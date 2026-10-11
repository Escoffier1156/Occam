"""The invariants must survive edits that look like improvements.

Three properties hold this project up, and a change that breaks one is
rejected even if it makes the engine faster. That rule cannot be enforced by
writing it down: `fetch_add` instead of `compare_exchange` is a correct-looking
optimisation, and deleting an assert that has never fired is a reasonable
tidy-up. Both would end the guarantee silently.

KeepOut marks those regions and refuses structural edits to them, ignoring
formatting, comments and renames. This runs it, and also pins which regions
must exist: `keepout init` rewrites the database from whatever markers it
finds, so removing a marker and re-initialising would otherwise retire a lock
without a trace.

    nix-shell --run 'python3 -m unittest -v tests/test_invariants_locked.py'

Set OCCAM_ALLOW_UNLOCKED=1 to run the rest of the suite without KeepOut
present. It prints what is unprotected rather than skipping quietly.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "keepout.json"
KEEPOUT_SRC = os.environ.get("KEEPOUT_SRC", "/home/shogo/KeepOut/src")

# Every region that must stay locked, and what it is there for. A lock going
# missing is a finding, not a detail.
REQUIRED_LOCKS = {
    "engine/src/carry.rs#lock_0":
        "invariant 1: Carry is Copy, so it can hold nothing heap-allocated",
    "engine/src/bulk.rs#lock_0":
        "invariant 3: a second live bulk is refused atomically",
    "engine/src/bulk.rs#lock_1":
        "invariant 2: one row group per file, so the footer cannot grow with D",
    "engine/src/process.rs#lock_0":
        "invariant 3: a second live operator is refused atomically",
    "engine/src/main.rs#lock_0":
        "the loop: bulk and operator dropped each iteration, only carry survives",
    "py/m3_order.py#lock_0":
        "the ranking guard that caught saturation degenerating into file order",
    "py/descriptors.py#lock_0":
        "the parse-failure limit that caught a SMILES dialect",
    "py/candidates.py#lock_0":
        "the deterministic tie-break that kept deduplication off file order",
    "py/smiles.py#lock_0":
        "the two steps that keep the repaired parse equal to RDKit's own "
        "default: RemoveHs, and the stereochemistry assignment the InChIKey "
        "tie-break depends on",
}


def keepout_available():
    return (Path(KEEPOUT_SRC) / "keepout" / "cli.py").exists()


class InvariantsLocked(unittest.TestCase):
    def test_keepout_is_present(self):
        if keepout_available():
            return
        if os.environ.get("OCCAM_ALLOW_UNLOCKED"):
            print(
                f"\n  WARNING: KeepOut not found at {KEEPOUT_SRC}. The memory "
                f"invariants are unprotected in this run:\n    "
                + "\n    ".join(REQUIRED_LOCKS.values()),
                file=sys.stderr,
            )
            return
        self.fail(
            f"KeepOut not found at {KEEPOUT_SRC}. It is what stops a tidy-up from "
            f"removing a guard the invariants depend on. Set KEEPOUT_SRC to its "
            f"src directory, or OCCAM_ALLOW_UNLOCKED=1 to run without it."
        )

    def test_every_required_region_is_still_locked(self):
        self.assertTrue(DB.exists(), f"{DB} missing; run `keepout init .`")
        locks = json.loads(DB.read_text()).get("locks", {})
        for key, why in REQUIRED_LOCKS.items():
            with self.subTest(lock=key):
                # assertIn would print the whole database; say what is missing.
                if key not in locks:
                    self.fail(f"lock retired: {key} -- {why}")
                self.assertEqual(locks[key]["mode"], "ast", f"{key} is no longer an ast lock")
                self.assertTrue(locks[key].get("ast_hash"), f"{key} has no hash recorded")

    @unittest.skipUnless(keepout_available(), "KeepOut not installed")
    def test_no_locked_region_has_been_altered(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = KEEPOUT_SRC + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-m", "keepout.cli", "check", str(ROOT)],
            capture_output=True, text=True, env=env, cwd=ROOT,
        )
        if result.returncode != 0:
            violations = [
                l for l in (result.stdout + result.stderr).splitlines()
                if "VIOLATION" in l or "MISSING LOCK" in l
            ]
            self.fail(
                "a locked region changed shape:\n  " + "\n  ".join(violations)
                + "\n\nIf the change is deliberate, say so in the lock comment "
                  "(UNLOCK_REASON: ...) and re-run `keepout init .`."
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
