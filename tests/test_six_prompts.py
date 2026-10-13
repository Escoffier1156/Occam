"""The six prompts, as a regression.

Six requests were run in order, repeatedly, through a day of changes. Most of
what was found that day was found by reading the six outputs side by side: a
library selector that answered about the wrong slice, a recommendation whose
ordering preferred the largest overshoot, a ranking verdict that passed on one
specification and failed on four otherwise identical ones. None of it was
visible in a single run. All of it was visible in six.

So the six become a test, in two layers, because they cost two different
things.

  THE REPLAY, which runs in the default suite. Each run's records --
  request.json, proposal.json, screenable.json, what_to_measure.json -- are
  stored under tests/fixtures/six_prompts/ and pushed back through the
  reporting layer. It fixes the screen: what a reader is shown about records
  that are held fixed. Every display fault found that day would have failed
  here, and it needs no library, no engine and no model.

  THE RERUN, which does not. Set OCCAM_SIX=1 and it runs the six requests for
  real and compares the verdicts against the fixtures. That is the layer that
  catches a change in the measurement rather than in its presentation, and it
  takes about fifty minutes.

    nix-shell --run 'OCCAM_SIX=1 python3 -m unittest tests/test_six_prompts.py'

WHAT THE FIXTURES ARE, AND WHAT THEY ARE NOT

They are records of one machine on one day against one library. They are not
ground truth about chemistry, and a test comparing against them says only
that the tool still answers what it answered. When a change is meant to move
a verdict -- as the sampling asymmetry moved run 6 from a pass to not settled
-- the fixtures are re-recorded and the commit says which verdict moved and
why. A fixture updated without that sentence is the failure this file exists
to make visible.

To re-record, after a change whose effect on the six is understood:

    tests/fixtures/six_prompts/RECORD.md has the commands and the prompts.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "six_prompts"
sys.path.insert(0, str(ROOT / "py"))

RERUN = os.environ.get("OCCAM_SIX") == "1"

# What each run is for. The prompt text lives in the fixture's request.json
# under "goal", so it is not written twice.
RUNS = {
    "run1": "a request the library can fill as asked",
    "run2": "a request naming no library, and a condition this pipeline "
            "cannot evaluate",
    "run3": "a request that needs one setting loosened",
    "run4": "the same, with a ranking request and a word that looks like a "
            "target",
    "run5": "the same, with a ranking request and no such word",
    "run6": "a request that needs two settings loosened, and whose survivor "
            "set is the largest of the six",
}


def load(name):
    """One run's records. Missing files are returned as None: run 2 stops
    before a proposal exists, which is the point of run 2."""
    out = {}
    for stem in ("request", "proposal", "screenable", "what_to_measure"):
        path = FIXTURES / name / f"{stem}.json"
        out[stem] = json.loads(path.read_text()) if path.exists() else None
    return out


@unittest.skipUnless(FIXTURES.is_dir(), f"no fixtures at {FIXTURES}")
class TheRecordsAreComplete(unittest.TestCase):
    """A fixture set that is missing a run passes every test below by
    skipping it, so the set itself is checked first."""

    def test_every_run_has_a_directory(self):
        for name in RUNS:
            self.assertTrue((FIXTURES / name).is_dir(), f"{name} is missing")

    def test_every_run_records_the_request_it_came_from(self):
        for name in RUNS:
            spec = load(name)["request"]
            self.assertIsNotNone(spec, f"{name} has no request.json")
            self.assertTrue(spec.get("goal"), f"{name} records no prompt")

    def test_the_run_that_stops_has_no_proposal(self):
        """Run 2 names no library. It must stop before proposing, and a
        fixture with a proposal in it would mean that stop was lost."""
        self.assertIsNone(load("run2")["proposal"])

    def test_every_other_run_has_a_proposal(self):
        for name in RUNS:
            if name == "run2":
                continue
            self.assertIsNotNone(load(name)["proposal"],
                                 f"{name} has no proposal.json")


@unittest.skipUnless(FIXTURES.is_dir(), f"no fixtures at {FIXTURES}")
class TheScreenIsReproducibleFromTheRecords(unittest.TestCase):
    """The reporting layer, against records held fixed.

    Every display fault found by reading the six would have failed here: a
    verdict reaching the screen as an identifier, a count printed beside a
    count from a different scope, a near miss computed and then dropped, a
    sentence with a hole in it.
    """

    def _answer(self, name):
        import render as render_mod

        rec = load(name)
        if rec["proposal"] is None:
            return None
        return "\n".join(render_mod.answer(
            rec["request"], rec["proposal"], body=rec["screenable"],
            cost=rec["what_to_measure"]))

    def test_the_answer_block_renders_for_every_run(self):
        for name in RUNS:
            if name == "run2":
                continue
            text = self._answer(name)
            self.assertIn("ANSWER", text, f"{name} produced no answer block")
            self.assertIn("No molecules are emitted", text, name)

    def test_no_verdict_reaches_the_screen_as_an_identifier(self):
        """A verdict this layer has no phrase for degrades to its key with
        the underscores taken out, which is a record shape on the screen."""
        import render as render_mod

        for name in RUNS:
            rec = load(name)
            body = rec["screenable"]
            if not body:
                continue
            for key in ("rankability", "applicability"):
                value = (body.get(key) or {}).get("verdict")
                if value is None:
                    continue
                self.assertIn(value, render_mod.VERDICT,
                              f"{name}: {key} verdict {value!r} has no phrase")

    def test_a_ranking_request_is_answered_where_one_was_made(self):
        """Runs 4 and 5 ask for a top 50. The ordering verdict is the answer
        to that, and it used to be printed thirty lines away from it."""
        for name in ("run4", "run5"):
            spec = load(name)["request"]
            self.assertEqual(spec.get("asked_to_rank", {}).get("top_n"), 50,
                             f"{name} lost its ranking request")
            text = self._answer(name)
            self.assertIn("you asked for the top 50", text, name)

    def test_no_ranking_answer_where_none_was_asked(self):
        for name in ("run1", "run3", "run6"):
            self.assertNotIn("you asked for", self._answer(name), name)

    def test_the_word_that_looks_like_a_target_is_reported(self):
        """Run 4 contains `purchase`. It ends in -ase, no label mentions it,
        and promoting it once stopped a run before anything was screened."""
        spec = load("run4")["request"]
        self.assertIn("purchase", spec.get("target_candidates") or [])

    def test_a_near_miss_is_reported_where_one_exists(self):
        """Run 6 asked for 500 and the closest exactly-counted specification
        held 494. That figure was being computed and discarded."""
        proposal = load("run6")["proposal"]
        miss = proposal.get("nearest_miss")
        self.assertIsNotNone(miss, "the near miss vanished again")
        self.assertEqual(miss["short_by"],
                         proposal["request"]["distinct_scaffolds_required"]
                         - miss["scaffolds_exact"])
        self.assertIn(f"{miss['scaffolds_exact']:,}", self._answer("run6"))

    def test_a_scaffold_count_never_exceeds_what_it_was_counted_from(self):
        """One row read "123 molecules pass, 536 distinct scaffolds", which
        cannot happen: the two figures came from passes over different
        amounts of the library."""
        for name in RUNS:
            proposal = load(name)["proposal"]
            if not proposal:
                continue
            rows = [proposal.get("recommended"), proposal.get("as_asked"),
                    *(proposal.get("frontier") or []),
                    proposal.get("nearest_miss")]
            for row in [r for r in rows if r]:
                exact = (row.get("exact") or {}).get("scaffolds_exact")
                if exact is None:
                    exact = row.get("scaffolds_exact")
                if exact is None or row.get("survivors") is None:
                    continue
                self.assertLessEqual(
                    exact, row["survivors"],
                    f"{name}: {exact} scaffolds from {row['survivors']} "
                    f"molecules")


@unittest.skipUnless(FIXTURES.is_dir(), f"no fixtures at {FIXTURES}")
class TheVerdictsAcrossTheSix(unittest.TestCase):
    """What the six say when read side by side. These are the comparisons
    that found the faults, and a single run cannot make them."""

    def _rank(self, name):
        body = load(name)["screenable"]
        return (body or {}).get("rankability") or {}

    def test_no_ordering_verdict_passes_on_a_sample(self):
        """The fault this layer exists for. A distinct share rises as fewer
        molecules are measured, so a sampled figure cannot clear the floor --
        and of the six, the one specification whose survivor set exceeded the
        sample was the only one reported rankable."""
        for name in RUNS:
            rank = self._rank(name)
            if not rank or rank.get("verdict") == "not_measurable":
                continue
            if rank.get("verdict") != "answerable":
                continue
            self.assertEqual(
                rank.get("sampled"), rank.get("of_survivors"),
                f"{name} says the model can order a set it only sampled")

    def test_a_sampled_verdict_says_what_it_rests_on(self):
        """Both directions lean on the sampled share bounding the full-set
        share, which is observed and not proven."""
        for name in RUNS:
            rank = self._rank(name)
            if not rank or not rank.get("sampled"):
                continue
            if rank["sampled"] == rank.get("of_survivors"):
                self.assertNotIn("rests_on", rank,
                                 f"{name} discloses an assumption it is not "
                                 f"making")
            else:
                self.assertIn("rests_on", rank,
                              f"{name} relies on the bound without saying so")

    def test_the_floor_is_disclosed_wherever_it_decides_something(self):
        for name in RUNS:
            rank = self._rank(name)
            if not rank or "distinct_share" not in rank:
                continue
            self.assertIn("declared, not derived", rank.get("floor_basis", ""),
                          f"{name} prints the floor without its basis")

    def test_the_four_identical_requests_agree(self):
        """Runs 3, 4 and 5 differ only in what they ask for on top of the
        same specification. Their measurements must be identical, and when
        they were not it was because one of them had been sampled."""
        seen = {name: (self._rank(name).get("distinct_share"),
                       self._rank(name).get("of_survivors"))
                for name in ("run3", "run4", "run5")}
        self.assertEqual(len(set(seen.values())), 1,
                         f"the same specification measured differently: {seen}")


@unittest.skipUnless(RERUN, "set OCCAM_SIX=1 to run the six for real")
class TheSixStillAnswerWhatTheyAnswered(unittest.TestCase):
    """The slow layer. Runs each request against the real library and
    compares the verdicts with the fixtures.

    Only the verdicts and the counts the verdicts turn on. Timings move,
    version stamps move, and a test that compares whole transcripts fails on
    every commit and gets switched off.
    """

    @classmethod
    def setUpClass(cls):
        import tempfile

        cls.tmp = tempfile.mkdtemp(prefix="occam-six-")
        cls.fresh = {}
        for name in RUNS:
            spec = load(name)["request"]
            out = Path(cls.tmp) / name
            result = subprocess.run(
                [sys.executable, "py/occam.py", "--yes", "--out-dir", str(out),
                 spec["goal"]],
                cwd=ROOT, capture_output=True, text=True)
            records = {}
            for stem in ("request", "proposal", "screenable"):
                path = out / f"{stem}.json"
                records[stem] = (json.loads(path.read_text())
                                 if path.exists() else None)
            records["stdout"] = result.stdout
            cls.fresh[name] = records

    def test_the_library_each_request_selects_has_not_moved(self):
        """A selector fault answered about a different slice and finished
        normally, which is the failure mode no verdict catches."""
        for name in RUNS:
            self.assertEqual(self.fresh[name]["request"].get("library"),
                             load(name)["request"].get("library"), name)

    def test_the_recommended_specification_has_not_moved(self):
        for name in RUNS:
            if name == "run2":
                continue
            was = (load(name)["proposal"].get("recommended") or {})
            now = (self.fresh[name]["proposal"].get("recommended") or {})
            self.assertEqual(now.get("specification"),
                             was.get("specification"), name)

    def test_the_counts_the_verdict_turns_on_have_not_moved(self):
        for name in RUNS:
            if name == "run2":
                continue
            was = (load(name)["proposal"].get("recommended") or {})
            now = (self.fresh[name]["proposal"].get("recommended") or {})
            for key in ("survivors", "scaffolds_estimated"):
                self.assertEqual(now.get(key), was.get(key), f"{name}.{key}")
            self.assertEqual((now.get("exact") or {}).get("scaffolds_exact"),
                             (was.get("exact") or {}).get("scaffolds_exact"),
                             name)

    def test_the_near_miss_has_not_moved(self):
        for name in RUNS:
            if name == "run2":
                continue
            was = load(name)["proposal"].get("nearest_miss")
            now = self.fresh[name]["proposal"].get("nearest_miss")
            self.assertEqual(bool(now), bool(was), name)
            if was:
                self.assertEqual(now["scaffolds_exact"],
                                 was["scaffolds_exact"], name)

    def test_the_screen_verdicts_have_not_moved(self):
        for name in RUNS:
            was = load(name)["screenable"]
            now = self.fresh[name]["screenable"]
            if was is None:
                self.assertIsNone(now, name)
                continue
            for key in ("rankability", "applicability"):
                self.assertEqual((now[key] or {}).get("verdict"),
                                 (was[key] or {}).get("verdict"),
                                 f"{name}.{key}")

    def test_the_run_that_stops_still_stops(self):
        self.assertIn("stopping", self.fresh["run2"]["stdout"])
