"""The prompts, as a regression.

A set of requests is run in order, repeatedly, through every day of changes.
Most of what has been found was found by reading their outputs side by side:
a library selector that answered about the wrong slice, a recommendation
whose ordering preferred the largest overshoot, a ranking verdict that passed
on one specification and failed on otherwise identical ones, a library read
as zero molecules that answered a confident no. None of it was visible in a
single run.

HOW MANY THERE ARE IS NOT WRITTEN DOWN

This was called six_prompts, and the name went wrong the moment a seventh was
added -- as one was, because four of one day's five faults came out of a
library nothing was exercising. A test in here was called
test_the_four_identical_requests_agree while checking three runs: the count
in the name had already drifted from the count in the body.

So the set is read from the fixture directory and the groupings are derived
from the records. Adding a prompt means adding a directory. Nothing has to be
renamed, and no sentence about how many there are can go stale.

They run in two layers, because they cost two different things.

  THE REPLAY, which runs in the default suite. Each run's records --
  request.json, proposal.json, screenable.json, what_to_measure.json -- are
  stored under tests/fixtures/prompts/ and pushed back through the reporting
  layer. It fixes the screen: what a reader is shown about records that are
  held fixed. Every display fault found so far would have failed here, and it
  needs no library, no engine and no model.

  THE RERUN, which does not. Set OCCAM_PROMPTS=1 and it runs every request for
  real and compares the verdicts against the fixtures. That is the layer that
  catches a change in the measurement rather than in its presentation, and it
  takes about twenty minutes, four at a time.

    nix-shell --run 'OCCAM_PROMPTS=1 python3 -m unittest tests/test_prompts.py'

WHAT THE FIXTURES ARE, AND WHAT THEY ARE NOT

They are records of one machine on one day against one library. They are not
ground truth about chemistry, and a test comparing against them says only
that the tool still answers what it answered. When a change is meant to move
a verdict -- as the sampling asymmetry moved run 6 from a pass to not settled
-- the fixtures are re-recorded and the commit says which verdict moved and
why. A fixture updated without that sentence is the failure this file exists
to make visible.

To re-record, after a change whose effect on the set is understood:

    tests/fixtures/prompts/RECORD.md has the commands and the prompts.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "prompts"
sys.path.insert(0, str(ROOT / "py"))

RERUN = os.environ.get("OCCAM_PROMPTS") == "1"

# What each run is for. The prompt text lives in the fixture's request.json
# under "goal", so it is not written twice; WHY is here because a directory
# cannot say why it was added.
WHY = {
    "run1": "a request the library can fill as asked",
    "run2": "a request naming no library, and a condition this pipeline "
            "cannot evaluate",
    "run3": "a request that needs one setting loosened",
    "run4": "the same, with a ranking request and a word that looks like a "
            "target",
    "run5": "the same, with a ranking request and no such word",
    "run6": "a request that needs two settings loosened, and the largest "
            "survivor set here",
    "run7": "a library held in one parquet file rather than a directory of "
            "parts, whose every row fails a condition held as asked",
}


def runs():
    """Every prompt on disk, in order. The set is what is recorded, not what
    a constant in this file remembers -- a count in a name goes stale the
    first time one is added, and one was."""
    return sorted(d.name for d in FIXTURES.iterdir()
                  if d.is_dir() and (d / "request.json").exists())


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

    def test_the_recorded_set_and_the_documented_set_agree(self):
        """runs() reads the directory, so asserting the directories exist
        would assert nothing. What can go wrong is a prompt recorded with no
        note saying why it is here, or a note for a prompt that was removed."""
        recorded, documented = set(runs()), set(WHY)
        self.assertEqual(recorded - documented, set(),
                         "recorded with no note saying why it is in the set")
        self.assertEqual(documented - recorded, set(),
                         "a note for a prompt that is not recorded")

    def test_every_run_records_the_request_it_came_from(self):
        for name in runs():
            spec = load(name)["request"]
            self.assertIsNotNone(spec, f"{name} has no request.json")
            self.assertTrue(spec.get("goal"), f"{name} records no prompt")

    def test_the_run_that_stops_has_no_proposal(self):
        """Run 2 names no library. It must stop before proposing, and a
        fixture with a proposal in it would mean that stop was lost."""
        self.assertIsNone(load("run2")["proposal"])

    def test_every_other_run_has_a_proposal(self):
        for name in runs():
            if name == "run2":
                continue
            self.assertIsNotNone(load(name)["proposal"],
                                 f"{name} has no proposal.json")


@unittest.skipUnless(FIXTURES.is_dir(), f"no fixtures at {FIXTURES}")
class TheScreenIsReproducibleFromTheRecords(unittest.TestCase):
    """The reporting layer, against records held fixed.

    Every display fault found by reading them would have failed here: a
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
        for name in runs():
            if name == "run2":
                continue
            text = self._answer(name)
            self.assertIn("ANSWER", text, f"{name} produced no answer block")
            self.assertIn("No molecules are emitted", text, name)

    def test_no_verdict_reaches_the_screen_as_an_identifier(self):
        """A verdict this layer has no phrase for degrades to its key with
        the underscores taken out, which is a record shape on the screen."""
        import render as render_mod

        for name in runs():
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
        for name in runs():
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
class TheVerdictsAcrossTheSet(unittest.TestCase):
    """What they say when read side by side. These are the comparisons that
    found the faults, and a single run cannot make them."""

    def _rank(self, name):
        body = load(name)["screenable"]
        return (body or {}).get("rankability") or {}

    def test_no_ordering_verdict_passes_on_a_sample(self):
        """The fault this layer exists for. A distinct share rises as fewer
        molecules are measured, so a sampled figure cannot clear the floor --
        and of the set, the one specification whose survivor set exceeded the
        sample was the only one reported rankable."""
        for name in runs():
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
        for name in runs():
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
        for name in runs():
            rank = self._rank(name)
            if not rank or "distinct_share" not in rank:
                continue
            self.assertIn("declared, not derived", rank.get("floor_basis", ""),
                          f"{name} prints the floor without its basis")

    def test_requests_landing_on_one_specification_measure_the_same(self):
        """Several prompts differ only in what they ask for on top of the
        same specification, and must measure identically. When they did not,
        it was because one had been sampled and the others had not.

        The groups are derived from the records. Naming the members here --
        as "runs 3, 4 and 5", under a method called four_identical -- fixed a
        count in two places, and the two had already drifted apart."""
        groups = {}
        for name in runs():
            proposal = load(name)["proposal"] or {}
            spec = (proposal.get("recommended") or {}).get("specification")
            library = (load(name)["request"] or {}).get("library")
            if spec and self._rank(name):
                groups.setdefault((library, spec), {})[name] = (
                    self._rank(name).get("distinct_share"),
                    self._rank(name).get("of_survivors"))
        shared = {k: v for k, v in groups.items() if len(v) > 1}
        self.assertTrue(shared, "no two prompts share a specification, so "
                                "this comparison has stopped being made")
        for key, seen in shared.items():
            self.assertEqual(
                len(set(seen.values())), 1,
                f"{key[1]} measured differently across {sorted(seen)}: {seen}")


@unittest.skipUnless(RERUN, "set OCCAM_PROMPTS=1 to run them for real")
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

        cls.tmp = tempfile.mkdtemp(prefix="occam-prompts-")
        cls.fresh = {}
        for name in runs():
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
        for name in runs():
            self.assertEqual(self.fresh[name]["request"].get("library"),
                             load(name)["request"].get("library"), name)

    def test_the_recommended_specification_has_not_moved(self):
        for name in runs():
            if name == "run2":
                continue
            was = (load(name)["proposal"].get("recommended") or {})
            now = (self.fresh[name]["proposal"].get("recommended") or {})
            self.assertEqual(now.get("specification"),
                             was.get("specification"), name)

    def test_the_counts_the_verdict_turns_on_have_not_moved(self):
        for name in runs():
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
        for name in runs():
            if name == "run2":
                continue
            was = load(name)["proposal"].get("nearest_miss")
            now = self.fresh[name]["proposal"].get("nearest_miss")
            self.assertEqual(bool(now), bool(was), name)
            if was:
                self.assertEqual(now["scaffolds_exact"],
                                 was["scaffolds_exact"], name)

    def test_the_screen_verdicts_have_not_moved(self):
        for name in runs():
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

    def test_no_recorded_field_has_appeared_or_changed_type(self):
        """The fixtures go stale silently, and that is how 108 survived.

        Every other test here names the fields it compares -- rankability's
        verdict, applicability's verdict, the exact counts. That is the right
        choice for VALUES, because timings and version stamps move and a test
        comparing whole transcripts fails on every commit and gets switched
        off.

        It is the wrong choice for SHAPE. The tie diagnosis was added to
        suggested_revisions.ranking.measured_alternatives as a dict where a
        sentence had been, and 23 further fields appeared beside it. Nothing
        compared that subtree, so the fixtures sat two commits behind the code
        with every verdict still matching, and the one field whose TYPE had
        changed was the one the renderer then formatted with an f-string.

        So: the shape is compared and the values are not. A field added or
        retyped is then a deliberate re-recording with a sentence in the
        commit, which is what RECORD.md already asks for and had no way to
        enforce.
        """
        def shape(value, path=""):
            if isinstance(value, dict):
                out = {}
                for key, inner in value.items():
                    out.update(shape(inner, f"{path}.{key}" if path else key))
                return out
            # A list's length is a measurement, not a shape: the frontier
            # holds as many rows as it holds. Its elements' shape is not,
            # so the first row stands for all of them.
            if isinstance(value, list):
                return shape(value[0], f"{path}[]") if value else {path: "list"}
            return {path: type(value).__name__}

        for name in runs():
            for stem in ("request", "proposal", "screenable"):
                was, now = load(name)[stem], self.fresh[name][stem]
                if was is None or now is None:
                    self.assertEqual(was is None, now is None, f"{name}.{stem}")
                    continue
                sw, sn = shape(was), shape(now)
                self.assertEqual(
                    sorted(set(sn) - set(sw)), [],
                    f"{name}.{stem}: fields the code writes and the fixture "
                    f"does not have. Re-record, and say in the commit what "
                    f"they are")
                self.assertEqual(
                    sorted(set(sw) - set(sn)), [],
                    f"{name}.{stem}: fields the fixture has and the code no "
                    f"longer writes")
                self.assertEqual(
                    sorted(k for k in sw if sw[k] != sn[k]), [],
                    f"{name}.{stem}: fields whose type changed. A field that "
                    f"changes type changes how it must be printed -- that is "
                    f"contraindication 108")


@unittest.skipUnless(FIXTURES.is_dir(), f"no fixtures at {FIXTURES}")
class ALibraryHeldInOneFile(unittest.TestCase):
    """Run 7 is here because four of one day's five faults came out of a
    library nothing in the set was exercising. CHAMP is one parquet file
    rather than a directory of parts, and every one of its rows fails a
    condition that was held as asked.

    The replay above renders records; these are the parts of that run the
    records alone do not reach.
    """

    NAME = "run7"

    def test_a_single_file_library_was_read(self):
        """iter_library() globbed part-*.parquet inside a directory, which
        matches nothing inside a FILE path and raises nothing. The run read
        zero molecules and answered that no setting could fill the request."""
        sweep = json.loads((FIXTURES / self.NAME / "sweep.json").read_text())
        self.assertGreater(sweep["summary"]["evaluated"], 0,
                           "the library was not read, and a verdict was "
                           "produced from an empty population anyway")

    def test_a_library_that_is_not_a_zinc_slice_is_named_not_pathed(self):
        """The echo printed `data/work/champ.parquet`: the record's spelling
        on the screen. library_phrase() only knows the ZINC tranches, so the
        other four fell through to the raw path."""
        import inventory as inventory_mod
        import render as render_mod

        spec = json.loads((FIXTURES / self.NAME / "request.json").read_text())
        phrase = render_mod.declared_library_phrase(
            spec["library"], inventory_mod.declared())
        self.assertIsNotNone(phrase, f"{spec['library']} is named nowhere")
        self.assertNotIn("data/work", phrase)
        self.assertNotIn(".parquet", phrase)
        self.assertIn("CHAMP", phrase)

    def test_the_zero_is_the_data_and_not_the_detector(self):
        """CHAMP holds no primary amine at all. That is a property of the
        library, and the same detector finds them everywhere else, so the
        measurement is recorded beside the zero rather than left as a
        surprising result nobody checked."""
        import json as json_mod
        import pathlib

        manifest = json_mod.loads(
            (ROOT / "data" / "raw" / "champ" / "manifest.json").read_text())
        counts = manifest["evidence"]["nitrogen_counts_measured"]
        self.assertEqual(counts["primary_amine"], 0)
        self.assertIn("GramNegAccum", counts["control"],
                      "the zero is recorded without the control that makes "
                      "it a measurement rather than a suspicion")

    def test_a_run_that_reaches_nothing_still_names_the_cause(self):
        """"no setting of these knobs can fill the request" with a ceiling of
        0 is true and tells a reader nothing. The sweep cannot name the cause
        -- a condition held as asked is never varied, so its effect never
        appears as a difference -- so it is measured in one more pass, spent
        only here."""
        import sweep as sweep_mod

        proposal = json.loads(
            (FIXTURES / self.NAME / "proposal.json").read_text())
        best = proposal.get("best_reachable") or {}
        self.assertEqual(best.get("survivors"), 0,
                         "this run no longer reaches zero, so it no longer "
                         "exercises the path it was added for")
        spec = json.loads((FIXTURES / self.NAME / "request.json").read_text())
        alone = sweep_mod.eliminating_conditions(spec["library"],
                                                 best["choices"])
        self.assertEqual(alone["read"], 1132)
        blocking = [r for r in alone["alone"] if not r["keeps"]]
        self.assertEqual([r["knob"] for r in blocking], ["nitrogen"])
        kept = {r["knob"]: r["keeps"] for r in alone["alone"]}
        self.assertGreater(kept["rotatable"], 1000)
        self.assertGreater(kept["weight"], 1000)
