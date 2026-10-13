"""The contraindications that an assertion can be written for.

docs/HANDOFF.md section 7 marks each of 83 items with whether a test enforces
it. 50 were sentences only, and the list says plainly why that matters: what
has actually held is the subset that became a test, and the rest has been
stepped on again in a changed shape.

This file converts the ten that an assertion reaches. The other forty are
about judgement -- which withdrawal to record, not drifting toward a
presentable deliverable -- and will not become tests. Saying so is cheaper
than pretending otherwise.

Each test names its item and fails if the trap comes back. Where the code
takes a different route to safety than the contraindication's own remedy, the
test asserts the PROPERTY, not the remedy: py/ingest.py avoids reading a
directory as a file with a suffix filter rather than with `is_file`, and
either is fine as long as no directory gets through.
"""

import ast
import csv
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))


def source(name):
    return (ROOT / "py" / name).read_text()


class Item1AndItem37GitignorePatterns(unittest.TestCase):
    """1: `data/raw/*` does not hold -- re-including a subdirectory
    re-includes its contents. 37: `engine/target/` alone left target-fp/ and
    target-lin/ tracked, because there are three builds."""

    def setUp(self):
        self.lines = [line.strip() for line
                      in (ROOT / ".gitignore").read_text().splitlines()
                      if line.strip() and not line.startswith("#")]

    def test_raw_data_is_excluded_recursively(self):
        self.assertIn("data/raw/**", self.lines)
        self.assertNotIn("data/raw/*", self.lines,
                         "a single star re-includes a subdirectory's contents")

    def test_the_manifests_are_still_tracked(self):
        """Excluding data/raw/** has to leave the records in."""
        for pattern in ("!data/raw/**/", "!data/raw/**/manifest.json"):
            self.assertIn(pattern, self.lines)

    def test_every_engine_build_directory_is_excluded(self):
        self.assertIn("engine/target*/", self.lines)
        self.assertNotIn("engine/target/", self.lines,
                         "the bare form misses target-fp/ and target-lin/")


class Item4OneRowGroupPerPart(unittest.TestCase):
    """4: a parquet footer grows with the row group count, which would make
    the engine's memory depend on the data volume."""

    def test_the_writer_sets_a_row_group_size(self):
        text = source("ingest_zinc.py")
        self.assertIn("row_group_size=PART_ROWS", text,
                      "without it parquet chooses, and the footer grows")

    def test_a_written_part_holds_exactly_one_row_group(self):
        import pyarrow.parquet as pq

        parts = sorted((ROOT / "data" / "work" / "zinc_P").glob("part-*.parquet"))
        if not parts:
            self.skipTest("no ingested parts on this disk to check")
        for path in parts:
            self.assertEqual(pq.ParquetFile(path).num_row_groups, 1,
                             f"{path.name} holds more than one row group")


class Item13NoDirectoryIsReadAsAFile(unittest.TestCase):
    """13: data/raw has subdirectories, so listing it and treating each entry
    as a file reads a directory."""

    def test_the_raw_scan_yields_only_files(self):
        import ingest as ingest_mod

        raw = ingest_mod.RAW_DIR
        if not raw.exists():
            self.skipTest("data/raw is not on this disk")
        # The same selection py/ingest.py makes in main().
        chosen = [p for p in raw.iterdir()
                  if p.suffix.lower() in ingest_mod.DIALECTS]
        self.assertTrue(chosen, "the scan found nothing, so it proves nothing")
        for path in chosen:
            self.assertTrue(path.is_file(), f"{path} is not a file")


class Item47NoSelfMatchingWaitLoop(unittest.TestCase):
    """47: `while pgrep -f 'py/x.py'; do sleep 20; done` never ends -- the
    loop's own command line contains the pattern, so there is always a match.
    The watcher outran the job it was watching by hours."""

    def test_no_executable_file_waits_on_a_bare_pgrep_pattern(self):
        offenders = []
        for path in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.sh")):
            # data/work holds generated scripts and is not committed.
            # `.claude` holds the app's worktrees: copies of this very file,
            # which the self-exclusion below cannot match because it compares
            # paths. Contraindication 47 is about the scripts of this
            # repository, not about a checkout of it sitting inside itself.
            if any(part in (".git", ".claude", "target", "__pycache__",
                            "work") for part in path.parts):
                continue
            # This file states the trap in order to search for it, so its own
            # text matches. Excluding it is not a loophole: a wait loop would
            # not be written in a test that scans for wait loops.
            if path == Path(__file__):
                continue
            for number, line in enumerate(path.read_text(
                    errors="replace").splitlines(), 1):
                if "pgrep" not in line:
                    continue
                if "while" not in line and "until" not in line:
                    continue
                # A bracketed character class in the pattern is the split that
                # stops the self-match: pgrep -f '[i]ngest.py' cannot match the
                # loop's own command line. Only an unbracketed pattern is the
                # trap.
                pattern = re.search(r"pgrep[^|;]*?-f\s*(['\"])(.+?)\1", line)
                if pattern and "[" not in pattern.group(2):
                    offenders.append(f"{path.relative_to(ROOT)}:{number}")
        self.assertEqual(offenders, [], f"self-matching wait loop: {offenders}")


class Item59FactClaimsAreNeverTakenAsInput(unittest.TestCase):
    """59: "1億件", "0.071", "7,394" were all wrong, and all of them were
    claims about the data rather than requests of it."""

    CLAIMS = [
        ("the 100-million-item cache", "a library size"),
        ("the 0.071 auc deficit", "an AUC"),
        ("7,394 distinct diversity domains", "a domain count"),
        ("the verified library", "a verification status"),
    ]

    def test_every_claim_shape_is_dropped_and_listed(self):
        import occam as occam_mod

        for phrase, what in self.CLAIMS:
            spec, notes = occam_mod.parse(
                f"256 distinct scaffolds from {phrase} of cationic ZINC, "
                f"primary amine")
            self.assertTrue(
                notes["dropped_claims"],
                f"{phrase!r} ({what}) was not reported as a dropped claim")

    def test_a_dropped_claim_never_becomes_a_requirement(self):
        import occam as occam_mod

        spec, notes = occam_mod.parse(
            "256 distinct scaffolds from the 100-million-item cationic ZINC "
            "library, 0.071 auc, primary amine")
        for requirement in spec["requirements"]:
            self.assertNotIn("100", str(requirement))
            self.assertNotIn("0.071", str(requirement))


class Item61NoWeightedScore(unittest.TestCase):
    """61: `w1*survivors + w2*diversity - w3*penalty` makes "best" a product
    of invented weights. The frontier is returned instead so a person makes
    the trade."""

    def test_the_proposer_does_not_multiply_criteria_by_coefficients(self):
        """Checked on the parsed code, so the docstring that explains why not
        does not count as doing it."""
        tree = ast.parse(source("propose.py"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value,
                                                         ast.Constant):
                continue  # a docstring
            if isinstance(node, ast.Name) and re.fullmatch(r"w[0-9]", node.id):
                self.fail(f"a weight named {node.id} is in the code")

    def test_the_frontier_is_what_is_returned(self):
        import propose as propose_mod

        self.assertIn("frontier", source("propose.py"))
        self.assertTrue(callable(propose_mod.propose))

    def test_the_ranking_key_is_steps_then_provenance_then_a_tiebreak(self):
        text = source("propose.py")
        self.assertIn("steps_from_baseline", text)
        self.assertIn("deterministic tie-break", text,
                      "the tie-break has to be recorded as a tie-break, not "
                      "as a preference")


class Item62TheSketchErrorStaysOutOfTheDecision(unittest.TestCase):
    """62: deciding on a conservative lower bound of the sketch leaves the
    error model inside the verdict.

    The sweep keeps the sketch, because its fixed width per specification is
    invariant 2 and an exact set would make the sweep's state grow with the
    distinct scaffold count. The error is kept out of the decision instead by
    counting exactly, in a second pass, every specification whose number a
    person will see or act on: the uncertain band, the frontier, the winner.

    An exact set was tried in place of the sketch and reverted. The cost was
    measured: 330,610 scaffolds across 27 specifications at 127.4 bytes each,
    40 MiB, growing with the library. Removing an error model is not worth
    putting a data-dependent term into the one invariant this project has
    verified over four orders of magnitude.
    """

    def test_the_band_is_three_standard_errors(self):
        import propose as propose_mod

        self.assertAlmostEqual(propose_mod.UNCERTAIN_BAND,
                               3 * propose_mod.SKETCH_RELATIVE_ERROR)

    def test_the_sketch_never_decides_inside_the_band(self):
        import propose as propose_mod

        required = 120
        row = propose_mod.classify(
            {"specification": "x", "choices": {"nitrogen": "primary_amine"},
             "survivors": 123, "distinct_scaffolds_estimated": required,
             "provenance": {"weakest": "published", "per_knob": {}}},
            required, {"nitrogen": "primary_amine"})
        self.assertEqual(row["state_by_sketch"], "uncertain")

    def test_a_clear_case_needs_no_second_pass(self):
        import propose as propose_mod

        required = 120
        for estimate, expected in ((1000, "feasible"), (10, "infeasible")):
            row = propose_mod.classify(
                {"specification": "x",
                 "choices": {"nitrogen": "primary_amine"},
                 "survivors": 1, "distinct_scaffolds_estimated": estimate,
                 "provenance": {"weakest": "published", "per_knob": {}}},
                required, {"nitrogen": "primary_amine"})
            self.assertEqual(row["state_by_sketch"], expected)

    def test_the_frontier_is_counted_exactly_even_outside_the_band(self):
        """Three standard errors is a probabilistic guarantee, and with 27
        specifications in one run the chance that none falls outside it is not
        1. Every row a reader compares is counted, band or no band."""
        text = source("propose.py")
        decision = text[text.index("def propose("):]
        self.assertIn("provisional", decision,
                      "the frontier is no longer chosen before the exact pass")
        self.assertIn("FRONTIER", decision)
        self.assertIn('"state_by_sketch"] == "uncertain"', decision,
                      "the band is no longer part of what gets counted")

    def test_what_was_counted_exactly_is_recorded(self):
        """A reader must be able to tell which numbers were counted and which
        were estimated, without working it out."""
        text = source("propose.py")
        self.assertIn('"counted_exactly"', text)
        self.assertIn('"counted_by_sketch_only"', text)


class Item73SentenceInitialMustBindsEverything(unittest.TestCase):
    """73: "The screening must satisfy..." makes every knob strict, which
    leaves one specification to compare and no relaxation to offer. That is
    correct, and a requester has to be able to see that it happened."""

    PROSE = ("The screening must satisfy the following criteria: "
             "256 distinct scaffolds from the cationic ZINC library, "
             "primary amine, rotatable bonds at most 5, "
             "molecular weight under 450")

    def test_every_requirement_comes_out_strict(self):
        import occam as occam_mod

        spec, _ = occam_mod.parse(self.PROSE)
        knobs = {r["knob"]: r["strict"] for r in spec["requirements"]
                 if "knob" in r}
        self.assertTrue(knobs, "nothing was read as a requirement")
        for knob, strict in knobs.items():
            self.assertTrue(strict, f"{knob} was left relaxable")

    def test_only_one_specification_remains_to_compare(self):
        import knobs as knobs_mod
        import occam as occam_mod
        import sweep as sweep_mod

        spec, _ = occam_mod.parse(self.PROSE)
        baseline, strict = {}, {}
        for requirement in spec["requirements"]:
            if "knob" in requirement:
                baseline[requirement["knob"]] = requirement["setting"]
                strict[requirement["knob"]] = requirement["strict"]
        for knob in knobs_mod.ORDER:
            baseline.setdefault(knob, knobs_mod.settings_of(knob)[0]["name"])
            strict.setdefault(knob, False)
        variants = [v for v in sweep_mod.enumerate_variants(baseline)
                    if all(v.choices[k] == baseline[k]
                           for k in baseline if strict[k])]
        self.assertEqual(len(variants), 1,
                         "a strict request still had alternatives to offer")


class Item80AThresholdNeverAppearsWithoutItsSource(unittest.TestCase):
    """80: "0.0% at or above the threshold" cannot be checked by a reader who
    is not told what the threshold is or where it came from."""

    def test_the_applicability_record_carries_both(self):
        text = source("screenable.py")
        self.assertIn('"threshold": EXTRAPOLATION_BELOW', text)
        self.assertIn('"threshold_source"', text)

    def test_the_source_is_printed_wherever_the_threshold_is(self):
        """Read off the renderer: a block that prints one prints the other."""
        text = source("screenable.py")
        shown = text[text.index("def show("):]
        self.assertIn("threshold_source", shown,
                      "the threshold is printed and its source is not")
        self.assertLess(shown.index("a['threshold']"),
                        shown.index("threshold_source") + len(shown),
                        "both have to reach the screen")


if __name__ == "__main__":
    unittest.main()


class TheExactPassHoldsOneSetAtATime(unittest.TestCase):
    """The frontier's limit of 8 bounds how many exact passes run, not the
    memory of each. Eight sets alive at once is 45 MB at the widest
    specification measured -- the figure the sketch exists to avoid. Invariant
    3, one bulk at a time, applies to the second pass too."""

    def test_the_exact_count_returns_counts_and_not_the_set(self):
        """Structural, not a measurement: a caller cannot accumulate what it
        is never handed."""
        import propose as propose_mod

        returns = [node for node in ast.walk(ast.parse(
            source("propose.py"))) if isinstance(node, ast.FunctionDef)
            and node.name == "exact_scaffold_count"]
        self.assertEqual(len(returns), 1)
        keys = set()
        for node in ast.walk(returns[0]):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                keys = {k.value for k in node.value.keys
                        if isinstance(k, ast.Constant)}
        self.assertEqual(keys, {"survivors", "scaffolds_exact",
                                "held_strings"},
                         "the exact pass returns something other than counts")

    def test_the_passes_run_one_after_another(self):
        """A comprehension or a map over the whole set would hold them all."""
        tree = ast.parse(source("propose.py"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                                     ast.GeneratorExp)):
                continue
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Name)
                        and inner.func.id == "exact_scaffold_count"):
                    self.fail("the exact pass is run inside a comprehension, "
                              "which holds every result at once")

    def test_the_frontier_width_is_a_declared_constant(self):
        import propose as propose_mod

        self.assertIsInstance(propose_mod.FRONTIER, int)
        self.assertGreater(propose_mod.FRONTIER, 0)


class Item25AndItem81TwoScopesNeverShareARow(unittest.TestCase):
    """25: say which count a number is. 81: if the count for the same label
    moves between outputs, suspect the scope.

    Found live. A sweep read 20,000 rows and an exact second pass was asked
    for 100,000, and the frontier printed "123 molecules pass, 536 distinct
    scaffolds" -- a distinct count larger than the population it came from."""

    SWEPT = {"summary": {"limit": 20000},
             "baseline": {"nitrogen": "primary_amine"},
             "specifications": []}

    def test_a_second_pass_over_a_different_amount_is_refused(self):
        import propose as propose_mod

        with self.assertRaises(ValueError) as caught:
            propose_mod.propose(self.SWEPT, 120, part_dir="data/work/zinc_O",
                                limit=100000)
        self.assertIn("20000", str(caught.exception))
        self.assertIn("100000", str(caught.exception))

    def test_a_sweep_that_recorded_no_limit_is_also_refused(self):
        """Records written before the limit was recorded cannot be matched."""
        import propose as propose_mod

        old = {"summary": {}, "baseline": {"nitrogen": "primary_amine"},
               "specifications": []}
        with self.assertRaises(ValueError):
            propose_mod.propose(old, 120, part_dir="data/work/zinc_O",
                                limit=100000)

    def test_the_same_amount_is_accepted(self):
        import propose as propose_mod

        same = {"summary": {"limit": 20000},
                "baseline": {"nitrogen": "primary_amine"},
                "specifications": []}
        out = propose_mod.propose(same, 120, part_dir="data/work/zinc_O",
                                  limit=20000)
        self.assertIn("evaluated no specifications", out["verdict"],
                      "an empty sweep used to raise from max() over nothing")

    def test_the_sweep_records_what_it_read(self):
        self.assertIn('"limit": limit,', source("sweep.py"))

    def test_a_distinct_count_cannot_exceed_its_population(self):
        """The cheapest possible check, and it catches every way two scopes
        get mixed."""
        self.assertIn("distinct scaffolds from", source("propose.py"))
        self.assertIn("is impossible", source("propose.py"))


class TheWeakestClaimIsGivenUpFirst(unittest.TestCase):
    """Found by the first external review of this project.

    Asked for 500 distinct scaffolds, the proposer offered 686 by moving the
    published rotatable limit from 5 to 9 -- two steps away from a condition
    Richter 2017 states -- without weighing the specification one step tighter
    that moved the molecular weight bound this study chose and no paper
    states. Two faults in one ordering: it could not tell a published setting
    from an arbitrary one, and its tie-break preferred the largest survivor
    count, which is the largest overshoot.

    WITHDRAWN: the review and this docstring put the tighter specification at
    about 520, which would have made it the answer. Counted exactly it holds
    494 -- six short of the request -- so nothing reportable was being hidden
    by the ordering. The ordering fault was real and that figure was not. 494
    is now reported as the nearest miss, which is how a wrong estimate gets
    corrected instead of repeated.
    """

    BASELINE = {"nitrogen": "primary_amine", "rotatable": "rotatable_le_5",
                "weight": "mw_under_450"}

    def _row(self, choices, scaffolds, survivors):
        import propose as propose_mod
        return propose_mod.classify(
            {"specification": " + ".join(choices[k] for k in sorted(choices)),
             "choices": choices, "survivors": survivors,
             "distinct_scaffolds_estimated": scaffolds,
             "provenance": {"weakest": "published", "per_knob": {}}},
            500, self.BASELINE)

    def test_a_bound_this_study_chose_is_given_up_before_a_published_one(self):
        import propose as propose_mod

        moved_published = dict(self.BASELINE, rotatable="rotatable_le_9")
        moved_ours = dict(self.BASELINE, rotatable="rotatable_le_7",
                          weight="mw_under_600")
        report = {"summary": {"limit": None}, "baseline": self.BASELINE,
                  "specifications": [
                      {"specification": "rb9", "choices": moved_published,
                       "survivors": 2000, "distinct_scaffolds_estimated": 686,
                       "provenance": {"weakest": "relaxation", "per_knob": {}}},
                      # Both clearly above the uncertain band, so this test
                      # isolates the ordering: a row inside the band is
                      # resolved by the exact pass before it is ranked, which
                      # needs a library and is covered by the end-to-end run.
                      {"specification": "rb7+mw600", "choices": moved_ours,
                       "survivors": 1300, "distinct_scaffolds_estimated": 560,
                       "provenance": {"weakest": "relaxation", "per_knob": {}}},
                  ]}
        out = propose_mod.propose(report, 500)
        self.assertEqual(out["recommended"]["specification"], "rb7+mw600",
                         "the published rotatable limit was given up while a "
                         "bound this study chose was still available")

    def test_the_closest_specification_wins_not_the_largest(self):
        import propose as propose_mod

        a = dict(self.BASELINE, rotatable="rotatable_le_7")
        b = dict(self.BASELINE, weight="mw_under_600")
        report = {"summary": {"limit": None}, "baseline": self.BASELINE,
                  "specifications": [
                      {"specification": "far", "choices": a, "survivors": 9000,
                       "distinct_scaffolds_estimated": 5000,
                       "provenance": {"weakest": "relaxation", "per_knob": {}}},
                      {"specification": "near", "choices": b, "survivors": 600,
                       "distinct_scaffolds_estimated": 560,
                       "provenance": {"weakest": "published", "per_knob": {}}},
                  ]}
        out = propose_mod.propose(report, 500)
        self.assertEqual(out["recommended"]["specification"], "near",
                         "the largest overshoot was preferred")

    def test_a_specification_short_of_the_request_sorts_last(self):
        """overshoot() returns infinity for a specification that does not
        reach the request, so a near miss can never be preferred to something
        that meets it. 499 against 500 is inside the sketch's band and is
        decided by the exact pass, not here; this checks the ordering."""
        import propose as propose_mod

        short = self._row(dict(self.BASELINE, rotatable="rotatable_le_7"),
                          300, 9)
        meets = self._row(dict(self.BASELINE, weight="mw_under_600"), 700, 1)
        self.assertEqual(short["state_by_sketch"], "infeasible")
        self.assertEqual(meets["state_by_sketch"], "feasible")


class ASampleCanRefuseASetAndCannotClearOne(unittest.TestCase):
    """Found by re-running the six prompts after the first review's fixes.

    The distinct-score share rises as fewer molecules are measured, so the
    sampled figure is an upper bound on what the whole set would give. The
    refusal therefore carries from a sample and the pass does not. The old
    rule used one licence for both directions, and of six runs the single
    specification whose survivor set exceeded the sample was the only one
    reported rankable: 0.5967 on 1,200 of 1,820, against 0.4648 on all 1,820.
    """

    def test_a_clearing_sample_does_not_clear_the_set(self):
        import screenable as sc
        self.assertEqual(
            sc.rank_verdict(0.5967, 0.005, sampled=True),
            "not_settled_at_this_sample",
            "a share measured on part of the set was reported as a pass")

    def test_a_clearing_full_measurement_does_clear_the_set(self):
        import screenable as sc
        self.assertEqual(sc.rank_verdict(0.5967, 0.005, sampled=False),
                         "answerable")

    def test_a_failing_sample_refuses_the_set(self):
        """The direction the module was built on: a sample below the floor
        puts the whole set below it, so the refusal needs no further rows."""
        import screenable as sc
        for sampled in (True, False):
            self.assertEqual(sc.rank_verdict(0.4648, 0.005, sampled=sampled),
                             "not_answerable")

    def test_the_tie_ceiling_still_refuses_a_clearing_share(self):
        import screenable as sc
        self.assertEqual(
            sc.rank_verdict(0.90, sc.MAX_TIE_SHARE + 0.01, sampled=False),
            "not_answerable",
            "one value held most of the set and the order was allowed")

    def test_every_verdict_this_rule_returns_has_a_phrase(self):
        """An unmapped verdict degrades to the key with its underscores
        stripped, which is a record shape on the screen."""
        import render as render_mod
        import screenable as sc

        seen = {sc.rank_verdict(d, t, s)
                for d in (0.2, 0.4648, 0.5967, 0.99)
                for t in (0.001, 0.5)
                for s in (True, False)}
        for value in seen:
            self.assertIn(value, render_mod.VERDICT, f"{value} has no phrase")

    def test_a_sampled_verdict_discloses_what_it_rests_on(self):
        """Monotonicity was declared measured and not proven, so the places
        relying on it say so. Both directions rely on it when sampled."""
        text = source("screenable.py")
        self.assertIn("not proven", text)
        self.assertRegex(text, r'"rests_on"')

    def test_the_floor_says_it_was_declared_and_not_derived(self):
        """Asked for the 50% floor's basis on the first day; it has none. The
        calibration curves explain why the share needs a floor, not why the
        floor is at 0.50."""
        import screenable as sc
        self.assertIn("declared, not derived", sc.FLOOR_BASIS)

    def test_the_retracted_sample_count_is_not_left_in_dead_code(self):
        """`rankable_alternatives()` defaulted to size=400 -- the count this
        module retracted -- and nothing called it. Dead code holding a
        retracted number is how the retraction gets undone."""
        text = source("screenable.py")
        self.assertNotIn("def rankable_alternatives", text)
        # Read the defaults, not the file. The note recording the removal
        # quotes `size=400`, so a substring search finds the prose that says
        # the number is gone and calls it the number coming back. The same
        # mistake a boundary check in this suite already made once.
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = [a.arg for a in node.args.args][-len(node.args.defaults):] \
                if node.args.defaults else []
            for name, default in zip(names, node.args.defaults):
                if name != "size":
                    continue
                self.assertNotEqual(
                    getattr(default, "value", None), 400,
                    f"{node.name}() defaults its sample to the retracted 400")


class TheNearestMissReachesTheScreen(unittest.TestCase):
    """Asked for 500: the frontier offered 686 and said nothing about the
    specification counted exactly at 494, six short and two steps tighter. A
    reviewer working from an estimate of 520 could not retract it, because the
    measured 494 existed only in a count that was taken and then discarded.
    `frontier` holds feasible rows, so it had nowhere to go.
    """

    BASELINE = {"nitrogen": "primary_amine", "rotatable": "rotatable_le_5",
                "weight": "mw_under_450"}

    def _proposal(self):
        import propose as propose_mod
        far = dict(self.BASELINE, rotatable="rotatable_le_9")
        near = dict(self.BASELINE, rotatable="rotatable_le_7",
                    weight="mw_under_600")
        report = {"summary": {"limit": None}, "baseline": self.BASELINE,
                  "specifications": [
                      {"specification": "far", "choices": far,
                       "survivors": 1820, "distinct_scaffolds_estimated": 686,
                       "provenance": {"weakest": "relaxation",
                                      "per_knob": {}}},
                      {"specification": "near", "choices": near,
                       "survivors": 1292, "distinct_scaffolds_estimated": 494,
                       "provenance": {"weakest": "relaxation",
                                      "per_knob": {}}},
                  ]}
        # exact=False: the point is the reporting, not the second pass. The
        # near row is given an exact count directly, which is the state the
        # exact pass leaves it in.
        out = propose_mod.propose(report, 500, exact=False)
        return out

    def test_a_row_counted_exactly_and_short_is_reported(self):
        import propose as propose_mod
        rows = [{"specification": "near",
                 "choices": dict(self.BASELINE, rotatable="rotatable_le_7"),
                 "survivors": 1292, "distinct_scaffolds_estimated": 499,
                 "provenance": {"weakest": "relaxation", "per_knob": {}}},
                {"specification": "far",
                 "choices": dict(self.BASELINE, rotatable="rotatable_le_9"),
                 "survivors": 1820, "distinct_scaffolds_estimated": 686,
                 "provenance": {"weakest": "relaxation", "per_knob": {}}}]
        report = {"summary": {"limit": None}, "baseline": self.BASELINE,
                  "specifications": rows}
        out = propose_mod.propose(report, 500, exact=False)
        # Without the library the near row stays uncertain and is held back,
        # which is the other half of the same rule.
        self.assertTrue(out["held_back"], "an unplaced row vanished")
        self.assertIsNone(out["nearest_miss"],
                          "a sketch estimate was printed as a measured miss")

    def test_a_sketch_estimate_is_never_printed_as_a_near_miss(self):
        """499 against 500 is inside the sketch's own error bar. Reporting it
        as "one short" would be a guess with a decimal point on it."""
        out = self._proposal()
        self.assertIsNone(out["nearest_miss"])

    def test_the_answer_block_carries_the_miss(self):
        import render as render_mod
        text = "\n".join(render_mod.answer(
            {"deliverable": {"distinct_scaffolds": 500}},
            {"request": {"distinct_scaffolds_required": 500},
             "recommended": {"distinct_scaffolds": 686, "survivors": 1820,
                             "counted_exactly": True,
                             "loosened": {"rotatable": "rotatable_le_9"},
                             "choices": {"rotatable": "rotatable_le_9"}},
             "nearest_miss": {"specification": "near",
                              "choices": {"rotatable": "rotatable_le_7"},
                              "scaffolds_exact": 494, "short_by": 6,
                              "survivors": 1292, "steps_from_baseline": 1,
                              "loosened": {}, "provenance": {}}}))
        self.assertIn("494", text)
        self.assertIn("6 short", text)

    def test_the_withdrawn_520_is_not_still_asserted(self):
        """The ordering note said a specification reaching about 520 was
        available. It was counted at 494. A number known to be wrong is
        withdrawn in place, not left standing."""
        text = source("propose.py")
        self.assertIn("494", text)
        self.assertIn("withdrawn", text)


class EveryBranchOfTheBarrierSentenceIsGrammatical(unittest.TestCase):
    """A template with a noun swapped into it printed "this library is about
    not recorded for this library." when no barrier was on file, and read the
    same way when no library had been named at all. A hole that has to be
    grammatical under every value it takes is a branch that is missing.

    The sentences are asserted, not the source. A search for the broken phrase
    finds the comment recording its removal and calls that the defect coming
    back -- the same mistake the sample-count check in this file made.
    """

    def test_a_recorded_barrier_is_named(self):
        import render as render_mod
        self.assertEqual(
            render_mod.barrier_sentence("data/work/zinc_P",
                                        "the Gram-negative outer membrane"),
            "this library is about the Gram-negative outer membrane.")

    def test_an_unrecorded_barrier_does_not_become_the_subject(self):
        import render as render_mod
        for missing in ("unrecorded", None):
            said = render_mod.barrier_sentence("data/work/zinc_P", missing)
            self.assertEqual(
                said, "which barrier this library is about is not recorded.")
            self.assertNotIn("is about not recorded", said)

    def test_no_library_is_not_described_as_a_library(self):
        import render as render_mod
        said = render_mod.barrier_sentence(None, "unrecorded")
        self.assertIn("no library was named", said)
        self.assertNotIn("this library is about", said)

    def test_every_branch_ends_in_one_full_stop(self):
        """Three sentences, each a sentence."""
        import render as render_mod
        for library, barrier in (("x", "a barrier"), ("x", "unrecorded"),
                                 (None, "unrecorded"), (None, None)):
            said = render_mod.barrier_sentence(library, barrier)
            self.assertTrue(said.endswith("."), said)
            self.assertEqual(said.count("."), 1, said)

    def test_the_call_site_no_longer_interpolates(self):
        self.assertNotIn('f"this library is about {named}."',
                         source("occam.py"))


class EveryQuestionTheMeasurementCanSettleIsMeasured(unittest.TestCase):
    """A sample cannot clear the floor, so a tool that always sampled would
    answer "not settled" for every set larger than the sample and the ranking
    verdict would be decorative. The rule has three cases and the middle one
    is the point: a set larger than the fallback sample but inside the time
    budget is read in full, so the verdict gets stated rather than deferred.

    The ceiling is a consequence of a declared time budget, not a count that
    was argued for, so these tests pass it explicitly. What is being fixed
    here is the rule.
    """

    def test_a_set_inside_the_ceiling_is_read_in_full(self):
        import screenable as sc
        self.assertEqual(sc.measure_count(1820, ceiling=5000, fallback=1200),
                         1820, "a set inside the budget was sampled")

    def test_a_set_at_the_ceiling_is_read_in_full(self):
        import screenable as sc
        self.assertEqual(sc.measure_count(5000, ceiling=5000, fallback=1200),
                         5000)

    def test_a_set_past_the_ceiling_falls_back_to_the_sample(self):
        import screenable as sc
        self.assertEqual(sc.measure_count(17775, ceiling=5000, fallback=1200),
                         1200)

    def test_full_overrides_the_ceiling(self):
        """--full is what makes "not settled" a choice rather than a dead
        end: the requester can spend the time and get the verdict."""
        import screenable as sc
        self.assertEqual(
            sc.measure_count(17775, full=True, ceiling=5000, fallback=1200),
            17775)

    def test_a_set_smaller_than_the_sample_is_never_padded(self):
        import screenable as sc
        self.assertEqual(sc.measure_count(430, ceiling=5000, fallback=1200),
                         430)
        self.assertEqual(sc.measure_count(430, ceiling=100, fallback=1200),
                         430, "the fallback exceeded the set and was not cut")

    def test_the_rule_and_the_verdict_agree(self):
        """A count chosen by measure_count() must produce a verdict that
        rank_verdict() can state. Inside the ceiling the measurement is full,
        so a clearing share is answerable; past it the measurement is sampled,
        so a clearing share is not settled."""
        import screenable as sc

        total, clearing = 1820, 0.5967
        read = sc.measure_count(total, ceiling=5000, fallback=1200)
        self.assertEqual(
            sc.rank_verdict(clearing, 0.005, sampled=read < total),
            "answerable")

        read = sc.measure_count(total, ceiling=1000, fallback=1200)
        self.assertEqual(
            sc.rank_verdict(clearing, 0.005, sampled=read < total),
            "not_settled_at_this_sample")


class ARankingRequestIsAnsweredAndNotOnlyDropped(unittest.TestCase):
    """"Rank them and give me the top 50" went into the unread list, which is
    right -- no knob is being set and this tool emits no molecules. Thirty
    lines below, the same run measured the set and reported that the model
    cannot order it, which is the direct answer to what was asked, and the two
    were never joined.
    """

    def test_a_top_n_request_is_recorded(self):
        import occam as occam_mod
        spec, _ = occam_mod.parse(
            "256 distinct scaffolds from the dicationic ZINC library, "
            "primary amine, rotatable bonds at most 5, molecular weight "
            "under 450. Rank them and give me the top 50.")
        self.assertEqual(spec["asked_to_rank"]["top_n"], 50)

    def test_a_request_with_no_ranking_records_nothing(self):
        """The join only appears where there is a request to join it to."""
        import occam as occam_mod
        spec, _ = occam_mod.parse(
            "256 distinct scaffolds from the dicationic ZINC library, "
            "primary amine, rotatable bonds at most 5")
        self.assertIsNone(spec.get("asked_to_rank"))

    def test_the_ranking_request_does_not_become_a_requirement(self):
        """It sets nothing. The vocabulary stays closed and the deliverable
        is untouched -- a ranking is not something this tool does."""
        import occam as occam_mod
        spec, _ = occam_mod.parse(
            "256 distinct scaffolds from the dicationic ZINC library, "
            "primary amine. Rank them and give me the top 50.")
        self.assertEqual(spec["deliverable"], {"distinct_scaffolds": 256})
        self.assertNotIn("rank", [r.get("knob") for r in spec["requirements"]])

    def test_an_unorderable_set_answers_the_request_it_was_asked_about(self):
        import render as render_mod
        said = " ".join(render_mod.ranking_answer({"phrases": ["top 50"],
                                                   "top_n": 50},
                                                  "not_answerable"))
        self.assertIn("top 50", said)
        self.assertIn("cannot be drawn", said)

    def test_an_unsettled_verdict_does_not_read_as_a_refusal(self):
        """"Not settled" is not "no". A reader must not come away thinking
        the set was shown to be unorderable when it was not measured."""
        import render as render_mod
        said = " ".join(render_mod.ranking_answer(
            {"phrases": ["top 50"], "top_n": 50},
            "not_settled_at_this_sample"))
        self.assertIn("not settled", said)
        self.assertNotIn("cannot be drawn", said)

    def test_an_orderable_set_still_promises_no_molecules(self):
        """The verdict licenses running the screen. It does not make this tool
        the thing that hands over a list."""
        import render as render_mod
        said = " ".join(render_mod.ranking_answer({"phrases": ["top 50"],
                                                   "top_n": 50},
                                                  "answerable"))
        self.assertIn("no molecules are emitted", said.lower())

    def test_nothing_asked_prints_nothing(self):
        import render as render_mod
        self.assertEqual(render_mod.ranking_answer(None, "not_answerable"), [])
        self.assertEqual(
            render_mod.ranking_answer({"phrases": []}, "not_answerable"), [])
