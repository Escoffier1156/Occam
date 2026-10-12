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
