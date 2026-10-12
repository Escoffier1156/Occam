"""The reading of a request, where refusing is the behaviour under test.

The parser's value is that it does not guess. Two ways of not guessing are
checked here: a number with no registered setting is refused rather than
rounded to the nearest one, and a word for a quantity that is no longer a
requirement is refused rather than redirected to the field that now carries
it. The second is the charge knob's retirement: charge is chosen by the
library slice, and silently turning "neutral" into a slice would be the same
move as turning "MW under 500" into 450.

The case that makes this non-trivial is that "neutral ZINC" is a legitimate
way to name the -N slice, so the same word has to be read one way inside a
library phrase and refused outside one.
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import knobs as knobs_mod  # noqa: E402
import occam as occam_mod  # noqa: E402


def settings_in(spec):
    return {r["knob"]: r["setting"] for r in spec["requirements"] if "knob" in r}


class TheRegistryAndItsRetirements(unittest.TestCase):

    def test_order_is_derived_from_the_registry(self):
        """It was written out in four places, and retiring a knob had to find
        all four."""
        self.assertEqual(knobs_mod.ORDER, tuple(knobs_mod.KNOBS))

    def test_a_retired_knob_is_not_also_live(self):
        for knob in knobs_mod.RETIRED:
            self.assertNotIn(knob, knobs_mod.KNOBS,
                             f"{knob} is both retired and in the registry")

    def test_a_retirement_records_what_it_rested_on(self):
        for knob, body in knobs_mod.RETIRED.items():
            for key in ("retired", "why", "where_it_went"):
                self.assertTrue(body.get(key),
                                f"{knob} was retired with no {key}")

    def test_charge_is_retired(self):
        self.assertIn("charge", knobs_mod.RETIRED)
        self.assertIn("slices", knobs_mod.BLOCKED["library_charge_slice"])


class ARequestNamingAChargeIsRefused(unittest.TestCase):

    def test_a_charge_word_outside_a_library_phrase_is_refused(self):
        spec, notes = occam_mod.parse(
            "64 distinct scaffolds from the cationic ZINC library, neutral, "
            "primary amine, rotatable bonds at most 5")
        phrases = [item["phrase"] for item in notes["refused"]]
        self.assertIn("neutral", phrases)
        self.assertNotIn("charge", settings_in(spec))

    def test_the_refusal_names_the_slices_rather_than_a_source_file(self):
        _, notes = occam_mod.parse(
            "64 distinct scaffolds from the cationic ZINC library, "
            "zero formal charge, primary amine")
        refusal = next(item for item in notes["refused"]
                       if "charge" in item["phrase"])
        self.assertTrue(any("-O" in str(v) for v in refusal["available"]))
        self.assertNotIn(".py", refusal["what_to_do"])

    def test_a_charge_word_is_never_read_as_a_requirement(self):
        for phrase in ("neutral only", "any charge", "zero formal charge",
                       "formal charge of zero", "cationic", "anionic"):
            spec, notes = occam_mod.parse(
                f"64 distinct scaffolds from the IMI library, {phrase}, "
                f"primary amine")
            self.assertTrue(notes["refused"],
                            f"{phrase!r} was read silently")
            for requirement in spec["requirements"]:
                self.assertNotIn(requirement.get("knob"), knobs_mod.RETIRED)

    def test_neutral_zinc_names_the_slice_and_is_not_refused(self):
        """The -N slice is named this way in the documentation, so the word
        inside a library phrase has to keep working."""
        spec, notes = occam_mod.parse(
            "64 distinct scaffolds from the neutral ZINC library, "
            "primary amine, rotatable bonds at most 5")
        self.assertEqual(spec["library"], "data/work/zinc")
        self.assertEqual(notes["refused"], [])

    def test_cationic_zinc_names_the_slice_and_is_not_refused(self):
        spec, notes = occam_mod.parse(
            "64 distinct scaffolds from the cationic ZINC library, "
            "primary amine")
        self.assertEqual(spec["library"], "data/work/zinc_O")
        self.assertEqual(notes["refused"], [])


class AStrictnessMarkerIsNotAlsoReportedAsLost(unittest.TestCase):
    """The list of phrases nothing was read from is the one list a requester
    has to read. A marker that WAS read belongs in neither it nor near it: a
    false entry there is how a reader learns to skip the list."""

    def test_a_marker_phrase_does_not_land_in_the_unread_list(self):
        spec, notes = occam_mod.parse(
            "256 distinct scaffolds from the neutral ZINC library, "
            "primary amine must be strictly required, "
            "rotatable bonds strictly at most 5, "
            "molecular weight strictly under 450")
        unread = " ".join(notes["unread"]).lower()
        for marker in ("must", "strictly", "required"):
            self.assertNotIn(marker, unread,
                             f"{marker!r} was read as strictness and also "
                             f"reported as unread")

    def test_the_strictness_it_carried_was_actually_applied(self):
        spec, _ = occam_mod.parse(
            "256 distinct scaffolds from the neutral ZINC library, "
            "primary amine must be strictly required, "
            "rotatable bonds strictly at most 5, "
            "molecular weight strictly under 450")
        strict = {r["knob"]: r["strict"] for r in spec["requirements"]
                  if "knob" in r}
        self.assertEqual(strict, {"nitrogen": True, "rotatable": True,
                                  "weight": True})

    def test_a_marker_still_does_not_leak_across_a_comma(self):
        """Recording the markers as read must not widen the clause window."""
        spec, _ = occam_mod.parse(
            "64 distinct scaffolds from the cationic ZINC library, "
            "molecular weight strictly under 450, primary amine")
        strict = {r["knob"]: r["strict"] for r in spec["requirements"]
                  if "knob" in r}
        self.assertTrue(strict["weight"])
        self.assertFalse(strict["nitrogen"])


class TheOutputDirectory(unittest.TestCase):

    def _run(self, *argv):
        saved = sys.argv
        sys.argv = ["occam", *argv]
        try:
            return occam_mod.main()
        finally:
            sys.argv = saved

    def test_a_missing_out_dir_is_created_rather_than_crashing(self):
        """It used to raise FileNotFoundError from write_text, after the whole
        request had already been echoed to the screen."""
        with tempfile.TemporaryDirectory() as box:
            target = Path(box) / "deep" / "not" / "there"
            code = self._run("--dry-run", "--out-dir", str(target),
                             "64 distinct scaffolds from the cationic ZINC "
                             "library, primary amine")
            self.assertEqual(code, 0)
            self.assertTrue((target / "request.json").exists())

    def test_an_unwritable_out_dir_stops_with_a_reason(self):
        with tempfile.TemporaryDirectory() as box:
            blocker = Path(box) / "a-file"
            blocker.write_text("not a directory\n")
            code = self._run("--dry-run", "--out-dir", str(blocker / "under"),
                             "64 distinct scaffolds from the cationic ZINC "
                             "library, primary amine")
            self.assertEqual(code, 1)


class AValueWithNoSettingIsRefusedNotRounded(unittest.TestCase):

    def test_an_unregistered_weight_is_refused(self):
        _, notes = occam_mod.parse(
            "64 distinct scaffolds from the cationic ZINC library, "
            "molecular weight under 500")
        refusal = next(item for item in notes["refused"]
                       if "500" in item["phrase"])
        self.assertEqual(refusal["available"], [450, 600])

    def test_an_unregistered_rotatable_count_is_refused(self):
        _, notes = occam_mod.parse(
            "64 distinct scaffolds from the cationic ZINC library, "
            "rotatable bonds at most 6")
        refusal = next(item for item in notes["refused"]
                       if "6" in item["phrase"])
        self.assertEqual(refusal["available"], [5, 7, 9])

    def test_a_registered_value_is_read(self):
        spec, notes = occam_mod.parse(
            "64 distinct scaffolds from the cationic ZINC library, "
            "primary amine, rotatable bonds at most 5, "
            "molecular weight strictly under 450")
        self.assertEqual(notes["refused"], [])
        self.assertEqual(settings_in(spec),
                         {"nitrogen": "primary_amine",
                          "rotatable": "rotatable_le_5",
                          "weight": "mw_under_450"})

    def test_strictness_stays_inside_its_clause(self):
        spec, _ = occam_mod.parse(
            "64 distinct scaffolds from the cationic ZINC library, "
            "molecular weight strictly under 450, primary amine")
        strict = {r["knob"]: r["strict"] for r in spec["requirements"]
                  if "knob" in r}
        self.assertTrue(strict["weight"])
        self.assertFalse(strict["nitrogen"])


if __name__ == "__main__":
    unittest.main()


class TheProductPathDoesNotReachTheDemotedModules(unittest.TestCase):
    """py/candidates.py, py/sequential.py and py/screen.py generate molecules
    and are kept for the record, not shipped. A product module importing one
    of them puts the demoted code back on the path -- and py/feasible.py did,
    for two SMARTS that the registry already defines. The copies were still
    identical when that was found; contraindication 31 is the same trap in the
    same file, found after they had diverged."""

    DEMOTED = ("candidates", "sequential", "screen")

    def _product_modules(self):
        import occam  # noqa: F401
        return [name for name in sys.modules
                if "." not in name and (ROOT / "py" / f"{name}.py").exists()]

    def test_no_product_module_imports_a_demoted_one(self):
        import re
        for name in self._product_modules():
            source = (ROOT / "py" / f"{name}.py").read_text()
            for demoted in self.DEMOTED:
                self.assertIsNone(
                    re.search(rf"^\s*import {demoted}\b", source, re.M),
                    f"py/{name}.py imports the demoted py/{demoted}.py")

    def test_the_amine_patterns_live_in_one_place(self):
        """Two definitions of a primary amine is one definition too many."""
        from rdkit import Chem

        import candidates as candidates_mod
        both = {
            "primary amine": (candidates_mod.ENTRY_AMINE,
                              knobs_mod.setting("nitrogen",
                                                "primary_amine")),
            "basic nitrogen": (candidates_mod.BASIC_N,
                               knobs_mod.setting("nitrogen",
                                                 "basic_nitrogen")),
        }
        # The demoted copy still exists. While it does, it has to agree with
        # the registry, because a reader comparing an old record against a new
        # one would otherwise be comparing two different patterns.
        for label, (old, _) in both.items():
            self.assertIsNotNone(Chem.MolToSmarts(old), label)
        import knobs
        self.assertEqual(Chem.MolToSmarts(candidates_mod.ENTRY_AMINE),
                         Chem.MolToSmarts(knobs.ENTRY_PRIMARY_AMINE))
        self.assertEqual(Chem.MolToSmarts(candidates_mod.BASIC_N),
                         Chem.MolToSmarts(knobs.BASIC_NITROGEN))


class TheExactRecountCanBeSkippedButNeverSilently(unittest.TestCase):
    """One pass over the library per shown specification dominates a short
    run, so there is a flag. What it must not do is make a faster answer look
    like the same answer."""

    def test_the_flag_exists(self):
        source = (ROOT / "py" / "occam.py").read_text()
        self.assertIn('"--no-exact"', source)
        self.assertIn("exact=not args.no_exact", source)

    def test_a_skipped_recount_says_the_number_is_an_estimate(self):
        import render as render_mod

        self.assertIn("sketch estimate",
                      render_mod.scaffold_phrase({"scaffolds_estimated": 118}))

    def test_an_undecided_specification_says_it_is_undecided(self):
        import render as render_mod

        self.assertIn("too close", render_mod.state("uncertain"))


class ConditionsBelongToOneBarrier(unittest.TestCase):
    """The registry's settings are the eNTRy rules, which are about entry
    into Gram-negative bacteria. Pointed at a library that measures another
    barrier they would run to completion and report nothing wrong -- the
    silence the charge-slice check exists to break, in another place."""

    def test_the_registry_records_which_barrier_it_is_about(self):
        self.assertIn("Gram-negative", knobs_mod.APPLIES_TO["barrier"])
        self.assertIn("Richter", knobs_mod.APPLIES_TO["source"])
        self.assertIn("blood-brain", knobs_mod.APPLIES_TO["not"])

    def test_the_bbb_data_is_nameable(self):
        """It has to be readable as a request before a mismatch can be
        reported; discarding the word would just lose it."""
        spec, notes = occam_mod.parse(
            "120 distinct scaffolds from the B3DB library, primary amine")
        self.assertEqual(spec["library"],
                         "data/work/B3DB_classification.parquet")
        for phrase in ("blood-brain barrier", "BBB"):
            spec, _ = occam_mod.parse(
                f"120 distinct scaffolds from the {phrase} set, primary amine")
            self.assertEqual(spec["library"],
                             "data/work/B3DB_classification.parquet")

    def test_every_named_library_has_its_barrier_recorded(self):
        """An unrecorded library must not default to matching the registry."""
        for _, path in occam_mod.LIBRARIES:
            self.assertIn(path, occam_mod.LIBRARY_BARRIER,
                          f"{path} has no barrier recorded, so a mismatch "
                          f"against it cannot be detected")

    def test_the_bbb_library_is_not_the_registry_barrier(self):
        self.assertNotEqual(
            occam_mod.LIBRARY_BARRIER["data/work/B3DB_classification.parquet"],
            knobs_mod.APPLIES_TO["barrier"])

    def test_zinc_has_no_barrier_because_it_is_unmeasured(self):
        """ZINC is a catalogue. It measures nothing, so it contradicts
        nothing, and the eNTRy conditions are what the request brings to it."""
        for path in ("data/work/zinc", "data/work/zinc_O", "data/work/zinc_P"):
            self.assertIsNone(occam_mod.LIBRARY_BARRIER[path])


class EveryDocumentedLibraryPhraseSelectsTheRightSlice(unittest.TestCase):
    """The library selector is the first thing every request touches, and it
    had two cases under test out of the ten the documentation promises.

    LIBRARIES is an ordered list matched first-wins, so a pattern that is too
    loose silently shadows every pattern after it. `dicationic ZINC` matched
    the -O pattern -- `cation(?:ic)?` with no word boundary finds "cationic"
    inside "dicationic" -- and a request for 17,775 compounds read 636,669 of
    a different slice instead. Nothing failed; it just answered about the
    wrong library.

    The phrases below are every one docs/prompt_format.md names. An ordered
    first-wins matcher needs exhaustive cases, not examples.
    """

    CASES = [
        ("cationic ZINC", "data/work/zinc_O"),
        ("protonated ZINC", "data/work/zinc_O"),
        ("ZINC O slice", "data/work/zinc_O"),
        ("the +1 ZINC slice", "data/work/zinc_O"),
        ("dicationic ZINC", "data/work/zinc_P"),
        ("ZINC +2 slice", "data/work/zinc_P"),
        ("multiply charged ZINC", "data/work/zinc_P"),
        ("ZINC P slice", "data/work/zinc_P"),
        ("neutral ZINC", "data/work/zinc"),
        ("ZINC N slice", "data/work/zinc"),
        ("ZINC", "data/work/zinc"),
        ("CHAMP", "data/work/champ.parquet"),
        ("IMI", "data/work/imi_clean.parquet"),
        ("GramNegAccum", "data/work/gna.parquet"),
        ("GNA", "data/work/gna.parquet"),
        ("B3DB", "data/work/B3DB_classification.parquet"),
        ("blood-brain", "data/work/B3DB_classification.parquet"),
        ("BBB", "data/work/B3DB_classification.parquet"),
    ]

    def test_every_phrase_selects_its_documented_slice(self):
        for phrase, expected in self.CASES:
            with self.subTest(phrase=phrase):
                spec, _ = occam_mod.parse(
                    f"120 distinct scaffolds from the {phrase} library, "
                    f"primary amine")
                self.assertEqual(spec["library"], expected,
                                 f"{phrase!r} selected {spec['library']}")

    def test_a_longer_word_is_not_matched_inside(self):
        """The failure: `cationic` found inside `dicationic`.

        Checked by running the patterns, not by inspecting them: an earlier
        version of this test searched the pattern text for bare words and
        reported `dication` as an unbounded `cation`, which is the same
        substring mistake it exists to catch.
        """
        import re
        # Only spellings the documentation names. Inventing one here would be
        # asserting a vocabulary nobody promised.
        longer = [("dicationic zinc", "cationic", "data/work/zinc_P")]
        for phrase, shadowed, expected in longer:
            with self.subTest(phrase=phrase):
                hits = [path for pattern, path in occam_mod.LIBRARIES
                        if re.search(pattern, phrase)]
                self.assertTrue(hits, f"{phrase!r} matched nothing")
                self.assertEqual(
                    hits[0], expected,
                    f"{phrase!r} was taken by the {shadowed!r} rule")

    def test_the_documentation_lists_no_phrase_the_parser_cannot_read(self):
        """A phrase promised in the documentation and not readable by the
        parser is worse than one that is absent: a requester uses it and the
        requirement is silently dropped.

        Checked on the whole phrase. It used to check the first word only, so
        "the +1 ZINC slice" passed on the strength of "the" appearing
        somewhere in a Japanese document -- a test that could not fail.
        """
        doc = (ROOT / "docs" / "prompt_format.md").read_text().lower()
        for phrase, _ in self.CASES:
            # The documentation writes these as the request would, so compare
            # on the words that carry the meaning rather than on articles.
            words = [w for w in phrase.lower().split()
                     if w not in ("the", "a", "an", "library", "set", "slice")]
            self.assertTrue(words, f"{phrase!r} is all function words")
            for word in words:
                self.assertIn(word, doc,
                              f"{phrase!r} is tested but {word!r} is not in "
                              f"docs/prompt_format.md")

    def test_which_pattern_matched_is_recorded(self):
        """First-wins ordering is invisible in the output, so the record has
        to say which rule fired."""
        spec, _ = occam_mod.parse(
            "120 distinct scaffolds from the dicationic ZINC library, "
            "primary amine")
        self.assertIn("library_matched", spec,
                      "nothing records which pattern selected the library")


class AnOrdinaryWordIsNotATarget(unittest.TestCase):
    """Found by the first external review. `top 50 to purchase` promoted
    `purchase` to a named target, the screen was abandoned as unanswerable,
    and the structural part was never evaluated.

    The -ase shape is the whole of what made it a target. The exclusion list
    held exactly two words, `disease` and `release`, which is a record of
    having been caught twice already; extending it does not help, because the
    next ordinary English word is not on it either."""

    def test_an_ordinary_word_ending_in_ase_is_not_promoted(self):
        # Words the -ase pattern actually catches. It needs four characters
        # before the suffix, so `phrase` and `lipase` are outside it either
        # way -- which also means the pattern misses real short enzyme names.
        for word in ("purchase", "increase", "database", "showcase"):
            with self.subTest(word=word):
                spec, notes = occam_mod.parse(
                    f"120 distinct scaffolds from the cationic ZINC library, "
                    f"primary amine, and give me the top 50 to {word}")
                targets = [r["target"] for r in spec["requirements"]
                           if "target" in r]
                self.assertNotIn(word, targets)
                self.assertIn(word, spec.get("target_candidates", []))

    def test_it_is_recorded_as_a_candidate_rather_than_silently_dropped(self):
        """parse() records it; check_targets() decides, because deciding needs
        the evidence and reading the evidence takes about a minute, which
        --dry-run must not pay."""
        spec, _ = occam_mod.parse(
            "120 distinct scaffolds from the cationic ZINC library, "
            "primary amine, and give me the top 50 to purchase")
        self.assertIn("purchase", spec.get("target_candidates", []))

    def test_reading_a_request_does_not_read_the_datasets(self):
        """It did, once per request, and a --dry-run that promises to answer
        immediately took a minute."""
        import time

        started = time.monotonic()
        occam_mod.parse("120 distinct scaffolds from the cationic ZINC "
                        "library, primary amine, top 50 to purchase")
        self.assertLess(time.monotonic() - started, 5.0,
                        "parse() is reading the evidence")

    def test_a_word_in_a_target_position_is_still_read(self):
        """The case this exists for: a target with no label anywhere, which
        has to reach the verdict that says so."""
        for phrase in ("an inhibitor of transglycosylase",
                       "a transglycosylase inhibitor",
                       "against transglycosylase"):
            with self.subTest(phrase=phrase):
                spec, _ = occam_mod.parse(
                    f"120 distinct scaffolds from the cationic ZINC library, "
                    f"primary amine, {phrase}")
                targets = [r["target"] for r in spec["requirements"]
                           if "target" in r]
                self.assertTrue(
                    any("transglycosylase" in t for t in targets),
                    f"{phrase!r} did not name a target")

    def test_the_structural_part_still_runs(self):
        """What the bug cost: the run stopped before screening anything."""
        spec, _ = occam_mod.parse(
            "120 distinct scaffolds from the cationic ZINC library, "
            "primary amine, rotatable bonds at most 5, "
            "and give me the top 50 to purchase")
        knobs = {r["knob"] for r in spec["requirements"] if "knob" in r}
        self.assertEqual(knobs, {"nitrogen", "rotatable"})
        self.assertEqual(spec["library"], "data/work/zinc_O")
