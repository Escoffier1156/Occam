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
