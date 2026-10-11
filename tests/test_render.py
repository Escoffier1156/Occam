"""The display layer must stay in step with the registry.

py/render.py holds the phrases a reader sees; py/knobs.py holds the settings a
screen runs. Keeping them in separate files keeps cosmetic edits out of the
file where an edit changes a result, and the price is that the two can drift.
These tests are that price being paid: a setting added to the registry without
a phrase, or a phrase whose number disagrees with the limit it claims to state,
fails here.
"""

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import knobs as knobs_mod  # noqa: E402
import render as render_mod  # noqa: E402

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


class EveryRegistryEntryIsSayable(unittest.TestCase):

    def test_every_knob_has_a_property_name(self):
        for knob in knobs_mod.KNOBS:
            self.assertIn(knob, render_mod.PROPERTY,
                          f"knob {knob!r} would print as its identifier")

    def test_every_setting_has_a_phrase_and_a_citation(self):
        for knob, body in knobs_mod.KNOBS.items():
            for setting in body["settings"]:
                key = (knob, setting["name"])
                self.assertIn(key, render_mod.READS,
                              f"{key} would print as its identifier")
                self.assertIn(key, render_mod.CITE,
                              f"{key} would print with no source")

    def test_no_phrase_names_a_setting_that_does_not_exist(self):
        """The drift runs both ways: a phrase left behind after a setting was
        removed would be dead text nobody notices."""
        for knob, name in render_mod.READS:
            self.assertIn(knob, knobs_mod.KNOBS, f"no knob {knob!r}")
            names = [s["name"] for s in knobs_mod.settings_of(knob)]
            self.assertIn(name, names, f"{knob} has no setting {name!r}")

    def test_a_phrase_states_the_limit_the_setting_actually_uses(self):
        """A readable phrase that rounds 450 to 500 would be worse than the
        identifier it replaced, because it reads as authoritative."""
        for knob, body in knobs_mod.KNOBS.items():
            for setting in body["settings"]:
                limit = setting.get("limit")
                if limit is None or limit == float("inf"):
                    continue
                phrase = render_mod.READS[(knob, setting["name"])]
                found = {float(n) for n in _NUMBER.findall(phrase)}
                self.assertIn(float(limit), found,
                              f"{setting['name']} limits at {limit} but reads "
                              f"{phrase!r}")
                self.assertEqual(found, {float(limit)},
                                 f"{phrase!r} states a number that is not "
                                 f"the limit {limit}")

    def test_a_registry_caution_is_not_left_unprinted(self):
        """A caution exists because the knob's name misleads. One added to the
        registry and not surfaced here would be a warning written and never
        read."""
        for knob, body in knobs_mod.KNOBS.items():
            if body.get("caution"):
                self.assertIn(knob, render_mod.CAUTION,
                              f"{knob} carries a caution that never prints")


class Verdicts(unittest.TestCase):
    """Every verdict py/screenable.py can return has to be sayable.

    The list is read out of the source rather than copied, so a verdict added
    there without a statement here fails instead of printing its field value.
    """

    def _literals(self):
        source = (ROOT / "py" / "screenable.py").read_text()
        return set(re.findall(r'"verdict":\s*"([^"]+)"', source)) | set(
            re.findall(r'else\s+"([a-z_]+)"', source))

    def test_every_verdict_in_the_source_has_a_statement(self):
        for literal in self._literals():
            self.assertIn(literal, render_mod.VERDICT,
                          f"verdict {literal!r} would print as a field value")

    def test_a_verdict_never_reaches_the_screen_as_an_identifier(self):
        for literal in render_mod.VERDICT:
            self.assertNotIn("_", render_mod.verdict(literal))

    def test_an_unknown_verdict_degrades_to_words(self):
        self.assertEqual(render_mod.verdict("some_new_state"),
                         "some new state")

    def test_a_control_says_what_a_control_says(self):
        """"the control can be done" is not what a passing control reports."""
        self.assertEqual(render_mod.control_verdict("pass"), "passes")
        self.assertNotIn("_", render_mod.control_verdict("not_measurable"))


class RecordShapesDoNotReachTheScreen(unittest.TestCase):

    def test_a_label_tally_is_words_not_a_dict(self):
        phrase = render_mod.counts_phrase({"positive": 8600, "negative": 13451})
        self.assertEqual(phrase, "8,600 positive, 13,451 negative")
        for character in "{}':":
            self.assertNotIn(character, phrase)

    def test_a_tally_key_with_an_underscore_is_said_as_words(self):
        self.assertEqual(render_mod.counts_phrase({"not_tested": 7}),
                         "7 not tested")

    def test_a_label_column_is_said_as_the_thing_it_records(self):
        self.assertEqual(render_mod.label_phrase("gram_positive_label"),
                         "gram positive")

    def test_an_exact_scaffold_count_is_not_reported_as_an_estimate(self):
        """propose.py counts exactly whenever the sketch cannot decide. Once
        that count has set the verdict, "about 118" understates what is known
        and misdescribes the verdict."""
        row = {"scaffolds_estimated": 118, "survivors": 123,
               "exact": {"scaffolds_exact": 119, "survivors": 123}}
        phrase = render_mod.scaffold_phrase(row)
        self.assertIn("119", phrase)
        self.assertNotIn("118", phrase)
        self.assertNotIn("about", phrase)

    def test_a_sketch_estimate_says_it_is_one(self):
        phrase = render_mod.scaffold_phrase({"scaffolds_estimated": 118})
        self.assertIn("about 118", phrase)
        self.assertIn("sketch", phrase)

    def test_a_provenance_says_what_it_licenses(self):
        self.assertEqual(render_mod.provenance("published"),
                         "inside a published rule")
        self.assertIn("no chemical claim",
                      render_mod.provenance("relaxation"))

    def test_a_provenance_keeps_its_secondary_source_note(self):
        """The 600 Da bound is published via a verbatim quote in a paper that
        was read, from one that was not. Dropping the parenthetical would
        promote it to a source this project checked."""
        value = knobs_mod.setting("weight", "mw_under_600")["provenance"]
        phrase = render_mod.provenance(value)
        self.assertIn("inside a published rule", phrase)
        self.assertIn("secondary", phrase)

    def test_every_provenance_in_the_registry_is_sayable(self):
        for knob, body in knobs_mod.KNOBS.items():
            for setting in body["settings"]:
                phrase = render_mod.provenance(setting["provenance"])
                self.assertNotEqual(phrase, setting["provenance"],
                                    f"{setting['name']} prints its field value")

    def test_a_feasibility_state_is_a_statement(self):
        self.assertEqual(render_mod.state("infeasible"),
                         "cannot fill the request")
        for value in ("feasible", "infeasible", "uncertain"):
            self.assertNotIn("_", render_mod.state(value))

    def test_a_choice_of_values_reads_as_a_list(self):
        self.assertEqual(render_mod.or_list([5, 7, 9]), "5, 7 or 9")
        self.assertEqual(render_mod.or_list([450, 600]), "450 or 600")
        self.assertEqual(render_mod.or_list([5]), "5")
        self.assertEqual(render_mod.or_list([]), "")

    def test_a_specification_reads_as_a_clause(self):
        phrase = render_mod.spec_phrase({
            "nitrogen": "primary_amine", "rotatable": "rotatable_le_5",
            "weight": "mw_under_450", "charge": "neutral_only"})
        self.assertNotIn("_", phrase)
        self.assertIn("under 450 Da", phrase)


class UnreadFragments(unittest.TestCase):

    def test_grammar_is_suppressed_and_counted(self):
        content, suppressed = render_mod.partition_unread(
            ["from the", "library", "effectively penetrate the outer membrane"])
        self.assertEqual(content, ["effectively penetrate the outer membrane"])
        self.assertEqual(suppressed, 2)

    def test_a_possible_requirement_is_never_suppressed(self):
        for fragment in ("efflux pump inhibitor", "logD below 1",
                         "porin", "must cross the outer membrane"):
            content, suppressed = render_mod.partition_unread([fragment])
            self.assertEqual(content, [fragment], f"{fragment!r} was hidden")
            self.assertEqual(suppressed, 0)

    def test_suppressing_everything_reports_it_without_raising_an_alarm(self):
        """The count is always shown, so a reader who distrusts the split can
        see that it happened. The warning is not, because nothing was lost and
        a warning printed on every run is a warning nobody reads."""
        lines = render_mod.unread_lines(["from the", "of"])
        self.assertTrue(any("2 leftover fragment(s)" in line for line in lines))
        self.assertFalse(any("lost" in line for line in lines))

    def test_a_dropped_requirement_is_always_warned_about(self):
        lines = render_mod.unread_lines(["from the", "crosses the inner membrane"])
        self.assertTrue(any("lost" in line for line in lines))
        self.assertTrue(any("crosses the inner membrane" in line for line in lines))
        self.assertTrue(any("1 further fragment(s)" in line for line in lines))


class LibraryPhrase(unittest.TestCase):

    def test_a_known_slice_reads_as_chemistry_not_a_path(self):
        phrase = render_mod.library_phrase({
            "known": True, "slice": "O", "means": "+1 at pH 7.4",
            "rows": 636669, "source": "x_H29P240-O.smi.gz",
            "library": "data/work/zinc_O"})
        self.assertEqual(
            phrase,
            "ZINC22 tranche H29P240, -O slice (+1 at pH 7.4), 636,669 compounds")
        self.assertNotIn("data/work", phrase)

    def test_an_unidentified_library_says_so_and_keeps_the_path(self):
        phrase = render_mod.library_phrase({
            "known": False, "library": "data/work/mystery",
            "why": "no parts.json"})
        self.assertIn("data/work/mystery", phrase)
        self.assertIn("not identified", phrase)

    def test_an_unrecorded_count_is_not_invented(self):
        phrase = render_mod.library_phrase({
            "known": True, "slice": "P", "means": "+2 or more at pH 7.4",
            "rows": None, "source": "x_H29P240-P.smi.gz"})
        self.assertIn("count unrecorded", phrase)


class NoIdentifierReachesTheScreen(unittest.TestCase):

    def test_requirement_lines_carry_no_underscores(self):
        for knob, body in knobs_mod.KNOBS.items():
            for setting in body["settings"]:
                lines = render_mod.requirement_lines(
                    knob, setting["name"], strict=False)
                for line in lines:
                    self.assertNotIn("_", line,
                                     f"identifier reached the screen: {line!r}")

    def test_strictness_is_words_not_a_boolean(self):
        lines = render_mod.requirement_lines("weight", "mw_under_450", True)
        self.assertTrue(any("not negotiable" in line for line in lines))
        lines = render_mod.requirement_lines("weight", "mw_under_450", False)
        self.assertTrue(any("may be relaxed" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
