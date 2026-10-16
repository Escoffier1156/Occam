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

from hypothesis import given, settings, strategies as st

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
        # The shape a record from the second-pass window has. py/sweep.py
        # counts exactly now, so new records carry distinct_scaffolds, and
        # both have to read correctly.
        phrase = render_mod.scaffold_phrase(row)
        self.assertIn("119", phrase)
        self.assertNotIn("118", phrase)
        self.assertNotIn("about", phrase)

    def test_a_sketch_estimate_says_it_is_one(self):
        phrase = render_mod.scaffold_phrase({"scaffolds_estimated": 118})
        self.assertIn("about 118", phrase)
        self.assertIn("sketch", phrase)

    def test_an_exact_count_from_the_sweep_says_so(self):
        phrase = render_mod.scaffold_phrase({"distinct_scaffolds": 539,
                                             "counted_exactly": True})
        self.assertIn("539", phrase)
        self.assertIn("counted exactly", phrase)
        self.assertNotIn("about", phrase)

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


class TheAnswerBlockCarriesTheWholeAnswer(unittest.TestCase):
    """The run prints 88 lines under seven headings with the verdict on line
    34. The detail stays -- it is where the numbers come from -- but a person
    at a prompt reads one block, in the order a decision is made in."""

    def _answer(self, result, **kw):
        return "\n".join(render_mod.answer({}, result, **kw))

    def test_an_infeasible_answer_states_the_ceiling(self):
        """The number most wanted when the answer is no. It was dropped,
        because best_reachable carries it under the key the sweep used."""
        text = self._answer({
            "request": {"distinct_scaffolds_required": 120},
            "best_reachable": {"scaffolds_estimated": 39},
        })
        self.assertIn("39", text)
        self.assertIn("120", text)
        self.assertIn("not feasible", text)

    def test_a_loosened_answer_names_what_was_loosened(self):
        text = self._answer({
            "as_asked": {"distinct_scaffolds": 119, "counted_exactly": True,
                         "survivors": 123},
            "recommended": {"distinct_scaffolds": 124, "counted_exactly": True,
                            "survivors": 128, "loosened": {"weight": "mw_under_600"},
                            "choices": {"nitrogen": "primary_amine",
                                        "rotatable": "rotatable_le_5",
                                        "weight": "mw_under_600"}},
        })
        self.assertIn("119", text)
        self.assertIn("124", text)
        self.assertIn("under 600 Da", text)

    def test_a_feasible_answer_says_so_without_a_substitution(self):
        text = self._answer({
            "recommended": {"distinct_scaffolds": 187, "counted_exactly": True,
                            "survivors": 430, "loosened": {},
                            "choices": {"nitrogen": "primary_amine"}},
        })
        self.assertIn("feasible as asked", text)
        self.assertIn("187", text)
        self.assertNotIn("loosened", text)

    def test_the_three_pillars_appear_in_decision_order(self):
        text = self._answer(
            {"recommended": {"distinct_scaffolds": 187, "counted_exactly": True,
                             "survivors": 430, "loosened": {},
                             "choices": {"nitrogen": "primary_amine"}}},
            body={"rankability": {"verdict": "not_answerable"},
                  "applicability": {"verdict": "answerable_but_extrapolating",
                                    "control": {"verdict": "pass"}}},
            cost={"measurable": True, "compounds_to_measure": 99,
                  "for_half_the_set": 15})
        feasible = text.index("feasible as asked")
        ordered = text.index("could be ordered")
        measure = text.index("99 compounds")
        self.assertLess(feasible, ordered)
        self.assertLess(ordered, measure)

    def test_the_answer_never_promises_molecules(self):
        text = self._answer({"recommended": {"distinct_scaffolds": 1,
                                             "counted_exactly": True,
                                             "survivors": 1, "loosened": {},
                                             "choices": {}}})
        self.assertIn("No molecules are emitted", text)


class TheDisplayBoundary(unittest.TestCase):
    """No line leaves this process carrying a Python repr.

    Three defects have come out of the display layer -- contraindication E (a
    sentence template whose slot read as nonsense), G (a universal claim
    printed beside the max() that contradicted it) and 108 (a dict formatted
    with an f-string). These tests hold the boundary that makes the third
    class unreachable rather than detectable; see
    render.display_boundary() for why it is not a recorded string to compare
    against.
    """

    def test_rejects_a_whole_dict(self):
        self.assertIsNotNone(
            render_mod.repr_shape("to settle it: {'measured_on': 1820}"))

    def test_rejects_a_fragment_with_no_brace_on_it(self):
        # wrap() breaks a long repr across lines. Of the 12 lines
        # contraindication 108 put on the screen, 9 carried no brace at all,
        # so a check for `{'` alone would have passed most of the damage.
        self.assertIsNotNone(
            render_mod.repr_shape("      'reads_as': 'the same count at a"))

    def test_rejects_a_list_of_numbers(self):
        self.assertIsNotNone(
            render_mod.repr_shape("  timed at [221.89, 220.56, 196.36] seconds"))

    def test_rejects_an_object_with_no_str(self):
        self.assertIsNotNone(
            render_mod.repr_shape("  library: <knobs.Registry object at 0x7f1>"))

    def test_rejects_an_unresolved_percent_format(self):
        self.assertIsNotNone(render_mod.repr_shape("  %s distinct scaffolds"))

    def test_passes_the_sentences_the_tool_actually_writes(self):
        for line in (
            "  radius 4, 8,192 bits: 846 distinct -- unchanged",
            "  ionisable nitrogen   an unhindered primary amine   keeps 0 of 1,132 (0.0%)",
            "  median 0.217, best 0.371, threshold 0.4",
            "      94.0% at or above the threshold, so the measurement can clear it",
            "  the count is a property of the threshold: 0.30: 196, 0.40: 318",
        ):
            self.assertIsNone(render_mod.repr_shape(line), line)

    def test_a_chosen_out_dir_may_contain_braces(self):
        # The last line of a run prints --out-dir, which is the caller's path.
        # Bare `{` is deliberately not a repr shape for this reason.
        self.assertIsNone(
            render_mod.repr_shape("-> /tmp/out-{weird}/request.json, sweep.json"))

    def test_the_boundary_raises_rather_than_printing(self):
        import io as io_mod
        sink = io_mod.StringIO()
        with self.assertRaises(render_mod.Unrenderable):
            with render_mod.display_boundary(sink):
                print("ordinary prose, 846 distinct of 1,820")
                print({"measured_on": 1820, "distinct_fingerprints": 846})
        self.assertIn("846 distinct of 1,820", sink.getvalue())
        self.assertNotIn("measured_on", sink.getvalue(),
                         "the offending line reached the reader anyway")

    def test_the_boundary_is_transparent_to_ordinary_output(self):
        import io as io_mod
        sink = io_mod.StringIO()
        with render_mod.display_boundary(sink):
            for line in render_mod.wrap("846 distinct fingerprints among "
                                        "1,820 molecules (46.5%)", indent=4):
                print(line)
        self.assertEqual(sink.getvalue(),
                         "    846 distinct fingerprints among 1,820 molecules (46.5%)\n")

    def test_a_partial_line_is_checked_at_flush(self):
        # input()'s prompt is written without a newline, so flush() emits a
        # partial line. It must still pass the boundary, and must not have a
        # newline added to it.
        import io as io_mod
        sink = io_mod.StringIO()
        with render_mod.display_boundary(sink):
            sys.stdout.write("> ")
        self.assertEqual(sink.getvalue(), "> ")

        sink = io_mod.StringIO()
        with self.assertRaises(render_mod.Unrenderable):
            with render_mod.display_boundary(sink):
                sys.stdout.write("{'groups': 577")

    def test_stdout_is_restored_even_when_the_boundary_fires(self):
        import io as io_mod
        original = sys.stdout
        with self.assertRaises(render_mod.Unrenderable):
            with render_mod.display_boundary(io_mod.StringIO()):
                print({"a": 1})
        self.assertIs(sys.stdout, original)

    @settings(max_examples=300, deadline=None, database=None)
    @given(value=st.one_of(
        st.dictionaries(st.text(min_size=1, max_size=12), st.integers(), min_size=1),
        st.lists(st.integers(), min_size=1),
        st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=2),
    ))
    def test_no_container_can_be_interpolated(self, value):
        """The property, not the instance: a container cannot reach the reader.

        This is what makes the class unreachable rather than caught. A
        detection test fixes the dict that was found; this one says that any
        dict or list dropped into an f-string stops the run.
        """
        self.assertIsNotNone(render_mod.repr_shape(f"to settle it: {value}"),
                             f"{value!r} would have printed as a repr")


class EveryRenderedLineFromEveryFixture(unittest.TestCase):
    """The recorded runs, rendered, line by line through the boundary.

    The fixtures hold JSON rather than text on purpose, so this does not
    compare against a recorded sentence -- it renders what the records say and
    asserts that no line of it carries a repr. A sentence can be improved
    without touching anything here, and a value interpolated without a
    formatter fails whichever run it appears in.
    """

    FIXTURES = ROOT / "tests" / "fixtures" / "prompts"

    def test_no_rendered_line_carries_a_repr(self):
        import json
        rendered = 0
        for run in sorted(self.FIXTURES.glob("run*")):
            def load(name):
                path = run / name
                return json.loads(path.read_text()) if path.exists() else None

            spec, proposal = load("request.json"), load("proposal.json")
            if proposal is None:
                continue  # run2 stops before a proposal exists
            body, cost = load("screenable.json"), load("what_to_measure.json")
            lines = list(render_mod.answer(spec, proposal, body=body, cost=cost,
                                           library=spec.get("library")))
            if body is not None:
                lines += list(render_mod.detail(body, cost, proposal))
            self.assertTrue(lines, f"{run.name} rendered nothing")
            for line in lines:
                self.assertIsNone(render_mod.repr_shape(line),
                                  f"{run.name}: {line}")
            rendered += len(lines)
        # Contraindication 102's shape: a loop that found nothing reads green.
        self.assertGreater(rendered, 200,
                           "too few lines rendered to have checked anything")
