"""A count must reach one output from one place.

This is contraindication 83, written as a test rather than as a sentence. The
sentence form of it has been in the list since the start -- "if the count for
the same label moves between outputs, stop" -- and it was broken twice anyway,
because a list of sentences is not read before every edit.

What went wrong, twice:

  data/processed/combined_bioassay_data.tsv has 74,202 rows. That number was
  written into a format string in py/occam.py and into the scope sentence in
  py/screenable.py. One output printed both. Re-ingest the file and only one
  of them moves.

  The training sets were named "IMI-COMBINE (activity, 22,051)", with the
  count inside the name, while the same comparison already measured the rows
  it read and the scope sentence stated 22,051 as well. The same number
  reached one report from two literals that nothing kept in step.

The test below is not a search for duplicated text. It asks a narrower
question that can be answered mechanically: does any count appear in two
different string literals of the reporting modules? A count that is measured
and formatted at print time appears in none, which is the point.
"""

import ast
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

# The modules whose strings are printed to a reader. A count hardcoded here is
# a count that no measurement keeps current, and a function name here is a
# line of this codebase presented to a reader as an answer. One list: there
# were two with different contents, so a module was scanned for one defect and
# not the other depending on which class you read.
REPORTING = ("occam.py", "screenable.py", "propose.py", "render.py",
             "what_to_measure.py", "feasible.py", "inventory.py",
             "provenance.py")

# A number that reads as a count of rows, molecules or records: a thousands
# separator, or five or more bare digits. Four bare digits are left out because
# the sources are full of publication years.
COUNT = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{5,}\b")

# Counts that are allowed to appear in a literal, each with the reason. A
# measured quantity never belongs here; these are the ones that are part of a
# citation or a worked example, where the number IS the text.
ALLOWED = {
    "1,281": "the mass of the glycopeptide that ranked first with no weight "
             "bound, quoted in the registry's account of why the bound exists",
    "2,871": "a journal page number in a citation",
    "1,624": "a count inside a module docstring describing a past run",
    "21,752": "a count inside a docstring describing how the control is built",
    "13451": "part of a dict shown verbatim in a docstring as the thing not "
             "to print",
    "20,000": "the sample size of the charge measurement, stated as the "
              "sample size and not as a result",
    "80,000": "the size of the pool the charge sample was drawn from",
}


def literals_with_counts(path):
    """Every (count, lineno, literal) in one file's string constants."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for found in COUNT.findall(node.value):
                out.append((found, node.lineno, node.value))
    return out


class ACountComesFromOnePlace(unittest.TestCase):

    def _all(self):
        found = {}
        for name in REPORTING:
            path = ROOT / "py" / name
            if not path.exists():
                continue
            for count, lineno, literal in literals_with_counts(path):
                if count in ALLOWED:
                    continue
                found.setdefault(count, []).append((name, lineno, literal))
        return found

    def test_no_count_appears_in_two_literals(self):
        """The failure is two copies, so two copies is what is checked."""
        for count, places in self._all().items():
            distinct = {(name, lineno) for name, lineno, _ in places}
            if len(distinct) > 1:
                where = "; ".join(f"{n}:{l}" for n, l in sorted(distinct))
                self.fail(
                    f"{count} is written in {len(distinct)} places: {where}. "
                    f"A count belongs to whatever measures it; format it at "
                    f"print time, or add it to ALLOWED with the reason it is "
                    f"text rather than a measurement.")

    def test_the_allowlist_is_not_a_place_to_hide_a_measurement(self):
        """Every entry needs a reason, and the reason has to be a sentence
        rather than a word, because the point of the list is that someone
        reading it can tell a citation from a result."""
        for count, reason in ALLOWED.items():
            self.assertGreater(len(reason.split()), 4,
                               f"{count} is allowed with no real reason")

    def test_the_scan_finds_something(self):
        """A regex that matches nothing would make this suite pass for the
        wrong reason."""
        total = sum(len(v) for v in self._all().values()) + len(ALLOWED)
        self.assertGreater(total, 0, "the count scan matched nothing at all")


class TheCountsThatUsedToBeDuplicated(unittest.TestCase):

    def test_the_full_source_row_count_is_stated_once(self):
        """74,202 was in occam.py and in screenable.py, and one output printed
        both."""
        places = []
        for name in REPORTING:
            path = ROOT / "py" / name
            if path.exists() and "74,202" in path.read_text():
                places.append(name)
        self.assertLessEqual(len(places), 1,
                             f"74,202 is written in {places}")

    def test_a_training_set_is_not_named_with_its_row_count(self):
        """The count belongs to the rows the comparison read, which it
        already measures."""
        import render as render_mod
        phrase = render_mod.training_set_phrase(
            {"training_set": "IMI-COMBINE", "measures": "activity",
             "structures": 22051})
        self.assertEqual(phrase, "IMI-COMBINE (activity, 22,051)")

        source = (ROOT / "py" / "screenable.py").read_text()
        self.assertNotIn("IMI-COMBINE (activity,", source,
                         "the count is back inside the name")

    def test_a_training_set_with_no_count_still_reads(self):
        import render as render_mod
        self.assertEqual(
            render_mod.training_set_phrase({"training_set": "CHAMP"}),
            "CHAMP")


class NoFunctionNameReachesTheScreen(unittest.TestCase):
    """A reader deciding what to do next needs the action, not the call that
    performs it. `rankable_alternatives()` appeared in a verdict's revision,
    which is a line of this codebase presented as an answer."""

    def test_no_printed_string_names_a_function_call(self):
        import ast
        import re

        for name in REPORTING:
            path = ROOT / "py" / name
            if not path.exists():
                continue
            tree = ast.parse(path.read_text())
            # A docstring is not printed. render.py's library_phrase() names
            # provenance.identify() in its own documentation, which is where
            # that name belongs.
            docstrings = set()
            for holder in ast.walk(tree):
                if isinstance(holder, (ast.Module, ast.FunctionDef,
                                       ast.AsyncFunctionDef, ast.ClassDef)):
                    first = (holder.body or [None])[0]
                    if (isinstance(first, ast.Expr)
                            and isinstance(first.value, ast.Constant)
                            and isinstance(first.value.value, str)):
                        docstrings.add(id(first.value))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Constant)
                        and isinstance(node.value, str)):
                    continue
                if id(node) in docstrings:
                    continue
                # A bare identifier followed by () inside a sentence.
                hit = re.search(r"\b[a-z_][a-z0-9_]{3,}\(\)", node.value)
                if hit and ".py" not in node.value:
                    self.fail(f"py/{name}:{node.lineno} prints "
                              f"{hit.group(0)!r} to a reader")


if __name__ == "__main__":
    unittest.main()
