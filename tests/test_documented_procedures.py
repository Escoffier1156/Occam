"""What the repository says to do, the suite does.

FOUR OF ONE DAY'S FAULTS WERE THE SAME FAULT

  the alternative    "measure the frontier, about 32 minutes"  -- never
                     measured whether it would change the answer. It could
                     not: the answer was already known
  the source label   `measured` beside SKETCH_RELATIVE_ERROR   -- the
                     measurement did not exist
  the sampling       `sample and say so instead` in a comment  -- the
                     sampling was implemented and the saying so was not
  the procedure      RECORD.md's re-recording steps            -- read a
                     prompts.txt that is not in this repository

Four contraindications, one root: correct when written, and never run by
anyone. A document is not a description of the repository, it is a claim about
it, and a claim nothing executes decays at the speed the code moves.

docs/prompt_format.md has been held this way for some time -- every spelling
it lists is put through the parser by tests/test_request.py, so an example in
it that does not work is a failing build. This module generalises that to the
rest of what the documents assert, as far as it can be asserted cheaply:

  paths        a path a procedure names exists, exactly
  filenames    a bare filename resolves somewhere in the repository
  flags        a documented flag is one the program's argparse accepts
  features     a documented cargo feature is declared in Cargo.toml
  suite        the documented module list is every test module there is
  runnable     the entry points the documents promise answer --help

WHAT IS DELIBERATELY NOT EXECUTED

The container build, the ZINC fetch and the twenty-minute re-recording. A
check that costs twenty minutes is a check that gets switched off, which is
the failure this module exists to prevent, arriving by another road. Those
three are verified by the parts of them that are cheap: their paths, their
flags and their features.
"""

import ast
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Every document that tells a reader to do something.
DOCS = ("README.md", "NOTICE", "docs/HANDOFF.md", "docs/prompt_format.md",
        "docs/algorithms.md", "docs/provenance.md", "docs/final_report.md",
        "tests/fixtures/prompts/RECORD.md")

RUNNABLE = re.compile(r"```bash\n(.*?)```", re.S)

# A path rooted in one of this repository's directories.
ROOTED = re.compile(r"(?:^|[\s'\"=(])((?:py|tests|docs|engine|data)/[A-Za-z0-9_./-]+)")

# A filename with no directory on it. `prompts.txt` was one of these.
#
# THE BOUNDARIES ARE SHELL PUNCTUATION, NOT WHITESPACE
#
# This required whitespace on both sides, and the fault it was written for
# reads
#
#     prompt=$(sed -n "${n}p" prompts.txt)
#
# so the name ends on a paren and the pattern missed it. Validating against
# the repaired RECORD.md reported nothing wrong, which is how a regex gets
# believed: it was measured on a document from which the fault had already
# been removed. Injecting the fault back is what found it.
BARE = re.compile(r"""(?:^|[\s'"=(])([A-Za-z0-9_-]+\.(?:py|txt|json|nix|toml|md|gz))"""
                  r"""(?=[\s'")\];|&]|$)""", re.M)

# The program must sit at a command position: the start of a line, or just
# after a shell operator. `podman build ... -t occam .` names a container tag.
PROGRAM = re.compile(r"(?:^|[|;]|&&|\$\()\s*(?:python3\s+(?:-u\s+)?)?"
                     r"(py/[a-z_]+\.py|\./occam|occam)((?:\s+[^|;&\n]*)?)", re.M)

FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")
ARGPARSE_BUILTIN = {"--help", "-h"}

# Any source file named anywhere in the prose, not only inside a block.
SOURCE = re.compile(r"\b((?:py|tests|engine/src)/[A-Za-z0-9_]+\.(?:py|rs))")


def read(doc):
    return (ROOT / doc).read_text()


def blocks(doc):
    """Each runnable block in `doc`, as (line number, text)."""
    text = read(doc)
    return [(text[:m.start()].count("\n") + 1, m.group(1))
            for m in RUNNABLE.finditer(text)]


def accepted_flags(script):
    """Every --flag the script's argparse accepts, read from its source."""
    out = set(ARGPARSE_BUILTIN)
    tree = ast.parse((ROOT / script).read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("-"):
                    out.add(str(arg.value))
    return out


def repository_filenames():
    """Every tracked file, by basename. git is the index of what exists."""
    listed = subprocess.run(["git", "ls-files"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.split()
    names = {}
    for path in listed:
        names.setdefault(Path(path).name, []).append(path)
    return names


class EveryProcedureNamesSomethingThatExists(unittest.TestCase):

    def test_a_rooted_path_exists_exactly(self):
        checked = 0
        for doc in DOCS:
            for line, block in blocks(doc):
                for token in ROOTED.findall(block):
                    # A token the shell completes: $n, a glob, a <placeholder>,
                    # or a line continuation. There is nothing to resolve.
                    tail = block[block.index(token) + len(token):][:1]
                    if re.search(r"[$<>{}*]", token) or tail in ("$", "\\"):
                        continue
                    checked += 1
                    self.assertTrue((ROOT / token).exists(),
                                    f"{doc}:{line} names {token}, which does not exist")
        # 43 today. A floor is here because a regex that stops matching
        # passes silently, which is contraindication 102's shape; it is set
        # below the count rather than at it so that deleting a procedure is
        # not a test failure.
        self.assertGreater(checked, 30, f"only {checked} paths found; the "
                                        f"regex has probably stopped matching")

    def test_a_bare_filename_resolves_somewhere_in_the_repository(self):
        """RECORD.md read a `prompts.txt` that was nowhere.

        Not required to sit at the root: `occam --spec request.json` names a
        file the reader writes, and `request.json` is a real name in this
        project -- seven of them are tracked under tests/fixtures. The claim
        worth checking is weaker and still catches the fault: a filename a
        document uses must be a filename this repository knows.
        """
        names = repository_filenames()
        checked = 0
        for doc in DOCS:
            for line, block in blocks(doc):
                for name in BARE.findall(block):
                    if re.search(r"[$<>{}*]", name):
                        continue
                    checked += 1
                    self.assertIn(
                        name, names,
                        f"{doc}:{line} uses {name}, which this repository has "
                        f"nowhere. That is how RECORD.md came to read a "
                        f"prompts.txt that was never committed")
        # TWO. The population really is this small, and saying so is better
        # than a floor that looks substantial.
        #
        # Almost every filename in these documents is written with its
        # directory -- 43 of them -- and a rooted path is checked exactly, by
        # the test above. That habit is why this fault was rare enough to
        # survive: RECORD.md's prompts.txt was one of the two places where it
        # was not followed. So the floor is 1, and it says only that the
        # regex still matches something.
        self.assertGreater(checked, 0, "the bare-filename regex matches nothing")

    def test_every_source_file_named_in_prose_exists(self):
        checked = 0
        for doc in DOCS:
            text = read(doc)
            for m in SOURCE.finditer(text):
                checked += 1
                self.assertTrue(
                    (ROOT / m.group(1)).exists(),
                    f"{doc}:{text[:m.start()].count(chr(10)) + 1} names "
                    f"{m.group(1)}, which does not exist")
        # 118 today.
        self.assertGreater(checked, 90, f"only {checked} names found; the "
                                        f"regex has probably stopped matching")


class EveryDocumentedFlagIsAFlag(unittest.TestCase):

    def test_no_procedure_passes_a_flag_the_program_rejects(self):
        checked = 0
        for doc in DOCS:
            for line, block in blocks(doc):
                for found in PROGRAM.finditer(block):
                    program, rest = found.group(1), found.group(2)
                    script = ("py/occam.py" if program in ("occam", "./occam")
                              else program)
                    if not (ROOT / script).exists():
                        continue
                    allowed = accepted_flags(script)
                    for flag in FLAG.findall(rest):
                        checked += 1
                        self.assertIn(flag, allowed,
                                      f"{doc}:{line} passes {flag} to {script}, "
                                      f"which does not accept it")
        # 7 today. A program position that stops matching reads as green,
        # and this check is one regex away from checking nothing.
        self.assertGreater(checked, 5, f"only {checked} flags found; the "
                                       f"program-position regex has probably "
                                       f"stopped matching")

    def test_every_documented_cargo_feature_is_declared(self):
        declared = set(re.findall(r"^([a-z_][a-z0-9_]*)\s*=\s*\[",
                                  (ROOT / "engine" / "Cargo.toml").read_text(),
                                  re.M))
        checked = 0
        for doc in DOCS:
            for line, block in blocks(doc):
                for feature in re.findall(r"--features\s+([a-z_]+)", block):
                    checked += 1
                    self.assertIn(feature, declared,
                                  f"{doc}:{line} builds --features {feature}, "
                                  f"which engine/Cargo.toml does not declare")
        # Two: the default build takes no --features.
        self.assertEqual(checked, 2, "the three engine builds are the "
                                     "documented ones; two of them name a feature")


class TheDocumentedSuiteIsTheWholeSuite(unittest.TestCase):
    """The module list is the only list.

    There is no discover() here on purpose: it once collected nothing and
    exited 0, which reads as green (contraindication 102). The price of an
    explicit list is that a new test module can be written and never run, and
    nothing would say so. This is that price being paid.
    """

    def documented_lists(self):
        """Each (where, modules) for a command naming more than one module.

        One module is a targeted command -- OCCAM_FACTORS against the memory
        invariant, say. A list is a claim about what the suite is.
        """
        out = {}
        for doc in DOCS:
            text = read(doc)
            for m in re.finditer(r"python3 -m unittest ((?:tests/\S+\.py|\\\n|[ \t])+)",
                                 text):
                modules = sorted(set(re.findall(r"tests/(\S+\.py)", m.group(1))))
                if len(modules) > 1:
                    out[f"{doc}:{text[:m.start()].count(chr(10)) + 1}"] = modules
        return out

    def test_the_documented_list_is_every_test_module(self):
        real = sorted(p.name for p in (ROOT / "tests").glob("test_*.py"))
        lists = self.documented_lists()
        self.assertTrue(lists, "no documented suite invocation found at all")
        for where, modules in lists.items():
            self.assertEqual(
                sorted(set(real) - set(modules)), [],
                f"{where} does not run these, so they run nowhere")
            self.assertEqual(
                sorted(set(modules) - set(real)), [],
                f"{where} names modules that do not exist")

    def test_every_documented_list_is_the_same_list(self):
        lists = self.documented_lists()
        distinct = {tuple(v) for v in lists.values()}
        self.assertEqual(len(distinct), 1,
                         f"the suite is documented {len(distinct)} different "
                         f"ways: {sorted(lists)}")


class ThePromisedEntryPointsRun(unittest.TestCase):
    """Static checks say a flag is spelled right. These say it runs.

    Kept to --help, which costs one process each. The container build, the
    ZINC fetch and the re-recording are not run: a twenty-minute check is one
    that gets switched off, which is this module's own failure arriving by
    another road.
    """

    def test_the_launcher_the_documents_name_answers_help(self):
        # py/occam.py directly, not ./occam: the launcher enters nix-shell,
        # and the suite is already inside it.
        done = subprocess.run([sys.executable, "py/occam.py", "--help"],
                              cwd=ROOT, capture_output=True, text=True)
        self.assertIn("usage", (done.stdout + done.stderr).lower())

    def test_every_script_a_procedure_runs_answers_help(self):
        scripts = set()
        for doc in DOCS:
            for _, block in blocks(doc):
                for found in PROGRAM.finditer(block):
                    program = found.group(1)
                    if program.startswith("py/") and (ROOT / program).exists():
                        scripts.add(program)
        self.assertTrue(scripts, "no scripts found in any procedure")
        for script in sorted(scripts):
            done = subprocess.run([sys.executable, script, "--help"],
                                  cwd=ROOT, capture_output=True, text=True)
            self.assertIn("usage", (done.stdout + done.stderr).lower(),
                          f"{script} --help printed no usage:\n{done.stderr[:400]}")


if __name__ == "__main__":
    unittest.main()
