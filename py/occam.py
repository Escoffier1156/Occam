"""One command: a request in, a proposal out.

    occam "256 distinct scaffolds from the neutral ZINC library, MW under 450,
           primary amine, rotatable bonds at most 5"

    occam --spec request.json

The prose form is a convenience for the terminal. The structured form is what
belongs in a record: it is exactly what was asked, with nothing to interpret.

HOW THE PROSE IS READ, AND WHY NOT BY A MODEL

A language model could translate prose into the specification, and that was
rejected. The whole value of this pipeline is that each step can be checked,
and a model's reading cannot be. It would also be the one component with an
incentive to be helpful: asked for an impossible target it would find
something near enough, which is precisely the failure this pipeline exists to
prevent.

So the reading is a deterministic match against a closed vocabulary. That is
only possible because the vocabulary IS closed -- four knobs, ten settings, a
count. Ten destinations need no language model.

Three rules make it safe:

  refuse, never round    "MW under 500" has no setting. The nearest are 450
                         and 600, and both are named rather than one being
                         chosen. Silently rounding would put a number in the
                         record that nobody asked for.

  report what was dropped  A request carries claims about the data as well as
                         requests of it -- "the 100-million-item cache", "the
                         0.071 deficit". Those are not inputs. They are
                         listed, with what the measurement actually says, so a
                         mistaken premise surfaces instead of passing through.

  echo before running    The specification is a dozen lines. It is printed and
                         confirmed before anything streams a million molecules.
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import feasible as feasible_mod  # noqa: E402
import inventory as inventory_mod  # noqa: E402
import knobs as knobs_mod  # noqa: E402
import propose as propose_mod  # noqa: E402
import provenance as provenance_mod  # noqa: E402
import render as render_mod  # noqa: E402
import screenable as screenable_mod  # noqa: E402
import what_to_measure as what_to_measure_mod  # noqa: E402
import smiles as smiles_mod  # noqa: E402
import sweep as sweep_mod  # noqa: E402

WORK = ROOT / "data" / "work"

# Phrases that select a setting. Every destination is a registry entry, so
# nothing here can name a setting that does not exist.
VOCABULARY = {
    "nitrogen": [
        (r"\bprimary\s+amine", "primary_amine"),
        (r"\bprimary\s+or\s+secondary\s+amine", "primary_or_secondary_amine"),
        (r"\bsecondary\s+amine", "primary_or_secondary_amine"),
        (r"\bbasic\s+nitrogen", "basic_nitrogen"),
        (r"\bionis?able\s+nitrogen", "basic_nitrogen"),
        (r"\bionizable\s+nitrogen", "basic_nitrogen"),
    ],
}

# Charge used to be a knob, and reading it as one was reading the wrong
# quantity: see knobs.RETIRED['charge']. A request about charge is now refused
# rather than quietly redirected to a library slice, because silently turning
# the word a requester used into a different field is the same move as
# rounding "MW under 500" to 450.
#
# A charge word that the library phrase already consumed is NOT refused:
# "neutral ZINC" names the -N slice, which is a legitimate way to say this, and
# the library patterns match before these do.
CHARGE_WORDS = [
    r"\bzero\s+formal\s+charge\b",
    r"\bformal\s+charge\s+of\s+zero\b",
    r"\bany\s+charge\b",
    r"\bneutral\s+only\b",
    r"\b(?:cationic|anionic|zwitterionic)\b",
    r"\bneutral\b",
]

# Numeric knobs: the phrase gives a number, and the number must match a
# setting exactly. No rounding.
# Filler that can sit between a property and its number. The first version of
# these patterns allowed only "rotatable bonds at most 5", so a request reading
# "the rotatable bonds must be at most 5" matched nothing and the requirement
# was silently dropped into the unread list. The words permitted here carry no
# meaning of their own -- they are copulas, articles and the strictness markers,
# which are read separately.
_FILLER = (r"(?:\s+(?:must|should|shall|is|are|be|being|to|the|a|an|kept|held|"
           r"strictly|exactly|hard|required|fixed|no)\b)*")
_COMPARATOR = (r"(?:\s*(?:(?:at\s+most|at\s+maximum|no\s+more\s+than|up\s+to|"
               r"under|below|less\s+than|fewer\s+than|of)\b|<=|<|≤|=))?")

NUMERIC = {
    "rotatable": {
        "patterns": [r"rotatable(?:\s+bonds?)?" + _FILLER + _COMPARATOR + r"\s*(\d+)"],
        "limits": {5: "rotatable_le_5", 7: "rotatable_le_7", 9: "rotatable_le_9"},
        "unit": "rotatable bonds",
    },
    "weight": {
        "patterns": [r"(?:molecular\s+weight|mol(?:ecular)?\.?\s*wt\.?|mw)"
                     + _FILLER + _COMPARATOR + r"\s*(\d{3,4})"],
        "limits": {450: "mw_under_450", 600: "mw_under_600"},
        "unit": "daltons",
    },
}

# Words that mark a requirement as not negotiable. The first version of this
# parser set every requirement to relaxable, so "molecular weight strictly
# under 450" was read as a value to loosen -- and the proposer duly offered to
# loosen it. The word was in the request and was thrown away.
STRICT_MARKERS = [
    r"\bstrict(?:ly)?\b", r"\bmust\b", r"\bhard\b", r"\brequired\b",
    r"\bno\s+more\s+than\b", r"\bnon-?negotiable\b", r"\bexactly\b",
    r"\bfixed\b", r"\bdo\s+not\s+relax\b", r"\bkeep\b",
]
# How far from the matched phrase a strictness marker still counts as applying
# to it. A clause, not a sentence: "MW strictly under 450, primary amine" must
# not make the amine strict too.
STRICT_WINDOW = 40

DELIVERABLE = [
    r"(?:top\s+)?(\d[\d,]*)\s+(?:unique\s+|distinct\s+|different\s+)?"
    r"(?:scaffolds?|骨格)",
    r"(?:top\s+)?(\d[\d,]*)\s+(?:unique\s+|distinct\s+)?candidates?",
]

# Libraries, matched first-wins, so order and boundaries both decide the
# answer. The slice is named here because leaving it out of the vocabulary is
# what let it be chosen by whoever typed the download command, with no record
# and no check -- see py/provenance.py.
#
# THREE RULES, EACH WRITTEN AFTER GETTING IT WRONG
#
#   Most specific first. The +2 slice is tested before the +1 slice, which is
#   tested before the bare name, because a looser pattern earlier silently
#   shadows every pattern after it and nothing in the output says so.
#
#   Every alternative is bounded. `cation(?:ic)?` with no boundary found
#   "cationic" inside "dicationic": a request for the 17,775-compound slice
#   read 636,669 of a different one, completed normally, and answered about
#   the wrong library.
#
#   Lowercase only. parse() lowercases the request before matching, so an
#   uppercase letter in a pattern can never fire. `\bO\s+slice` and
#   `\bP\s+slice` were dead from the day they were written, and "ZINC O
#   slice" -- a spelling the documentation promises -- fell through to the
#   neutral slice, which is the one the published rule says does not
#   accumulate.
#
# tests/test_request.py checks every spelling docs/prompt_format.md names. An
# ordered first-wins matcher needs exhaustive cases, not examples.
LIBRARIES = [
    (r"\bzinc\b[^.]{0,40}?(?:\+2|\bdication(?:ic)?\b|\bmultiply\s+charged\b"
     r"|\bp\s+slice\b)", "data/work/zinc_P"),
    (r"(?:\+2|\bdication(?:ic)?\b|\bmultiply\s+charged\b)[^.]{0,40}?\bzinc\b",
     "data/work/zinc_P"),
    (r"\bzinc\b[^.]{0,40}?(?:\+1|\bcation(?:ic)?\b|\bprotonated\b"
     r"|\bo\s+slice\b)", "data/work/zinc_O"),
    (r"(?:\+1|\bcation(?:ic)?\b|\bprotonated\b)[^.]{0,40}?\bzinc\b",
     "data/work/zinc_O"),
    (r"\bzinc\b[^.]{0,40}?(?:\bneutral\b|\bn\s+slice\b)", "data/work/zinc"),
    (r"\bneutral\b[^.]{0,40}?\bzinc\b", "data/work/zinc"),
    (r"\bzinc", "data/work/zinc"),
    (r"\bimi", "data/work/imi_clean.parquet"),
    (r"\bchamp", "data/work/champ.parquet"),
    (r"\bgramnegaccum|\bgna\b", "data/work/gna.parquet"),
    # Blood-brain barrier. Nameable so that a request about it is read rather
    # than discarded, and so that the barrier mismatch below can fire. The
    # registry has no BBB settings, so a request to SCREEN this with the
    # Gram-negative conditions stops; a request ABOUT it -- is my set inside
    # its domain, how much would have to be measured -- is answerable.
    (r"\bb3db\b|blood[- ]brain|\bbbb\b",
     "data/work/B3DB_classification.parquet"),
]

# Which barrier a library is about, where that is recorded. A library absent
# from here is not assumed to match the registry: it is reported as unknown.
LIBRARY_BARRIER = {
    "data/work/zinc_O": None,
    "data/work/zinc_P": None,
    "data/work/zinc": None,
    "data/work/imi_clean.parquet": "the Gram-negative outer membrane",
    "data/work/champ.parquet": "the Gram-negative outer membrane",
    "data/work/gna.parquet": "the Gram-negative outer membrane",
    "data/work/B3DB_classification.parquet": "the blood-brain barrier",
}

# A word ending this way, or named as the object of "inhibitor", is read as a
# claimed biological target and checked against the evidence rather than used.
TARGET_SHAPES = [
    r"\b([a-z]{4,}(?:ase|ases))\b",
    r"\binhibitors?\s+(?:of|for)\s+(?:bacterial\s+|the\s+)?([a-z][a-z\- ]{3,30}?)\b",
    r"\b([a-z][a-z\-]{3,25})\s+inhibitors?\b",
]

# Numbers that assert something about the data rather than request something.
# Matching these is how a mistaken premise gets reported instead of ignored.
FACT_CLAIMS = [
    (r"(\d[\d,.]*)\s*[- ]?(?:million|billion|m\b)[- ]?item", "a library size"),
    (r"(\d+\.\d+)\s*auc", "an AUC value"),
    (r"auc\s*(?:deficit|gap|difference)\s*(?:of\s*)?(\d+\.\d+)", "an AUC difference"),
    (r"(\d[\d,]{3,})\s+(?:distinct\s+)?(?:diversity\s+)?domains?", "a domain count"),
    (r"\b(pre-?packed|verified|confirmed|proven|established)\b", "a verification status"),
]


def _is_strict(text, span):
    """Is there a strictness marker in the clause around this phrase, and
    where were the markers?

    The window stops at clause punctuation so a marker cannot leak across a
    comma into the next requirement.

    The spans are returned so the caller can record them as read. They are
    read here and nowhere else, so without that, "primary amine must be
    strictly required" put its markers in the list of phrases nothing was read
    from -- while the strictness they carried had in fact been applied. A
    reader checking that list for a lost requirement would have found a
    requirement that was not lost, which is the way to teach them to stop
    checking it.
    """
    start, end = span
    left = text.rfind(",", max(0, start - STRICT_WINDOW), start)
    left = left + 1 if left != -1 else max(0, start - STRICT_WINDOW)
    right = text.find(",", end, end + STRICT_WINDOW)
    right = right if right != -1 else min(len(text), end + STRICT_WINDOW)
    clause = text[left:right]
    spans = []
    for marker in STRICT_MARKERS:
        for found in re.finditer(marker, clause):
            spans.append((left + found.start(), left + found.end()))
    return bool(spans), spans


# A lead-in that ends in a colon governs everything after it: "The screening
# must satisfy the following criteria: A, B, C" means all three are hard. The
# clause window below deliberately stops at a comma so a marker cannot leak
# into the next requirement, which is right for a marker sitting beside one
# requirement and wrong for one introducing a list. Without this, "primary
# amine" in that sentence was read as relaxable and the proposer would have
# offered to loosen it -- contraindication 72, in the shape of
# contraindication 73.
LEAD_IN = re.compile(r"^[^:]{0,160}:")


def _binds_everything(text):
    """A strictness marker in a lead-in clause, with the span it occupies."""
    lead = LEAD_IN.match(text)
    if not lead:
        return False, []
    spans = []
    for marker in STRICT_MARKERS:
        for found in re.finditer(marker, lead.group(0)):
            spans.append(found.span())
    return bool(spans), spans


def _overlaps(span, consumed):
    """Was this span already read as part of something else?"""
    start, end = span
    return any(start < done_end and done_start < end
               for done_start, done_end in consumed)


def _unmatched_spans(text, consumed):
    """The parts of the request nothing was read from."""
    out, cursor = [], 0
    for start, end in sorted(consumed):
        if start > cursor:
            piece = text[cursor:start].strip(" ,.;:\n\t")
            if len(piece) > 3:
                out.append(piece)
        cursor = max(cursor, end)
    tail = text[cursor:].strip(" ,.;:\n\t")
    if len(tail) > 3:
        out.append(tail)
    return out


def parse(prose):
    """Prose to specification. Returns (spec, notes). Never guesses."""
    text = prose.lower()
    consumed, notes = [], {"dropped_claims": [], "refused": [], "unread": []}
    spec = {"goal": prose.strip(), "library": None, "deliverable": {},
            "requirements": []}

    # Read before the requirements, so each one can be marked strict as it is
    # added rather than revisited afterwards.
    all_strict, lead_spans = _binds_everything(text)
    consumed.extend(lead_spans)

    for pattern, kind in FACT_CLAIMS:
        for match in re.finditer(pattern, text):
            consumed.append(match.span())
            notes["dropped_claims"].append(
                {"phrase": match.group(0).strip(), "reads_as": kind,
                 "why": "a claim about the data, not a request of it. The "
                        "measurement is taken from the library, not from here."})

    for pattern, path in LIBRARIES:
        match = re.search(pattern, text)
        if match:
            spec["library"] = path
            # First-wins ordering is invisible in the output, so the record
            # says which rule fired and on what. Without it, a library chosen
            # by a pattern that should not have matched looks exactly like one
            # chosen correctly.
            spec["library_matched"] = {"phrase": match.group(0).strip(),
                                       "pattern": pattern}
            consumed.append(match.span())
            break

    for pattern in DELIVERABLE:
        match = re.search(pattern, text)
        if match:
            spec["deliverable"]["distinct_scaffolds"] = int(match.group(1).replace(",", ""))
            consumed.append(match.span())
            break

    for knob, entries in VOCABULARY.items():
        for pattern, setting in entries:
            match = re.search(pattern, text)
            if match:
                strict, markers = _is_strict(text, match.span())
                spec["requirements"].append({"knob": knob, "setting": setting,
                                             "strict": strict or all_strict})
                consumed.append(match.span())
                consumed.extend(markers)
                break

    for knob, body in NUMERIC.items():
        for pattern in body["patterns"]:
            match = re.search(pattern, text)
            if not match:
                continue
            consumed.append(match.span())
            value = int(match.group(1))
            if value in body["limits"]:
                strict, markers = _is_strict(text, match.span())
                spec["requirements"].append({"knob": knob,
                                             "setting": body["limits"][value],
                                             "strict": strict or all_strict})
                consumed.extend(markers)
            else:
                notes["refused"].append(
                    {"phrase": match.group(0).strip(),
                     "why": f"{value} {body['unit']} is not a registered setting",
                     "available": sorted(body["limits"]),
                     "unit": body["unit"],
                     # The reader of this line is the person making the
                     # request, who is not necessarily the person who
                     # maintains the registry. Naming a source file told them
                     # to go and edit Python.
                     "what_to_do": "ask for one of those instead. A new value "
                                   "is a change to the registry, and has to "
                                   "arrive with the source that justifies it"})
            break

    # A charge word outside everything already read. Inside a library phrase
    # it was the slice being named, which is how charge is chosen now.
    for pattern in CHARGE_WORDS:
        match = re.search(pattern, text)
        if not match or _overlaps(match.span(), consumed):
            continue
        consumed.append(match.span())
        notes["refused"].append(
            {"phrase": match.group(0).strip(),
             "why": "charge is chosen by the library slice, not by a "
                    "requirement. The retired knob read the formal charge of "
                    "the structure as drawn, which is a different quantity "
                    "from the protonation state ZINC predicts at pH 7.4",
             "available": [f"{letter} ({means})" for letter, means
                           in knobs_mod.BLOCKED["library_charge_slice"]
                           ["slices"].items()],
             "available_label": "registered slices",
             "what_to_do": "name the slice as part of the library instead, "
                           "for example \"cationic ZINC\" for -O"})
        break

    seen = set()
    for pattern in TARGET_SHAPES:
        for match in re.finditer(pattern, text):
            name = match.group(1).strip()
            if name in seen or name in ("disease", "release"):
                continue
            seen.add(name)
            consumed.append(match.span())
            spec["requirements"].append({"target": name})

    notes["unread"] = _unmatched_spans(prose, consumed)
    return spec, notes


def read_request():
    """A request typed at the prompt, ending with a blank line.

    Multi-line because a request is: a library, a deliverable and three
    conditions do not fit comfortably on one. Ends on a blank line rather
    than a key combination so that pasting several lines works.
    """
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print("Request? Type it, then a blank line. Ctrl-C to abandon.")
    print("What can be asked for: docs/prompt_format.md")
    lines = []
    try:
        while True:
            line = input("  " if lines else "> ")
            if not line.strip():
                if lines:
                    break
                continue
            lines.append(line.strip())
    except (EOFError, KeyboardInterrupt):
        print()
        return ""
    return " ".join(lines)


def stop(reason):
    """The one line a run ends on, wrapped so it reads as a sentence."""
    print()
    for line in render_mod.wrap(f"stopping: {reason}", hanging=2):
        print(line)
    return 1


def show(spec, notes, library_info=None):
    """The request said back to the reader, in the words of the chemistry.

    The identifiers the registry uses are not printed here. They are in
    request.json, which is the artefact meant to be read without
    interpretation; this is the one meant to be read once, quickly, by someone
    checking that the request they made is the request that was understood.
    """
    width = 20
    print("read as:")
    library = (render_mod.library_phrase(library_info) if library_info
               else (spec["library"] or "(not named)"))
    print(f"  {'library':{width}} {library}")
    for key, value in spec["deliverable"].items():
        print(f"  {'deliverable':{width}} {value:,} {key.replace('_', ' ')}")
    print()
    for requirement in spec["requirements"]:
        if "target" in requirement:
            print(f"  {'target':{width}} {requirement['target']}")
            print(f"  {'':{width}} to be checked against the labels that exist")
        else:
            for line in render_mod.requirement_lines(
                    requirement["knob"], requirement["setting"],
                    requirement["strict"], width):
                print(line)

    if notes["dropped_claims"]:
        print("\nnot taken as input (claims about the data):")
        for item in notes["dropped_claims"]:
            print(f"  \"{item['phrase']}\" reads as {item['reads_as']}")
            print(f"    {item['why']}")
    if notes["refused"]:
        print("\nrefused rather than rounded:")
        for item in notes["refused"]:
            for line in render_mod.wrap(f"\"{item['phrase']}\": {item['why']}",
                                        indent=2, hanging=4):
                print(line)
            available = render_mod.or_list(item["available"])
            unit = f" {item['unit']}" if item.get("unit") else ""
            label = item.get("available_label", "registered")
            for line in render_mod.wrap(f"{label}: {available}{unit}",
                                        indent=4, hanging=6):
                print(line)
            for line in render_mod.wrap(item["what_to_do"], indent=4,
                                        hanging=6):
                print(line)
    for line in render_mod.unread_lines(notes["unread"]):
        print(line)


def check_targets(spec):
    """Every named target, against the labels that actually exist."""
    names = [r["target"] for r in spec["requirements"] if "target" in r]
    if not names:
        return []
    datasets = inventory_mod.take(WORK)
    out = []
    for name in names:
        out.append(feasible_mod.label_available(datasets, [name]))
    return out


def _survivor_smiles(library, choices, limit=None):
    """SMILES of the molecules one specification keeps. Not returned to the
    caller of this tool -- only counted and measured over."""
    variant = sweep_mod.Variant(choices)
    nitrogen = {s["name"]: s for s in knobs_mod.settings_of("nitrogen")}
    from rdkit.Chem import Descriptors, rdMolDescriptors

    out = []
    for text in sweep_mod.iter_library(library, limit):
        mol = smiles_mod.parse(text)
        if mol is None:
            continue
        if Descriptors.MolWt(mol) >= variant.mw_limit:
            continue
        if rdMolDescriptors.CalcNumRotatableBonds(mol) > variant.rot_limit:
            continue
        if not nitrogen[choices["nitrogen"]]["test"](mol):
            continue
        out.append(text)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("request", nargs="?", default=None, help="the request, as prose")
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many molecules (0 = the whole library)")
    parser.add_argument("--yes", action="store_true", help="do not ask before running")
    parser.add_argument("--dry-run", action="store_true",
                        help="read the request and stop")
    parser.add_argument("--out-dir", type=Path, default=WORK)
    parser.add_argument("--no-exact", action="store_true",
                        help="skip the exact recount of the frontier. One "
                             "pass over the library per shown specification, "
                             "so it dominates a short run; the numbers come "
                             "back as sketch estimates and say so")
    parser.add_argument("--no-measure", action="store_true",
                        help="skip the count of what would have to be "
                             "measured when the set is out of domain")
    parser.add_argument("--no-screenable", action="store_true",
                        help="skip the two verdicts about what a screen would "
                             "establish (they cost one scoring pass over a sample)")
    args = parser.parse_args()

    # Made before anything is read, so an unwritable --out-dir is a sentence
    # rather than a traceback thrown after the whole request has been echoed.
    # Creating a directory the caller named is not a guess; it is the only
    # reading that phrase has.
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return stop(f"cannot write to --out-dir {args.out_dir}: "
                    f"{error.strerror or error}")

    if args.spec:
        spec = json.loads(args.spec.read_text())
        notes = {"dropped_claims": [], "refused": [], "unread": []}
    elif args.request:
        spec, notes = parse(args.request)
    else:
        # No request on the command line: ask for one. Putting a four-line
        # request inside shell quotes every time is not how anyone works, and
        # it is how this was used all day.
        request = read_request()
        if not request:
            parser.error("give a request as prose, or --spec a file")
        spec, notes = parse(request)

    # The library itself is an input. It had no record and no guard once, and
    # the screen it produced reported the opposite of the right answer. It is
    # identified before the request is echoed, so that the echo can name the
    # library the way a chemist would rather than by its path on this disk.
    # The slice check is about ZINC's charge suffixes. Running it on a
    # measured dataset reports "the library does not record where it came
    # from", which is true and says nothing: that dataset has no slices.
    is_zinc = str(spec.get("library") or "").startswith("data/work/zinc")
    library_check = provenance_mod.report(spec["library"], spec["requirements"]) \
        if is_zinc else None

    show(spec, notes, library_check["library"] if library_check else None)
    (args.out_dir / "request.json").write_text(json.dumps(spec, indent=2) + "\n")

    target_checks = check_targets(spec)
    if target_checks:
        print("\ntargets named in the request:")
        for check in target_checks:
            answerable = check["verdict"] == "answerable"
            state = "can be checked" if answerable else "cannot be answered"
            print(f"  {check['requirement']}: {state}")
            if check.get("missing"):
                for line in render_mod.wrap(check["missing"], indent=4,
                                            hanging=6):
                    print(line)

    if library_check:
        info = library_check["library"]
        print("\nabout that library:")
        note = (info["note"] if info["known"]
                else f"the charge slice could not be read: {info['why']}")
        for line in render_mod.wrap(note, indent=2):
            print(line)
        contradiction = library_check["contradiction"]
        if contradiction:
            headline = ("this contradicts the request"
                        if contradiction["severity"] == "contradiction"
                        else "consistent with the request")
            for line in render_mod.wrap(f"{headline}: {contradiction['what']}",
                                        indent=2, hanging=4):
                print(line)
            for key, label in (("evidence", "published evidence"),
                               ("measured_here", "measured here"),
                               ("what_to_do", "what to do")):
                if contradiction.get(key):
                    for line in render_mod.wrap(
                            f"{label}: {contradiction[key]}",
                            indent=4, hanging=6):
                        print(line)

    # The registry's settings are the eNTRy rules and say so. A library about
    # another barrier screened with them would run to completion and report
    # nothing wrong -- the same silence the charge-slice check exists to break.
    barrier = LIBRARY_BARRIER.get(spec.get("library"), "unrecorded")
    asked_knobs = [r for r in spec["requirements"] if "knob" in r]
    if asked_knobs and barrier not in (None, knobs_mod.APPLIES_TO["barrier"]):
        print("\nabout these conditions:")
        for line in render_mod.wrap(
                f"they are {knobs_mod.APPLIES_TO['source']}, and they are "
                f"about {knobs_mod.APPLIES_TO['barrier']}.", indent=2,
                hanging=4):
            print(line)
        named = ("not recorded for this library" if barrier == "unrecorded"
                 else barrier)
        for line in render_mod.wrap(f"this library is about {named}.",
                                    indent=2, hanging=4):
            print(line)
        for line in render_mod.wrap(knobs_mod.APPLIES_TO["not"], indent=2,
                                    hanging=4):
            print(line)
        for line in render_mod.wrap(
                f"what would extend it: "
                f"{knobs_mod.APPLIES_TO['what_would_extend_it']}", indent=2,
                hanging=4):
            print(line)
        if not args.yes:
            return stop("the conditions asked for are a published rule about "
                        "a different barrier from the one this library "
                        "measures. Pass --yes to run anyway, having read why.")

    blocking = [c for c in target_checks if c["verdict"] == "not_answerable"]
    if library_check and (library_check["contradiction"] or {}).get("severity") \
            == "contradiction" and not args.yes:
        # A refusal with no revision is half an answer (contraindication 77).
        # The slices that are on this disk are named, with what the swap is
        # worth, so the next command is in front of the reader.
        alternatives = provenance_mod.alternative_slices(spec["library"])
        if alternatives["slices"]:
            print("\nwhat to screen instead (measured, on this disk):")
            for row in alternatives["slices"]:
                share = row["primary_amine_share"]
                rows = (f"{row['rows']:,} compounds"
                        if isinstance(row["rows"], int) else "count unrecorded")
                enriched = (f", primary amines {share:.2%} of it"
                            if share is not None else "")
                for line in render_mod.wrap(
                        f"{row['slice']} slice ({row['means']}): {rows}"
                        f"{enriched}", indent=2, hanging=4):
                    print(line)
            for line in render_mod.wrap(
                    f"shares measured on {alternatives['measured_on']}. The "
                    f"requirements are unchanged by the swap: say the slice in "
                    f"the library instead, for example \"cationic ZINC\" for "
                    f"-O.", indent=2, hanging=4):
                print(line)
        return stop("the library contradicts the published rule the request "
                    "is built on. Pass --yes to run anyway, having read why.")
    if blocking:
        print()
        for line in render_mod.wrap(
                "verdict: this cannot be answered. A target named in the "
                "request has no label anywhere in the evidence, so no setting "
                "of any requirement makes it answerable.", hanging=2):
            print(line)
        # The revision. A refusal without one is half an answer.
        available = screenable_mod.answerable_questions()
        print("\nwhat the evidence can answer instead "
              "(measured, not a chemical suggestion):")
        for line in render_mod.wrap(
                f"missing resolution: {available['resolution_absent']} - "
                f"{available['why']}", indent=2, hanging=4):
            print(line)
        if available.get("scope"):
            # The row counts live in the scope sentence screenable.py writes.
            # They were once restated here as well, and two copies of a count
            # in one output is the failure this pipeline is built to catch.
            for line in render_mod.wrap(f"scope of the counts: "
                                        f"{available['scope']}",
                                        indent=2, hanging=4):
                print(line)
        if available["resolution_available"]:
            print("  labels that do exist:")
            for row in available["resolution_available"]:
                print(f"    {render_mod.label_phrase(row['label']):24} "
                      f"active {row['active']:>6,}  "
                      f"inactive {row['inactive']:>6,}")
        if available.get("continuous"):
            for line in render_mod.wrap(available["continuous"], indent=4,
                                        hanging=6):
                print(line)
        for line in render_mod.wrap(f"substitution: "
                                    f"{available['substitution']}",
                                    indent=2, hanging=4):
            print(line)
        print()
        for line in render_mod.wrap(
                "The structural part of the request can still be screened. "
                "Rerun without the target to do that.", hanging=2):
            print(line)

    if notes["refused"]:
        # Two kinds of refusal arrive here: a value with no registered
        # setting, and a word for a quantity that is not a requirement. The
        # reason is worded for both, because guessing at either would be the
        # same failure.
        return stop("the request named something this cannot read as a "
                    "requirement. Nothing was rounded and nothing was "
                    "redirected to a different field.")
    if not spec.get("library"):
        return stop("no library named.")
    if not spec["deliverable"]:
        return stop("no deliverable named. 'N distinct scaffolds' is what "
                    "the feasibility test needs.")
    if args.dry_run or blocking:
        return 0

    if not args.yes:
        try:
            # NFKC before comparing: an IME leaves the full-width 'ｙ', which
            # is not 'y', and the run stopped on a yes.
            reply = unicodedata.normalize(
                "NFKC", input("\nrun this? [y/N] ")).strip().lower()
            if reply not in ("y", "yes", "はい"):
                print("stopped.")
                return 0
        except EOFError:
            print("\nno terminal to confirm on; pass --yes to run unattended.")
            return 1

    baseline = {}
    strict = {}
    for requirement in spec["requirements"]:
        if "knob" in requirement:
            baseline[requirement["knob"]] = requirement["setting"]
            strict[requirement["knob"]] = requirement.get("strict", False)
    for knob in knobs_mod.KNOBS:
        baseline.setdefault(knob, knobs_mod.settings_of(knob)[0]["name"])
        strict.setdefault(knob, False)

    variants = [v for v in sweep_mod.enumerate_variants(baseline)
                if all(v.choices[k] == baseline[k] for k in baseline if strict[k])]
    held = sum(strict.values())
    print(f"\ncomparing {len(variants)} specifications in a single pass over "
          f"the library")
    print(f"  {held} requirement(s) held as asked, the rest varied")

    summary = sweep_mod.run(spec["library"], variants, args.limit or None)
    rows = [v.report(summary["evaluated"]) for v in variants]
    rows.sort(key=lambda r: -r["survivors"])
    sweep_report = {"baseline": baseline, "summary": summary, "specifications": rows}
    (args.out_dir / "sweep.json").write_text(json.dumps(sweep_report, indent=2) + "\n")
    print(f"  {summary['evaluated']:,} molecules read in {summary['seconds']}s, "
          f"holding {summary['state_bytes_total'] / 1024:,.0f} KiB of state")

    inert = provenance_mod.inert_knobs(rows)
    if inert:
        print("\nrequirements that did not change the result:")
        for line in render_mod.inert_lines(inert):
            print(line)
    sweep_report["inert_knobs"] = inert
    (args.out_dir / "sweep.json").write_text(json.dumps(sweep_report, indent=2) + "\n")

    required = spec["deliverable"].get("distinct_scaffolds", 0)
    result = propose_mod.propose(sweep_report, required, spec["library"],
                                 args.limit or None, exact=not args.no_exact)
    (args.out_dir / "proposal.json").write_text(json.dumps(result, indent=2) + "\n")

    print()
    for line in render_mod.wrap(f"verdict: {result['verdict']}", hanging=2):
        print(line)
    if result.get("as_asked"):
        a = result["as_asked"]
        print(f"\nas asked: {a['survivors']:,} molecules pass, "
              f"{render_mod.scaffold_phrase(a)} - "
              f"{render_mod.state(a['state'])}")
    if result.get("recommended"):
        r = result["recommended"]
        # When nothing was loosened, the recommendation IS the request. Saying
        # "recommended instead" there would imply a substitution that did not
        # happen.
        print("\nthe request as asked:" if not r["loosened"]
              else "\nrecommended instead:")
        for line in render_mod.wrap(render_mod.spec_phrase(r["choices"]),
                                    indent=2, hanging=4):
            print(line)
        print(f"  {r['survivors']:,} molecules pass, "
              f"{render_mod.scaffold_phrase(r)}")
        for knob, setting in r["loosened"].items():
            body = knobs_mod.setting(knob, setting)
            print(f"  loosened {render_mod.prop(knob)} to "
                  f"{render_mod.reads(knob, setting)}")
            for line in render_mod.wrap(body["source"], indent=4, hanging=6):
                print(line)
    if result.get("best_reachable"):
        print("\nwhat would unblock it:")
        for item in result["what_would_unblock"]:
            for line in render_mod.wrap(item, indent=2, hanging=4):
                print(line)
    # What a screen of this specification would and would not establish. These
    # used to be gates inside the candidate pipeline, where they were reached
    # only after a list of molecules had been produced. They are verdicts.
    measured_cost = False
    screen_body = None
    screen_cost = None
    if not args.no_screenable and result.get("recommended"):
        chosen = result["recommended"]["choices"]
        print()
        survivors = _survivor_smiles(spec["library"], chosen, args.limit or None)
        if len(survivors) < 50:
            for line in render_mod.wrap(
                    f"what a screen of this specification would establish: "
                    f"nothing yet. {len(survivors)} survivors is too few to "
                    f"measure either an ordering or an applicability domain "
                    f"on.", hanging=2):
                print(line)
        else:
            body = screenable_mod.report(survivors)
            screen_body = body

            # Computed before the report is printed so that the count of what
            # would close the verdict appears inside the verdict rather than
            # as a second document after it.
            extrapolating = str(body["applicability"]["verdict"]).startswith(
                "answerable_but_extrapolating")
            cost = None
            if extrapolating and not args.no_measure:
                cost = what_to_measure_mod.report(
                    survivors,
                    dataset=render_mod.library_phrase(library_check["library"])
                    if library_check else spec["library"])
                measured_cost = True
                screen_cost = cost

            screenable_mod.show(body, cost)
            (args.out_dir / "screenable.json").write_text(
                json.dumps(body, indent=2) + "\n")
            if cost is not None:
                (args.out_dir / "what_to_measure.json").write_text(
                    json.dumps(cost, indent=2) + "\n")

    for line in render_mod.answer(
            spec, result, body=screen_body, cost=screen_cost,
            library=render_mod.library_phrase(library_check["library"])
            if library_check else spec.get("library")):
        print(line)

    written = ["request.json", "sweep.json", "proposal.json"]
    if not args.no_screenable:
        written.append("screenable.json")
    if measured_cost:
        written.append("what_to_measure.json")
    print(f"\n-> {args.out_dir}/" + ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
