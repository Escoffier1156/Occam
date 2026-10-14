"""How a request and a verdict are said to a reader, as opposed to recorded.

Everything here is presentation. The record written to data/work/ keeps the
registry's own identifiers -- primary_amine, mw_under_450, rotatable_le_5 --
because the point of the record is that it can be read years later with no
interpretation. The terminal is the opposite case: it is read once, by a
chemist, who should not have to know that a threshold is spelled
`mw_under_450` to see that it says 450 daltons.

So the identifiers never reach the screen and never leave the JSON.

WHY THE PHRASES LIVE HERE AND NOT IN THE REGISTRY

A setting in py/knobs.py carries its test, its provenance and its source. Those
decide what a screen does. A display phrase decides nothing, and putting it in
the registry would invite editing the registry for cosmetic reasons, which is
the one file where an accidental edit changes a result.

The cost of the separation is that the two tables can drift: a setting added to
the registry with no phrase here would print as a bare identifier again.
tests/test_render.py fails in that case, which is the whole reason it exists.
A missing phrase also degrades visibly rather than silently -- the identifier
is printed as-is, never guessed at from its spelling.
"""

import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import knobs as knobs_mod  # noqa: E402

# Prose is wrapped rather than left to the terminal, because the lines that
# matter most here are the longest: a caution runs to three lines and a single
# unwrapped paragraph is the form a reader skips.
WIDTH = 78


def wrap(text, indent=0, hanging=None):
    """`text` as lines no wider than WIDTH, indented, continuation deeper."""
    first = " " * indent
    rest = " " * (indent if hanging is None else hanging)
    return textwrap.wrap(text, width=WIDTH, initial_indent=first,
                         subsequent_indent=rest) or [first.rstrip()]

# The property a knob constrains, named as a chemist would name it rather than
# as the code names it. "rotatable" and "weight" are column names; "flexibility"
# and "molecular weight" are properties.
PROPERTY = {
    "nitrogen": "ionisable nitrogen",
    "rotatable": "flexibility",
    "weight": "molecular weight",
}

# What each setting asks for, as a phrase that can be read aloud. The numbers
# here must match the registry's own limits; tests/test_render.py checks every
# number that appears in a phrase against the limit it claims to state.
READS = {
    ("nitrogen", "primary_amine"): "an unhindered primary amine",
    ("nitrogen", "primary_or_secondary_amine"):
        "a primary or secondary amine on carbon",
    ("nitrogen", "basic_nitrogen"): "any nitrogen basic near pH 7.4",
    ("rotatable", "rotatable_le_5"): "at most 5 rotatable bonds",
    ("rotatable", "rotatable_le_7"): "at most 7 rotatable bonds",
    ("rotatable", "rotatable_le_9"): "at most 9 rotatable bonds",
    ("weight", "mw_under_450"): "under 450 Da",
    ("weight", "mw_under_600"): "under 600 Da",
    ("weight", "mw_unbounded"): "no upper bound on weight",
}

# One line naming where the setting comes from. The registry's `source` is the
# full statement and is printed by --provenance; this is the version that fits
# beside the phrase without pushing it off the line.
CITE = {
    ("nitrogen", "primary_amine"):
        "Richter 2017, first eNTRy condition, best case",
    ("nitrogen", "primary_or_secondary_amine"):
        "Richter 2017, first eNTRy condition",
    ("nitrogen", "basic_nitrogen"):
        "Richter 2017, first eNTRy condition, broadest reading",
    ("rotatable", "rotatable_le_5"): "Richter 2017, third eNTRy condition",
    ("rotatable", "rotatable_le_7"): "the published limit of 5, moved",
    ("rotatable", "rotatable_le_9"): "the published limit of 5, moved",
    ("weight", "mw_under_450"): "chosen by this study; no published bound",
    ("weight", "mw_under_600"):
        "O'Shea & Moser 2008, quoted verbatim in Richter 2017",
    ("weight", "mw_unbounded"): "Richter 2017 sets no bound on size",
}

# The one sentence a reader has to see even if they read nothing else about the
# knob. Both of these record a measurement that contradicts the obvious reading
# of the knob's name, so they are printed every time the knob is used, not
# filed behind a flag. The registry holds the full text.
CAUTION = {
    "weight": "not a permeability bound. Richter 2017 reports no correlation "
              "between accumulation and molecular weight; measured here, "
              "450 -> 600 changes survivors 1.1x against 2.8x for rotatable "
              "bonds. Read it as a size bound.",
}

# Function words, plus the nouns that name the request's own machinery. A
# fragment built only from these carries no requirement, so listing it under
# "nothing was read from" only teaches the reader to skim that list -- which is
# the one list that has to be read. Anything outside this set is content and is
# always shown in full.
FUNCTION_WORDS = frozenset("""
a an and are as at be been being but by do does for from had has have how if in
into is it its of on or over so than that the their then there these this those
to under up upon was were what when where which while with within without
library libraries dataset data set collection compounds molecules molecule
items entries rows please want need give me us i we you
""".split())

_WORD = re.compile(r"[a-z0-9]+")
_TRANCHE = re.compile(r"^(?:x_)?(?P<tranche>.+?)-[NOMP](?:\.[a-z])?\.smi\.gz$")


def reads(knob, setting):
    """The phrase for a setting, or the identifier if none is written here.

    Falling back to the identifier is deliberate. A guess assembled from the
    spelling of the name would read as fluent English and could be wrong about
    a number, which is worse than printing something the reader can see is
    machine output and report.
    """
    return READS.get((knob, setting), setting)


def prop(knob):
    return PROPERTY.get(knob, knob)


def strictness(strict):
    return "not negotiable" if strict else "may be relaxed"


def requirement_lines(knob, setting, strict, width=20):
    """What was asked, where it comes from, and any caution on the knob."""
    # Strictness sits on the first line, with the phrase it governs. Wrapped
    # into the end of the citation it was the last thing on the last line,
    # which is where it was least likely to be read -- and it decides whether
    # the proposer may trade the requirement away.
    left = 2 + width + 1
    out = [f"  {prop(knob):{width}} {reads(knob, setting)}  "
           f"({strictness(strict)})"]
    cite = CITE.get((knob, setting))
    if cite:
        out += wrap(cite, indent=left)
    if knob in CAUTION:
        out += wrap(f"note: {CAUTION[knob]}", indent=left, hanging=left + 6)
    # N: the output names the first and third eNTRy conditions. Someone
    # reading only the output cannot tell that the second one is missing, and
    # naming two of three implies the third was applied.
    if cite and "eNTRy" in cite:
        out += wrap("the second eNTRy condition, globularity at most 0.25, is "
                    "not evaluated here: it needs a 3D conformer and nothing "
                    "in this pipeline generates one", indent=left,
                    hanging=left + 2)
    return out


# The verdicts screenable.py returns, as statements. A verdict is the one word
# a reader carries away from a run, and `not_measurable` is not a word: it is a
# field value that happens to be legible.
VERDICT = {
    "pass": "can be done",
    "pass for the range, weak per molecule":
        "can be done across the range, but weakly for any single molecule",
    "answerable": "can be answered",
    "answerable_but_extrapolating":
        "can be answered only by extrapolating past the training set",
    "not_answerable": "cannot be answered",
    "not_measurable": "could not be measured",
    # The share that decides this verdict rises as fewer molecules are
    # measured, so a sample can put a set below the floor but cannot put one
    # above it. Neither "can" nor "cannot" is the honest word for that: the
    # measurement did not reach.
    "not_settled_at_this_sample": "not settled by what was measured",
}


def verdict(value):
    """A verdict as a statement; unknown values degrade to readable text."""
    return VERDICT.get(value, str(value).replace("_", " "))


# The two kinds of provenance a setting may have, as what they license rather
# than as the words the registry files them under. The distinction is the whole
# audit: either a paper already says it, or a number was moved and no claim was
# made.
PROVENANCE = {
    "published": "inside a published rule",
    "relaxation": "a threshold moved; no chemical claim",
    "unjustified": "A NEW CHEMICAL CLAIM - must never be proposed",
}


def provenance(value):
    """A provenance as what it licenses. A trailing parenthetical on a
    published entry, such as the secondary-source note on the 600 Da bound,
    is kept."""
    for key, phrase in PROVENANCE.items():
        if str(value).startswith(key):
            tail = str(value)[len(key):].strip()
            return f"{phrase} {tail}" if tail else phrase
    return str(value).replace("_", " ")


def ranking_answer(asked, rank_verdict):
    """The ordering verdict, stated against the request that asked for it.

    A request for a ranked top N is not a requirement and goes in the unread
    list, which is right. But the run then measures whether the model can
    order the set at all, thirty lines below, and the two were never joined:
    the reader saw their request listed as not acted on, and separately a
    verdict about ordering, with nothing saying these were the same subject.

    Returns [] when nothing was asked, so the join only appears where there is
    a request to join it to.
    """
    if not asked or not asked.get("phrases"):
        return []
    n = asked.get("top_n")
    what = f"the top {n:,}" if n else "a ranked list"
    head = f"you asked for {what}."
    if rank_verdict == "not_answerable":
        return wrap(f"{head} It cannot be drawn from this set: the scores are "
                    f"too tied for the order to be the model's, so whatever "
                    f"broke the tie would be choosing them. The verdict above "
                    f"is about this request.", indent=2, hanging=4)
    if rank_verdict == "not_settled_at_this_sample":
        return wrap(f"{head} Whether it can be drawn is not settled by what "
                    f"was measured, and must not be drawn on the strength of "
                    f"the figure above. The verdict above is about this "
                    f"request.", indent=2, hanging=4)
    if rank_verdict == "not_measurable":
        return wrap(f"{head} Whether it can be drawn was not measured: too "
                    f"few molecules survive to tell an order from a "
                    f"tie-break.", indent=2, hanging=4)
    # Answerable, and still not produced. The verdict says the order would be
    # the model's; it does not make this tool the thing that hands over a list.
    return wrap(f"{head} The model can order this set, so such a list would "
                f"be its ordering rather than a tie-break -- but no molecules "
                f"are emitted here. What the verdict licenses is running the "
                f"screen, not receiving the list from this tool.",
                indent=2, hanging=4)


def declared_library_phrase(path, declarations):
    """A library that is not a ZINC slice, named as its manifest names it.

    library_phrase() reads provenance.identify(), which only knows the ZINC
    tranches. The other four libraries the request vocabulary names -- CHAMP,
    IMI, GramNegAccum, B3DB -- fell through to the raw path, so the screen
    printed `data/work/champ.parquet`: a record shape, which is the one thing
    this module exists to keep off it.

    The manifests already carry the name and the row count, and the inventory
    already reads them. Returns None when nothing declares this path, because
    an unidentified library is a fact worth leaving visible rather than
    papering over.
    """
    for entry in declarations or []:
        if path not in (entry.get("artefacts") or []):
            continue
        name = entry.get("display") or entry.get("dataset")
        rows = entry.get("rows")
        measures = entry.get("measures")
        parts = [str(name)]
        if measures:
            parts.append(f"({measures}{f', {rows:,}' if rows else ''})")
        elif rows:
            parts.append(f"({rows:,} compounds)")
        phrase = " ".join(parts)
        if entry.get("caveat"):
            phrase += f" - {entry['caveat']}"
        return phrase
    return None


def barrier_sentence(library, barrier):
    """Which barrier the library is about, as one of three statements.

    Not one sentence with a noun swapped into it. Substituting the words for
    an absent record produced "this library is about not recorded for this
    library.", and the no-library case read the same way while there was no
    library to be about. A hole that has to be grammatical under every value
    it can take is a branch that has not been written.

    Here rather than at the call site so the three readings can be checked
    without running a screen. The defect was visible by reading and was not
    read for several runs.
    """
    if not library:
        return ("no library was named, so there is nothing here for these "
                "conditions to be checked against.")
    if barrier in (None, "unrecorded"):
        return "which barrier this library is about is not recorded."
    return f"this library is about {barrier}."


def cannot_say(what, why, instead=None):
    """One form for everything this tool declines to claim.

    Saying what a number cannot do is the product. Scattering that across
    sentences written one at a time is how some of them ended up not being
    said at all -- a passing verdict printed no figure, and a count measured
    on a sample and a count measured on everything were put in the same table.

    `what` is the claim not being made, `why` is the reason in one clause, and
    `instead` is what the number does support, where it supports something.
    """
    out = [f"this does not say {what}: {why}"]
    if instead:
        out.append(f"what it does say: {instead}")
    return out


def spec_phrase(choices):
    """A whole specification as one readable clause.

    sweep.py names a specification by joining its settings' identifiers, which
    is the right name for a key in sweep.json and the wrong one for a line a
    reader has to compare against another line.
    """
    return "; ".join(reads(knob, choices[knob])
                     for knob in knobs_mod.ORDER if knob in choices)


# How a specification's feasibility state is said. These are field values
# meant for sweep.json, and "uncertain" on its own does not say uncertain
# about what.
STATE = {
    "feasible": "fills the request",
    "infeasible": "cannot fill the request",
    "uncertain": "too close to the sketch's error to call",
}


def state(value):
    return STATE.get(value, str(value).replace("_", " "))


def scaffold_phrase(row):
    """The scaffold count, saying which kind of number it is.

    py/sweep.py counts distinct scaffolds exactly, so the usual answer is a
    count. A sweep written while the HyperLogLog sketch was still in place
    carries an estimate instead, and a record from the window where both
    existed carries the estimate plus a second-pass exact count. All three are
    read, and each says what it is -- a reader comparing an old proposal
    against a new one must not have to work out which was which.
    """
    if row.get("distinct_scaffolds") is not None:
        exact = row.get("counted_exactly", True)
        count = row["distinct_scaffolds"]
        return (f"{count:,} distinct scaffolds, counted exactly" if exact
                else f"about {count:,} distinct scaffolds (sketch estimate)")
    second_pass = (row.get("exact") or {}).get("scaffolds_exact")
    if second_pass is not None:
        return f"{second_pass:,} distinct scaffolds, counted exactly"
    # L: the same specification reads "187, counted exactly" in one run and
    # "about 187 (sketch estimate)" in another, because the exact pass is only
    # spent where the verdict needs it. Reading the two side by side, that is
    # a contradiction unless the reason is attached.
    return (f"about {row['scaffolds_estimated']:,} distinct scaffolds "
            f"(sketch estimate; counted exactly only where the verdict turns "
            f"on it)")


def inert_lines(inert):
    """Knobs that changed no survivor count, said without their identifiers.

    The sentence is built from the comparison counts rather than reusing the
    one provenance.py writes, because that one embeds the knob's key: "every
    pair differing only in rotatable" is not a sentence about chemistry.
    """
    out = []
    for knob in inert:
        name = prop(knob["knob"])
        if knob["all_identical"]:
            out += wrap(f"{name} filtered nothing on this library: every pair "
                        f"of specifications differing only in it gave the same "
                        f"survivor count, in all {knob['comparisons']} "
                        f"comparisons. Do not report it as a constraint that "
                        f"was applied.", indent=2, hanging=4)
        else:
            out += wrap(f"{name} changed nothing in {knob['identical_in']} of "
                        f"{knob['comparisons']} comparisons.",
                        indent=2, hanging=4)
    return out


def answer(spec, result, body=None, cost=None, library=None):
    """The whole answer, in one block, at the end.

    The run prints 88 lines under seven headings and the verdict is on line
    34. That is a transcript of the reasoning, and a transcript is what a
    reader has to work through rather than read. Everything above this block
    stays -- it is where the numbers come from and it is what makes the answer
    checkable -- but a person at a prompt gets the three pillars here, in the
    order they decide in: can it be done, what will it establish, what to do
    next.
    """
    out = ["", "=" * 72, "ANSWER", "=" * 72]
    r = result.get("recommended") or {}
    asked = result.get("as_asked")

    out.append("")
    if r and not r.get("loosened"):
        out.append(f"  feasible as asked: {scaffold_phrase(r)}, "
                   f"from {r['survivors']:,} molecules")
    elif r:
        out.append(f"  not as asked: {scaffold_phrase(asked)}"
                   if asked else "  not as asked")
        out += wrap(f"feasible if loosened: {spec_phrase(r['choices'])} "
                    f"-> {scaffold_phrase(r)}", indent=2, hanging=4)
    else:
        # The ceiling is the number a reader wants most when the answer is
        # no, and it was being dropped: best_reachable carries it under
        # whichever key the sweep that produced it used, which is exactly
        # what scaffold_phrase exists to read.
        best = result.get("best_reachable")
        out.append("  not feasible at any registered setting")
        if best:
            required = result.get("request", {}).get(
                "distinct_scaffolds_required")
            against = (f" against {required:,} required" if required else "")
            # "any of them" is plural and implies there were options. When
            # every requirement is strict there is exactly one specification
            # to compare, and saying it reached the most of several is a
            # choice the reader never had.
            compared = result.get("compared")
            lead = ("the most any of them reaches is"
                    if compared is None or compared > 1
                    else "the one specification compared reaches")
            out += wrap(f"{lead} {scaffold_phrase(best)}{against}",
                        indent=4, hanging=6)
    # Six short of what was asked, two steps tighter: the one line that says
    # whether to widen the library or lower the request. It belongs beside the
    # recommendation, not only in the detail above, because a reader who stops
    # at this block is the reader deciding.
    miss = result.get("nearest_miss")
    if miss:
        out += wrap(f"closest that did not reach: {spec_phrase(miss['choices'])} "
                    f"-> {miss['scaffolds_exact']:,} distinct scaffolds, "
                    f"{miss['short_by']:,} short", indent=2, hanging=4)
    if library:
        out += wrap(f"library: {library}", indent=2, hanging=4)

    if body:
        out.append("")
        rank = body.get("rankability", {})
        app = body.get("applicability", {})
        out.append(f"  a screen of it could be ordered: "
                   f"{verdict(rank.get('verdict'))}")
        out.append(f"  its activity could be predicted: "
                   f"{verdict(app.get('verdict'))}")
        control = (app.get("control") or {}).get("verdict")
        if control:
            out.append(f"    positive control on that verdict: "
                       f"{control_verdict(control)}")
        # A reader who stops at this block is the reader who asked for a top
        # N. The verdict two lines up is the answer to it, and saying so here
        # is the difference between a verdict and an answer.
        out += ranking_answer(spec.get("asked_to_rank") if spec else None,
                              rank.get("verdict"))

    if cost and cost.get("measurable"):
        out.append("")
        out += wrap(f"to bring it inside the model's domain: "
                    f"{cost['compounds_to_measure']:,} compounds measured, "
                    f"{cost['for_half_the_set']:,} of them for half the set",
                    indent=2, hanging=4)

    out.append("")
    out += wrap("No molecules are emitted. What to make of this is the "
                "chemist's call; the lines above are what was measured and "
                "the detail above them is where each number came from.",
                indent=2, hanging=2)
    out.append("=" * 72)
    return out


def library_phrase(info):
    """The library named as a chemist would name it, not as a path.

    `info` is provenance.identify()'s record. When it could not read the slice
    the phrase says so and keeps the path, because an unidentified library is
    the condition the provenance check exists to surface.
    """
    if not info.get("known"):
        return f"{info.get('library', '(not named)')} - slice not identified"
    # "N compounds" after a slice's name reads as the slice's size, and for
    # -N it was the size of the ingest: 1,000,000 read out of 4,572,322. Two
    # of the three slices are complete, so the same words meant two different
    # things depending on which one a reader was looking at.
    rows, total = info.get("rows"), info.get("slice_rows")
    if not isinstance(rows, int):
        count = "count unrecorded"
    elif info.get("capped") and isinstance(total, int):
        count = f"{rows:,} of its {total:,} compounds read"
    else:
        count = f"{rows:,} compounds"
    source = info.get("source") or ""
    match = _TRANCHE.match(source)
    tranche = f"tranche {match.group('tranche')}, " if match else ""
    return (f"ZINC22 {tranche}-{info['slice']} slice "
            f"({info['means']}), {count}")


def control_verdict(value):
    """A control's verdict, which is a statement about the control itself.

    "the control can be done" is not what a passing control says. It says the
    measurement is capable of clearing the threshold, so the failure it rules
    out is a broken measurement rather than a library that is genuinely out of
    domain.
    """
    return {"pass": "passes",
            "not_measurable": "could not be measured"}.get(
                value, verdict(value))


def or_list(values):
    """"5, 7 or 9". A Python list literal on screen is the record's shape."""
    items = [str(v) for v in values]
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} or {items[-1]}"


def counts_phrase(labels):
    """A label tally as words, not as the dict it is stored as.

    `{'positive': 8600, 'negative': 13451}` printed straight out of the record
    is the clearest case of the record's shape reaching the screen.
    """
    if not isinstance(labels, dict):
        return str(labels)
    return ", ".join(f"{value:,} {str(key).replace('_', ' ')}"
                     for key, value in labels.items())


def training_set_phrase(row):
    """A training set named with the count the comparison actually read.

    The count used to be part of the set's name, written as a literal, and the
    same number was printed again by the sentence stating the scope of the
    counts. One output carrying a count from two independent literals is the
    arrangement that drifts: re-ingest the set and only one of them moves.
    """
    parts = [part for part in (row.get("measures"),
                              f"{row['structures']:,}"
                              if row.get("structures") is not None else None)
             if part]
    return (f"{row['training_set']} ({', '.join(parts)})" if parts
            else row["training_set"])


def label_phrase(key):
    """A dataset's label column named as the thing it records.

    `gram_positive_label` is a column name. "gram positive" is what it says.
    """
    return re.sub(r"_label$", "", key).replace("_", " ")


def has_content_word(fragment):
    words = _WORD.findall(fragment.lower())
    return any(word not in FUNCTION_WORDS for word in words)


def partition_unread(pieces):
    """Split unread fragments into the ones that could have been requirements
    and the ones that are grammar.

    The suppressed ones are counted, never dropped in silence: a reader who
    distrusts the split has to be able to see that it happened.
    """
    content = [p for p in pieces if has_content_word(p)]
    return content, len(pieces) - len(content)


def unread_lines(pieces, show=8):
    content, suppressed = partition_unread(pieces)
    if not content and not suppressed:
        return []
    if not content:
        # Nothing was lost, so nothing is asked of the reader. Saying it
        # anyway, every run, is how a warning stops being read.
        return [""] + wrap(f"every part of the request was read. {suppressed} "
                           f"leftover fragment(s) were articles, prepositions "
                           f"or the words for the data itself.")
    out = ["\nnothing in the request was read as a requirement from:"]
    for piece in content[:show]:
        out.append(f"  \"{piece[:90]}\"")
    if len(content) > show:
        out.append(f"  ... and {len(content) - show} more")
    if suppressed:
        out += wrap(f"({suppressed} further fragment(s) suppressed as "
                    f"articles, prepositions or the words for the data "
                    f"itself)", indent=2)
    out += wrap("If any of that was a requirement, it has been lost: this "
                "reads a closed vocabulary and does not guess. Say it in the "
                "wording shown above.", indent=2)
    return out
