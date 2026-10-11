"""The knobs a screening specification can turn, and where each setting comes from.

This is the registry a proposal is allowed to draw on. Every setting carries a
provenance, and only two kinds are permitted:

  published    the setting is written in a published rule, so substituting it
               stays inside what that rule already claims. Richter et al. 2017
               state the first eNTRy condition as an ionisable nitrogen, with
               an unhindered primary amine as the best case -- so moving from
               the primary-amine pattern to any basic nitrogen is a move within
               the published rule, not a new chemical claim.

  relaxation   the setting only moves a numeric threshold. A threshold this
               study chose can be loosened or tightened without asserting any
               chemistry, because the assertion was never there to begin with.
               The molecular weight ceiling is the clear case: the published
               rules set no upper bound on size at all.

Anything that would be a new chemical claim is marked `unjustified` and must
never be proposed. The point of the registry is that a proposal can be audited
by reading one line: either a paper says it, or it is a number being moved.

Nothing here encodes medicinal chemistry judgement. "Widen to secondary amines
because they are also basic" is a claim about chemistry that this project
cannot verify, so it appears only where a published rule already covers it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

from rdkit import Chem  # noqa: E402

PUBLISHED = "published"
RELAXATION = "relaxation"
UNJUSTIFIED = "unjustified"

# The eNTRy rule's own pattern, from Richter et al., Nature 2017 (n = 68).
ENTRY_PRIMARY_AMINE = Chem.MolFromSmarts("[$([N;H2;X3][CX4]),$([N;H3;X4+][CX4])]")
# A secondary amine on carbon. Inside "ionisable nitrogen" as the paper words
# the condition, and narrower than the basic-nitrogen pattern below.
SECONDARY_AMINE = Chem.MolFromSmarts("[$([N;H1;X3]([CX4])[CX4]),$([N;H2;X4+]([CX4])[CX4])]")
# Any nitrogen basic around pH 7.4: the broadest reading of the same condition.
BASIC_NITROGEN = Chem.MolFromSmarts(
    "[$([NX3;H0,H1,H2;!$(N[!#6]);!$(N*=[O,N,S]);!$(N#*);!$([N][a])]),$([NX4+;!$(N*=[O,N,S])])]"
)


def _match(pattern):
    return lambda mol: mol.HasSubstructMatch(pattern)


# Each knob lists its settings from strictest to loosest. A proposal moves
# along this order and reports how far it moved.
KNOBS = {
    "nitrogen": {
        "question": "which nitrogen counts as the ionisable one",
        "settings": [
            {"name": "primary_amine",
             "test": _match(ENTRY_PRIMARY_AMINE),
             "provenance": PUBLISHED,
             "source": "Richter et al. 2017: 'unhindered primary amine best'"},
            {"name": "primary_or_secondary_amine",
             "test": lambda mol: (mol.HasSubstructMatch(ENTRY_PRIMARY_AMINE)
                                  or mol.HasSubstructMatch(SECONDARY_AMINE)),
             "provenance": PUBLISHED,
             "source": "Richter et al. 2017 states the condition as an ionisable "
                       "nitrogen; a secondary amine on carbon is one"},
            {"name": "basic_nitrogen",
             "test": _match(BASIC_NITROGEN),
             "provenance": PUBLISHED,
             "source": "the broadest reading of the same published condition: "
                       "any nitrogen basic around pH 7.4"},
        ],
    },
    "rotatable": {
        "question": "how flexible a molecule may be",
        "settings": [
            {"name": "rotatable_le_5", "limit": 5, "provenance": PUBLISHED,
             "source": "Richter et al. 2017, third eNTRy condition"},
            {"name": "rotatable_le_7", "limit": 7, "provenance": RELAXATION,
             "source": "the published limit moved; no chemical claim"},
            {"name": "rotatable_le_9", "limit": 9, "provenance": RELAXATION,
             "source": "the published limit moved; no chemical claim"},
        ],
    },
    "weight": {
        "question": "the upper bound on molecular weight",
        "caution": "This is NOT a permeability knob. Richter et al. 2017, the "
                   "source of the other conditions here, states of its own 68 "
                   "compounds that 'there is also no correlation between "
                   "accumulation and molecular weight', and that accumulation "
                   "'does not increase with lower ClogD7.4'. Measured here, "
                   "loosening 450 to 600 changed the survivor count by 1.1x "
                   "against 2.8x for rotatable bonds, which agrees. Treat this "
                   "knob as a size and drug-likeness bound, not as a reason to "
                   "expect entry.",
        "settings": [
            {"name": "mw_under_450", "limit": 450.0, "provenance": RELAXATION,
             "source": "chosen by the request; the published rules set no bound"},
            {"name": "mw_under_600", "limit": 600.0,
             "provenance": PUBLISHED + " (secondary: verbatim quote in Richter "
                                       "et al. 2017; the 2008 paper was not read "
                                       "directly)",
             "source": "O'Shea & Moser 2008 J Med Chem 51:2871 report that "
                       "antibiotics effective against Gram-negative pathogens "
                       "'almost always have a molecular weight of less than 600 Da' "
                       "(quoted verbatim by Richter et al. 2017). This is a range "
                       "statement about what exists, not a cut-off they recommend. "
                       "This study reached the same bound independently after a "
                       "1,281 Da glycopeptide ranked first with no bound at all."},
            {"name": "mw_unbounded", "limit": float("inf"), "provenance": PUBLISHED,
             "source": "Richter et al. 2017 set no upper bound on size"},
        ],
    },
}

# The order a specification's settings are enumerated and reported in. Derived
# from the registry rather than written out, because it was written out in four
# places and retiring a knob had to find all four.
ORDER = tuple(KNOBS)

# Knobs that were in the registry and were taken out, with what the decision
# rested on. A retired knob is kept here rather than deleted so that a request
# naming it can be refused with a reason instead of falling into the list of
# phrases nothing was read from.
RETIRED = {
    "charge": {
        "retired": "2026-10-11",
        "was": "the formal charge of the enumerated structure",
        "why": "the knob did not measure the quantity a request means by "
               "charge. ZINC22 splits each tranche by PREDICTED PROTONATION "
               "STATE at pH 7.4; this knob read the FORMAL CHARGE OF THE "
               "STRUCTURE AS DRAWN. The two are different quantities that "
               "share a word, and only the first is what a request about "
               "charge is asking for. The measurement below shows the knob "
               "cannot reach the population it appears to select: in the "
               "sample taken from the -M slice, the slice whose molecules are "
               "predicted to carry -1, no molecule was drawn with a negative "
               "formal charge at all.",
        "not_the_reason": "the knob was not retired for being inert. It is "
                          "not inert: 23 of 20,000 molecules sampled from the "
                          "-O slice carry a formal charge of +1, so the "
                          "setting does remove molecules. Whether that "
                          "removal is visible depends on how many survivors a "
                          "screen has, which is a property of the run and not "
                          "of the knob.",
        "where_it_went": "a request about charge selects a library slice. See "
                         "BLOCKED['library_charge_slice'] for the slices and "
                         "what each one means. py/occam.py refuses a charge "
                         "word rather than reading it as a slice, for the same "
                         "reason it refuses 'MW under 500' rather than "
                         "rounding it to 450.",
    },
}

# Knobs that cannot be turned, with the reason. A proposal that would need one
# of these has to say so instead of substituting something else.
BLOCKED = {
    "library_charge_slice": {
        "why": "ZINC22 splits each tranche by predicted protonation state at pH "
               "7.4 -- N neutral, O +1, M -1, P +2 or more -- and that choice is "
               "made when the file is downloaded, not by any knob here. It is "
               "recorded as blocked because getting it wrong silently selects "
               "against the published rule. Measured in zinc-22x H29P240: "
               "primary amines are 0.21% of -N, 2.46% of -O and 13.06% of -P, a "
               "62-fold range. Richter et al. 2017 report 12 of 41 positively "
               "charged compounds accumulating against 0 of 39 neutral and 0 of "
               "20 negative, so -N is the slice their data says does not "
               "accumulate. The screen of 2026-10-09 used -N alone.",
        "slices": {
            "-N": "predicted neutral at pH 7.4",
            "-O": "predicted +1 at pH 7.4",
            "-M": "predicted -1 at pH 7.4",
            "-P": "predicted +2 or more at pH 7.4",
        },
        # Structured so that a refusal can offer a measured alternative rather
        # than the advice "screen -O or -P instead", which is a sentence and
        # not an answer. The same figures appear in the prose of "why" above;
        # these are the ones code reads.
        "primary_amine_share": {"-N": 0.0021, "-O": 0.0246, "-P": 0.1306},
        "primary_amine_share_measured_on": "zinc-22x H29P240, 20,000 per slice",
        "where": {"-N": "data/work/zinc", "-O": "data/work/zinc_O",
                  "-M": "data/work/zinc_M", "-P": "data/work/zinc_P"},
        # This replaces a sentence that read "every slice is written in neutral
        # form". Three of the four slices sampled contain molecules drawn with
        # a formal charge, so that sentence was too strong. The numbers below
        # carry the tranche, the sample size and the seed, because the sentence
        # they replace stated a count of 19,971 that nothing recorded how to
        # reproduce -- and a rerun got 19,977.
        "formal_charge_as_drawn": {
            "what_this_is": "a measurement of THIS sample, not a property of "
                            "ZINC22. Run py/charge_slices.py to reproduce it.",
            "tranche": "zinc-22x H29P240",
            "sample": 20000,
            "seed": 0,
            "measured": "2026-10-11",
            "per_slice": {
                "-N": {"sampled": 20000, "formal_charge_zero": 20000,
                       "other": {}},
                "-O": {"sampled": 20000, "formal_charge_zero": 19977,
                       "other": {"+1": 23}},
                "-P": {"sampled": 17775, "formal_charge_zero": 17775,
                       "other": {},
                       "note": "the whole slice, not a sample"},
                "-M": {"sampled": 20000, "formal_charge_zero": 19989,
                       "other": {"+1": 11},
                       "note": "drawn from a pool of the first 80,000 "
                               "molecules of a PARTIAL download of this "
                               "slice. The pool is a prefix of a prefix, so "
                               "this is a measurement of that pool and not of "
                               "the slice"},
            },
            "what_it_does_not_say": "a count of 0 in a sample of 20,000 is an "
                                    "observation of none, not an absence. It "
                                    "bounds the share at roughly 0.015% at "
                                    "95%, and no lower. In particular, no "
                                    "molecule drawn with a NEGATIVE formal "
                                    "charge appeared in the -M sample, which "
                                    "is a statement about that sample of that "
                                    "partial download.",
        },
        "what_would_unblock": "download the -O and -P slices of the same tranche "
                              "and screen them too, or make the slice an "
                              "explicit field of the request so it cannot be "
                              "chosen by accident",
    },
    "target_identity": {
        "why": "no dataset here records what a compound acted on. Every label "
               "found is binary active/inactive, so no setting of any knob can "
               "make a target-specific question answerable.",
        "what_would_unblock": "an assay with the target recorded per compound",
    },
    "globularity": {
        "why": "the second eNTRy condition is globularity <= 0.25, which needs a "
               "3D conformer. Nothing in this pipeline generates one, so the "
               "published rule is applied with two of its three conditions.",
        "what_would_unblock": "a conformer generator, and a decision about which "
                              "conformer counts",
        "definition_if_implemented": "Richter et al. 2017 define globularity as "
                                     "the inverse condition number of the "
                                     "covariance matrix of the atomic "
                                     "coordinates: 1 is a perfect sphere, 0 a "
                                     "flat or linear object. Benzene is 0, "
                                     "adamantane 1. Observed values in that "
                                     "paper: ampicillin 0.12, penicillin G 0.17, "
                                     "deoxynybomycin 0.02.",
        "measured_inert": {
            "library": "zinc-22x H29P240",
            "share_passing": 1.0,
            "sampled": 300,
            "median": 0.075,
            "max": 0.247,
            "threshold": 0.25,
            "what": "every molecule sampled passes, so applying this condition "
                    "changes nothing on this library. Reporting it as applied "
                    "would be a true statement about the code and a false one "
                    "about the screen.",
        },
        "note": "GramNegAccum's raw table4.csv carries a computed 'glob' column, "
                "so the training side could use it even while the ZINC side "
                "cannot. That makes it usable for fitting and unusable for "
                "screening, which is a worse trap than not having it.",
    },
}


class Retired(KeyError):
    """A record names a knob that has since been taken out of the registry.

    Raised rather than worked around. A record written under an older registry
    describes a screen this code would not run, and quietly dropping the knob
    would reinterpret what was asked -- which is the failure this whole module
    is arranged to prevent.
    """


def settings_of(knob):
    if knob in RETIRED and knob not in KNOBS:
        body = RETIRED[knob]
        raise Retired(
            f"{knob!r} was retired on {body['retired']}. {body['why']} "
            f"Where it went: {body['where_it_went']}")
    return KNOBS[knob]["settings"]


def unreadable(names):
    """Which of these knob names this registry can no longer act on."""
    return [name for name in names if name not in KNOBS]


def setting(knob, name):
    for candidate in settings_of(knob):
        if candidate["name"] == name:
            return candidate
    raise KeyError(f"{knob} has no setting {name!r}; "
                   f"available: {[s['name'] for s in settings_of(knob)]}")


def strictness(knob, name):
    """Where a setting sits in its knob's order, 0 being strictest."""
    return [s["name"] for s in settings_of(knob)].index(name)


def describe():
    out = {"knobs": {}, "blocked": BLOCKED}
    for knob, body in KNOBS.items():
        out["knobs"][knob] = {
            "question": body["question"],
            "settings": [{"name": s["name"], "provenance": s["provenance"],
                          "source": s["source"]} for s in body["settings"]],
        }
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
