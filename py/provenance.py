"""What a library actually is, and whether the request contradicts it.

Written because a decisive input had no record and no guard. ZINC22 splits
every tranche by predicted protonation state at pH 7.4 -- N neutral, O +1,
M -1, P +2 or more -- and only one slice was downloaded. That choice set the
outcome: under the specification exactly as asked, the -N slice yields 199
distinct scaffolds against a requirement of 256 and fails, while the -O slice
yields 1,624 and passes with nothing relaxed. The screen reported "cannot fill
the request, relax the rotatable-bond limit" when the correct answer was "you
are reading the wrong slice".

Nothing caught it. The feasibility check, the sweep and the proposer all ran
normally, because the slice was not a field of anything. A parameter that
decides the answer and appears in no record is worse than a wrong parameter.

So this module does three things, each of which would have caught it:

  identify   reads the slice back out of the library's own provenance, so it
             appears in the request echo instead of living in whoever typed
             the download command

  contradict compares the slice against the published rule the request is
             built on. Richter et al. 2017 report 12 of 41 positively charged
             compounds accumulating against 0 of 39 neutral and 0 of 20
             negative. Asking for an ionisable nitrogen while reading the
             slice predicted to be neutral is a contradiction, and saying so
             is cheap

  inert      finds knobs that changed nothing. On the -N sweep, 27 of 27
             pairs differing only in the charge setting gave identical
             survivor counts: the knob was a no-op and the evidence for that
             was printed in the output nobody read. A knob that cannot move
             the result is not a constraint, and reporting it as one invites
             the reader to believe a filter was applied
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

# ZINC22's charge suffix, and what each one means. This is the predicted
# protonation state at pH 7.4, NOT the formal charge of the written structure:
# measured on zinc-22x H29P240, 19,971 of 20,000 molecules in the -O slice
# carry formal charge 0 as drawn.
ZINC_CHARGE_SLICES = {
    "N": {"predicted_charge": 0, "name": "neutral at pH 7.4"},
    "O": {"predicted_charge": +1, "name": "+1 at pH 7.4"},
    "M": {"predicted_charge": -1, "name": "-1 at pH 7.4"},
    "P": {"predicted_charge": "+2 or more", "name": "+2 or more at pH 7.4"},
}

# Richter et al. 2017, Nature 545:299, verbatim counts. The basis of the
# contradiction check.
ENTRY_ACCUMULATION_BY_CHARGE = {
    "positive": {"accumulated": 12, "total": 41},
    "neutral": {"accumulated": 0, "total": 39},
    "negative": {"accumulated": 0, "total": 20},
}

# ZINC's own name is H29P240-N.x.smi.gz, where the single letter before
# .smi.gz is the generation. This also accepts the form without it, because
# the first download of this library renamed the files and dropped the
# generation letter -- destroying part of the provenance this module exists to
# read. Keep ZINC's filenames intact.
_SLICE = re.compile(r"-([NOMP])(?:\.[a-z])?\.smi\.gz$")


def identify(part_dir):
    """What this library is, from its own recorded provenance."""
    sidecar = Path(part_dir) / "parts.json"
    if not sidecar.exists():
        return {"library": str(part_dir), "slice": None,
                "known": False,
                "why": "no parts.json; the library does not record where it "
                       "came from, so its charge slice cannot be checked"}
    record = json.loads(sidecar.read_text())
    source = record.get("source", "")
    match = _SLICE.search(source)
    if not match:
        return {"library": str(part_dir), "slice": None, "known": False,
                "source": source,
                "why": "the source filename carries no ZINC charge suffix; this "
                       "may not be a ZINC22 tranche"}
    letter = match.group(1)
    body = ZINC_CHARGE_SLICES[letter]
    return {"library": str(part_dir), "slice": letter, "known": True,
            "source": Path(source).name,
            "predicted_charge_at_ph_7_4": body["predicted_charge"],
            "means": body["name"],
            "rows": record.get("rows_written"),
            # This used to end "the SMILES in every slice are written
            # neutral", which was too strong: three of the four slices sampled
            # hold molecules drawn with a formal charge. The measurement, with
            # its tranche, sample size and seed, is in
            # knobs.BLOCKED['library_charge_slice']['formal_charge_as_drawn'].
            "note": "this is the predicted protonation state, not the formal "
                    "charge of the written structure. Nearly every SMILES in "
                    "every slice is written in neutral form, so the two "
                    "quantities do not line up and only this one selects the "
                    "charge state."}


def alternative_slices(current):
    """The slices that are actually on this disk, and what each is worth.

    A refusal without a revision is half an answer. The contradiction check
    used to end at "screen the -O and -P slices instead", which is advice: it
    does not say whether those slices are here, how big they are, or what the
    swap buys. Everything needed to say that is already recorded, so it is
    said.
    """
    import knobs as knobs_mod

    body = knobs_mod.BLOCKED["library_charge_slice"]
    out = []
    for letter, path in body["where"].items():
        if Path(path) == Path(current):
            continue
        record = Path(path) / "parts.json"
        if not record.exists():
            continue
        rows = json.loads(record.read_text()).get("rows_written")
        out.append({
            "slice": letter,
            "means": body["slices"][letter],
            "library": path,
            "rows": rows,
            "primary_amine_share": body["primary_amine_share"].get(letter),
            "on_this_disk": True,
        })
    # Richest in the published rule's first condition first: that is the axis
    # the contradiction was about.
    out.sort(key=lambda row: -(row["primary_amine_share"] or 0))
    return {"measured_on": body["primary_amine_share_measured_on"],
            "slices": out}


def contradiction(library_info, requirements):
    """Does the slice contradict the published rule being applied?"""
    if not library_info.get("known"):
        return None
    wants_ionisable = any(
        r.get("knob") == "nitrogen" for r in requirements
    )
    if not wants_ionisable:
        return None
    charge = library_info["predicted_charge_at_ph_7_4"]
    if charge == 0:
        positive = ENTRY_ACCUMULATION_BY_CHARGE["positive"]
        neutral = ENTRY_ACCUMULATION_BY_CHARGE["neutral"]
        return {
            "severity": "contradiction",
            "what": f"the request asks for an ionisable nitrogen, which is the "
                    f"first eNTRy condition, but the library is the slice "
                    f"predicted to be neutral at pH 7.4",
            "evidence": f"Richter et al. 2017 report "
                        f"{positive['accumulated']} of {positive['total']} "
                        f"positively charged compounds accumulating against "
                        f"{neutral['accumulated']} of {neutral['total']} "
                        f"neutral ones",
            "measured_here": "on zinc-22x H29P240, primary amines are 0.21% of "
                             "the -N slice, 2.46% of -O and 13.06% of -P",
            "what_to_do": "screen the -O and -P slices of the same tranche, or "
                          "say explicitly why the neutral slice is wanted",
        }
    if charge == -1:
        negative = ENTRY_ACCUMULATION_BY_CHARGE["negative"]
        return {
            "severity": "contradiction",
            "what": "the library is the slice predicted to carry a negative "
                    "charge at pH 7.4",
            "evidence": f"Richter et al. 2017 report "
                        f"{negative['accumulated']} of {negative['total']} "
                        f"negatively charged compounds accumulating",
            "what_to_do": "screen the -O or -P slice instead",
        }
    # A positive charge reads as "+1", not "1". The sign is the whole content
    # of the sentence: the published rule separates positive from neutral.
    reading = f"+{charge}" if isinstance(charge, int) and charge > 0 else str(charge)
    return {"severity": "consistent",
            "what": f"the library is the slice predicted to carry {reading} at "
                    f"pH 7.4, which is the group the published rule associates "
                    f"with accumulation"}


def inert_knobs(specifications):
    """Knobs whose setting never changed a survivor count.

    A knob that cannot move the result is not a constraint. Reporting one as
    applied lets a reader believe a filter ran when nothing was filtered.
    """
    if not specifications:
        return []
    knobs = list(specifications[0]["choices"])
    out = []
    for knob in knobs:
        groups = {}
        for spec in specifications:
            others = tuple(spec["choices"][k] for k in knobs if k != knob)
            groups.setdefault(others, {})[spec["choices"][knob]] = spec["survivors"]
        comparable = [g for g in groups.values() if len(g) > 1]
        if not comparable:
            continue
        identical = sum(1 for g in comparable if len(set(g.values())) == 1)
        if identical == len(comparable):
            out.append({
                "knob": knob,
                "comparisons": len(comparable),
                "all_identical": True,
                "what": f"every pair of specifications differing only in "
                        f"{knob} gave the same survivor count, in all "
                        f"{len(comparable)} comparisons. On this library this "
                        f"knob filters nothing.",
                "what_to_do": "do not report it as a constraint that was "
                              "applied; find out why it is inert before "
                              "relying on it",
            })
        elif identical:
            out.append({
                "knob": knob,
                "comparisons": len(comparable),
                "all_identical": False,
                "identical_in": identical,
                "what": f"{identical} of {len(comparable)} comparisons were "
                        f"unchanged by {knob}",
            })
    return out


def report(part_dir, requirements, specifications=None):
    info = identify(part_dir)
    out = {"library": info, "contradiction": contradiction(info, requirements)}
    if specifications:
        out["inert_knobs"] = inert_knobs(specifications)
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, default=None)
    args = parser.parse_args()

    specifications = None
    requirements = [{"knob": "nitrogen"}]
    if args.sweep and args.sweep.exists():
        body = json.loads(args.sweep.read_text())
        specifications = body["specifications"]

    result = report(args.parts, requirements, specifications)
    info = result["library"]
    print(f"library: {info['library']}")
    if info["known"]:
        print(f"  slice {info['slice']}: {info['means']}, "
              f"{info.get('rows') or '?'} rows")
        print(f"  {info['note']}")
    else:
        print(f"  slice unknown: {info['why']}")

    c = result["contradiction"]
    if c:
        print(f"\n[{c['severity']}] {c['what']}")
        for key in ("evidence", "measured_here", "what_to_do"):
            if c.get(key):
                print(f"  {key}: {c[key]}")

    for knob in result.get("inert_knobs", []):
        tag = "INERT" if knob["all_identical"] else "partly inert"
        print(f"\n[{tag}] {knob['knob']}")
        print(f"  {knob['what']}")
        if knob.get("what_to_do"):
            print(f"  {knob['what_to_do']}")


if __name__ == "__main__":
    main()
