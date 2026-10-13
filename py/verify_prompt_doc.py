# Every spelling docs/prompt_format.md names, put through the parser it
# documents. A document about how to write a prompt is wrong in the way that
# matters if one of its rows does not parse.
import sys
sys.path.insert(0, 'py')
import occam

BASE = "256 distinct scaffolds, "
bad = 0

def check(label, prose, want_key, want):
    global bad
    spec, notes = occam.parse(prose)
    if want_key == "library":
        got = spec.get("library")
    elif want_key == "deliverable":
        got = spec["deliverable"].get("distinct_scaffolds")
    else:
        got = next((r["setting"] for r in spec["requirements"]
                    if r.get("knob") == want_key), None)
    ok = got == want
    bad += not ok
    print(f"  {'ok ' if ok else 'BAD'} {label:<52} -> {got}")

print("LIBRARIES")
for phrase, want in [
    ("cationic ZINC", "data/work/zinc_O"),
    ("protonated ZINC", "data/work/zinc_O"),
    ("ZINC O slice", "data/work/zinc_O"),
    ("+1 ZINC", "data/work/zinc_O"),
    ("ZINC +2 slice", "data/work/zinc_P"),
    ("dicationic ZINC", "data/work/zinc_P"),
    ("multiply charged ZINC", "data/work/zinc_P"),
    ("ZINC P slice", "data/work/zinc_P"),
    ("neutral ZINC", "data/work/zinc"),
    ("ZINC N slice", "data/work/zinc"),
    ("ZINC", "data/work/zinc"),
]:
    check(phrase, BASE + "from the " + phrase + " library", "library", want)

print("\nNITROGEN")
for phrase, want in [
    ("primary amine", "primary_amine"),
    ("secondary amine", "primary_or_secondary_amine"),
    ("primary or secondary amine", "primary_or_secondary_amine"),
    ("basic nitrogen", "basic_nitrogen"),
    ("ionizable nitrogen", "basic_nitrogen"),
    ("ionisable nitrogen", "basic_nitrogen"),
]:
    check(phrase, BASE + phrase, "nitrogen", want)

print("\nROTATABLE")
for phrase, want in [
    ("rotatable bonds at most 5", "rotatable_le_5"),
    ("rotatable <= 7", "rotatable_le_7"),
    ("rotatable bonds ≤ 9", "rotatable_le_9"),
    ("rotatable bonds must be at most 5", "rotatable_le_5"),
    ("the rotatable bonds should be no more than 5", "rotatable_le_5"),
    ("rotatable bonds under 7", "rotatable_le_7"),
    ("rotatable bonds up to 9", "rotatable_le_9"),
    ("rotatable bonds less than 5", "rotatable_le_5"),
    ("rotatable bonds < 7", "rotatable_le_7"),
    ("rotatable bonds = 9", "rotatable_le_9"),
]:
    check(phrase, BASE + phrase, "rotatable", want)

print("\nWEIGHT")
for phrase, want in [
    ("molecular weight under 450", "mw_under_450"),
    ("MW under 600", "mw_under_600"),
    ("molecular weight is strictly under 450", "mw_under_450"),
    ("MW must be under 600", "mw_under_600"),
    ("mol. wt. 450", "mw_under_450"),
]:
    check(phrase, BASE + phrase, "weight", want)

print("\nDELIVERABLE")
for phrase, want in [("256 distinct scaffolds", 256),
                     ("256 unique candidates", 256)]:
    check(phrase, phrase + " from the cationic ZINC library", "deliverable",
          want)
spec, _ = occam.parse("256 from the cationic ZINC library, primary amine")
print(f"  {'ok ' if not spec['deliverable'] else 'BAD'} "
      f"{'256 alone (must NOT be read)':<52} -> {spec['deliverable']}")
bad += bool(spec["deliverable"])

print("\nSTRICTNESS")
for phrase, knob, want in [
    ("molecular weight strictly under 450", "weight", True),
    ("MW must be under 450", "weight", True),
    ("keep rotatable bonds at most 5", "rotatable", True),
    ("rotatable bonds at most 5", "rotatable", False),
]:
    spec, _ = occam.parse(BASE + phrase)
    got = next((r["strict"] for r in spec["requirements"]
                if r.get("knob") == knob), None)
    ok = got == want
    bad += not ok
    print(f"  {'ok ' if ok else 'BAD'} {phrase:<52} -> strict={got}")

# The clause window: a marker must not leak to the neighbour.
spec, _ = occam.parse(BASE + "MW strictly under 450, primary amine")
flags = {r["knob"]: r["strict"] for r in spec["requirements"] if "knob" in r}
ok = flags.get("weight") is True and flags.get("nitrogen") is False
bad += not ok
print(f"  {'ok ' if ok else 'BAD'} "
      f"{'MW strictly under 450, primary amine':<52} -> {flags}")

# Sentence-initial must binds everything.
spec, _ = occam.parse(
    "256 distinct scaffolds from the cationic ZINC library. The screening "
    "must satisfy the following criteria: primary amine, rotatable bonds "
    "must be at most 5, molecular weight under 450")
flags = {r["knob"]: r["strict"] for r in spec["requirements"] if "knob" in r}
ok = all(flags.values()) and len(flags) == 3
bad += not ok
print(f"  {'ok ' if ok else 'BAD'} {'lead-in must (all three strict)':<52} -> {flags}")

print("\nREFUSED, NOT ROUNDED")
for phrase in ("rotatable bonds at most 6", "molecular weight under 500"):
    spec, notes = occam.parse(BASE + phrase)
    ok = bool(notes["refused"])
    bad += not ok
    print(f"  {'ok ' if ok else 'BAD'} {phrase:<52} -> refused={bool(notes['refused'])}")
for phrase in ("neutral only", "any charge", "zero formal charge",
               "cationic", "anionic", "zwitterionic"):
    spec, notes = occam.parse("256 distinct scaffolds from the ZINC O slice "
                              "library, primary amine, " + phrase)
    ok = bool(notes["refused"])
    bad += not ok
    print(f"  {'ok ' if ok else 'BAD'} {phrase + ' (as a condition)':<52} "
          f"-> refused={bool(notes['refused'])}")

print("\nDROPPED CLAIMS")
for phrase in ("the 100-million-item cache", "the 0.071 AUC deficit",
               "7,394 distinct diversity domains", "pre-packed", "verified",
               "confirmed", "proven", "established"):
    spec, notes = occam.parse(BASE + "primary amine, " + phrase)
    ok = bool(notes["dropped_claims"])
    bad += not ok
    print(f"  {'ok ' if ok else 'BAD'} {phrase:<52} "
          f"-> dropped={[d['phrase'] for d in notes['dropped_claims']]}")

print(f"\n{'ALL ROWS PARSE AS DOCUMENTED' if not bad else f'{bad} ROW(S) DISAGREE WITH THE DOCUMENT'}")
sys.exit(1 if bad else 0)
