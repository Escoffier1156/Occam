"""Turns a sweep into a proposal: which specification to use, and why.

The sweep measures every variant. This decides which ones are worth handing
back, and it does so without a weighted score.

WHY NOT A WEIGHTED SCORE

  score = w1*survivors + w2*diversity - w3*penalty

would make the recommendation a product of w1, w2 and w3, which are invented.
This project has already retracted three claims that turned out to be
artefacts of a chosen constant. So the selection is a constraint plus an
order, and both come from the request rather than from here:

  1  feasible      a variant must be able to fill the request. The deliverable
                   is N distinct scaffolds, one representative each, so the
                   constraint is on the variant's distinct scaffold count, not
                   on its survivor count. Survivors are only the pool.
  2  least loosened among the feasible ones, the fewest relaxation steps away
                   from the specification as asked. Loosen as far as needed and
                   no further.
  3  provenance    ties break towards settings a published rule already covers.
  4  frontier      the whole trade-off is returned, not only the winner. The
                   granularity of the steps is a human choice -- whether
                   rotatable bonds go 5, 7, 9 or 5, 6, 7 is not measured
                   anywhere -- so the ranking is a suggestion and the frontier
                   is the evidence.

WHERE THE SKETCH IS ALLOWED TO ACT

HyperLogLog decides nothing. It is a coarse filter: variants whose estimate is
far below the requirement are dropped, and the few that remain are counted
exactly in a second pass. Using a conservative lower bound instead would leave
the sketch's error model inside the decision, which is worse than paying for
one more pass.

The band where the estimate cannot settle the question is reported as such.
"""

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import knobs as knobs_mod  # noqa: E402
import render as render_mod  # noqa: E402
import smiles as smiles_mod  # noqa: E402
import sweep as sweep_mod  # noqa: E402

from rdkit import RDLogger  # noqa: E402
from rdkit.Chem import Descriptors, rdMolDescriptors  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

RDLogger.DisableLog("rdApp.*")

WORK = ROOT / "data" / "work"
# HyperLogLog's standard error at 2^12 registers: 1.04 / sqrt(4096) = 1.63%,
# which is the textbook figure and is where 0.016 comes from. Three standard
# errors is the band within which the estimate is not allowed to decide.
#
# WITHDRAWN: "measured in this repository at 0.29% to 1.24% over four
# cardinalities". No such measurement exists. The 0.29% it cited is a memory
# figure from a different table -- 36,224 KiB against 36,328 -- and the
# constant was always the theoretical value. A derived number wearing the
# word "measured" is the fault contraindication 102 is about, found twice in
# one day.
#
# Measured now, 30 seeds per cardinality against exact counts:
#
#     n          median   p95     max     over the 4.8% band
#     100        0.81%    1.83%   1.83%   0/30
#     686        0.78%    2.17%   2.31%   0/30
#     2,000      0.84%    2.10%   2.30%   0/30
#     10,000     2.72%    5.02%   5.87%   2/30
#     50,000     1.13%    3.64%   3.64%   0/30
#     200,000    1.39%    3.92%   4.05%   0/30
#
# The band holds at every cardinality but one. Around 10,000 the error roughly
# triples and crosses it twice in thirty draws, which is 2.5 x 4,096 -- where
# HyperLogLog leaves its small-cardinality correction. That is a property of
# the estimator at this register count, not of a bad draw, and the exact pass
# is what the band exists to trigger: a row near the requirement is counted
# rather than trusted. A request landing near 10,000 scaffolds is the case to
# watch, and it is recorded here rather than left for the next person to find.
SKETCH_RELATIVE_ERROR = 0.016
UNCERTAIN_BAND = 3 * SKETCH_RELATIVE_ERROR
# How many specifications the frontier shows, and so how many get an exact
# count beyond the uncertain band. A handful, fixed: the cost of exactness has
# to stay independent of the library's size.
FRONTIER = 8


def total_steps(choices, baseline):
    return sum(knobs_mod.strictness(k, choices[k]) - knobs_mod.strictness(k, baseline[k])
               for k in choices)


def loosened_knobs(choices, baseline):
    return {k: v for k, v in choices.items() if v != baseline[k]}


def classify(spec, required, baseline):
    """feasible / uncertain / infeasible, by the sketch alone."""
    estimate = spec["distinct_scaffolds_estimated"]
    if estimate >= required * (1 + UNCERTAIN_BAND):
        state = "feasible"
    elif estimate <= required * (1 - UNCERTAIN_BAND):
        state = "infeasible"
    else:
        state = "uncertain"
    return {
        "specification": spec["specification"],
        "choices": spec["choices"],
        "survivors": spec["survivors"],
        "scaffolds_estimated": estimate,
        "state_by_sketch": state,
        "steps_from_baseline": total_steps(spec["choices"], baseline),
        "loosened": loosened_knobs(spec["choices"], baseline),
        "provenance": spec["provenance"],
    }


# How many scaffold sets may be held at once. The exact pass used to hold one
# and read the library once per specification; it now holds up to this many
# and reads the library once per batch of them.
#
# WHY THE TRADE WAS TURNED AROUND
#
# The old shape was invariant 3 applied to the second pass -- one bulk alive
# at a time -- and its own docstring named the condition for changing it: "if
# a budget ever justifies running k of these at once, k follows from the
# budget by division". The budget arrived. On the -P slice (17,775 molecules)
# a pass costs about 30 seconds, so eight of them cost four minutes and
# nobody noticed. On -O (636,669) a pass costs about eighteen minutes, and a
# request with three relaxable axes would have read the library eight times
# for two and a half hours of I/O to produce numbers one pass already had.
#
# The sweep reads the library once and evaluates every specification against
# each molecule. The exact pass threw that structure away and re-read. The
# scaffold is also computed once per molecule now instead of once per
# specification, which was the larger waste: MurckoScaffoldSmiles was 27.8%
# of the per-molecule cost when it ran once.
#
# WHAT IT COSTS, MEASURED
#
# A scaffold set is its SMILES strings: 108 bytes each as Python holds them,
# measured 2026-10-14 over 4,330 scaffolds on the -P slice. The widest
# specification in any sweep on this disk is 112,903 scaffolds, on -O
# (data/work/sweep_O.json), so eight sets at that width is about 97 MB.
#
# WITHDRAWN: contraindication 84 and the old docstring both put the widest
# measured specification at 46,437 scaffolds and 127.4 B each, for 45 MB. No
# record on this disk holds 46,437 -- the four sweeps kept here top out at
# 4,354 (-P), 41,397 (-N, one million read) and 112,903 (-O) -- and which
# library it came from was never written down. The figure is replaced by the
# measured ones rather than carried forward.
#
# This is a count, not a budget divided by anything, because the divisor
# would be the scaffold count and that is what the pass exists to find out.
# A sketch estimate of it carries a 4.8% band, and sizing memory from an
# estimate means the estimate being wrong is a memory fault rather than a
# number being slightly off.
SETS_AT_ONCE = FRONTIER


def exact_scaffold_counts(part_dir, rows, limit=None):
    """Counts distinct scaffolds exactly for several variants, in one pass.

    EVERY SET IS LOCAL AND WHAT COMES BACK IS COUNTS

    The sets live in this function and the caller is handed numbers, so it
    cannot accumulate them even by accident -- the property the one-at-a-time
    shape was protecting, which does not depend on how many are held here.

    At most SETS_AT_ONCE are held together; a longer list is read in batches
    of that size, so the memory is bounded by the constant and the number of
    passes is len(rows) / SETS_AT_ONCE rather than len(rows).

    Returns one count dict per row, in the order given.
    """
    nitrogen = {s["name"]: s for s in knobs_mod.settings_of("nitrogen")}
    out = []
    for first in range(0, len(rows), SETS_AT_ONCE):
        batch = rows[first:first + SETS_AT_ONCE]
        variants = [sweep_mod.Variant(choices) for choices in batch]
        scaffolds = [set() for _ in batch]
        survivors = [0] * len(batch)
        for text in sweep_mod.iter_library(part_dir, limit):
            mol = smiles_mod.parse(text)
            if mol is None:
                continue
            weight = Descriptors.MolWt(mol)
            rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
            # Computed at most once per molecule, and only once some
            # specification has kept it. It is 27.8% of the per-molecule cost
            # (profiled in sweep.py, where the table is), and the old shape
            # paid it once per specification per molecule.
            scaffold = None
            for index, (choices, variant) in enumerate(zip(batch, variants)):
                if weight >= variant.mw_limit or rotatable > variant.rot_limit:
                    continue
                if not nitrogen[choices["nitrogen"]]["test"](mol):
                    continue
                survivors[index] += 1
                if scaffold is None:
                    try:
                        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or ""
                    except ValueError:
                        scaffold = ""
                scaffolds[index].add(scaffold)
        for index, choices in enumerate(batch):
            # A distinct count cannot exceed the population it was counted
            # from. The cheapest possible check, and it catches every way two
            # scopes can get mixed into one row.
            if len(scaffolds[index]) > survivors[index]:
                raise ValueError(
                    f"{len(scaffolds[index]):,} distinct scaffolds from "
                    f"{survivors[index]:,} survivors is impossible. Two "
                    f"different populations have been counted as one.")
            out.append({"survivors": survivors[index],
                        "scaffolds_exact": len(scaffolds[index]),
                        "held_strings": len(scaffolds[index])})
        del scaffolds
    return out


def exact_scaffold_count(part_dir, choices, limit=None):
    """One variant. The single-row case of exact_scaffold_counts()."""
    return exact_scaffold_counts(part_dir, [choices], limit)[0]


def propose(sweep_report, required, part_dir=None, limit=None, exact=True):
    baseline = sweep_report["baseline"]
    # A sweep written under an older registry can name a knob that has since
    # been retired. Saying so beats a KeyError from four frames down, and
    # beats dropping the knob, which would silently turn the record into a
    # description of a different screen.
    stale = knobs_mod.unreadable(baseline)
    if stale:
        raise knobs_mod.Retired(
            f"this sweep was written with {', '.join(sorted(stale))}, which "
            f"this registry no longer has. Rerun the sweep; the record stays "
            f"valid for what it recorded, but a proposal cannot be built on a "
            f"knob that is gone.")
    rows = [classify(s, required, baseline) for s in sweep_report["specifications"]]

    # WHICH SPECIFICATIONS GET COUNTED EXACTLY, AND WHY THAT SET
    #
    # The sweep holds a fixed-width sketch per specification, so its state
    # does not grow with the library. That is invariant 2 and it is the most
    # verified result here, so the sweep is not where exactness is bought.
    #
    # It is bought in a second pass over a set whose SIZE DOES NOT DEPEND ON
    # THE LIBRARY:
    #
    #   every specification inside the uncertain band, where the sketch by
    #   construction cannot decide;
    #
    #   every specification that will be shown on the frontier, because those
    #   are the numbers a person compares when making the trade, and a 1.6%
    #   error between two rows of a table is read as a difference;
    #
    #   the one that ends up recommended.
    #
    # The band is a probabilistic guarantee: at three standard errors a single
    # specification falls outside it rarely, but with 27 of them in one run
    # "rarely" is not negligible. Counting the frontier regardless removes the
    # band from everything a reader sees, and widening the band costs more
    # exact passes rather than more memory.
    for row in rows:
        row["state"] = row["state_by_sketch"]

    def abandoned_published(row):
        """How many settings this specification left that a paper states.

        Leaving a published setting is the expensive move; leaving a threshold
        this study chose costs nothing, because there was no claim there to
        give up. The old ordering could not tell them apart: it read the
        weakest provenance of the whole specification, so moving the published
        rotatable limit two steps ranked level with moving a bound this study
        picked, and the tie fell to whichever kept more molecules.

        Counted in STEPS, not in knobs. Giving up one published setting by two
        steps and giving up one by a single step are not the same move, and
        counting knobs made them tie: rotatable 5 -> 9 and rotatable 5 -> 7
        both read as "one published setting abandoned".
        """
        steps = 0
        for knob, setting in row["loosened"].items():
            was = knobs_mod.setting(knob, baseline[knob])["provenance"]
            if not str(was).startswith("published"):
                continue
            # A move that lands on another published setting gives up nothing:
            # the weight bound at 600 Da is stated by a paper, so arriving
            # there is not leaving the published rules.
            now = knobs_mod.setting(knob, setting)["provenance"]
            if str(now).startswith("published"):
                continue
            steps += abs(knobs_mod.strictness(knob, setting)
                         - knobs_mod.strictness(knob, baseline[knob]))
        return steps

    def overshoot(row):
        """How far past the request this specification goes.

        Asked for 500, the old tie-break offered 686: it was -survivors,
        which prefers the biggest set and so prefers the biggest overshoot.
        What was asked for is the target, not a floor to clear by as much as
        possible. A specification that does not reach the request sorts last
        rather than first.

        The review that found this put a reachable 520 one step tighter, and
        that figure is withdrawn: counted exactly, the tighter specification
        holds 494. The ordering was wrong about which overshoot to prefer; it
        was not hiding an answer.
        """
        count = (row.get("exact") or {}).get("scaffolds_exact")
        if count is None:
            count = row.get("scaffolds_estimated")
        if count is None or count < required:
            return float("inf")
        return count - required

    def rank(row):
        # Fewest steps, then fewest published settings given up, then closest
        # to what was asked, then a deterministic tie-break that is a
        # tie-break and not a preference.
        return (row["steps_from_baseline"],
                abandoned_published(row),
                overshoot(row),
                row["specification"])

    # Provisional, on the sketch: enough to decide WHICH rows are worth an
    # exact pass. The order is settled again afterwards on the exact counts.
    provisional = sorted((r for r in rows if r["state"] != "infeasible"), key=rank)
    wanted = {id(r): r for r in provisional[:FRONTIER]}
    wanted.update({id(r): r for r in rows
                   if r["state_by_sketch"] == "uncertain"})

    # A sweep read some amount of the library; a second pass must read the
    # same amount or its numbers belong to a different population. Mixing them
    # produced a row reading "123 molecules pass, 536 distinct scaffolds",
    # which cannot happen -- a distinct count cannot exceed what it was
    # counted from. Scope, as in contraindications 25 and 81.
    swept_limit = sweep_report.get("summary", {}).get("limit", "unrecorded")
    if exact and part_dir and swept_limit != limit:
        raise ValueError(
            f"the sweep read limit={swept_limit!r} and the exact pass was "
            f"asked for limit={limit!r}. Counts from two different amounts of "
            f"the library cannot go in one row. Rerun with the same limit, or "
            f"pass exact=False to keep the sketch's numbers.")

    if exact and part_dir and wanted:
        # One pass for every specification that needs an exact count, rather
        # than one pass each. The list is materialised deliberately: the
        # memory it bounds is the SCAFFOLD SETS inside, which are batched at
        # SETS_AT_ONCE, and a row here is a dict of counts and choices.
        targets = list(wanted.values())
        measurements = exact_scaffold_counts(
            part_dir, [row["choices"] for row in targets], limit)
        for row, measured in zip(targets, measurements):
            row["exact"] = measured
            # The exact pass recounted the survivors too. Keeping the sketch
            # pass's figure beside the exact scaffold count is what produced
            # the impossible row.
            row["survivors"] = measured["survivors"]
            row["state"] = ("feasible"
                            if measured["scaffolds_exact"] >= required
                            else "infeasible")

    feasible = sorted((r for r in rows if r["state"] == "feasible"), key=rank)

    # CONTRAINDICATION 88: THE CANDIDATES NEAREST THE REQUEST ARE THE ONES THE
    # SKETCH CANNOT PLACE
    #
    # A specification lands in the uncertain band when its estimate is within
    # three standard errors of the requirement -- that is, when it is CLOSE TO
    # THE ANSWER. The band is where the best candidates live, by construction.
    #
    # Those rows were then excluded from `feasible`, so the recommendation was
    # drawn from everything except the near misses. Asked for 500, the nearest
    # candidates estimated at 499 and 481 fell in the band, were dropped
    # unmeasured, and the answer became 686. The tie-break was blamed first;
    # the tie-break was not the whole of it.
    #
    # Counted exactly those rows hold 494 and 475, so on this library none of
    # them was the answer -- an earlier note here said 520 was available and
    # that is withdrawn. The fault stands: the rows closest to the request
    # were the ones being discarded before anyone counted them.
    #
    # Resolved, they are placed correctly, and that is the normal path. Left
    # unresolved -- no library to re-read, or the exact pass switched off --
    # they used to vanish with nothing said. A candidate that is unplaced is
    # not a candidate that failed, and the difference has to be on the screen.
    held_back = sorted((r for r in rows
                        if r["state"] == "uncertain" and "exact" not in r),
                       key=rank)

    # THE NEAREST MISS IS THE ONE LINE A READER WANTS
    #
    # `held_back` covers the rows nothing could place. It does not cover the
    # rows that WERE placed, exactly, just short -- and those are the ones the
    # request was nearest to. Asked for 500, the frontier offers 686 and says
    # nothing about the specification counted at 494, six short, two steps
    # tighter and keeping a published limit.
    #
    # Six short is actionable in a way 686 is not: widen the library, or drop
    # the request to 494. Neither is visible from the recommendation alone.
    # This is also how a wrong premise gets corrected -- a reviewer working
    # from an estimate of 520 could not retract it, because the measured 494
    # existed only in a count that was taken and then discarded. `frontier`
    # holds feasible rows, so it had nowhere to go.
    #
    # Only exactly-counted rows qualify. A sketch estimate six short of the
    # requirement is inside its own error bar and would be a guess printed as
    # a near miss.
    missed = [r for r in rows
              if (r.get("exact") or {}).get("scaffolds_exact") is not None
              and r["exact"]["scaffolds_exact"] < required]
    nearest_miss = max(missed, key=lambda r: (r["exact"]["scaffolds_exact"],
                                              -r["steps_from_baseline"]),
                       default=None)

    out = {
        "request": {"distinct_scaffolds_required": required,
                    "specification_as_asked": baseline},
        "verdict": None,
        "recommended": None,
        "frontier": feasible[:FRONTIER],
        "infeasible_count": sum(1 for r in rows if r["state"] == "infeasible"),
        "counted_exactly": [r["specification"] for r in rows if "exact" in r],
        "counted_by_sketch_only": [r["specification"] for r in rows
                                   if "exact" not in r],
        "held_back": [{"specification": r["specification"],
                       "choices": r["choices"],
                       "scaffolds_estimated": r["scaffolds_estimated"],
                       "steps_from_baseline": r["steps_from_baseline"],
                       "loosened": r["loosened"]}
                      for r in held_back[:FRONTIER]],
        "held_back_why": "the sketch could not place these against the "
                         "requirement: their estimate is inside three standard "
                         "errors of it. That is where the closest candidates "
                         "are, so they are named rather than dropped. An exact "
                         "second pass decides them, which needs the library",
        "nearest_miss": nearest_miss and {
            "specification": nearest_miss["specification"],
            "choices": nearest_miss["choices"],
            "scaffolds_exact": nearest_miss["exact"]["scaffolds_exact"],
            "short_by": required - nearest_miss["exact"]["scaffolds_exact"],
            "survivors": nearest_miss["survivors"],
            "steps_from_baseline": nearest_miss["steps_from_baseline"],
            "loosened": nearest_miss["loosened"],
            "provenance": nearest_miss["provenance"],
        },
        "nearest_miss_why": "the closest specification that was counted "
                            "exactly and did not reach the request. It is not "
                            "an alternative -- it does not fill the order -- "
                            "but how far short it fell is the figure that says "
                            "whether to widen the library or lower the "
                            "request, and a recommendation further away "
                            "cannot say it",
        "blocked_knobs": knobs_mod.BLOCKED,
        "how_this_was_chosen": [
            "a variant must reach the required distinct scaffold count; the "
            "deliverable is one representative per scaffold, so the constraint "
            "is on scaffolds and not on survivors",
            "among those, the fewest relaxation steps from the specification as "
            "asked",
            "then the fewest PUBLISHED settings given up. Leaving a bound this "
            "study chose costs nothing, because no claim was being made there; "
            "leaving one a paper states is the expensive move, and the two "
            "were not told apart before",
            "then the specification closest to what was asked. The tie-break "
            "used to be the largest survivor count, which prefers the largest "
            "overshoot: 686 distinct scaffolds were offered against a request "
            "for 500, and the two steps between them were never weighed. The "
            "specification said here to reach about 520 was counted exactly "
            "afterwards at 494, six short of the request, so it was not an "
            "alternative that had been missed -- that figure is withdrawn. It "
            "is reported as the nearest miss instead",
            "the step granularity is a human choice, so the frontier is "
            "returned and the ranking is a suggestion",
        ],
        # How many specifications there were to choose between. One is not a
        # choice, and the answer block says so rather than writing "the most
        # any of them reaches" about a set of size one.
        "compared": len(rows),
        "sketch_role": "coarse filter only; variants near the requirement were "
                       "counted exactly in a second pass",
    }

    as_asked = next((r for r in rows if r["choices"] == baseline), None)
    if as_asked and as_asked["state"] == "feasible":
        out["verdict"] = "the specification as asked is feasible"
        out["recommended"] = as_asked
    elif feasible:
        out["verdict"] = "the specification as asked cannot fill the request; a " \
                         "looser one can"
        out["recommended"] = feasible[0]
        out["as_asked"] = as_asked
    elif not rows:
        # A sweep with no specifications in it. Reached by a record that was
        # written before anything was evaluated; max() over nothing raised
        # here, which said nothing about why.
        out["verdict"] = ("the sweep evaluated no specifications, so there is "
                          "nothing to propose. Rerun the sweep")
    else:
        best = max(rows, key=lambda r: r["scaffolds_estimated"])
        out["verdict"] = "no setting of these knobs can fill the request"
        out["best_reachable"] = best
        out["what_would_unblock"] = [
            f"a larger or different library: the best reachable is "
            f"{best['scaffolds_estimated']:,} distinct scaffolds against "
            f"{required:,} required",
            "or a lower requirement, which is the requester's decision",
        ]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", type=Path, default=WORK / "sweep.json")
    parser.add_argument("--required", type=int, default=256,
                        help="distinct scaffolds the deliverable needs")
    parser.add_argument("--parts", type=Path, default=None,
                        help="library, for the exact second pass")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-exact", action="store_true")
    parser.add_argument("--out", type=Path, default=WORK / "proposal.json")
    args = parser.parse_args()

    report = json.loads(args.sweep.read_text())
    try:
        result = propose(report, args.required, args.parts, args.limit or None,
                         exact=not args.no_exact)
    except knobs_mod.Retired as retired:
        print(f"stopping: {retired.args[0]}")
        return 1
    except ValueError as mismatch:
        print(f"stopping: {mismatch.args[0]}")
        return 1
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    show(result, args.required)
    print(f"\n-> {args.out}")
    return 0


def show(result, required):
    """The proposal, with the frontier as the thing a reader is here for.

    The frontier is the whole reason there is no weighted score: the ranking
    cannot encode a preference between fewer rotatable bonds and a higher
    molecular weight, so every specification that clears the request is
    returned and the trade is made by a person. A table of registry
    identifiers cannot be read that way, which means printing them here
    deletes the design.
    """
    asked = result["request"]["specification_as_asked"]
    print(f"request: {required:,} distinct scaffolds")
    for line in render_mod.wrap(f"as asked: {render_mod.spec_phrase(asked)}",
                                hanging=2):
        print(line)
    print()
    for line in render_mod.wrap(f"verdict: {result['verdict']}", hanging=2):
        print(line)

    if result.get("recommended"):
        r = result["recommended"]
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

    if result.get("as_asked"):
        a = result["as_asked"]
        print(f"\nas asked would give: {a['survivors']:,} molecules, "
              f"{render_mod.scaffold_phrase(a)} - "
              f"{render_mod.state(a['state'])}")

    if result.get("best_reachable"):
        b = result["best_reachable"]
        print("\nthe best any registered setting reaches:")
        for line in render_mod.wrap(render_mod.spec_phrase(b["choices"]),
                                    indent=2, hanging=4):
            print(line)
        print(f"  {render_mod.scaffold_phrase(b)}, "
              f"against {required:,} required")
        print("\nwhat would unblock it:")
        for item in result["what_would_unblock"]:
            for line in render_mod.wrap(item, indent=2, hanging=4):
                print(line)

    frontier = result["frontier"]
    print(f"\nthe trade, {len(frontier)} specification(s) that clear the "
          f"request:")
    print("  there is no score ranking these. They are ordered by how little "
          "was loosened,")
    print("  and which one to run is a judgement about the chemistry.")
    for index, row in enumerate(frontier, 1):
        steps = row["steps_from_baseline"]
        moved = ("nothing loosened" if not steps
                 else f"{steps} step(s) looser")
        print()
        for line in render_mod.wrap(
                f"{index}. {render_mod.spec_phrase(row['choices'])}",
                indent=2, hanging=5):
            print(line)
        print(f"     {row['survivors']:,} molecules pass, "
              f"{render_mod.scaffold_phrase(row)}")
        for line in render_mod.wrap(
                f"{moved}; weakest provenance: "
                f"{render_mod.provenance(row['provenance']['weakest'])}",
                indent=5, hanging=7):
            print(line)

    print("\nrequirements that cannot be turned at all:")
    for knob, body in result["blocked_knobs"].items():
        for line in render_mod.wrap(f"{knob.replace('_', ' ')}: {body['why']}",
                                    indent=2, hanging=4):
            print(line)
        for line in render_mod.wrap(f"would need: "
                                    f"{body['what_would_unblock']}",
                                    indent=4, hanging=6):
            print(line)


if __name__ == "__main__":
    raise SystemExit(main())
