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
import smiles as smiles_mod  # noqa: E402
import sweep as sweep_mod  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Descriptors, rdMolDescriptors  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

RDLogger.DisableLog("rdApp.*")

WORK = ROOT / "data" / "work"
# HyperLogLog's standard error at 2^12 registers, measured in this repository
# at 0.29% to 1.24% over four cardinalities. Three standard errors is the band
# within which the estimate is not allowed to decide.
SKETCH_RELATIVE_ERROR = 0.016
UNCERTAIN_BAND = 3 * SKETCH_RELATIVE_ERROR


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


def exact_scaffold_count(part_dir, choices, limit=None):
    """Counts distinct scaffolds exactly for one variant, in a second pass.

    Only run for the handful of variants the sketch could not separate. Holding
    an exact set is bounded by the survivor count of that one variant, not by
    the library, and the caller is told what it cost.
    """
    variant = sweep_mod.Variant(choices)
    nitrogen = {s["name"]: s for s in knobs_mod.settings_of("nitrogen")}
    scaffolds = set()
    survivors = 0
    for text in sweep_mod.iter_library(part_dir, limit):
        mol = smiles_mod.parse(text)
        if mol is None:
            continue
        weight = Descriptors.MolWt(mol)
        rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
        charge = Chem.GetFormalCharge(mol)
        if variant.charge is not None and charge != variant.charge:
            continue
        if weight >= variant.mw_limit or rotatable > variant.rot_limit:
            continue
        if not nitrogen[choices["nitrogen"]]["test"](mol):
            continue
        survivors += 1
        try:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "")
        except ValueError:
            scaffolds.add("")
    return {"survivors": survivors, "scaffolds_exact": len(scaffolds),
            "held_strings": len(scaffolds)}


def propose(sweep_report, required, part_dir=None, limit=None, exact=True):
    baseline = sweep_report["baseline"]
    rows = [classify(s, required, baseline) for s in sweep_report["specifications"]]

    uncertain = [r for r in rows if r["state_by_sketch"] == "uncertain"]
    if exact and uncertain and part_dir:
        for row in uncertain:
            measured = exact_scaffold_count(part_dir, row["choices"], limit)
            row["exact"] = measured
            row["state"] = ("feasible" if measured["scaffolds_exact"] >= required
                            else "infeasible")
    for row in rows:
        row.setdefault("state", row["state_by_sketch"])

    feasible = [r for r in rows if r["state"] == "feasible"]
    # Least loosened first, then published provenance, then more survivors as a
    # deterministic tie-break rather than a preference.
    feasible.sort(key=lambda r: (r["steps_from_baseline"],
                                 0 if r["provenance"]["weakest"] == "published" else 1,
                                 -r["survivors"]))

    out = {
        "request": {"distinct_scaffolds_required": required,
                    "specification_as_asked": baseline},
        "verdict": None,
        "recommended": None,
        "frontier": feasible[:8],
        "infeasible_count": sum(1 for r in rows if r["state"] == "infeasible"),
        "uncertain_resolved_exactly": [r["specification"] for r in uncertain],
        "blocked_knobs": knobs_mod.BLOCKED,
        "how_this_was_chosen": [
            "a variant must reach the required distinct scaffold count; the "
            "deliverable is one representative per scaffold, so the constraint "
            "is on scaffolds and not on survivors",
            "among those, the fewest relaxation steps from the specification as "
            "asked",
            "ties towards settings a published rule already covers",
            "the step granularity is a human choice, so the frontier is "
            "returned and the ranking is a suggestion",
        ],
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
    result = propose(report, args.required, args.parts, args.limit or None,
                     exact=not args.no_exact)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"request: {args.required} distinct scaffolds")
    print(f"as asked: {' + '.join(result['request']['specification_as_asked'].values())}")
    print(f"\nverdict: {result['verdict']}\n")

    if result.get("recommended"):
        r = result["recommended"]
        print(f"recommended: {r['specification']}")
        print(f"  survivors {r['survivors']:,}, scaffolds ~{r['scaffolds_estimated']:,}")
        print(f"  {r['steps_from_baseline']} relaxation step(s) from what was asked")
        for knob, setting in r["loosened"].items():
            body = knobs_mod.setting(knob, setting)
            print(f"  loosened {knob} -> {setting}  [{body['provenance']}]")
            print(f"    {body['source']}")
        if not r["loosened"]:
            print("  nothing loosened")

    if result.get("as_asked"):
        a = result["as_asked"]
        print(f"\nas asked would give: survivors {a['survivors']:,}, "
              f"scaffolds ~{a['scaffolds_estimated']:,} -> {a['state']}")

    if result.get("best_reachable"):
        b = result["best_reachable"]
        print(f"best reachable: {b['specification']}")
        print(f"  scaffolds ~{b['scaffolds_estimated']:,} against {args.required:,} required")
        for line in result["what_would_unblock"]:
            print(f"  -> {line}")

    print(f"\nfrontier ({len(result['frontier'])} shown):")
    print(f"  {'specification':66} {'surv':>9} {'scaf':>9} {'steps':>6}  provenance")
    for row in result["frontier"]:
        print(f"  {row['specification']:66} {row['survivors']:>9,} "
              f"{row['scaffolds_estimated']:>9,} {row['steps_from_baseline']:>6}  "
              f"{row['provenance']['weakest']}")

    print("\nknobs that cannot be turned:")
    for knob, body in result["blocked_knobs"].items():
        print(f"  {knob}: {body['why']}")
        print(f"    would need: {body['what_would_unblock']}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
