"""
NOT THE PRODUCT.

This emits a list of molecules. The product is the assist: py/occam.py takes a
request and returns what a screen of it would and would not establish, and
emits nothing. Choosing which molecules to make is the researcher's.

It is kept, rather than deleted, because it produced the confirmed numbers in
section 4 of HANDOFF.md and because py/candidates.py carries a KeepOut
lock. Deleting it would make those numbers unverifiable.

It is guarded, rather than left alone, because on 2026-10-10 the assistant ran
it directly, twice, without being asked -- a candidate list looks like a
deliverable in a way that a verdict does not. Emitting molecules now needs
--emit-molecules, so reaching for it has to be deliberate.
Sequential screen over an unmeasured library, permeability first.

Stages, in this order, because the order is the point: each stage only pays
for what the previous one let through.

  1  physical rule     molecular weight below the limit, formal charge zero
  2  permeation proxy  group A: a primary amine, the eNTRy pattern
                       group B: any nitrogen basic around pH 7.4
                       both also require rotatable bonds <= 5
  3  activity model    the binary antimicrobial model trained on IMI-COMBINE,
                       applied to the survivors only
  4  diversity         Bemis-Murcko scaffold clustering, one representative
                       per scaffold, top N by score

What the activity score is, stated once and carried into the output: the
training labels are `binding` and `true_hit`, both binary active/inactive with
no target recorded. The model therefore predicts undifferentiated antimicrobial
activity. It does not predict activity against any named enzyme, and nothing
here should be read as a target-specific prediction.

What the permeation stage is: a rule decision on counted structure, not a
learned prediction. The eNTRy conditions come from Richter et al. 2017; the
molecular weight ceiling is added by this study, and the published rules set
no upper bound on size. The rules are applied rather than a trained
permeability model because the only open permeability set large enough to
train on is entirely azide probes, which would import that bias into a library
that contains almost no azides.

Nothing here is verified on this library. These are proxies with a published
provenance, not measurements.
"""

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import candidates as candidates_mod  # noqa: E402
import fingerprints as fingerprints_mod  # noqa: E402
import folds as folds_mod  # noqa: E402
import smiles as smiles_mod  # noqa: E402
from m3_order import assert_ranking_is_usable, sh, stats_from, write_parts  # noqa: E402

from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

RDLogger.DisableLog("rdApp.*")

WORK = ROOT / "data" / "work"
ENGINE_FP = ROOT / "engine" / "target-fp" / "release" / "engine"
PART_ROWS = 8192

# Guards on the ranking, fixed here rather than chosen after seeing it. A
# saturated sigmoid once collapsed a ranking into file order without failing,
# so every path that produces an order carries these.
MAX_TIE_SHARE = 0.05
MIN_DISTINCT_SHARE = 0.50


def generator():
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=fingerprints_mod.FP_RADIUS, fpSize=fingerprints_mod.FP_BITS
    )


def stage_rules(part_dir, max_mw, max_rotatable, nitrogen="both",
                charge="neutral_only"):
    """Stages 1 and 2: streams the library, keeps group A and group B.

    One part is open at a time (taboo 3). Returns the survivors and the counts
    needed to say what the filter actually saved.
    """
    counts = Counter()
    survivors = []
    started = time.monotonic()
    for path in sorted(part_dir.glob("part-*.parquet")):
        table = pq.read_table(path)
        rows = list(zip(table.column("smiles").to_pylist(),
                        table.column("zinc_id").to_pylist()))
        del table
        for text, zinc_id in rows:
            counts["read"] += 1
            # The cheap tests come first and the InChIKey last. Deduplication
            # and the tie-break need the key, but only for survivors, and
            # generating it is 38% of the per-molecule cost: 1.78 ms against
            # 1.11 ms measured over 2,000 rows of this library. On a million
            # molecules that is eleven minutes spent on keys that are thrown
            # away.
            mol = smiles_mod.parse(text)
            if mol is None:
                counts["unparseable"] += 1
                continue
            counts["evaluated"] += 1
            if charge == "neutral_only" and Chem.GetFormalCharge(mol) != 0:
                counts["charged"] += 1
                continue
            weight = Descriptors.MolWt(mol)
            if weight >= max_mw:
                counts["too_heavy"] += 1
                continue
            counts["mw_ok"] += 1
            rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
            if rotatable > max_rotatable:
                counts["too_flexible"] += 1
                continue
            in_a = bool(mol.HasSubstructMatch(candidates_mod.ENTRY_AMINE))
            in_b = bool(mol.HasSubstructMatch(candidates_mod.BASIC_N))
            # Keep only what the request named. "both" is the old behaviour,
            # kept so the runs measured under it stay reproducible.
            wanted = {"primary_amine": in_a, "basic_nitrogen": in_b,
                      "both": in_a or in_b}[nitrogen]
            if not wanted:
                counts["no_ionisable_nitrogen"] += 1
                continue
            # Only now is the key worth computing.
            props = candidates_mod.properties(text)
            if props is None:
                counts["unparseable"] += 1
                continue
            counts["kept"] += 1
            counts["group_a"] += in_a
            counts["group_b"] += in_b
            counts["group_a_and_b"] += in_a and in_b
            survivors.append({
                "smiles": text, "zinc_id": zinc_id,
                "inchikey": props["inchikey"], "skeleton": props["skeleton"],
                "mw": weight, "rotb": float(rotatable),
                "group_a": in_a, "group_b": in_b,
                "basic_n_count": int(len(mol.GetSubstructMatches(candidates_mod.BASIC_N))),
            })
    counts["rule_seconds"] = round(time.monotonic() - started, 1)
    return survivors, counts


def fingerprint_table(records, label=0):
    """ECFP8 columns for the survivors, in the layout the engine reads.

    A label column is required by the part format; the library is unmeasured,
    so it is a constant and is never read back as truth.
    """
    gen = generator()
    names = [f"b{i}" for i in range(fingerprints_mod.FP_BITS)]
    schema = pa.schema([(n, pa.float32()) for n in names]
                       + [("label", pa.int8()), ("scaffold", pa.string()),
                          ("smiles", pa.string())])
    columns = [[] for _ in schema]
    for record in records:
        mol = smiles_mod.parse(record["smiles"])
        bits = gen.GetFingerprintAsNumPy(mol)
        for column, value in zip(columns, bits):
            column.append(float(value))
        columns[-3].append(label)
        columns[-2].append(record["skeleton"] or "")
        columns[-1].append(record["smiles"])
    return pa.table(columns, schema=schema), names


def train_on_imi(imi_features, epochs, learning_rate, tmp):
    """Fits the activity model on every IMI row; returns (weights, stats)."""
    table = pq.read_table(imi_features)
    names = [n for n in table.column_names if n not in ("label", "scaffold", "smiles")]
    part = tmp / "imi_train"
    write_parts(table, names, [folds_mod.TRAIN] * table.num_rows, part)
    stats = part / "stats.json"
    stats.write_text(json.dumps(stats_from(table, names, [True] * table.num_rows)) + "\n")
    weights = tmp / "imi.txt"
    sh(ENGINE_FP, part, "--stats", stats, "--evaluate", "train",
       "--epochs", epochs, "--learning-rate", learning_rate, "--save-weights", weights)
    positives = sum(table.column("label").to_pylist())
    return weights, stats, names, {"rows": table.num_rows, "positives": positives}


def score(records, names, weights, stats, tmp):
    """Scores every survivor; returns logits aligned to `records` by index."""
    table, _ = fingerprint_table(records)
    part = tmp / "zinc_score"
    write_parts(table, names, [folds_mod.TEST] * table.num_rows, part)
    dump = tmp / "zinc_score.tsv"
    sh(ENGINE_FP, part, "--stats", stats, "--evaluate", "test",
       "--load-weights", weights, "--dump-scores", dump)
    logits = []
    with open(dump) as handle:
        next(handle)
        for line in handle:
            _, _, logit = line.split("\t")
            logits.append(float(logit))
    if len(logits) != len(records):
        sys.exit(f"scored {len(logits)} rows for {len(records)} survivors; "
                 f"the dump does not line up")
    return logits


def known_reference():
    """Named antibiotics from GramNegAccum, with their fingerprints."""
    gen = generator()
    known = candidates_mod.known_antibiotics()
    return {name: gen.GetFingerprint(mol) for name, mol in known.items()}


def nearest_known(mol, reference, gen):
    fingerprint = gen.GetFingerprint(mol)
    best_name, best = None, 0.0
    for name, other in reference.items():
        similarity = DataStructs.TanimotoSimilarity(fingerprint, other)
        if similarity > best:
            best_name, best = name, similarity
    return best_name, best


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", type=Path, default=WORK / "zinc")
    parser.add_argument("--imi-features", type=Path,
                        default=WORK / "imi_fp" / "features.parquet")
    parser.add_argument("--max-mw", type=float, default=450.0)
    parser.add_argument("--max-rotatable", type=int, default=5)
    # The settings a request actually names. Without these the stages ran their
    # own hardcoded conditions, so a specification could be declared feasible
    # by py/occam.py and then not be the thing that was screened: group A and
    # group B were both collected and the top N drawn from the union, even when
    # the request asked for a primary amine alone.
    parser.add_argument("--nitrogen", default="both",
                        choices=["primary_amine", "basic_nitrogen", "both"])
    parser.add_argument("--charge", default="neutral_only",
                        choices=["neutral_only", "any_charge"])
    parser.add_argument("--spec", type=Path, default=None,
                        help="a request.json from py/occam.py; its settings "
                             "override the flags so the screen and the "
                             "proposal cannot disagree")
    parser.add_argument("--top", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=WORK / "zinc_candidates.json")
    parser.add_argument("--emit-molecules", action="store_true",
                        help="actually write the molecule list. Without this the "
                             "run stops after saying what it would have emitted.")
    args = parser.parse_args()

    print(__doc__[:__doc__.index("NOT THE PRODUCT") + 400].strip()
          if "NOT THE PRODUCT" in __doc__ else "", file=sys.stderr)
    if not args.emit_molecules:
        print(f"\nstopping: this writes a list of molecules, which is not the "
              f"assist output. Pass --emit-molecules if that is really what is "
              f"wanted, or use py/occam.py for a verdict on a specification.",
              file=sys.stderr)
        return 1

    print(__doc__.strip()[:0] or "", end="")
    if args.spec and args.spec.exists():
        spec = json.loads(args.spec.read_text())
        chosen = {r["knob"]: r["setting"] for r in spec.get("requirements", [])
                  if "knob" in r}
        if "nitrogen" in chosen:
            args.nitrogen = ("primary_amine" if chosen["nitrogen"] == "primary_amine"
                             else "basic_nitrogen")
        args.charge = chosen.get("charge", args.charge)
        if chosen.get("weight") == "mw_under_600":
            args.max_mw = 600.0
        if chosen.get("rotatable") in ("rotatable_le_7", "rotatable_le_9"):
            args.max_rotatable = int(chosen["rotatable"].rsplit("_", 1)[1])
        print(f"[0] settings taken from {args.spec}")

    print(f"[1-2] rules: MW < {args.max_mw:g}, charge {args.charge}, "
          f"rotatable <= {args.max_rotatable}, nitrogen {args.nitrogen}")
    survivors, counts = stage_rules(args.parts, args.max_mw, args.max_rotatable,
                                    args.nitrogen, args.charge)
    print(f"      read {counts['read']:,}  evaluated {counts['evaluated']:,}  "
          f"kept {counts['kept']:,} ({counts['kept']/max(counts['evaluated'],1):.2%})  "
          f"in {counts['rule_seconds']}s")
    print(f"      group A {counts['group_a']:,}   group B {counts['group_b']:,}   "
          f"both {counts['group_a_and_b']:,}")
    if not survivors:
        sys.exit("no survivors; nothing to score")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print("[3] training the binary activity model on IMI-COMBINE")
        weights, stats, names, imi = train_on_imi(
            args.imi_features, args.epochs, args.learning_rate, tmp)
        print(f"    {imi['rows']:,} rows, {imi['positives']:,} active")
        print(f"[3] scoring {len(survivors):,} survivors")
        started = time.monotonic()
        logits = score(survivors, names, weights, stats, tmp)
        score_seconds = round(time.monotonic() - started, 1)

    ranking = assert_ranking_is_usable(logits, "activity on the library")
    if ranking["tie_share"] > MAX_TIE_SHARE or ranking["distinct_share"] < MIN_DISTINCT_SHARE:
        sys.exit(f"ranking unusable: ties {ranking['tie_share']:.2%}, "
                 f"distinct {ranking['distinct_share']:.1%}")

    for record, logit in zip(survivors, logits):
        record["logit"] = logit

    print(f"[4] scaffold clustering {len(survivors):,} survivors")
    gen = generator()
    reference = known_reference()
    # Deterministic throughout: score, then InChIKey, then position. Every
    # "take the first of the tied" is otherwise an entry for file order.
    order = sorted(range(len(survivors)),
                   key=lambda i: (-survivors[i]["logit"],
                                  survivors[i]["inchikey"] or "", i))
    seen, chosen = set(), []
    for i in order:
        key = survivors[i]["skeleton"] or survivors[i]["inchikey"] or str(i)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(i)
        if len(chosen) >= args.top:
            break
    scaffolds_total = len({s["skeleton"] or s["inchikey"] for s in survivors})

    out = []
    for rank, i in enumerate(chosen, start=1):
        record = survivors[i]
        mol = smiles_mod.parse(record["smiles"])
        name, similarity = nearest_known(mol, reference, gen)
        out.append({
            "rank": rank,
            "smiles": record["smiles"],
            "zinc_id": record["zinc_id"],
            "inchikey": record["inchikey"],
            "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "",
            "mw": round(record["mw"], 2),
            "rotatable_bonds": record["rotb"],
            "formal_charge": Chem.GetFormalCharge(mol),
            "group_a_primary_amine": record["group_a"],
            "group_b_basic_nitrogen": record["group_b"],
            "basic_nitrogen_count": record["basic_n_count"],
            "activity_logit": round(record["logit"], 4),
            "nearest_known_antibiotic": name,
            "tanimoto_to_nearest_known": round(similarity, 3),
        })

    bands = Counter()
    for row in out:
        t = row["tanimoto_to_nearest_known"]
        bands[">=0.85 analogue of a known drug" if t >= 0.85 else
              "0.70-0.85" if t >= 0.70 else
              "0.40-0.70 the intended range" if t >= 0.40 else
              "<0.40 unlike anything known"] += 1

    summary = {
        "library": str(args.parts),
        "stages": {
            "1_physical": {"max_mw": args.max_mw, "formal_charge": 0,
                           "read": counts["read"], "evaluated": counts["evaluated"],
                           "unparseable": counts["unparseable"],
                           "rejected_charged": counts["charged"],
                           "rejected_too_heavy": counts["too_heavy"],
                           "passed": counts["mw_ok"]},
            "2_permeation_rules": {
                "max_rotatable": args.max_rotatable,
                "nitrogen_condition": args.nitrogen,
                "charge_condition": args.charge,
                "rejected_too_flexible": counts["too_flexible"],
                "rejected_no_ionisable_nitrogen": counts["no_ionisable_nitrogen"],
                "group_a_primary_amine": counts["group_a"],
                "group_b_basic_nitrogen": counts["group_b"],
                "in_both": counts["group_a_and_b"],
                "survivors": counts["kept"],
                "seconds": counts["rule_seconds"],
                "what_this_is": "a rule decision on counted structure, not a learned "
                                "prediction. eNTRy conditions from Richter et al. 2017; "
                                "the molecular weight ceiling is added by this study.",
            },
            "3_activity_model": {
                "training_set": str(args.imi_features),
                "training_rows": imi["rows"], "training_positives": imi["positives"],
                "epochs": args.epochs, "learning_rate": args.learning_rate,
                "fingerprint": f"ECFP{2*fingerprints_mod.FP_RADIUS}, "
                               f"{fingerprints_mod.FP_BITS} bits",
                "scored": len(survivors), "seconds": score_seconds,
                "ranking_guard": ranking,
                "what_this_is": "undifferentiated antimicrobial activity. The training "
                                "labels are binary active/inactive with no target "
                                "recorded, so this is NOT a target-specific prediction.",
            },
            "4_diversity": {"distinct_scaffolds_among_survivors": scaffolds_total,
                            "returned": len(out)},
        },
        "cost_of_putting_permeability_first": {
            "activity_evaluations_run": len(survivors),
            "activity_evaluations_if_unfiltered": counts["evaluated"],
            "avoided": counts["evaluated"] - len(survivors),
            "but": "the rules themselves cost one RDKit pass over every molecule "
                   f"({counts['rule_seconds']}s for {counts['evaluated']:,}), so total "
                   "work is not reduced -- only the fingerprint and engine work is.",
        },
        "tanimoto_bands": dict(bands),
        "candidates": out,
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nsurvivors {counts['kept']:,}  distinct scaffolds {scaffolds_total:,}  "
          f"returned {len(out)}")
    print(f"activity evaluations avoided: "
          f"{counts['evaluated'] - len(survivors):,} of {counts['evaluated']:,}")
    for band, n in sorted(bands.items()):
        print(f"  {band}: {n}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
