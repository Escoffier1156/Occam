"""What a screen of this specification would and would not establish.

Two measurements that used to sit inside the candidate pipeline, where they
were gates on a list of molecules. They are verdicts, not gates: a researcher
deciding whether to run this screen needs them before the screen, and the list
of molecules is not the product.

  rankability    can the activity model put this set in an order at all? If
                 most scores are duplicates, the top N is settled by the
                 tie-break rather than by the model, and "the highest scoring
                 256" is not a true description of the output. Measured on the
                 -O slice at 47.6% distinct against a 50% floor declared
                 before the run.

  applicability  is the set inside the range the activity model was fitted on?
                 Measured by each molecule's greatest Tanimoto to a training
                 structure. On the earlier candidate list the median was 0.165
                 and the maximum 0.285, so every molecule was outside.

Both run on a sample, which is said in the output. Deciding that a model cannot
order a set does not need every row: a thousand rows showing 47% distinct
settles it, and fingerprinting four thousand took twelve minutes.

Each verdict that comes back negative carries a revision: what to put in the
request instead so that the same question, or the nearest question the evidence
can answer, becomes answerable. The revisions are measured here and not
invented -- a target with no label is answered by naming the resolution the
evidence does have, an unrankable set by measuring which looser specification
clears the floor, an extrapolating set by measuring which training set it is
actually nearest to. Nothing proposes chemistry.

A third thing is reported rather than measured: which label the activity model
was fitted on. The IMI-COMBINE data carries four organism classes, and the
model in use was fitted on the Gram-positive one. That is a fact a researcher
has to know and not a verdict this module should reach.
"""

import argparse
import json
import random
import statistics as st
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import fingerprints as fingerprints_mod  # noqa: E402
import folds as folds_mod  # noqa: E402
import render as render_mod  # noqa: E402
import smiles as smiles_mod  # noqa: E402
from m3_order import sh, stats_from, write_parts  # noqa: E402

from rdkit import DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

RDLogger.DisableLog("rdApp.*")

WORK = ROOT / "data" / "work"
ENGINE_FP = ROOT / "engine" / "target-fp" / "release" / "engine"

# Declared before any of this was measured, and the same numbers the candidate
# pipeline used as gates.
MIN_DISTINCT_SHARE = 0.50
MAX_TIE_SHARE = 0.05
# The applicability-domain threshold. Not derived here: nearest-neighbour
# Tanimoto cut-offs in the 0.4 to 0.6 range are the common practice, and 0.4 is
# the bottom of that band, chosen so the verdict errs towards saying a set is
# inside rather than outside. Written down rather than tuned.
EXTRAPOLATION_BELOW = 0.40

# Positive controls, measured 2026-10-10. A verdict of INERT or out-of-domain
# looks identical to a broken calculation, so neither is issued without these.
CONTROLS = {
    "tanimoto_pipeline": {
        "test": "hold 300 rows out of IMI-COMBINE and measure each one's "
                "greatest Tanimoto to the remaining 21,752",
        "expected": "high, because the molecules come from the same collection",
        "measured": {"median": 0.747, "p5": 0.358, "max": 1.000,
                     "share_at_or_above_0_4": 0.940},
        "verdict": "pass",
        "what_it_rules_out": "a fingerprint radius or bit-length mismatch, or "
                             "salts and stereochemistry left on one side. Any of "
                             "those would depress the control too.",
    },
    "globularity": {
        "test": "compute globularity for the 68 compounds of GramNegAccum "
                "table4 and compare against the `glob` column its authors "
                "computed",
        "expected": "correlated, and spanning a similar range",
        "measured": {"published_median": 0.181, "measured_median": 0.156,
                     "published_range": [0.046, 0.450],
                     "measured_range": [0.037, 0.343],
                     "pearson_r": 0.529,
                     "median_absolute_difference": 0.043,
                     "threshold_agreement": 0.824},
        "verdict": "pass for the range, weak per molecule",
        "what_it_rules_out": "the failure mode where no conformer is embedded "
                             "and every molecule reads as flat: that would pin "
                             "the measured values near 0, and they reach 0.343.",
        "caveat": "r = 0.529 and 12 of 68 disagree on the 0.25 threshold, so an "
                  "individual molecule's value is not reliable. The conformer "
                  "choice moves it by up to 0.236. Use it for a distribution, "
                  "not for a single compound.",
        "bears_on": "the published `glob` column itself has 55 of 68 compounds "
                    "at or below 0.25, so the condition passes 81% of the set it "
                    "was derived from. The INERT finding on ZINC is the same "
                    "direction, more extreme.",
    },
}
SAMPLE = 1200
SAMPLE_SEED = 0


def generator():
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=fingerprints_mod.FP_RADIUS, fpSize=fingerprints_mod.FP_BITS
    )


def training_label(features):
    """Which label the activity model was fitted on, from its own sidecar."""
    sidecar = Path(features).with_name("features.json")
    if not sidecar.exists():
        return {"known": False,
                "why": f"no features.json beside {features}"}
    record = json.loads(sidecar.read_text())
    source = record.get("source", "")
    # py/ingest_imi.py maps the organism columns onto these two names. The
    # column names themselves are not quoted here: `gram-positive_label` is
    # how the file spells it, and a reader is owed the thing it records.
    mapping = {
        "binding": "the gram-positive label, used as a stand-in for target "
                   "binding without the permeability barrier",
        "binding=active": "the gram-positive label, used as a stand-in for "
                          "target binding without the permeability barrier",
        "true_hit": "the gram-negative label, which is binding AND entry",
        "true_hit=active": "the gram-negative label, which is binding AND entry",
    }
    column = record.get("label_column")
    if column is None:
        # Older fingerprint sets did not record it. Fall back to matching the
        # positive count against the source library's columns rather than
        # guessing from the file name.
        column = _infer_from_counts(record, source)
    return {"known": True, "source": source,
            "rows": record.get("rows"),
            "labels": record.get("labels"),
            "column": column,
            "means": mapping.get(column),
            "note": "IMI-COMBINE carries four organism classes "
                    "(gram-negative, gram-positive, fungi, acid-fast) with pMIC "
                    "values. Which one to fit on is a decision, not a default."}


def _infer_from_counts(record, source):
    """Which column has exactly this many actives? Measured, not guessed.

    A fingerprint set built before the label column was recorded can still be
    identified: the positive count has to match one of the source's columns.
    If two match, or none, the answer is None and the caller says so.
    """
    labels = record.get("labels") or {}
    positives = labels.get("positive")
    if positives is None or not Path(source).exists():
        return None
    try:
        table = pq.read_table(source)
    except Exception:
        return None
    matches = []
    for name in table.column_names:
        if name == "smiles":
            continue
        values = table.column(name).to_pylist()
        for value in set(values):
            if sum(1 for v in values if v == value) == positives:
                matches.append(f"{name}={value}")
    return matches[0] if len(matches) == 1 else None


def sample_survivors(survivors, size=SAMPLE, seed=SAMPLE_SEED):
    if len(survivors) <= size:
        return list(survivors), len(survivors)
    return random.Random(seed).sample(list(survivors), size), len(survivors)


def rankability(survivors, features, epochs=10, learning_rate=0.5,
                size=SAMPLE):
    """Can the activity model put this set in an order?"""
    chosen, total = sample_survivors(survivors, size)
    if len(chosen) < 50:
        return {"verdict": "not_measurable",
                "why": f"only {len(chosen)} molecules survive; too few to say "
                       f"whether an order is meaningful"}

    table = pq.read_table(features)
    names = [n for n in table.column_names if n not in ("label", "scaffold", "smiles")]
    gen = generator()

    import numpy as np
    import pyarrow as pa

    bits = fingerprints_mod.FP_BITS
    block = np.zeros((len(chosen), bits), dtype=np.float32)
    kept = 0
    for text in chosen:
        mol = smiles_mod.parse(text)
        if mol is None:
            continue
        block[kept] = gen.GetFingerprintAsNumPy(mol)
        kept += 1
    schema = pa.schema([(n, pa.float32()) for n in names]
                       + [("label", pa.int8()), ("scaffold", pa.string()),
                          ("smiles", pa.string())])
    columns = [pa.array(block[:kept, i]) for i in range(bits)]
    columns += [pa.array(np.zeros(kept, dtype=np.int8)),
                pa.array([""] * kept), pa.array(chosen[:kept])]
    scored_table = pa.table(columns, schema=schema)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        train = tmp / "train"
        write_parts(table, names, [folds_mod.TRAIN] * table.num_rows, train)
        stats = train / "stats.json"
        stats.write_text(json.dumps(
            stats_from(table, names, [True] * table.num_rows)) + "\n")
        weights = tmp / "w.txt"
        sh(ENGINE_FP, train, "--stats", stats, "--evaluate", "train",
           "--epochs", epochs, "--learning-rate", learning_rate,
           "--save-weights", weights)
        part = tmp / "score"
        write_parts(scored_table, names, [folds_mod.TEST] * kept, part)
        dump = tmp / "score.tsv"
        sh(ENGINE_FP, part, "--stats", stats, "--evaluate", "test",
           "--load-weights", weights, "--dump-scores", dump)
        logits = []
        with open(dump) as handle:
            next(handle)
            for line in handle:
                logits.append(float(line.split("\t")[2]))

    counts = Counter(logits)
    distinct = len(counts) / len(logits)
    top_value, top_count = counts.most_common(1)[0]
    ties = 1.0 - distinct
    usable = distinct >= MIN_DISTINCT_SHARE and (top_count / len(logits)) <= MAX_TIE_SHARE
    return {
        "verdict": "answerable" if usable else "not_answerable",
        "sampled": len(logits),
        "of_survivors": total,
        "distinct_share": round(distinct, 4),
        "tie_share": round(ties, 4),
        "most_common_value": round(top_value, 6),
        "most_common_count": top_count,
        "floor_distinct": MIN_DISTINCT_SHARE,
        "ceiling_single_value": MAX_TIE_SHARE,
        "logit_range": [round(min(logits), 4), round(max(logits), 4)],
        "fitted_on": str(features),
        "means": ("the model can order this set"
                  if usable else
                  f"the model cannot order this set: {distinct:.1%} of scores are "
                  f"distinct against a {MIN_DISTINCT_SHARE:.0%} floor declared "
                  f"before the run. A top N drawn from it would be settled by "
                  f"the tie-break, not by the model."),
    }


def applicability(survivors, training_smiles, size=SAMPLE):
    """Is this set inside the range the activity model was fitted on?"""
    chosen, total = sample_survivors(survivors, size)
    gen = generator()
    reference = []
    for text in training_smiles:
        mol = smiles_mod.parse(text)
        if mol is not None:
            reference.append(gen.GetFingerprint(mol))
    if not reference:
        return {"verdict": "not_measurable", "why": "no readable training structures"}
    nearest = []
    for text in chosen:
        mol = smiles_mod.parse(text)
        if mol is None:
            continue
        nearest.append(max(DataStructs.BulkTanimotoSimilarity(
            gen.GetFingerprint(mol), reference)))
    if not nearest:
        return {"verdict": "not_measurable", "why": "no readable survivors"}
    median = st.median(nearest)
    inside = sum(1 for v in nearest if v >= EXTRAPOLATION_BELOW) / len(nearest)
    return {
        "verdict": "answerable" if median >= EXTRAPOLATION_BELOW
                   else "answerable_but_extrapolating",
        "sampled": len(nearest),
        "of_survivors": total,
        "training_structures": len(reference),
        "nearest_tanimoto": {"median": round(median, 3),
                             "max": round(max(nearest), 3),
                             "min": round(min(nearest), 3)},
        "share_at_or_above": round(inside, 4),
        "threshold": EXTRAPOLATION_BELOW,
        "threshold_source": "nearest-neighbour Tanimoto cut-offs of 0.4 to 0.6 "
                            "are common applicability-domain practice; 0.4 is "
                            "the lenient end of that band",
        "control": CONTROLS["tanimoto_pipeline"],
        "means": ("the set overlaps the training structures at the declared "
                  "threshold"
                  if median >= EXTRAPOLATION_BELOW else
                  f"every part of this set sits outside the structures the model "
                  f"was fitted on (median {median:.3f}, max {max(nearest):.3f} "
                  f"against a {EXTRAPOLATION_BELOW} threshold). An order can be "
                  f"produced; its meaning is not established."),
    }


def nearest_training_set(survivors, candidates=None, size=300):
    """Which of the available training sets is this set actually nearest to?

    The revision for an extrapolating verdict. Measured over every labelled
    library present, so the answer is "this one, at this distance" rather than
    "get more data".
    """
    # Name and what it measures. The row count used to be written into the
    # name as well, and the same count was then printed twice in one report --
    # once from this literal and once from the sentence that states the scope.
    # Two copies of a count drift apart the moment a set is re-ingested. The
    # count below comes from the rows this comparison actually read.
    candidates = candidates or {
        "IMI-COMBINE": ("activity", WORK / "imi_clean.parquet"),
        "CHAMP": ("permeability", WORK / "champ.parquet"),
        "GramNegAccum": ("permeability", WORK / "gna.parquet"),
        # A different barrier, and listed for exactly that reason. If a set
        # is nearer to the blood-brain barrier data than to anything about
        # Gram-negatives, that is worth a reader seeing: it says the library
        # resembles brain-penetrant chemistry, which is not what was asked
        # for. Naming the barrier beside each set is what makes the row
        # readable rather than a bare distance.
        "B3DB": ("blood-brain barrier",
                 WORK / "B3DB_classification.parquet"),
    }
    chosen, total = sample_survivors(survivors, size)
    gen = generator()
    probes = []
    for text in chosen:
        mol = smiles_mod.parse(text)
        if mol is not None:
            probes.append(gen.GetFingerprint(mol))
    if not probes:
        return []
    out = []
    for name, (measures, path) in candidates.items():
        if not Path(path).exists():
            continue
        try:
            names = pq.read_schema(path).names
            column = "smiles" if "smiles" in names else "SMILES"
            rows = pq.read_table(path, columns=[column]).column(0).to_pylist()
        except Exception:
            continue
        reference = [gen.GetFingerprint(m) for m in
                     (smiles_mod.parse(s) for s in rows) if m is not None]
        if not reference:
            continue
        nearest = [max(DataStructs.BulkTanimotoSimilarity(p, reference))
                   for p in probes]
        out.append({"training_set": name,
                    "measures": measures,
                    "structures": len(reference),
                    "median_nearest": round(st.median(nearest), 3),
                    "max_nearest": round(max(nearest), 3),
                    "inside_threshold": round(
                        sum(1 for v in nearest if v >= EXTRAPOLATION_BELOW)
                        / len(nearest), 4)})
    out.sort(key=lambda r: -r["median_nearest"])
    return out


def answerable_questions(words=None):
    """What resolution does the evidence actually have?

    The revision for a target that has no label. Read from the datasets rather
    than asserted: IMI-COMBINE separates organism classes, so a request for a
    molecular target can be restated as a request about an organism class, and
    the cost of that substitution is stated.
    """
    import inventory as inventory_mod

    manifest = ROOT / "data" / "raw" / "imi_combine" / "manifest.json"
    out = {"resolution_available": [], "resolution_absent": "molecular target",
           "why": "no dataset here records which protein a compound acted on",
           "scope": "counts below are over the FULL source "
                    "(data/processed/combined_bioassay_data.tsv, 74,202 rows). "
                    "The model in use was fitted on data/work/imi_clean.parquet, "
                    "which keeps only the 22,051 rows carrying BOTH the "
                    "gram-positive and gram-negative labels -- so its "
                    "gram-positive actives number 8,601, not 17,530. The two "
                    "figures are the same label at different scopes."}
    if manifest.exists():
        body = json.loads(manifest.read_text())
        verified = body.get("verified_contents", {})
        for key, value in verified.items():
            if key.endswith("_label") and isinstance(value, dict):
                out["resolution_available"].append({
                    "label": key,
                    "active": value.get("active"),
                    "inactive": value.get("inactive"),
                })
        if verified.get("continuous_values"):
            out["continuous"] = verified["continuous_values"]
    out["substitution"] = (
        "a request naming a molecular target can be restated as a request about "
        "an organism class, which the evidence does resolve. That answers a "
        "different question: whether a compound is active against the organism, "
        "not whether it inhibits the named protein."
    )
    return out


def rankable_alternatives(frontier, library, features, limit=None, size=400,
                          epochs=10, learning_rate=0.5):
    """Which looser specification can the model actually order?

    The revision for an unrankable verdict. Each frontier member is measured,
    so the answer names a specification and its measured distinct share rather
    than suggesting that loosening something might help.
    """
    import knobs as knobs_mod
    import sweep as sweep_mod

    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    nitrogen = {s["name"]: s for s in knobs_mod.settings_of("nitrogen")}
    out = []
    for spec in frontier:
        variant = sweep_mod.Variant(spec["choices"])
        survivors = []
        for text in sweep_mod.iter_library(library, limit):
            mol = smiles_mod.parse(text)
            if mol is None:
                continue
            if variant.charge is not None and Chem.GetFormalCharge(mol) != variant.charge:
                continue
            if Descriptors.MolWt(mol) >= variant.mw_limit:
                continue
            if rdMolDescriptors.CalcNumRotatableBonds(mol) > variant.rot_limit:
                continue
            if not nitrogen[spec["choices"]["nitrogen"]]["test"](mol):
                continue
            survivors.append(text)
            if len(survivors) >= size * 3:
                break
        if len(survivors) < 50:
            continue
        measured = rankability(survivors, features, epochs, learning_rate, size)
        out.append({"specification": spec["specification"],
                    "steps_from_baseline": spec.get("steps_from_baseline"),
                    "distinct_share": measured.get("distinct_share"),
                    "verdict": measured["verdict"]})
        if measured["verdict"] == "answerable":
            break
    return out


def report(survivors, features=WORK / "imi_fp" / "features.parquet",
           training_library=WORK / "imi_clean.parquet", size=SAMPLE,
           epochs=10, learning_rate=0.5):
    """All three, for one survivor set. Molecules are never returned."""
    training_smiles = pq.read_table(training_library,
                                    columns=["smiles"]).column(0).to_pylist()
    rank = rankability(survivors, features, epochs, learning_rate, size)
    apply_ = applicability(survivors, training_smiles, size)
    model = training_label(features)

    # The third pillar. A negative verdict without a revision is half an answer.
    revisions = {}
    if apply_["verdict"] == "answerable_but_extrapolating":
        revisions["activity prediction"] = {
            "problem": "the set sits outside the structures the model was fitted on",
            "measured_alternatives": nearest_training_set(survivors),
            "what_this_means": "the training set listed first is the one this "
                               "library is actually nearest to. If none reaches "
                               "the threshold, no model fitted on what is here "
                               "can speak about this library, and the revision "
                               "is data rather than a setting.",
        }
    if rank["verdict"] == "not_answerable":
        revisions["ranking"] = {
            "problem": f"only {rank.get('distinct_share', 0):.1%} of scores are "
                       f"distinct against a {MIN_DISTINCT_SHARE:.0%} floor",
            # Named a function, which is a line of this codebase and not an
            # answer. A reader deciding what to do next needs the action, not
            # the call that performs it.
            "measured_alternatives": "a looser specification spreads the "
                                     "scores further, so one of the "
                                     "alternatives on the frontier may clear "
                                     "the floor. Which of them does is "
                                     "measurable and has not been measured "
                                     "here",
            "what_this_means": "the set can still be screened; what cannot be "
                               "done is to take the top N from it, because "
                               "the tie-break rather than the model would "
                               "choose them",
        }

    return {
        "survivors": len(survivors),
        "sample_size": size,
        "sample_seed": SAMPLE_SEED,
        "rankability": rank,
        "applicability": apply_,
        "activity_model": model,
        "suggested_revisions": revisions,
        "molecules_emitted": False,
        "note": "this module answers what a screen would establish. It does not "
                "return molecules; choosing which to make is the researcher's.",
    }


def show(body, cost=None):
    """The two verdicts, said as claims about the screen rather than as fields.

    A verdict is the one line a reader carries away, so none of it is printed
    as a field value: `not_measurable` in square brackets is legible but it is
    not a sentence, and a reader who is a chemist and not a programmer should
    not have to decide whether the brackets mean anything.

    The control stays beside the applicability verdict and is never printed
    without it. An out-of-domain verdict with no control is the failure this
    function exists to make visible: the same measurement run inside the
    training set has to show the threshold can be met at all.
    """
    print("what a screen of this specification would establish:")

    r = body["rankability"]
    print(f"\n  ordering the survivors: {render_mod.verdict(r['verdict'])}")
    if r.get("means"):
        for line in render_mod.wrap(r["means"], indent=6, hanging=8):
            print(line)
    if r.get("sampled"):
        print(f"      measured on {r['sampled']:,} of {r['of_survivors']:,} "
              f"survivors (sample seed {body['sample_seed']})")

    a = body["applicability"]
    print(f"\n  predicting activity: {render_mod.verdict(a['verdict'])}")
    if a.get("means"):
        for line in render_mod.wrap(a["means"], indent=6, hanging=8):
            print(line)
    if a.get("sampled"):
        print(f"      measured on {a['sampled']:,} of {a['of_survivors']:,} "
              f"survivors against {a['training_structures']:,} training "
              f"structures")
    if a.get("nearest_tanimoto"):
        n = a["nearest_tanimoto"]
        # Three decimals throughout: the same median appears again under the
        # alternatives below, formatted there, and 0.2 beside 0.200 reads as
        # two different measurements.
        print(f"      nearest training structure, Tanimoto: median "
              f"{n['median']:.3f}, best {n['max']:.3f}, "
              f"threshold {a['threshold']}")
        for line in render_mod.wrap(f"the threshold comes from: "
                                    f"{a['threshold_source']}",
                                    indent=6, hanging=8):
            print(line)
    control = a.get("control")
    if control:
        m = control["measured"]
        print(f"      positive control: "
              f"{render_mod.control_verdict(control['verdict'])}")
        for line in render_mod.wrap(
                f"the same measurement run inside the training set gives "
                f"median {m['median']}, with "
                f"{m['share_at_or_above_0_4']:.1%} at or above the threshold, "
                f"so the measurement can clear it",
                indent=8, hanging=10):
            print(line)
        for line in render_mod.wrap(f"this rules out: "
                                    f"{control['what_it_rules_out']}",
                                    indent=8, hanging=10):
            print(line)

    # The count of what would close this verdict belongs to the verdict, not
    # to a second document printed after it. py/what_to_measure.py computes
    # it; this prints it where the reader is already standing.
    if cost and cost.get("measurable"):
        sizes = cost["neighbourhood_sizes"]
        band = ", ".join(f"{row['covered_at']:.2f}: {row['compounds']:,}"
                         for row in cost["sensitivity"])
        print(f"      what would close it: "
              f"{cost['compounds_to_measure']:,} compounds measured, one per "
              f"neighbourhood")
        print(f"        at Tanimoto {cost['covered_at']}. "
              f"{cost['for_half_the_set']:,} of them cover half the set, "
              f"{cost['for_four_fifths']:,} cover four fifths")
        print(f"        {sizes['singletons']:,} neighbourhoods hold one "
              f"compound, so the tail is one purchase each")
        for line in render_mod.wrap(
                f"the count is a property of the threshold: {band}, moving "
                f"{cost['sensitivity_spread']}x across the band",
                indent=8, hanging=10):
            print(line)
        for line in render_mod.wrap(
                "this is coverage, not accuracy: it leaves no part of the set "
                "further than the threshold from something measured, and "
                "promises nothing about how a model fitted on the results "
                "ranks the rest. No compound is emitted.", indent=8,
                hanging=10):
            print(line)

    m = body["activity_model"]
    if m.get("known"):
        label = (str(m["column"]).replace("_", " ") if m.get("column")
                 else "not recorded; could not be identified")
        print(f"\n  the activity model was trained on: {label}")
        if m.get("means"):
            for line in render_mod.wrap(m["means"], indent=6, hanging=8):
                print(line)
        print(f"      {render_mod.counts_phrase(m['labels'])}, "
              f"over {m['rows']:,} rows")
        for line in render_mod.wrap(m["note"], indent=6, hanging=8):
            print(line)

    revisions = body.get("suggested_revisions") or {}
    if revisions:
        # Two blocks answer "what instead", and they answer different
        # questions: this one asks which measured set is nearest, and
        # py/what_to_measure.py asks how much new measuring would close the
        # gap. The headings name the question so the pair does not read as one
        # answer split in half.
        print("\nwhich measured set is nearest "
              "(measured, not a chemical suggestion):")
        for name, body_ in revisions.items():
            print()
            for line in render_mod.wrap(
                    f"{name.replace('_', ' ')}: {body_['problem']}",
                    indent=2, hanging=4):
                print(line)
            alternatives = body_.get("measured_alternatives")
            if isinstance(alternatives, list):
                for row in alternatives:
                    # The tick marks the training sets this library is inside.
                    # It is the column a reader scans, so it earns the mark.
                    mark = "*" if row["inside_threshold"] >= 0.5 else " "
                    label = render_mod.training_set_phrase(row)
                    print(f"    {mark} {label:38} "
                          f"median nearest {row['median_nearest']:.3f}, "
                          f"{row['inside_threshold']:.1%} at or above "
                          f"Tanimoto {EXTRAPOLATION_BELOW}")
            elif alternatives:
                for line in render_mod.wrap(str(alternatives), indent=4,
                                            hanging=6):
                    print(line)
            if body_.get("what_this_means"):
                for line in render_mod.wrap(body_["what_this_means"],
                                            indent=4, hanging=6):
                    print(line)

    print()
    for line in render_mod.wrap(
            "No molecules are emitted. The verdicts above state what a screen "
            "of this specification would and would not establish; choosing "
            "what to make from it is the chemist's call, not this tool's.",
            hanging=2):
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--survivors", type=Path, required=True,
                        help="JSON list of SMILES, or a parquet with a smiles column")
    parser.add_argument("--features", type=Path,
                        default=WORK / "imi_fp" / "features.parquet")
    parser.add_argument("--training-library", type=Path,
                        default=WORK / "imi_clean.parquet")
    parser.add_argument("--sample", type=int, default=SAMPLE)
    parser.add_argument("--out", type=Path, default=WORK / "screenable.json")
    args = parser.parse_args()

    path = args.survivors
    if path.suffix == ".json":
        body = json.loads(path.read_text())
        survivors = body if isinstance(body, list) else body.get("smiles", [])
    else:
        survivors = pq.read_table(path, columns=["smiles"]).column(0).to_pylist()

    result = report(survivors, args.features, args.training_library, args.sample)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    show(result)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
