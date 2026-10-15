"""How much measurement would make an out-of-domain set answerable.

    python3 py/what_to_measure.py --survivors data/work/survivors.json

THE GAP THIS CLOSES

py/screenable.py says a screen cannot be ranked because the library sits
outside the structures the model was fitted on. That is true and it is where
the answer used to stop. A bench scientist reads it and asks "so what?", which
is the right question: a verdict that names no next step is half an answer
(contraindication 77), and the half it gives is the half nobody can act on.

The next step is measurable rather than rhetorical. The applicability test
asks whether each compound has a training neighbour within a Tanimoto
threshold. Nothing in the set does. So cluster the set at that same threshold:
every cluster is a neighbourhood with no measured member, and measuring one
compound from a cluster brings the whole cluster within the threshold of
something measured. The number of clusters is the number of compounds to
measure, and the cluster sizes say which ones buy the most.

WHAT THIS NUMBER IS NOT

It is a coverage requirement, not an accuracy promise. Measuring one compound
per neighbourhood makes the set no longer an extrapolation. It does not
establish that a model fitted on those measurements ranks the rest correctly;
that is a separate question with a separate test (py/answerable.py). Saying
"measure 47 compounds and the model will work" would be the kind of claim this
pipeline exists to refuse.

The count is also a function of the threshold, entirely. At a stricter
threshold every neighbourhood is smaller and there are more of them. So the
count is never reported alone: the sensitivity across the usual band is
reported beside it, and if the number moves by a factor of three across that
band then the number is a property of the threshold and the reader is owed
that fact.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import fingerprints as fingerprints_mod  # noqa: E402
from render import wrap  # noqa: E402
import smiles as smiles_mod  # noqa: E402

WORK = ROOT / "data" / "work"

# The applicability threshold py/screenable.py uses, as a similarity. A
# neighbour closer than this counts as covering a compound.
COVERED_AT = 0.40
# The band the threshold is chosen from, reported so the count is never a
# single number standing on its own.
BAND = (0.30, 0.40, 0.50, 0.60)
# Butina holds the full lower triangle. Contraindication 11: above this the
# process is killed, so sample and say so instead.
MAX_MOLECULES = 6000
SAMPLE_SEED = 0


RADII = (2, 3, 4)


def matrix(survivors, training=None, radii=RADII, band=BAND):
    """Neighbourhood counts and both medians, per fingerprint radius.

    TWO MEDIANS, AND THEY ANSWER DIFFERENT QUESTIONS

    within_set is the median nearest-neighbour similarity inside this set. It
    is what the clustering is made of, so it moves with the radius and so does
    the number of compounds to measure.

    to_training is the median nearest-neighbour similarity to the structures
    the activity model was fitted on. That is the applicability domain, and it
    is NOT a free choice of ruler: the model's domain is defined by the
    model's own fingerprint. Measuring it at a radius the model does not use
    produces a number about some other model. It is computed here at each
    radius only to show how much of the extrapolation verdict is the radius --
    which is a fact about the measurement, not a way to escape the verdict.
    """
    sampled = list(survivors)[:MAX_MOLECULES]
    reference = list(training or [])[:MAX_MOLECULES]
    rows = []
    for radius in radii:
        prints, _ = _fingerprints(sampled, radius)
        if len(prints) < 2:
            continue
        row = {"radius": radius, "fingerprint": f"ECFP{2 * radius}",
               "within_set_median": _median_nearest(prints),
               "to_training_median": None,
               "neighbourhoods": {}, "singletons": {}}
        for value in band:
            clusters = _clusters(prints, value)
            row["neighbourhoods"][f"{value:.2f}"] = len(clusters)
            row["singletons"][f"{value:.2f}"] = sum(
                1 for c in clusters if len(c) == 1)
        if reference:
            others, _ = _fingerprints(reference, radius)
            if others:
                row["to_training_median"] = _median_nearest(prints, others)
        rows.append(row)
    return {"fingerprint_bits": fingerprints_mod.FP_BITS,
            "bits_are_fixed": "the bit length does not change with the "
                              "radius. What changes is how many distinct "
                              "substructures are hashed into the same "
                              f"{fingerprints_mod.FP_BITS} bits",
            "model_radius": fingerprints_mod.FP_RADIUS,
            "set_size": len(survivors),
            "clustered": len(sampled),
            "training_size": len(reference),
            "rows": rows,
            "within_set_means": "the median nearest neighbour inside this "
                                "set. Drives the clustering and the count of "
                                "compounds to measure",
            "to_training_means": "the median nearest neighbour in the "
                                 "structures the activity model was fitted "
                                 "on. This is the applicability domain, and "
                                 "it is only meaningful at the radius the "
                                 "model itself uses "
                                 f"(radius {fingerprints_mod.FP_RADIUS})"}


def _median_nearest(prints, reference=None):
    """Median nearest-neighbour Tanimoto, within a set or against another."""
    import statistics as st

    from rdkit import DataStructs

    out = []
    if reference is None:
        if len(prints) < 2:
            return None
        for index, one in enumerate(prints):
            others = prints[:index] + prints[index + 1:]
            out.append(max(DataStructs.BulkTanimotoSimilarity(one, others)))
    else:
        for one in prints:
            out.append(max(DataStructs.BulkTanimotoSimilarity(one, reference)))
    return round(st.median(out), 3)


def _fingerprints(texts, radius=None):
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    if True:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=fingerprints_mod.FP_RADIUS if radius is None else radius,
            fpSize=fingerprints_mod.FP_BITS)
        out, kept = [], []
        for text in texts:
            mol = smiles_mod.parse(text)
            if mol is None:
                continue
            out.append(generator.GetFingerprint(mol))
            kept.append(text)
    return out, kept


def _clusters(fingerprints, covered_at):
    """Neighbourhoods at this similarity. One per compound to measure."""
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    distances = []
    for i in range(1, len(fingerprints)):
        similarity = DataStructs.BulkTanimotoSimilarity(
            fingerprints[i], fingerprints[:i])
        distances.extend(1 - value for value in similarity)
    # Butina cuts on DISTANCE. The applicability test is a SIMILARITY
    # threshold. They are the same 0.4 pointing opposite ways, and clustering
    # at 0.4 distance instead of 0.6 would count neighbourhoods that the
    # applicability test already considers covered -- roughly doubling the
    # number of compounds this module asks anyone to buy.
    cutoff = 1.0 - covered_at
    return Butina.ClusterData(distances, len(fingerprints), cutoff,
                              isDistData=True)


def _fingerprint_name():
    """What the Tanimoto was computed on, named the way a chemist names it.

    A similarity threshold means nothing without this. Morgan radius r is
    ECFP(2r), so the radius this project uses is ECFP8 -- and 0.4 on ECFP8 is
    a considerably harder bar than 0.4 on the ECFP4 most tables report,
    because the substructures are larger and the bits sparser. A reader
    comparing this 0.4 against a published 0.4 without being told the radius
    is comparing two different quantities.
    """
    radius = fingerprints_mod.FP_RADIUS
    return (f"Morgan radius {radius}, {fingerprints_mod.FP_BITS} bits "
            f"(ECFP{2 * radius})")


def _dataset_note(dataset):
    """What the set is, including the part that is absent.

    Naming a target here would be inventing one. Nothing in the evidence
    records what a compound acted on: every label is binary active/inactive,
    which is why knobs.BLOCKED['target_identity'] exists. So the absence is
    stated rather than filled.
    """
    return {
        "set": dataset or "not named by the caller",
        "target": None,
        "target_note": "every label in the evidence is binary "
                       "active/inactive, so this set is not a hit set against "
                       "anything -- it is the subset of a catalogue that "
                       "satisfies the structural requirements that were asked "
                       "for",
    }


def _representative_note():
    """How a compound would be chosen from a neighbourhood, if one were.

    Nothing is chosen here: the output is a count, not a selection. The
    criterion is still declared, because a count of representatives invites
    the question and the usual answer -- highest activity in the cluster -- is
    unavailable for this library. ZINC22 carries no measurements; its own
    ingest record says so.
    """
    return {
        "selected": False,
        "why_not": "this is a count of what would have to be measured, not a "
                   "list of molecules. No compound is emitted",
        "criterion_if_selected": "the Butina cluster centroid, which RDKit "
                                 "returns as the first member of each cluster. "
                                 "Structural and reproducible",
        "criterion_not_available": "highest activity in the cluster, the usual "
                                   "choice, needs activity for these "
                                   "compounds. ZINC22 is unmeasured; the "
                                   "ingest record for this library records "
                                   "that as 'no label'",
    }


def report(survivors, covered_at=COVERED_AT, band=BAND, dataset=None,
           radii=None, training=None):
    """What measuring would cost, in compounds, at this threshold and across
    the band it was chosen from."""
    sampled = list(survivors)
    note = None
    if len(sampled) > MAX_MOLECULES:
        import random

        random.Random(SAMPLE_SEED).shuffle(sampled)
        sampled = sampled[:MAX_MOLECULES]
        note = (f"clustered on a seeded sample of {MAX_MOLECULES:,} of "
                f"{len(survivors):,}: the full distance matrix is held at once "
                f"and does not fit above that. The counts below are of the "
                f"sample, and scale with the set only if the sample's "
                f"diversity does.")

    fingerprints, kept = _fingerprints(sampled)
    if len(fingerprints) < 2:
        return {"measurable": False,
                "why": f"{len(fingerprints)} readable structures is too few to "
                       f"cluster"}

    clusters = _clusters(fingerprints, covered_at)
    sizes = sorted((len(c) for c in clusters), reverse=True)
    total = len(fingerprints)

    # Measuring one compound from the biggest neighbourhoods first: how much
    # of the set each purchase covers. This is the curve a budget is set
    # against.
    curve, running = [], 0
    for index, size in enumerate(sizes, 1):
        running += size
        curve.append({"compounds_measured": index,
                      "share_covered": round(running / total, 4)})

    def at_least(share):
        for row in curve:
            if row["share_covered"] >= share:
                return row["compounds_measured"]
        return None

    sensitivity = []
    for value in band:
        other = _clusters(fingerprints, value)
        sensitivity.append({"covered_at": value, "compounds": len(other)})
    counts = [row["compounds"] for row in sensitivity]
    spread = (max(counts) / min(counts)) if min(counts) else None

    return {
        "measurable": True,
        "dataset": _dataset_note(dataset),
        "fingerprint": _fingerprint_name(),
        "fingerprint_note": "the threshold's strictness is a property of this "
                            "fingerprint. The same number on a smaller radius "
                            "is a looser bar",
        "representative": _representative_note(),
        "set_size": len(survivors),
        "clustered": total,
        "unreadable": len(sampled) - len(kept),
        "covered_at": covered_at,
        "threshold_source": "the applicability threshold py/screenable.py "
                            "uses; nearest-neighbour Tanimoto cut-offs of 0.4 "
                            "to 0.6 are common practice and 0.4 is the lenient "
                            "end",
        "compounds_to_measure": len(clusters),
        # WHAT THIS COUNT IS A COUNT OF
        #
        # MAX_MOLECULES says "sample and say so instead". The sample happened
        # and the saying did not: the screen printed "3,992 compounds
        # measured, one per neighbourhood" for a set of 518,027, and 3,992 is
        # the neighbourhoods found in the 6,000 that were clustered.
        #
        # It does not saturate. Measured on that set, 2026-10-16:
        #
        #     500 sampled ->   477 neighbourhoods (95.4%)
        #   1,000        ->   905 (90.5%)
        #   2,000        -> 1,680 (84.0%)
        #   4,000        -> 2,932 (73.3%)
        #   6,000        -> 3,992 (66.5%)
        #
        # Two thirds of every sample are still their own neighbourhood at the
        # largest size this can cluster, so the count tracks the sample and
        # the whole set's figure is larger by an amount this cannot see.
        # Butina holds the full lower triangle, so 518,027 molecules is not a
        # bigger run, it is 134 billion distances.
        #
        # So the figure is reported as what it is: a floor, measured on a
        # sample, with the sample named beside it. A reader deciding whether
        # to fund the measuring needs to know the real number is above this
        # one, and by how much is not known.
        "counted_on": total,
        "of_survivors": len(survivors),
        "is_a_floor": total < len(survivors),
        "why_a_floor": (
            "clustered on a sample, and the neighbourhood count does not "
            "saturate: two thirds of the sample are still their own "
            "neighbourhood at the largest size this can cluster. The whole "
            "set needs more than this, and how many more is not measured "
            "here" if total < len(survivors) else None),
        "neighbourhood_sizes": {"largest": sizes[0],
                                "median": sizes[len(sizes) // 2],
                                "singletons": sum(1 for s in sizes if s == 1)},
        "for_half_the_set": at_least(0.50),
        "for_four_fifths": at_least(0.80),
        "curve": curve[:40],
        "sensitivity": sensitivity,
        "sensitivity_spread": round(spread, 2) if spread else None,
        "what_this_is": "a coverage requirement: measuring one compound per "
                        "neighbourhood leaves no part of the set further than "
                        "the threshold from something measured",
        "what_this_is_not": "a promise that a model fitted on those "
                            "measurements ranks the rest correctly. That is a "
                            "separate question, and py/answerable.py is the "
                            "test for it",
        # Opt-in: the count moves with the fingerprint as well as with the
        # threshold, and a reader is owed that. Left off by default because it
        # refingerprints the set once per radius, and against the training set
        # it is the slowest thing here.
        "by_fingerprint": (matrix(survivors, training, radii, band)
                           if radii else None),
        "note": note,
    }


def show(body):
    """Laid out for a reader deciding what to put on a plate.

    The order is the order the decision is made in: what the set is, what the
    similarity was computed on, what measuring would cost, and how much of
    that cost is an artefact of the threshold.
    """
    if not body["measurable"]:
        print(f"what to measure: cannot say - {body['why']}")
        return

    data = body["dataset"]
    sizes = body["neighbourhood_sizes"]
    rep = body["representative"]

    # Two counts appear here -- the size of this set and the size of the
    # library it came out of -- and they used to run together on one line.
    print(f"Dataset: {body['set_size']:,} compounds that met the requirements")
    for line in wrap(f"out of {data['set']}", indent=2):
        print(line)
    for line in wrap(f"Target: none. {data['target_note']}", indent=2):
        print(line)
    print(f"Fingerprint: {body['fingerprint']}")
    for line in wrap(body["fingerprint_note"], indent=2):
        print(line)

    print(f"\n[How much measuring would make this set answerable, "
          f"at Tanimoto {body['covered_at']}]")
    print(f"- {body['compounds_to_measure']:,} neighbourhoods, so "
          f"{body['compounds_to_measure']:,} compounds to cover the set")
    print(f"- half the set is covered by "
          f"{body['for_half_the_set']:,} of them")
    print(f"- four fifths by {body['for_four_fifths']:,}")
    print(f"- largest neighbourhood: {sizes['largest']:,} compounds")
    print(f"- singletons (one compound, covers only itself): "
          f"{sizes['singletons']:,}")

    print(f"\n[Threshold sensitivity]")
    for row in body["sensitivity"]:
        print(f"- Tanimoto {row['covered_at']:.2f} : "
              f"{row['compounds']:>6,} neighbourhoods")
    if body["sensitivity_spread"]:
        print(f"* across the band it moves {body['sensitivity_spread']}x")

    print(f"\n[How a compound would be picked from a neighbourhood]")
    for line in wrap(f"Not picked here: {rep['why_not']}.", indent=2):
        print(line)
    for line in wrap(f"If picked: {rep['criterion_if_selected']}.", indent=2):
        print(line)
    for line in wrap(f"Unavailable: {rep['criterion_not_available']}.",
                     indent=2):
        print(line)

    if body.get("note"):
        print()
        for line in wrap(body["note"], indent=2):
            print(line)

    other = body.get("by_fingerprint")
    if other and other.get("rows"):
        at = f"{body['covered_at']:.2f}"
        print(f"\n[The count moves with the fingerprint too]")
        print(f"  FP      at {at}   alone   within   to training*")
        for row in other["rows"]:
            mine = " <- the model's own" if row["radius"] == other["model_radius"] else ""
            to_train = (f"{row['to_training_median']:>14.3f}"
                        if row["to_training_median"] is not None
                        else f"{'not measured':>14}")
            print(f"  {row['fingerprint']:<6}"
                  f"{row['neighbourhoods'][at]:>8,}"
                  f"{row['singletons'][at]:>8,}"
                  f"{row['within_set_median']:>9.3f}"
                  f"{to_train}{mine}")
        for line in wrap(other["bits_are_fixed"] + ".", indent=2):
            print(line)
        for line in wrap(
                "A coarser fingerprint returns fewer neighbourhoods because it "
                "stops distinguishing structures the finer one separates. "
                "Nothing about the set changed when the ruler did, so that is "
                "a decision about what counts as the same compound for the "
                "purpose of buying one, not a reduction in what has to be "
                "measured.", indent=2):
            print(line)
        for line in wrap(other["to_training_means"] + ".", indent=2):
            print(line)
        if other["training_size"]:
            for line in wrap(
                    f"* against {other['training_size']:,} training "
                    f"structures, the most a distance matrix holds at once, "
                    f"so these are lower bounds.", indent=2):
                print(line)

    print()
    for line in wrap(f"This is {body['what_this_is']}.", indent=0, hanging=2):
        print(line)
    for line in wrap(f"It is not {body['what_this_is_not']}.", indent=0,
                     hanging=2):
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--survivors", type=Path, required=True,
                        help="JSON list of SMILES, or a parquet with a smiles "
                             "column")
    parser.add_argument("--covered-at", type=float, default=COVERED_AT)
    parser.add_argument("--dataset", default=None,
                        help="what this set is, for the record (e.g. the "
                             "library and slice it was screened from)")
    parser.add_argument("--out", type=Path,
                        default=WORK / "what_to_measure.json")
    args = parser.parse_args()

    if args.survivors.suffix == ".json":
        loaded = json.loads(args.survivors.read_text())
        survivors = loaded if isinstance(loaded, list) else loaded.get("smiles", [])
    else:
        import pyarrow.parquet as pq

        survivors = pq.read_table(
            args.survivors, columns=["smiles"]).column(0).to_pylist()

    body = report(survivors, args.covered_at, dataset=args.dataset)
    show(body)
    args.out.write_text(json.dumps(body, indent=2) + "\n")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
