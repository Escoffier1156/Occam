"""Can this evidence answer this question, and if not, what is missing?

The point of putting this before a run rather than after it. Every verdict
below was reached by hand at least once in this project, each time after the
work had already been done:

  a target-specific requirement was accepted and only then found to have no
  target-resolved label anywhere (IMI-COMBINE's labels are binary);

  a ranking over a million molecules was produced and only then found to sit
  at a maximum Tanimoto of 0.285 from its own training set;

  "transfer holds one way" was published internally and only then found to be
  an artefact of one side having 129 evaluation rows rather than 755.

VERDICTS

  answerable                    the evidence exists and the output would be
                                inside the range it was fitted on
  answerable_but_extrapolating  an order can be produced, but the library sits
                                outside the training set, so the order's
                                meaning is not established
  not_answerable                a required piece of evidence does not exist
  undecidable                   it was measured and the measurement cannot
                                separate the alternatives -- a real outcome,
                                and not the same as a negative result

THRESHOLDS, fixed here rather than chosen after seeing a result

  a library is treated as extrapolation when the median of each molecule's
  greatest Tanimoto to the training set falls below 0.40. The number is
  arbitrary; it is the bottom of the band this project has been reporting
  since the first candidate list, and it is written down instead of adjusted.

  a confidence interval that must exclude 0.5 needs its lower half-width below
  the observed margin. Required n follows from the half-width shrinking with
  1/sqrt(n), which assumes the class balance holds and that the observed
  effect is the real one. The second assumption is optimistic and is reported
  alongside the number.
"""

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import inventory as inventory_mod  # noqa: E402
import smiles as smiles_mod  # noqa: E402

from rdkit import DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

RDLogger.DisableLog("rdApp.*")

WORK = ROOT / "data" / "work"
EXTRAPOLATION_BELOW = 0.40
SAMPLE = 3000
FP_BITS, FP_RADIUS = 2048, 4


def generator():
    return rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def library_smiles(path, limit):
    """SMILES from a parts directory or a single parquet, one part at a time."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("part-*.parquet")) or [path / "features.parquet"]
    else:
        files = [path]
    out = []
    for file in files:
        if not file.exists():
            continue
        table = pq.read_table(file, columns=["smiles"])
        out.extend(table.column("smiles").to_pylist())
        del table
        if len(out) >= limit:
            break
    return out[:limit]


def label_available(datasets, words):
    """Is there any label mentioning these words? Returns a verdict dict."""
    hits = inventory_mod.find_label(datasets, words)
    real = [h for h in hits if h["values"]]
    return {
        "requirement": f"a label for {', '.join(words)}",
        "verdict": "answerable" if real else "not_answerable",
        "hits": hits,
        "missing": None if real else
            f"no label column or value mentions {', '.join(words)}. "
            f"A model fitted here predicts whatever its binary label recorded, "
            f"which is not this.",
    }


def applicability(library_path, training_path, sample=SAMPLE):
    """Is the library inside the training set, by nearest-neighbour Tanimoto?

    This is the gate. It runs on a sample, and says so: the question is whether
    an order would mean anything, which does not need every row.
    """
    gen = generator()
    training = []
    for text in library_smiles(training_path, 100000):
        mol = smiles_mod.parse(text)
        if mol is not None:
            training.append(gen.GetFingerprint(mol))
    if not training:
        return {"requirement": f"library inside {training_path}",
                "verdict": "not_answerable",
                "missing": f"no readable training structures at {training_path}"}
    nearest = []
    for text in library_smiles(library_path, sample):
        mol = smiles_mod.parse(text)
        if mol is None:
            continue
        nearest.append(max(DataStructs.BulkTanimotoSimilarity(
            gen.GetFingerprint(mol), training)))
    if not nearest:
        return {"requirement": f"library inside {training_path}",
                "verdict": "not_answerable",
                "missing": f"no readable structures at {library_path}"}
    median = st.median(nearest)
    inside = sum(1 for v in nearest if v >= EXTRAPOLATION_BELOW) / len(nearest)
    return {
        "requirement": f"library inside the training set at {training_path}",
        "verdict": "answerable" if median >= EXTRAPOLATION_BELOW
                   else "answerable_but_extrapolating",
        "training_structures": len(training),
        "library_sampled": len(nearest),
        "nearest_tanimoto": {"median": round(median, 3),
                             "max": round(max(nearest), 3),
                             "min": round(min(nearest), 3)},
        "share_at_or_above_threshold": round(inside, 4),
        "threshold": EXTRAPOLATION_BELOW,
        "means": ("an order over this library can be produced, but the molecules sit "
                  "outside the range the model was fitted on, so the order's meaning "
                  "is not established. Report it as a procedure, not as a prediction.")
                 if median < EXTRAPOLATION_BELOW else
                 "the library overlaps the training set at the declared threshold.",
    }


def required_n(observed_half_width, observed_n, margin, label):
    """How many evaluation rows to bring the half-width under `margin`."""
    if margin <= 0:
        return None
    n = observed_n * (observed_half_width / margin) ** 2
    return {"question": label,
            "observed": {"n": observed_n, "half_width": observed_half_width},
            "target_half_width": margin,
            "required_n": int(math.ceil(n)),
            "assumes": ["the half-width shrinks with 1/sqrt(n)",
                        "the class balance stays as it is",
                        "the observed effect is the real one, which is optimistic: "
                        "a smaller true effect needs more rows"]}


def rule_share(library_path, patterns, sample=SAMPLE):
    """What share of the library satisfies each structural pattern?

    Answers "will this filter leave me anything" before a filter is run over a
    million molecules. The eNTRy primary amine turned out to admit 254 of a
    million, which is a design verdict, not a detail.
    """
    import knobs as knobs_mod

    from rdkit.Chem import Descriptors, rdMolDescriptors

    # The patterns and the limits come from the registry, which is where a
    # proposal draws them from too. They were taken from py/candidates.py
    # here, so the primary-amine definition existed in two files -- and
    # py/candidates.py is the demoted candidate generator, so the product path
    # depended on a module it does not ship. The two copies were still
    # identical when this was changed; contraindication 31 is the same trap in
    # the same file, found after they had already diverged.
    known = {
        "primary_amine": knobs_mod.setting("nitrogen", "primary_amine")["test"],
        "basic_nitrogen": knobs_mod.setting("nitrogen", "basic_nitrogen")["test"],
        "rotatable_le_5": lambda m: rdMolDescriptors.CalcNumRotatableBonds(m)
                                    <= knobs_mod.setting("rotatable",
                                                         "rotatable_le_5")["limit"],
        "mw_under_450": lambda m: Descriptors.MolWt(m) < knobs_mod.setting(
            "weight", "mw_under_450")["limit"],
        "mw_under_600": lambda m: Descriptors.MolWt(m) < knobs_mod.setting(
            "weight", "mw_under_600")["limit"],
        "neutral": lambda m: sum(a.GetFormalCharge() for a in m.GetAtoms()) == 0,
    }
    unknown = [p for p in patterns if p not in known]
    counts = {p: 0 for p in patterns if p in known}
    total = 0
    for text in library_smiles(library_path, sample):
        mol = smiles_mod.parse(text)
        if mol is None:
            continue
        total += 1
        for name in counts:
            counts[name] += bool(known[name](mol))
    out = {"requirement": f"structural filters on {library_path}",
           "verdict": "answerable" if total else "not_answerable",
           "sampled": total,
           "shares": {k: round(v / total, 5) for k, v in counts.items()} if total else {},
           "available_patterns": sorted(known)}
    if unknown:
        out["not_implemented"] = unknown
        out["verdict"] = "not_answerable"
        out["missing"] = (f"no rule is implemented for {', '.join(unknown)}. "
                          f"A filter that is not implemented cannot be reported as "
                          f"applied.")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True,
                        help="JSON: what is being asked for")
    parser.add_argument("--out", type=Path, default=WORK / "feasibility.json")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())

    datasets = inventory_mod.take(WORK)
    checks = []
    for requirement in spec.get("requirements", []):
        kind = requirement.get("kind")
        if kind == "label":
            checks.append(label_available(datasets, requirement["words"]))
        elif kind == "applicability":
            checks.append(applicability(requirement["library"], requirement["training"],
                                        requirement.get("sample", SAMPLE)))
        elif kind == "rules":
            checks.append(rule_share(requirement["library"], requirement["patterns"],
                                     requirement.get("sample", SAMPLE)))
        elif kind == "power":
            checks.append(required_n(requirement["observed_half_width"],
                                     requirement["observed_n"],
                                     requirement["target_half_width"],
                                     requirement.get("question", "unnamed")))
        else:
            checks.append({"requirement": json.dumps(requirement),
                           "verdict": "not_answerable",
                           "missing": f"no check implemented for kind={kind!r}"})

    order = ["not_answerable", "answerable_but_extrapolating", "undecidable", "answerable"]
    verdicts = [c.get("verdict") for c in checks if c.get("verdict")]
    overall = next((v for v in order if v in verdicts), "answerable")
    report = {"goal": spec.get("goal"), "overall": overall, "checks": checks}
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"goal: {spec.get('goal')}")
    print(f"overall: {overall}\n")
    for check in checks:
        if "required_n" in check:
            print(f"[power] {check['question']}")
            print(f"  observed n={check['observed']['n']}, "
                  f"half-width {check['observed']['half_width']}")
            print(f"  for half-width {check['target_half_width']}: "
                  f"need n ~ {check['required_n']:,}")
            continue
        print(f"[{check['verdict']}] {check['requirement']}")
        for key in ("nearest_tanimoto", "share_at_or_above_threshold", "shares",
                    "missing", "means"):
            if check.get(key) not in (None, {}):
                print(f"  {key}: {check[key]}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
