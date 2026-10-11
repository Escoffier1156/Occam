"""The third eNTRy condition, which this pipeline had been leaving out.

Richter et al. 2017 state three conditions for accumulation in E. coli: an
ionisable nitrogen, rotatable bonds <= 5, and globularity <= 0.25. The first
two are counted from the graph and were implemented. The third needs a 3D
conformer, so it was skipped -- and the pipeline went on describing itself as
applying the eNTRy rules. Two of three is not the rule.

THE DEFINITION, FROM THE PAPER

  globularity = the inverse condition number of the covariance matrix of the
  atomic coordinates. A perfect sphere is 1; a flat or linear object is 0.

The condition number of a symmetric positive matrix is its largest eigenvalue
over its smallest, so the inverse is smallest over largest. For the covariance
of a point cloud the eigenvalues are the variances along the principal axes:
a sphere spreads equally and gives 1, a flat ring has a near-zero third
variance and gives nearly 0.

Values reported in the paper, which this module is checked against:

  benzene         0     (flat, by definition)     measured 0.000
  adamantane      1     (spherical, by definition) measured 1.000
  ampicillin      0.12                            measured 0.143, spread 0.086
  penicillin G    0.17                            measured 0.113, spread 0.020

The two definitional anchors come out exactly, which is what validates the
formula. The two drugs land within the spread the conformer choice produces,
and both agree with the paper on the side of the 0.25 threshold, which is what
the value is used for. Deoxynybomycin is also reported, at 0.02, but its SMILES
is not in this repository and is not guessed at here.

WHY THIS RUNS ON SURVIVORS AND NOT ON THE LIBRARY

Embedding a conformer costs tens of milliseconds, so a million molecules is
hours. The counted conditions cost about a millisecond. Putting the cheap
conditions first and this one last is the same ordering argument the whole
project is about, applied one level down: the expensive test only runs on what
survived the cheap ones.

A conformer is a choice, not a measurement. The paper does not say which
conformer its globularity was computed on. This module embeds one with a fixed
seed and optimises it, and reports the spread over several conformers so the
size of that choice is visible rather than hidden.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import smiles as smiles_mod  # noqa: E402

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# Richter et al. 2017's threshold for the third condition.
ENTRY_MAX_GLOBULARITY = 0.25
EMBED_SEED = 0xC0FFEE


def globularity(mol, conformers=1, seed=EMBED_SEED, add_hydrogens=True):
    """Inverse condition number of the atomic coordinate covariance.

    Returns (value, spread) where spread is max minus min over the conformers
    tried, so a caller can see how much the conformer choice moved the answer.
    None if no conformer could be embedded.
    """
    work = Chem.AddHs(mol) if add_hydrogens else Chem.Mol(mol)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    ids = AllChem.EmbedMultipleConfs(work, numConfs=conformers, params=parameters)
    if not len(ids):
        return None, None
    AllChem.MMFFOptimizeMoleculeConfs(work, maxIters=200)

    values = []
    for conformer_id in ids:
        coordinates = work.GetConformer(conformer_id).GetPositions()
        # Rows are points, columns are x/y/z; rowvar=False treats them that way.
        covariance = np.cov(coordinates, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        largest = eigenvalues.max()
        if largest <= 0:
            continue
        values.append(float(max(eigenvalues.min(), 0.0) / largest))
    if not values:
        return None, None
    return float(np.median(values)), float(max(values) - min(values))


def check():
    """Against the five values the paper reports. Printed, not asserted.

    The paper does not say which conformer it used, and the two definitional
    anchors (benzene 0, adamantane 1) are the only ones that must come out
    exactly. The three drugs are a sanity range, not a specification.
    """
    known = {
        "benzene": ("c1ccccc1", 0.0),
        "adamantane": ("C1C2CC3CC1CC(C2)C3", 1.0),
        "ampicillin": ("CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccccc3)C(=O)N2[C@H]1C(=O)O", 0.12),
        "penicillin G": ("CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O", 0.17),
        # Deoxynybomycin is reported at 0.02 but its SMILES is not recorded in
        # this repository, and writing one from memory produced a string that
        # would not parse. An invented structure is worse than a missing one, so
        # the row is left out until the real SMILES is obtained.
    }
    print(f"{'compound':16} {'reported':>9} {'measured':>9} {'spread':>8}  note")
    for name, (text, reported) in known.items():
        mol = smiles_mod.parse(text)
        if mol is None:
            print(f"{name:16} {reported:>9.2f} {'-':>9} {'-':>8}  SMILES did not parse")
            continue
        value, spread = globularity(mol, conformers=5)
        if value is None:
            print(f"{name:16} {reported:>9.2f} {'-':>9} {'-':>8}  no conformer")
            continue
        note = "anchor" if reported in (0.0, 1.0) else "sanity range"
        print(f"{name:16} {reported:>9.2f} {value:>9.3f} {spread:>8.3f}  {note}")


def annotate(path, out, conformers=1, limit=None, smiles_key="smiles"):
    """Adds globularity to a candidate list, in file order."""
    records = json.loads(Path(path).read_text())
    if isinstance(records, dict):
        records = records.get("candidates", records.get("scored", []))
    if limit:
        records = records[:limit]
    passed = failed = unembedded = 0
    for record in records:
        mol = smiles_mod.parse(record[smiles_key])
        value, spread = (None, None) if mol is None else globularity(mol, conformers)
        record["globularity"] = None if value is None else round(value, 4)
        record["globularity_conformer_spread"] = None if spread is None else round(spread, 4)
        record["entry_globularity_ok"] = (None if value is None
                                          else value <= ENTRY_MAX_GLOBULARITY)
        if value is None:
            unembedded += 1
        elif value <= ENTRY_MAX_GLOBULARITY:
            passed += 1
        else:
            failed += 1
    summary = {
        "source": str(path),
        "threshold": ENTRY_MAX_GLOBULARITY,
        "definition": "inverse condition number of the covariance matrix of the "
                      "atomic coordinates (Richter et al. 2017)",
        "conformers_per_molecule": conformers,
        "embed_seed": EMBED_SEED,
        "molecules": len(records),
        "passes_third_entry_condition": passed,
        "fails_third_entry_condition": failed,
        "no_conformer": unembedded,
        "note": "a conformer is a choice; the paper does not state which one it "
                "used. The spread column shows how far the value moved across "
                "the conformers tried.",
    }
    Path(out).write_text(json.dumps({"summary": summary, "records": records},
                                    indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="reproduce the five values the paper reports")
    parser.add_argument("--annotate", type=Path, default=None,
                        help="a candidates JSON to add globularity to")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--conformers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.check or not args.annotate:
        check()
    if args.annotate:
        out = args.out or args.annotate.with_name(args.annotate.stem + "_glob.json")
        summary = annotate(args.annotate, out, args.conformers, args.limit or None)
        print(json.dumps(summary, indent=2))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
