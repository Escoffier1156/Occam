"""SMILES -> Morgan fingerprint columns, for the binding model.

Nine physicochemical numbers describe how bulky, polar and greasy a molecule
is, which is what crossing a membrane depends on. Whether a molecule binds a
particular pocket depends on which substructures it carries, and no amount of
molecular weight expresses that. The binding model therefore reads a
fingerprint, and this writes one column per bit in the same layout the engine
already reads: F floats, a label, a scaffold.

The engine must be built with --features fingerprint to match FP_BITS.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import smiles as smiles_mod  # noqa: E402

WORK_DIR = ROOT / "data" / "work"
# Must equal F in engine/src/carry.rs under the fingerprint feature.
FP_BITS = 2048
FP_RADIUS = 4  # ECFP8, the radius the published baseline for this data uses.
PART_ROWS = 8192
MAX_RADICAL_RATE = 0.01

RDLogger.DisableLog("rdApp.*")


def scaffold_of(mol):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except ValueError:
        return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--positive-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=FP_BITS)
    parser.add_argument("--radius", type=int, default=FP_RADIUS)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "features.parquet"
    names = [f"b{i}" for i in range(args.bits)]
    schema = pa.schema(
        [(n, pa.float32()) for n in names]
        + [("label", pa.int8()), ("scaffold", pa.string()), ("smiles", pa.string())]
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=args.radius, fpSize=args.bits
    )

    source = pq.ParquetFile(args.source)
    dropped = Counter()
    rows = 0
    labels = Counter()
    # Rows that were kept but needed repairing; see py/smiles.py. Separate from
    # `dropped`, which counts rows that are gone.
    repairs = Counter()
    columns = [[] for _ in schema]
    writer = pq.ParquetWriter(out, schema)

    def flush():
        nonlocal columns
        if columns[0]:
            writer.write_table(pa.table(columns, schema=schema))
            columns = [[] for _ in schema]

    for batch in source.iter_batches(
        batch_size=PART_ROWS, columns=[args.smiles_column, args.label_column]
    ):
        for smiles, label in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if not smiles or label is None:
                dropped["missing"] += 1
                continue
            mol, repaired, reordered = smiles_mod.parse_report(smiles)
            if mol is None:
                dropped["smiles_unparseable"] += 1
                continue
            if repaired:
                repairs["hydrogens_recounted"] += 1
            if reordered:
                repairs["brackets_reordered"] += 1
            if smiles_mod.radical_atoms(mol):
                repairs["radical_remaining"] += 1
            bits = generator.GetFingerprintAsNumPy(mol)
            for col, v in zip(columns, bits):
                col.append(float(v))
            y = 1 if label == args.positive_label else 0
            columns[-3].append(y)
            columns[-2].append(scaffold_of(mol))
            columns[-1].append(smiles)
            labels[y] += 1
            rows += 1
            if len(columns[0]) >= PART_ROWS:
                flush()
    flush()
    writer.close()

    # A dialect that parses but loses hydrogen counts changes which bits are
    # set and raises nothing. Same limit as py/descriptors.py.
    if rows and repairs["radical_remaining"] / rows > MAX_RADICAL_RATE:
        sys.exit(
            f"{args.source}: {repairs['radical_remaining']}/{rows} molecules still carry "
            f"unpaired electrons ({repairs['radical_remaining']/rows:.2%}), above the "
            f"{MAX_RADICAL_RATE:.0%} limit. These parse without error but their "
            f"fingerprints are wrong; py/smiles.py does not understand this dialect yet."
        )

    summary = {
        "source": str(args.source),
        # The column and value, recorded because py/screenable.py has to be
        # able to say what the fitted model predicts. Without this it reported
        # the label as unknown, which is correct but useless.
        "label_column": args.label_column,
        "positive_label": args.positive_label,
        "fingerprint": f"Morgan radius {args.radius} ({args.bits} bits), ECFP{2*args.radius}",
        "rows": rows,
        "labels": {"positive": labels[1], "negative": labels[0]},
        "dropped": dict(dropped),
        "repairs": dict(repairs),
    }
    (args.out_dir / "features.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
