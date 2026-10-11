"""ECFP8 parts for an unlabelled library, at a scale where the engine matters.

py/fingerprints.py needs a label column, because every dataset it was written
for has one. ZINC22 is unmeasured, so this writes the same layout with a
constant label that is never read back as truth. It is a separate module rather
than a flag on fingerprints.py so the confirmed numbers that module produced
cannot shift.

Two differences that matter at a million rows:

  The columns are built through numpy rather than Python lists. A part of 8192
  rows is 8192 x 2048 values; as Python floats that is roughly 400 MB per
  flush, which defeats the point of measuring memory.

  One part holds one row group. A parquet footer grows with the row group
  count, so a single file holding thousands of them would make the engine's
  memory depend on the data volume -- the invariant this library exists to
  test.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import fingerprints as fingerprints_mod  # noqa: E402
import smiles as smiles_mod  # noqa: E402

from rdkit import RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

RDLogger.DisableLog("rdApp.*")

WORK = ROOT / "data" / "work"
PART_ROWS = 8192
TRAIN, TEST = 0, 2  # py/folds.py's codes, repeated here to avoid importing it


def iter_library(part_dir):
    """Streams SMILES from the numbered parts, one part open at a time."""
    for path in sorted(part_dir.glob("part-*.parquet")):
        table = pq.read_table(path, columns=["smiles"])
        values = table.column("smiles").to_pylist()
        del table
        yield from values


def run(part_dir, out_dir, split, limit, with_scaffold):
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("part-*.parquet"):
        stale.unlink()

    bits = fingerprints_mod.FP_BITS
    names = [f"b{i}" for i in range(bits)]
    schema = pa.schema([(n, pa.float32()) for n in names]
                       + [("label", pa.int8()), ("split", pa.int8())])
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=fingerprints_mod.FP_RADIUS, fpSize=bits
    )

    block = np.zeros((PART_ROWS, bits), dtype=np.float32)
    scaffolds = []
    filled = part = rows = unparseable = 0
    started = time.monotonic()

    def flush():
        nonlocal filled, part
        if not filled:
            return
        columns = [pa.array(block[:filled, i]) for i in range(bits)]
        columns.append(pa.array(np.zeros(filled, dtype=np.int8)))
        columns.append(pa.array(np.full(filled, split, dtype=np.int8)))
        pq.write_table(pa.table(columns, schema=schema),
                       out_dir / f"part-{part:08d}.parquet", row_group_size=PART_ROWS)
        part += 1
        filled = 0

    for text in iter_library(part_dir):
        if limit and rows >= limit:
            break
        mol = smiles_mod.parse(text)
        if mol is None:
            unparseable += 1
            continue
        block[filled] = generator.GetFingerprintAsNumPy(mol)
        if with_scaffold:
            try:
                scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "")
            except ValueError:
                scaffolds.append("")
        filled += 1
        rows += 1
        if filled >= PART_ROWS:
            flush()
            if part % 20 == 0:
                print(f"  {rows:,} rows, {part} parts, "
                      f"{time.monotonic() - started:.0f}s", flush=True)
    flush()
    seconds = time.monotonic() - started

    total_bytes = sum(p.stat().st_size for p in out_dir.glob("part-*.parquet"))
    summary = {
        "library": str(part_dir),
        "rows": rows,
        "parts": part,
        "unparseable": unparseable,
        "fingerprint": f"ECFP{2*fingerprints_mod.FP_RADIUS}, {bits} bits",
        "split": split,
        "seconds": round(seconds, 1),
        "rows_per_second": round(rows / seconds, 1) if seconds else None,
        "parquet_bytes": total_bytes,
        "uncompressed_feature_bytes": rows * bits * 4,
        "note": "label is a constant 0: ZINC22 is unmeasured. It is never read as truth.",
    }
    (out_dir / "parts.json").write_text(json.dumps(summary, indent=2) + "\n")
    if with_scaffold:
        (out_dir / "scaffolds.json").write_text(json.dumps(scaffolds) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", type=Path, default=WORK / "zinc")
    parser.add_argument("--out-dir", type=Path, default=WORK / "zinc_fp")
    parser.add_argument("--split", type=int, default=TEST, choices=[TRAIN, TEST])
    parser.add_argument("--limit", type=int, default=0, help="0 = the whole library")
    parser.add_argument("--with-scaffold", action="store_true")
    args = parser.parse_args()
    summary = run(args.parts, args.out_dir, args.split, args.limit or None,
                  args.with_scaffold)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
