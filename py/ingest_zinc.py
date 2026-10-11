"""ZINC22 tranche (.smi.gz) -> parquet parts, one row group per part.

The library is unmeasured, so there is no label: this writes structures only.
The permeability rules in py/candidates.py decide what survives, and only the
survivors need a fingerprint, which is the whole point of putting the filter
first.

Format notes, measured rather than assumed:
  - tab separated, two columns: SMILES then ZINC id
  - NO header row. The first line is already a molecule, and skipping it loses
    one. (The plan this was written from said there was a header.)
  - the directory name's H number is exactly the heavy atom count
  - a tranche is one heavy-atom / logP / charge cell, so the molecular weight
    inside one file is narrow: zinc-22x H29P240-N runs 350 to 596 Da

Counts are reported three ways, because they differ and the difference has
been a source of confusion before: lines read, molecules parsed, and distinct
InChIKey skeletons.

One part holds one row group: a parquet footer grows with the row group count,
which would make the engine's memory depend on the data volume.
"""

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import smiles as smiles_mod  # noqa: E402

WORK = ROOT / "data" / "work"
PART_ROWS = 8192
# Above this share of unparseable lines the reader is suspect, not the data.
# Measured on zinc-22x H29P240-N: 0.000% over 144,929 molecules.
MAX_PARSE_FAILURE_RATE = 0.01


def iter_molecules(path, limit=None):
    """Streams (smiles, zinc_id) for every readable line, in file order."""
    notes = Counter()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            notes["lines"] += 1
            fields = line.split("\t")
            if len(fields) != 2:
                notes["malformed"] += 1
                continue
            text, zinc_id = fields[0].strip(), fields[1].strip()
            mol, repaired, reordered = smiles_mod.parse_report(text)
            if mol is None:
                notes["unparseable"] += 1
                continue
            if repaired:
                notes["hydrogens_recounted"] += 1
            if reordered:
                notes["brackets_reordered"] += 1
            if smiles_mod.radical_atoms(mol):
                notes["radical_remaining"] += 1
            notes["parsed"] += 1
            yield text, zinc_id, mol
            if limit and notes["parsed"] >= limit:
                break
    notes["done"] = 1
    yield None, notes, None


def run(source, out_dir, limit):
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("part-*.parquet"):
        stale.unlink()
    schema = pa.schema([("smiles", pa.string()), ("zinc_id", pa.string())])

    columns = [[], []]
    part = 0
    rows = 0
    notes = Counter()

    def flush():
        nonlocal columns, part
        if not columns[0]:
            return
        pq.write_table(pa.table(columns, schema=schema),
                       out_dir / f"part-{part:08d}.parquet", row_group_size=PART_ROWS)
        columns = [[], []]
        part += 1

    for text, zinc_id, mol in iter_molecules(source, limit):
        if text is None:
            notes = zinc_id
            break
        # The canonical form, so downstream deduplication and the engine see
        # one spelling per structure.
        columns[0].append(smiles_mod.Chem.MolToSmiles(mol))
        columns[1].append(zinc_id)
        rows += 1
        if len(columns[0]) >= PART_ROWS:
            flush()
    flush()

    failures = notes["unparseable"] + notes["malformed"]
    total = notes["lines"] or 1
    if failures / total > MAX_PARSE_FAILURE_RATE:
        sys.exit(
            f"{source}: {failures}/{total} lines unreadable ({failures/total:.2%}), "
            f"above the {MAX_PARSE_FAILURE_RATE:.0%} limit. The reader is more likely "
            f"at fault than the data; stopping instead of continuing."
        )

    summary = {
        # The path alone was the whole record of where a library came from,
        # and it pointed into a session temporary directory that no longer
        # exists -- so every figure derived from this library had no reachable
        # provenance. The basename survives a move; the manifest names the
        # tranche and the URL to fetch it from.
        "source": str(source),
        "source_name": Path(source).name,
        "manifest": "data/raw/zinc22/manifest.json",
        "parts": part,
        "rows_written": rows,
        "lines_read": notes["lines"],
        "molecules_parsed": notes["parsed"],
        "repairs": {k: notes[k] for k in
                    ("hydrogens_recounted", "brackets_reordered", "radical_remaining")},
        "dropped": {k: notes[k] for k in ("malformed", "unparseable") if notes[k]},
        "note": "no label: ZINC22 is unmeasured. Distinct-structure counts are "
                "computed by py/candidates.py, which deduplicates on InChIKey.",
    }
    (out_dir / "parts.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="a ZINC22 .smi.gz tranche")
    parser.add_argument("--out-dir", type=Path, default=WORK / "zinc")
    parser.add_argument("--limit", type=int, default=1_000_000,
                        help="stop after this many parsed molecules (0 = all)")
    args = parser.parse_args()
    summary = run(args.source, args.out_dir, args.limit or None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
