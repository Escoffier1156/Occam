"""GramNegAccum accumulation tables -> parquet, with tertile labels.

Richter et al. report accumulation in nmol per 10^12 CFU for compounds dosed
into E. coli. Tables 1-4 carry a numeric `Accum`; tables 5 and 6 carry named
antibiotics without one and are used elsewhere as a structural reference, not
here.

Tables 1-4 share one scale, which is checked rather than assumed: 27 compounds
are measured in more than one table and not one of them disagrees. Table 4's
higher median comes from its composition -- 53% High against 19% in tables 1-3
-- not from a separate calibration, so the values pool directly.

Table 4 writes its structures in a `mol` column rather than `smiles`, and in a
dialect that needs py/smiles.py: 64 of its rows put the charge ahead of the
hydrogen count, and `[N+]` stands for a protonated amine rather than the
zero-hydrogen cation SMILES says it is. Parsed as written, 7 of the 68 rows
carry a primary amine; parsed correctly, 67 do. The whole point of this table
is that it is the n=68 training set behind the eNTRy rules, whose first
condition is an ionisable nitrogen, so reading it as written inverts the one
property it exists to demonstrate.

Compounds are keyed on the InChIKey skeleton, so a structure annotated with
stereochemistry in one table and without it in another counts once. That is
what takes 234 rows to 197 compounds.

The paper defines no accumulation cut-off, so the split is by tertile on the
same rule as py/ingest_champ.py: top third positive, bottom third negative,
middle third held out rather than forced to a side it cannot support.
"""

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import smiles as smiles_mod  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "gramnegaccum"
WORK_DIR = ROOT / "data" / "work"
PART_ROWS = 8192

# The tables carrying a numeric accumulation, and the column each one uses for
# structure. Tables 5 and 6 have no Accum column at all.
TABLES = ((1, "smiles"), (2, "smiles"), (3, "smiles"), (4, "mol"))

# Above this share of molecules left holding unpaired electrons, the dialect is
# not understood. Measured here: 0.000% with py/smiles.py, 38.2% without it.
MAX_RADICAL_RATE = 0.01

RDLogger.DisableLog("rdApp.*")


def read_table(n):
    """Rows of one table. The files use CR-only line endings."""
    raw = (RAW_DIR / f"table{n}.csv").read_text(encoding="utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    return list(csv.DictReader(io.StringIO(raw, newline="")))


def tertile_labels(values):
    """The two cuts that split sorted values into thirds, as ingest_champ does."""
    ordered = sorted(values)
    return ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]


def collect():
    """Every compound with a numeric accumulation, keyed on InChIKey skeleton."""
    compounds = {}
    notes = Counter()
    for n, column in TABLES:
        for row in read_table(n):
            text = (row.get(column) or "").strip()
            accum = (row.get("Accum") or "").strip()
            if not text:
                notes["smiles_missing"] += 1
                continue
            try:
                value = float(accum)
            except ValueError:
                notes["accum_not_numeric"] += 1
                continue
            mol, repaired, reordered = smiles_mod.parse_report(text)
            if mol is None:
                notes["smiles_unparseable"] += 1
                continue
            notes["rows"] += 1
            if repaired:
                notes["hydrogens_recounted"] += 1
            if reordered:
                notes["brackets_reordered"] += 1
            if smiles_mod.radical_atoms(mol):
                notes["radical_remaining"] += 1
            key = Chem.MolToInchiKey(mol)
            if not key:
                notes["inchikey_failed"] += 1
                continue
            skeleton = key.split("-")[0]
            entry = compounds.setdefault(
                skeleton, {"values": [], "tables": set(), "mol": mol, "inchikey": key}
            )
            entry["values"].append(value)
            entry["tables"].add(n)
    return compounds, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=WORK_DIR / "gna.parquet")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    compounds, notes = collect()

    # The quiet failure: a molecule that parses but lost its hydrogen counts
    # raises nothing and simply misses the amine pattern it should match.
    if notes["rows"] and notes["radical_remaining"] / notes["rows"] > MAX_RADICAL_RATE:
        sys.exit(
            f"{RAW_DIR}: {notes['radical_remaining']}/{notes['rows']} molecules still carry "
            f"unpaired electrons ({notes['radical_remaining']/notes['rows']:.2%}), above the "
            f"{MAX_RADICAL_RATE:.0%} limit. py/smiles.py does not understand this dialect yet."
        )

    # The claim that the tables share one scale, re-checked on every run rather
    # than recorded once: a compound measured twice must give the same number.
    shared = {k: v for k, v in compounds.items() if len(v["tables"]) > 1}
    disagreeing = [
        k for k, v in shared.items() if max(v["values"]) - min(v["values"]) > 1e-6
    ]
    if disagreeing:
        sys.exit(
            f"{len(disagreeing)} of {len(shared)} compounds measured in more than one "
            f"table disagree. Tables 1-4 were pooled on the assumption that they share "
            f"one scale; that assumption no longer holds, so they must not be pooled."
        )

    records = []
    for skeleton, entry in compounds.items():
        records.append(
            {
                "smiles": Chem.MolToSmiles(entry["mol"]),
                "inchikey": entry["inchikey"],
                "skeleton": skeleton,
                "accum": sum(entry["values"]) / len(entry["values"]),
                "tables": ",".join(str(t) for t in sorted(entry["tables"])),
            }
        )
    # Sorted on the InChIKey skeleton, not on file order: every downstream
    # tie-break and top-k cut inherits whatever order this writes.
    records.sort(key=lambda r: r["skeleton"])

    low_cut, high_cut = tertile_labels([r["accum"] for r in records])
    for r in records:
        v = r["accum"]
        r["tertile"] = "high" if v >= high_cut else ("low" if v < low_cut else "mid")

    schema = pa.schema(
        [
            ("smiles", pa.string()),
            ("inchikey", pa.string()),
            ("skeleton", pa.string()),
            ("accum", pa.float32()),
            ("tables", pa.string()),
            ("tertile", pa.string()),
        ]
    )
    table = pa.table({f.name: [r[f.name] for r in records] for f in schema}, schema=schema)
    pq.write_table(table, args.out, row_group_size=PART_ROWS)

    summary = {
        "source": str(RAW_DIR),
        "tables": [n for n, _ in TABLES],
        "compounds": len(records),
        "rows_read": notes["rows"],
        "measured_in_more_than_one_table": len(shared),
        "value_disagreements": len(disagreeing),
        "accumulation": "nmol per 10^12 CFU; larger means more compound accumulated",
        "tertile_cuts": {"low_below": low_cut, "high_at_or_above": high_cut},
        "tertiles": dict(Counter(r["tertile"] for r in records)),
        "repairs": {k: notes[k] for k in
                    ("hydrogens_recounted", "brackets_reordered", "radical_remaining")},
        "dropped": {k: notes[k] for k in
                    ("smiles_missing", "accum_not_numeric", "smiles_unparseable",
                     "inchikey_failed") if notes[k]},
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
