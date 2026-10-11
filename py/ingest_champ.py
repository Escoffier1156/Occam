"""CHAMP accumulation xlsx -> parquet, with tertile labels.

The CHAMP workbook stores MFI fold change over untreated. Fluorescence is
inverse to cytosolic accumulation in a chloroalkane penetration assay, so
accumulation = 1 - FC, and larger means more compound reached the cytosol.

That direction was checked against the workbook rather than assumed, using the
named antibiotics in the Fig. 2c sheet:
  - Vancomycin-butyl sits at FC ~1.01, i.e. accumulation ~0. Vancomycin cannot
    cross the E. coli outer membrane, and it stays high even under PMBN.
  - Trimethoprim and metronidazole sit near FC ~0.01, i.e. accumulation ~0.99,
    and both enter readily.
  - Deleting tolC lowers FC in 10 of 11 antibiotics, and permeabilising the
    outer membrane with PMBN lowers it in 9 of 11. Both raise accumulation.

Values must not be compared across sheets: the same "Porinated WT" column in
two sheets is identical, but the plain-WT column in Fig. 2l comes from another
experiment and is normalised separately. Diagnostics therefore pair conditions
only within one sheet.

The label is WT alone. The engine's permeability filter is asked "does this
get into an ordinary E. coli", not "does it get in once the barrier is
removed"; the other three conditions explain why a compound was kept or
dropped. The paper defines no cut-off, so the split is by tertile: the top
third is positive, the bottom third negative, and the middle third is held out
rather than forced to a side it cannot support.
"""

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev
from xml.etree import ElementTree as ET

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "champ" / "CHAMP_accumulation_E_coli.xlsx"
WORK_DIR = ROOT / "data" / "work"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
PART_ROWS = 8192

# Sheet part -> the columns to take from it. SMILES is column D everywhere.
# Each tuple is (output name, column letter). Grouped per sheet because values
# are only comparable within a sheet.
SHEETS = {
    "xl/worksheets/sheet13.xml": [("wt", "F")],
    "xl/worksheets/sheet16.xml": [("pmbn_wt", "E"), ("porinated_wt", "F")],
    "xl/worksheets/sheet18.xml": [("porinated_wt_check", "E"), ("porinated_tolc_ko", "F")],
}
SMILES_COL = "D"
HEADER_ROWS = 2
LABEL_SOURCE = "wt"


def sheet_rows(path, part):
    """Yields each row of a worksheet as {column letter: text}."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
    for r in ET.fromstring(z.read(part)).iter(NS + "row"):
        cells = {}
        for c in r.iter(NS + "c"):
            m = re.match(r"([A-Z]+)", c.get("r") or "")
            if not m:
                continue
            v = c.find(NS + "v")
            inline = c.find(NS + "is")
            if c.get("t") == "s" and v is not None:
                text = shared[int(v.text)]
            elif inline is not None:
                text = "".join(t.text or "" for t in inline.iter(NS + "t"))
            else:
                text = v.text if v is not None else ""
            cells[m.group(1)] = text
        yield cells


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_conditions(path):
    """Collects every condition, keyed by SMILES; returns (records, report)."""
    per_smiles, report = {}, {}
    for part, columns in SHEETS.items():
        rows = list(sheet_rows(path, part))[HEADER_ROWS:]
        report[part] = len(rows)
        for row in rows:
            smiles = (row.get(SMILES_COL) or "").strip()
            if not smiles:
                continue
            entry = per_smiles.setdefault(smiles, {})
            for name, letter in columns:
                fc = number(row.get(letter))
                if fc is not None:
                    # Fluorescence is inverse to accumulation.
                    entry.setdefault(name, []).append(1.0 - fc)
    return per_smiles, report


def tertile_labels(values):
    """Returns (low_cut, high_cut) splitting values into equal thirds."""
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 3], ordered[(2 * n) // 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or WORK_DIR / "champ.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    per_smiles, sheet_report = read_conditions(args.source)
    for part, n in sheet_report.items():
        print(f"{part}: {n} data rows", file=sys.stderr)

    # The same compound can appear twice in the library; average its replicates.
    spreads = []
    records = []
    for smiles, conditions in per_smiles.items():
        if LABEL_SOURCE not in conditions:
            continue
        row = {"smiles": smiles}
        for name, values in conditions.items():
            row[name] = fmean(values)
            if len(values) > 1:
                spreads.append(pstdev(values))
        records.append(row)

    if spreads:
        print(
            f"replicate SMILES: {len(spreads)} measurements averaged, "
            f"mean sd {fmean(spreads):.4f}, max {max(spreads):.4f}",
            file=sys.stderr,
        )

    # "Porinated WT" appears in two sheets; confirm they really are one column.
    mismatch = sum(
        1
        for r in records
        if "porinated_wt" in r and "porinated_wt_check" in r
        and abs(r["porinated_wt"] - r["porinated_wt_check"]) > 1e-9
    )
    print(f"porinated_wt disagreement between sheets: {mismatch} rows", file=sys.stderr)
    for r in records:
        r.pop("porinated_wt_check", None)

    low_cut, high_cut = tertile_labels([r[LABEL_SOURCE] for r in records])
    for r in records:
        v = r[LABEL_SOURCE]
        r["tertile"] = "high" if v >= high_cut else ("low" if v < low_cut else "mid")

    records.sort(key=lambda r: r["smiles"])
    names = ["smiles", "wt", "pmbn_wt", "porinated_wt", "porinated_tolc_ko", "tertile"]
    schema = pa.schema(
        [("smiles", pa.string())]
        + [(n, pa.float32()) for n in names[1:-1]]
        + [("tertile", pa.string())]
    )
    table = pa.table({n: [r.get(n) for r in records] for n in names}, schema=schema)
    pq.write_table(table, out, row_group_size=PART_ROWS)

    counts = Counter(r["tertile"] for r in records)
    summary = {
        "source": str(args.source),
        "compounds": len(records),
        "label_source": LABEL_SOURCE,
        "accumulation": "1 - MFI fold change; larger means more cytosolic accumulation",
        "tertile_cuts": {"low_below": low_cut, "high_at_or_above": high_cut},
        "tertiles": dict(counts),
    }
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
