"""Applies the permeability rules to a ZINC22 library, before any model runs.

This is the step the whole ordering argument is about. The rules are the eNTRy
conditions as py/candidates.py already defines them -- nothing is redefined
here, so the library and the IMI evaluation pool are judged by one set of
constants.

No learned model is used. The rules are thresholds on counted structure, so
there is no applicability domain to leave and no extrapolation to guard
against. The binding model is the part that extrapolates, and it runs later,
on the survivors.

What this measures, and the honest form of the saving:

  The filter cuts the number of molecules that reach the fingerprint and the
  engine. It does NOT cut total work, because deciding the rules costs one
  RDKit pass over every molecule in the library. Both costs are timed and
  reported separately, so "putting permeability first lowers the cost" can be
  checked rather than asserted. HANDOFF.md section 9 already records that
  the claim is unverified; this is where the numbers come from.

Counts are reported as lines read, molecules parsed and distinct InChIKey
skeletons, because those three differ.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import candidates as candidates_mod  # noqa: E402
import smiles as smiles_mod  # noqa: E402

WORK = ROOT / "data" / "work"
PART_ROWS = 8192

# The three groups, as py/candidates.py defines them. Group C keeps everything
# so the filtered groups have something to be compared against.
GROUPS = {
    "A": "primary amine + rotatable <= 5 + MW <= 600",
    "B": "basic nitrogen + rotatable <= 5 + MW <= 600",
    "C": "no filter",
}


def passes(group, props):
    if group == "C":
        return True
    if not (props["rotb_ok"] and props["mw_ok"]):
        return False
    return props["amine"] if group == "A" else props["basic_n"]


def iter_parts(part_dir):
    """Streams rows from the numbered parts, one part open at a time.

    Taboo 3: parquet's own row iterator holds two row groups across a
    boundary, so the parts are opened and closed here instead.
    """
    for path in sorted(part_dir.glob("part-*.parquet")):
        table = pq.read_table(path)
        smiles_list = table.column("smiles").to_pylist()
        ids = table.column("zinc_id").to_pylist()
        del table
        yield from zip(smiles_list, ids)


def run(part_dir, out_dir, groups):
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("smiles", pa.string()), ("zinc_id", pa.string()), ("inchikey", pa.string()),
        ("mw", pa.float32()), ("rotb", pa.float32()),
        ("amine", pa.int8()), ("basic_n", pa.int8()),
    ])
    writers, columns, counts = {}, {}, Counter()
    for group in groups:
        directory = out_dir / f"group_{group}"
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("part-*.parquet"):
            stale.unlink()
        writers[group] = {"dir": directory, "part": 0}
        columns[group] = [[] for _ in schema]

    def flush(group):
        cols = columns[group]
        if not cols[0]:
            return
        state = writers[group]
        pq.write_table(pa.table(cols, schema=schema),
                       state["dir"] / f"part-{state['part']:08d}.parquet",
                       row_group_size=PART_ROWS)
        state["part"] += 1
        columns[group] = [[] for _ in schema]

    started = time.monotonic()
    for text, zinc_id in iter_parts(part_dir):
        counts["read"] += 1
        props = candidates_mod.properties(text)
        if props is None:
            counts["unparseable"] += 1
            continue
        counts["evaluated"] += 1
        row = [text, zinc_id, props["inchikey"], props["mw"], float(props["rotb"]),
               int(props["amine"]), int(props["basic_n"])]
        for group in groups:
            if not passes(group, props):
                continue
            counts[f"kept_{group}"] += 1
            for column, value in zip(columns[group], row):
                column.append(value)
            if len(columns[group][0]) >= PART_ROWS:
                flush(group)
    for group in groups:
        flush(group)
    rule_seconds = time.monotonic() - started

    summary = {
        "library_parts": str(part_dir),
        "lines_read": counts["read"],
        "molecules_evaluated": counts["evaluated"],
        "unparseable": counts["unparseable"],
        "rule_seconds": round(rule_seconds, 1),
        "rule_seconds_per_molecule": round(rule_seconds / max(counts["evaluated"], 1), 6),
        "groups": {},
        "note": "MW <= 600 is added by this study; the published eNTRy rules set no "
                "upper bound on size. Permeability here is a rule decision, not a "
                "learned prediction.",
    }
    for group in groups:
        kept = counts[f"kept_{group}"]
        summary["groups"][group] = {
            "condition": GROUPS[group],
            "kept": kept,
            "share": kept / max(counts["evaluated"], 1),
            "parts": writers[group]["part"],
        }
    unfiltered = summary["groups"].get("C", {}).get("kept")
    if unfiltered:
        for group in groups:
            kept = summary["groups"][group]["kept"]
            summary["groups"][group]["binding_evaluations_avoided"] = unfiltered - kept
    (out_dir / "screen.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", type=Path, default=WORK / "zinc")
    parser.add_argument("--out-dir", type=Path, default=WORK / "zinc_screen")
    parser.add_argument("--groups", nargs="+", default=["A", "B", "C"], choices=list(GROUPS))
    args = parser.parse_args()
    summary = run(args.parts, args.out_dir, args.groups)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
