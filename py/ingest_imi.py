"""IMI-COMBINE bioassay tsv -> parquet, keeping rows labelled on both axes.

Gram-positive bacteria have no outer membrane, so activity there stands in for
target binding without the permeability barrier, while Gram-negative activity
needs binding *and* entry. Rows carrying both labels therefore give a library
where the two axes are separately known, which is what the order experiment
needs.

SPARK rows are excluded by default: the repository is MIT, but SPARK's own
terms are not commercial-use clean, and the MIT licence covers the compilation
rather than the upstream data.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "imi_combine" / "data" / "processed" / "combined_bioassay_data.tsv"
WORK_DIR = ROOT / "data" / "work"
GP, GN = "gram-positive_label", "gram-negative_label"

csv.field_size_limit(2**31 - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SRC)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--include-spark", action="store_true")
    args = parser.parse_args()

    out = args.out or WORK_DIR / ("imi_all.parquet" if args.include_spark else "imi_clean.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, joint, skipped = [], Counter(), Counter()
    with args.source.open(newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if not (r[GP] and r[GN]):
                skipped["missing_a_label"] += 1
                continue
            if not args.include_spark and "spark" in r["compound_source"]:
                skipped["spark"] += 1
                continue
            smiles = (r["compound_smiles"] or "").strip()
            if not smiles:
                skipped["no_smiles"] += 1
                continue
            joint[(r[GP], r[GN])] += 1
            rows.append((smiles, r[GP], r[GN]))

    table = pa.table(
        {
            "smiles": [a for a, _, _ in rows],
            "binding": [b for _, b, _ in rows],      # Gram-positive: binding proxy
            "true_hit": [c for _, _, c in rows],     # Gram-negative: binding AND entry
        }
    )
    pq.write_table(table, out, row_group_size=8192)

    gp_active = joint[("active", "active")] + joint[("active", "inactive")]
    gn_active = joint[("active", "active")] + joint[("inactive", "active")]
    summary = {
        "source": str(args.source),
        "rows": len(rows),
        "includes_spark": args.include_spark,
        "skipped": dict(skipped),
        "joint": {f"{a}/{b}": n for (a, b), n in sorted(joint.items())},
        "proxy_error_rate": {
            "gram_negative_actives": gn_active,
            "missed_by_proxy": joint[("inactive", "active")],
            "rate": joint[("inactive", "active")] / gn_active if gn_active else None,
            "meaning": "Gram-negative actives that the Gram-positive proxy calls inactive, "
                       "i.e. the share of true hits the binding stand-in cannot see.",
        },
        "blocked_by_membrane": {
            "gram_positive_actives": gp_active,
            "inactive_against_gram_negative": joint[("active", "inactive")],
            "rate": joint[("active", "inactive")] / gp_active if gp_active else None,
            "meaning": "Binds but cannot get in; the population a permeability filter "
                       "is meant to remove before the expensive step.",
        },
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
