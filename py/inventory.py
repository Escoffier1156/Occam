"""What evidence is actually here, read from the files rather than remembered.

Every feasibility verdict has to rest on this. A claim about what a dataset
can answer is worth nothing if the label column turns out to hold two values
with no target recorded -- which is exactly what IMI-COMBINE turned out to
hold when it was finally looked at, after the activity model had been
described as a binding model for some time.

So this reads schemas and label distributions off disk. It never takes a
dataset's name or a note in a document as evidence of what is in it.

Cardinality is the useful signal. A string column with a handful of distinct
values is a label; one with as many values as there are rows is an identifier
or a structure. A label column with two values cannot resolve a target, no
matter what the dataset is called.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "work"
RAW = ROOT / "data" / "raw"

# A string column with at most this many distinct values is treated as a label
# rather than an identifier.
MAX_LABEL_CARDINALITY = 24
# Rows read to characterise a column. Whole files are not loaded: some are a
# million rows wide by two thousand columns.
SAMPLE_ROWS = 20000
STRUCTURE_COLUMNS = ("smiles", "mol", "inchikey", "skeleton", "scaffold", "zinc_id")


def describe_parquet(path):
    """Schema, row count and the distribution of every low-cardinality column."""
    try:
        handle = pq.ParquetFile(path)
    except Exception as error:
        return {"path": str(path), "unreadable": str(error)}
    names = handle.schema_arrow.names
    rows = handle.metadata.num_rows
    out = {
        "path": str(path.relative_to(ROOT)),
        "rows": rows,
        "columns": len(names),
        "column_names": names if len(names) <= 32 else names[:8] + ["..."] + names[-4:],
        "labels": {},
        "structure_columns": [n for n in names if n.lower() in STRUCTURE_COLUMNS],
    }
    # Only the narrow columns are worth characterising; a fingerprint file has
    # two thousand float columns and none of them is a label.
    candidates = [n for n in names
                  if n.lower() not in STRUCTURE_COLUMNS
                  and str(handle.schema_arrow.field(n).type) in
                  ("string", "large_string", "int8", "int16", "int32", "int64", "bool")]
    if candidates:
        batch = next(handle.iter_batches(batch_size=min(SAMPLE_ROWS, max(rows, 1)),
                                         columns=candidates), None)
        if batch is not None:
            for index, name in enumerate(candidates):
                values = batch.column(index).to_pylist()
                counts = Counter(values)
                if len(counts) <= MAX_LABEL_CARDINALITY:
                    out["labels"][name] = {
                        "distinct": len(counts),
                        "values": dict(counts.most_common(MAX_LABEL_CARDINALITY)),
                        "sampled_rows": len(values),
                    }
                else:
                    out["labels"][name] = {"distinct": f">{MAX_LABEL_CARDINALITY}",
                                           "treated_as": "identifier or continuous"}
    return out


def describe_parts(directory):
    """A numbered-part library: counts the parts without reading them all."""
    parts = sorted(directory.glob("part-*.parquet"))
    if not parts:
        return None
    first = describe_parquet(parts[0])
    rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    sidecar = directory / "parts.json"
    return {
        "path": str(directory.relative_to(ROOT)),
        "kind": "parts directory",
        "parts": len(parts),
        "rows": rows,
        "columns": first.get("columns"),
        "column_names": first.get("column_names"),
        "labels": first.get("labels", {}),
        "structure_columns": first.get("structure_columns", []),
        "sidecar": json.loads(sidecar.read_text()) if sidecar.exists() else None,
    }


def label_columns(datasets):
    """Every low-cardinality column found, with its values. Just the facts."""
    out = {}
    for entry in datasets:
        for name, info in (entry.get("labels") or {}).items():
            if isinstance(info.get("distinct"), int):
                out.setdefault(entry["path"], []).append(
                    {"column": name, "distinct": info["distinct"],
                     "values": list(info.get("values", {}))}
                )
    return out


def find_label(datasets, needles):
    """Does any label column or value mention one of these words?

    This is the only honest way to ask whether the evidence resolves a named
    target. The first version of this function guessed from cardinality --
    "more than two values, so it could be a target" -- and promptly reported
    that `split` (train/valid/test) and `tertile` (high/mid/low) could
    distinguish a biological target. The number of values says nothing about
    what the values mean; only reading them does.

    Returns the hits, which may be empty. Empty is a real answer: it means the
    question cannot be asked of this evidence.
    """
    wanted = [n.lower() for n in needles]
    hits = []
    for entry in datasets:
        for name, info in (entry.get("labels") or {}).items():
            haystack = [name.lower()] + [str(v).lower() for v in info.get("values", {})]
            for needle in wanted:
                if any(needle in piece for piece in haystack):
                    hits.append({"dataset": entry["path"], "column": name,
                                 "matched": needle,
                                 "values": list(info.get("values", {}))})
                    break
        for name in entry.get("column_names") or []:
            if any(needle in str(name).lower() for needle in wanted):
                hits.append({"dataset": entry["path"], "column": name,
                             "matched": "column name", "values": None})
    return hits


def searched_sources():
    path = RAW / "_searched.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return [{"name": s["name"], "result": s.get("result"),
             "checked_here": s.get("checked_here"),
             "licence_open": s.get("licence") or s.get("license")}
            for s in record.get("sources", [])]


def take(work=WORK):
    datasets = []
    for path in sorted(work.glob("*.parquet")):
        datasets.append(describe_parquet(path))
    for directory in sorted(d for d in work.iterdir() if d.is_dir()):
        described = describe_parts(directory)
        if described:
            datasets.append(described)
        else:
            single = directory / "features.parquet"
            if single.exists():
                described = describe_parquet(single)
                described["path"] = str(directory.relative_to(ROOT)) + "/features.parquet"
                datasets.append(described)
    return datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--find", nargs="+", default=None, metavar="WORD",
                        help="report every label column or value mentioning these words")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    datasets = take(args.work)
    report = {
        "datasets": datasets,
        "label_columns": label_columns(datasets),
        "searched_sources": searched_sources(),
    }
    if args.find:
        report["find"] = {"needles": args.find, "hits": find_label(datasets, args.find)}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.quiet:
        return

    print(f"{'dataset':44} {'rows':>10} {'cols':>6}  labels")
    for entry in datasets:
        if entry.get("unreadable"):
            print(f"{entry['path']:44} unreadable: {entry['unreadable']}")
            continue
        labels = ", ".join(
            f"{n}({i['distinct']})" for n, i in (entry.get("labels") or {}).items()
            if isinstance(i.get("distinct"), int)
        ) or "-"
        print(f"{entry['path']:44} {entry.get('rows', 0):>10,} "
              f"{entry.get('columns', 0):>6}  {labels}")

    if args.find:
        needles = ", ".join(args.find)
        hits = report["find"]["hits"]
        print(f"\nis there any label mentioning {needles}?")
        if not hits:
            print("  no. not as a column name and not as a label value anywhere.")
            print("  A question about it cannot be answered from this evidence.")
        for hit in hits:
            print(f"  {hit['dataset']}: {hit['column']} (matched {hit['matched']}) "
                  f"{hit['values'] or ''}")


if __name__ == "__main__":
    main()
