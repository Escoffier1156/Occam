"""Normalize raw tsv/csv files into parquet.

Each input file is read as RFC 4180 records: a quoted field may contain the
delimiter, doubled quotes and line breaks, so one record can span several
physical lines. The header fixes the column count, every field is stored as a
string, and no column is interpreted.

A record is dropped, and counted by reason, when
  - width:    its field count differs from the header after parsing,
  - required: a column named with --required is empty (none by default),
  - utf8:     it is not valid UTF-8,
  - parse:    the parser rejects it (e.g. text after a closing quote),
  - blank:    it is an empty line.
Missing values stay null and are never imputed.

Records are streamed in fixed-size chunks, so memory does not grow with the
input. data/raw/ is only read; every output goes to data/work/.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
WORK_DIR = ROOT / "data" / "work"

# Both follow RFC 4180 quoting; strict makes malformed quoting an error for
# that record instead of silently merging it with the following lines.
DIALECTS = {
    ".tsv": {"delimiter": "\t", "quoting": csv.QUOTE_MINIMAL, "strict": True},
    ".csv": {"delimiter": ",", "quoting": csv.QUOTE_MINIMAL, "strict": True},
}
CHUNK_ROWS = 8192

csv.field_size_limit(2**31 - 1)


def column_names(header):
    """Make header names unique and non-empty without reading their meaning."""
    names, seen = [], set()
    for i, raw in enumerate(header):
        base = raw.strip() or f"col{i}"
        name, n = base, 1
        while name in seen:
            n += 1
            name = f"{base}_{n}"
        seen.add(name)
        names.append(name)
    return names


def is_valid_utf8(row):
    # surrogateescape maps undecodable bytes to lone surrogates.
    return not any("\udc80" <= ch <= "\udcff" for field in row for ch in field)


def ingest_file(src, dst, required=()):
    """Convert one file; returns (kept, Counter of drop reasons)."""
    kept, dropped = 0, Counter()
    tmp = dst.with_name(dst.name + ".tmp")
    with src.open(encoding="utf-8", errors="surrogateescape", newline="") as fh:
        reader = csv.reader(fh, **DIALECTS[src.suffix.lower()])
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{src}: empty file, no header")
        names = column_names(header)
        missing = [name for name in required if name not in names]
        if missing:
            raise ValueError(f"{src}: required columns not in header: {missing}")
        required_at = [names.index(name) for name in required]
        width = len(names)
        schema = pa.schema([(name, pa.string()) for name in names])
        columns = [[] for _ in range(width)]

        def flush(writer):
            if columns[0]:
                writer.write_batch(pa.record_batch(columns, schema=schema))
                for col in columns:
                    col.clear()

        with pq.ParquetWriter(tmp, schema) as writer:
            while True:
                try:
                    row = next(reader)
                except StopIteration:
                    break
                except csv.Error:
                    dropped["parse"] += 1
                    continue
                if not row:
                    dropped["blank"] += 1
                    continue
                if len(row) != width:
                    dropped["width"] += 1
                    continue
                if not is_valid_utf8(row):
                    dropped["utf8"] += 1
                    continue
                if any(row[i] == "" for i in required_at):
                    dropped["required"] += 1
                    continue
                for col, value in zip(columns, row):
                    col.append(value if value != "" else None)
                kept += 1
                if len(columns[0]) >= CHUNK_ROWS:
                    flush(writer)
            flush(writer)
    tmp.replace(dst)
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=WORK_DIR)
    parser.add_argument(
        "--required", nargs="+", default=[], metavar="COLUMN",
        help="drop records where any of these columns is empty",
    )
    args = parser.parse_args()

    sources = sorted(p for p in args.raw_dir.iterdir() if p.suffix.lower() in DIALECTS)
    if not sources:
        sys.exit(f"no tsv/csv files in {args.raw_dir}")
    stems = [p.stem for p in sources]
    if len(set(stems)) != len(stems):
        sys.exit(f"duplicate file stems in {args.raw_dir}: outputs would collide")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for src in sources:
        dst = args.out_dir / f"{src.stem}.parquet"
        kept, dropped = ingest_file(src, dst, args.required)
        n = sum(dropped.values())
        total += n
        reasons = " ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
        shown = dst.relative_to(ROOT) if dst.is_relative_to(ROOT) else dst
        print(f"{src.name}\tkept={kept}\tdropped={n}" + (f" ({reasons})" if reasons else "") + f"\t-> {shown}")
    print(f"total_dropped={total}")


if __name__ == "__main__":
    main()
