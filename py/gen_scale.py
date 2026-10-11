"""Build scaled inputs by concatenating one parquet file with itself n times.

Only the volume changes; the content is not interpreted. The output for factor
n is a directory data/work/<stem>_x<n>/ of parts named part-00000000.parquet,
part-00000001.parquet, ..., each holding exactly one row group of at most
PART_ROWS rows. A single parquet file would not do: its footer grows with the
row group count, i.e. with data volume, and the engine would have to hold it.

Batches are streamed from the source and at most one part is buffered, so the
generator's memory does not grow with the factor either.
"""

import argparse
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = ROOT / "data" / "work"
PART_ROWS = 8192


def part_name(index):
    # Must match BulkReader::part_path in engine/src/bulk.rs.
    return f"part-{index:08d}.parquet"


def scale(src, dst, factor):
    """Write src repeated factor times as a part directory; returns (rows, parts)."""
    source = pq.ParquetFile(src)
    schema = source.schema_arrow
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    pending, pending_rows, rows, parts = [], 0, 0, 0

    def flush():
        nonlocal pending, pending_rows, parts
        if pending_rows:
            table = pa.Table.from_batches(pending, schema=schema)
            pq.write_table(table, tmp / part_name(parts), row_group_size=PART_ROWS)
            parts += 1
        pending, pending_rows = [], 0

    for _ in range(factor):
        for batch in source.iter_batches(batch_size=PART_ROWS):
            while batch.num_rows:
                take = min(batch.num_rows, PART_ROWS - pending_rows)
                pending.append(batch.slice(0, take))
                pending_rows += take
                rows += take
                batch = batch.slice(take)
                if pending_rows == PART_ROWS:
                    flush()
    flush()

    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    return rows, parts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="parquet file produced by ingest.py")
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--out-dir", type=Path, default=WORK_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for factor in args.factors:
        if factor < 1:
            parser.error(f"factor must be >= 1, got {factor}")
        dst = args.out_dir / f"{args.source.stem}_x{factor}"
        rows, parts = scale(args.source, dst, factor)
        size = sum(p.stat().st_size for p in dst.iterdir())
        print(f"{dst}\trows={rows}\tparts={parts}\tbytes={size}")


if __name__ == "__main__":
    main()
