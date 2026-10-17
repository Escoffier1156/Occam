"""py/ingest.py must keep every valid RFC 4180 record and count what it drops.

    nix-shell --run 'python3 -m unittest -v tests/test_ingest.py'
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq
from hypothesis import given, settings, strategies as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import ingest  # noqa: E402

HEADER = ["a", "b", "c"]
# Delimiters, quotes and line breaks inside fields are the cases that matter.
#
# The two non-ASCII characters are deliberate. RFC 4180 is defined over bytes
# and this parser reads UTF-8, so a field has to survive a two-byte character
# (é) and a three-byte one (日) intact -- including when a quote or a line
# break lands beside one. Dropping them to make the file ASCII would leave the
# multi-byte path untested.
FIELD = st.text(alphabet=list("xyz \t,\"\n\ré日"), max_size=12)


def roundtrip(rows, suffix):
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / f"t{suffix}", Path(tmp) / "t.parquet"
        delimiter = "\t" if suffix == ".tsv" else ","
        with src.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(HEADER)
            writer.writerows(rows)
        kept, dropped = ingest.ingest_file(src, dst)
        return kept, dropped, pq.read_table(dst).to_pylist()


class Ingest(unittest.TestCase):
    @settings(max_examples=200, deadline=None, database=None)
    @given(
        rows=st.lists(st.lists(FIELD, min_size=3, max_size=3), max_size=30),
        suffix=st.sampled_from([".tsv", ".csv"]),
    )
    def test_quoted_records_survive(self, rows, suffix):
        kept, dropped, table = roundtrip(rows, suffix)
        self.assertEqual(sum(dropped.values()), 0, dropped)
        self.assertEqual(kept, len(rows))
        expected = [{k: (v if v != "" else None) for k, v in zip(HEADER, row)} for row in rows]
        self.assertEqual(table, expected)

    def test_drop_reasons_are_counted(self):
        raw = 'a\tb\tc\n1\t2\t3\n\n1\t2\n"x"y\t2\t3\n\t2\t3\n"multi\nline"\t2\t3\n'
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "t.tsv", Path(tmp) / "t.parquet"
            src.write_text(raw, encoding="utf-8", newline="")
            kept, dropped = ingest.ingest_file(src, dst, required=["a"])
            self.assertEqual(kept, 2)
            self.assertEqual(dict(dropped), {"blank": 1, "width": 1, "parse": 1, "required": 1})
            self.assertEqual(pq.read_table(dst).column("a").to_pylist(), ["1", "multi\nline"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
