"""One command for the data: fetch what the manifests declare, then ingest it.

    python3 py/fetch.py              the four measured sets, about 18 MB
    python3 py/fetch.py --zinc       and the ZINC22 slices, about 300 MB
    python3 py/fetch.py --dry-run    what it would do, and nothing else

WHY THE MANIFESTS AND NOT A SCRIPT WITH URLS IN IT

A URL in a script is a second copy of a fact the manifest already holds, and
the two drift. Each data/raw/*/manifest.json carries a `fetch` block -- the
pinned url, where the bytes go, their sha256, and which ingest script turns
them into what the tool reads -- so this module is a loop over declarations
and holds no knowledge of its own. Adding a dataset is a manifest, not a
patch here.

THE CHECKSUM IS THE GUARANTEE, NOT THE PIN

Every url names a tag or a commit, which is what makes a fetch reproducible,
and a tag can be moved. So the bytes are hashed and compared against the
manifest, and a mismatch stops with what moved rather than ingesting it. The
numbers in HANDOFF.md were measured on the bytes these checksums describe; data
that hashes differently is a different set and the figures would silently
become figures about it.

WRITING THIS FOUND A FALSE LABEL

Three of the five manifests could not be used to re-fetch as they stood, and
one was wrong in a way that reached the screen:

  - CHAMP recorded the LOCAL filename. Zenodo serves the file under the
    paper's title, so the manifest named something that cannot be downloaded.
  - IMI-COMBINE pinned a commit and recorded no checksum, so a re-fetch could
    be verified only by trusting the ref.
  - B3DB said the local copy predated upstream v1.0.0 and that re-fetching
    "would change the row count by 175 and move every number measured from
    this set". Measured: the file is byte-identical at every ref that has
    ever held it, and a run was printing that claim to the reader as a
    provenance caveat. Contraindication 112.

ZINC22 IS DIFFERENT, AND SAYS SO

Its terms permit screening and the sharing of screening results and require
written permission to redistribute large portions, so it is fetched from the
source rather than shipped, and it is opt-in. No checksum was ever recorded
for any slice, so what is verified is the row count after ingest. That is
weaker -- it would not notice a slice whose contents changed without changing
length -- and the run says so rather than leaving it to be assumed.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# The four measured sets, in the order a reader meets them, then ZINC.
MEASURED = ("champ", "gramnegaccum", "imi_combine", "b3db")
OPTIONAL = ("zinc22",)

TIMEOUT = 180
CHUNK = 1 << 20
AGENT = "occam/py/fetch.py (one command for the declared data)"


def manifest(dataset):
    return json.loads((RAW / dataset / "manifest.json").read_text())


def declared(dataset):
    """The `fetch` block, or a stop: a dataset with no declaration is not
    something this guesses at."""
    block = manifest(dataset).get("fetch")
    if block is None:
        raise SystemExit(
            f"data/raw/{dataset}/manifest.json declares no `fetch` block, so "
            f"there is nothing here that says where its bytes come from. Add "
            f"one rather than adding a url to this file.")
    return block


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def already_there(entry):
    """True when the file is present and hashes to what was declared.

    A fetch that re-downloads what it already has is a fetch nobody runs
    twice. A file present with the wrong hash is not skipped: it is reported.
    """
    target = ROOT / entry["to"]
    if not target.exists():
        return False
    if "sha256" not in entry:
        return True
    return digest(target) == entry["sha256"]


def download(entry, say):
    target = ROOT / entry["to"]
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(entry["url"], headers={"User-Agent": AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response, \
                open(partial, "wb") as out:
            got = 0
            for block in iter(lambda: response.read(CHUNK), b""):
                out.write(block)
                got += len(block)
    except (urllib.error.URLError, OSError) as err:
        partial.unlink(missing_ok=True)
        return f"could not fetch {entry['url']}: {err}"

    if "sha256" in entry:
        found = digest(partial)
        if found != entry["sha256"]:
            partial.unlink(missing_ok=True)
            return (f"{entry['to']} does not hash to what the manifest "
                    f"declares.\n"
                    f"      declared {entry['sha256']}\n"
                    f"      fetched  {found}\n"
                    f"      {got:,} bytes from {entry['url']}\n"
                    f"      The pin has moved, or the source has. Nothing was "
                    f"ingested: every figure in HANDOFF.md was measured on the "
                    f"bytes that checksum describes, and data that hashes "
                    f"differently is a different set.")
    partial.replace(target)
    say(f"      {got:,} bytes" + ("  checksum verified" if "sha256" in entry
                                  else "  no checksum declared"))
    return None


def count_rows(entry, say):
    """What a slice with no checksum gets instead: its row count.

    Weaker than a checksum and said to be weaker. It would not notice a slice
    whose contents changed without changing length, and it is checked against
    `rows_expected`, which is what this project measured -- for the -N slice
    that is a truncation at 1,000,000, not the size of the slice.
    """
    expected = entry.get("rows_expected")
    if expected is None:
        say(f"      {entry['slice']}: no row count declared, nothing to check")
        return None
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return (f"cannot count rows in {entry['ingest_to']}: pyarrow is not "
                f"importable, and the row count is the only check this slice "
                f"gets")
    out = ROOT / entry["ingest_to"]
    parts = sorted(out.glob("part-*.parquet"))
    if not parts:
        return (f"{entry['ingest_to']} holds no part-*.parquet after ingest, "
                f"so nothing was written where {expected:,} rows were "
                f"expected")
    rows = sum(pq.read_metadata(part).num_rows for part in parts)
    if rows != expected:
        return (f"{entry['ingest_to']} holds {rows:,} rows and the manifest "
                f"declares {expected:,}.\n"
                f"      The slice has moved, or the ingest has. No checksum "
                f"exists for this data, so the row count is the whole check "
                f"and it did not pass.")
    say(f"      {entry['slice']}: {rows:,} rows, as declared "
        f"({len(parts)} part{'s' if len(parts) != 1 else ''})")
    return None


def check_slice_size(entry, say):
    """The slice's own size, counted from the bytes just downloaded.

    Separate from count_rows(), which checks what landed on disk. This checks
    how large the slice IS, which for -N is 4.6x what the ingest keeps, and
    it is the figure the screen needs to say "1,000,000 of its 4,572,322
    compounds read" rather than "1,000,000 compounds" -- contraindication 92.

    Counted here rather than copied, by the same method as the original
    measurement, so the declaration is verified by the fetch instead of being
    trusted by it. Cheap: 66 MB gzipped for the largest slice.
    """
    declared = entry.get("slice_rows")
    if declared is None:
        return None
    path = ROOT / entry["to"]
    lines = 0
    try:
        with gzip.open(path, "rb") as handle:
            for block in iter(lambda: handle.read(CHUNK), b""):
                lines += block.count(b"\n")
    except OSError as err:
        return f"cannot count the lines of {entry['to']}: {err}"
    if lines != declared:
        return (f"{entry['to']} holds {lines:,} lines and the manifest "
                f"declares a slice of {declared:,}.\n"
                f"      The slice has grown or shrunk upstream. Every figure "
                f"about this library's size, and the screen's own "
                f"'N of its M compounds read', rests on that number.")
    if lines != entry.get("rows_expected"):
        say(f"      {entry['slice']}: slice is {lines:,} lines, as declared, "
            f"so this disk holds {entry['rows_expected']:,} of it")
    else:
        say(f"      {entry['slice']}: slice is {lines:,} lines, as declared, "
            f"and all of it was read")
    return None


def ingest(dataset, block, say):
    """Run what the manifest says turns the bytes into what the tool reads."""
    for script in block.get("then", []):
        command = [sys.executable, script]
        # ZINC is one invocation per slice: the script takes the tranche as a
        # positional argument and writes one directory per slice.
        if dataset == "zinc22":
            for entry in block["files"]:
                per_slice = command + [entry["to"], "--out-dir", entry["ingest_to"]]
                if entry.get("limit"):
                    per_slice += ["--limit", str(entry["limit"])]
                say(f"    {' '.join(per_slice[1:])}")
                done = subprocess.run(per_slice, cwd=ROOT)
                if done.returncode != 0:
                    return f"{script} failed on slice {entry['slice']}"
                # THE ROW COUNT IS THE ONLY CHECK THESE SLICES GET
                #
                # No checksum was ever recorded for any ZINC slice, so the
                # manifest says the row count after ingest is what gets
                # verified, and --dry-run says it, and the docstring at the
                # top of this file says it. For an hour it was said in three
                # places and implemented in none: `rows_expected` appeared
                # nowhere in this module. Contraindication 105's shape --
                # `sample and say so instead`, with the saying so missing --
                # found an hour after writing contraindication 111 about
                # exactly this.
                problem = count_rows(entry, say)
                if problem:
                    return problem
                problem = check_slice_size(entry, say)
                if problem:
                    return problem
            continue
        say(f"    {script}")
        done = subprocess.run(command, cwd=ROOT)
        if done.returncode != 0:
            return f"{script} failed"
    return None


def plan(datasets, say):
    """What each dataset would do, with nothing fetched."""
    total = 0
    for dataset in datasets:
        block = declared(dataset)
        say(f"  {dataset}")
        if block.get("no_checksum"):
            say(f"    no checksum is declared for this one; "
                f"the row count after ingest is what gets checked")
        for entry in block["files"]:
            state = "already here" if already_there(entry) else "would fetch"
            size = entry.get("bytes")
            total += size or 0
            say(f"    {state:13} {entry['to']}"
                + (f"  ({size:,} bytes)" if size else ""))
        for script in block.get("then", []):
            say(f"    then          {script}")
        for skipped in block.get("not_fetched", []):
            say(f"    not fetched   {skipped}")
    if total:
        say(f"\n  {total:,} bytes where a size is declared")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zinc", action="store_true",
                        help="also fetch the ZINC22 slices (about 300 MB, and "
                             "no checksum is recorded for them)")
    parser.add_argument("--only", action="append", metavar="DATASET",
                        help="one dataset by name; repeatable")
    parser.add_argument("--dry-run", action="store_true",
                        help="say what would happen and stop")
    parser.add_argument("--no-ingest", action="store_true",
                        help="fetch the bytes and do not build the parquet")
    args = parser.parse_args()

    wanted = list(MEASURED) + (list(OPTIONAL) if args.zinc else [])
    if args.only:
        known = set(MEASURED) | set(OPTIONAL)
        unknown = [d for d in args.only if d not in known]
        if unknown:
            raise SystemExit(f"no such dataset: {', '.join(unknown)}. "
                             f"declared: {', '.join(sorted(known))}")
        wanted = args.only

    def say(line):
        print(line, flush=True)

    if args.dry_run:
        say("would fetch, and nothing else:")
        plan(wanted, say)
        return 0

    failures = []
    for dataset in wanted:
        block = declared(dataset)
        say(f"\n{dataset}")
        if block.get("no_checksum"):
            say("  note: no checksum is declared for this dataset. The row "
                "count after ingest is what gets checked, which would not "
                "notice contents that changed without changing length.")
        fetched = 0
        for entry in block["files"]:
            if already_there(entry):
                say(f"  have    {entry['to']}")
                continue
            say(f"  fetch   {entry['to']}")
            problem = download(entry, say)
            if problem:
                say(f"  STOPPED {problem}")
                failures.append(f"{dataset}: {entry['to']}")
                break
            fetched += 1
        else:
            if args.no_ingest:
                continue
            say(f"  ingest")
            problem = ingest(dataset, block, say)
            if problem:
                say(f"  STOPPED {problem}")
                failures.append(f"{dataset}: {problem}")

    say("")
    if failures:
        say("not complete:")
        for one in failures:
            say(f"  {one}")
        say("\nNothing was ingested for those, which is the point: a figure "
            "measured on one set of bytes is not a figure about another.")
        return 1
    say("the declared data is in place. `./occam --help` for what to ask it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
