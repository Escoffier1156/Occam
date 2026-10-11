"""Fetch ChEMBL activities for one source into data/raw/.

The ChEMBL REST API returns HTTP 500 on about five requests in six, regardless
of the query. The failures are not rate limiting and not parameter-dependent:
the same URL succeeds on a later try, and successes arrive in bursts. So every
request is retried, each assay is written as it completes, and a rerun skips
assays already on disk. Interrupting and restarting is safe.

Pulling the whole 5.5 GB database dump would be the other way to get this, but
one source is about 0.5% of ChEMBL.

    nix-shell --run 'python3 py/fetch_chembl.py --src-id 40'

Output: data/raw/chembl_src<N>/<assay>.jsonl, one activity per line, plus
manifest.json recording what was fetched and when.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
BASE = "https://www.ebi.ac.uk/chembl/api/data"
PAGE = 1000
MAX_TRIES = 120
# The 500s are random, not rate limiting: successes come in bursts, and about
# one request in six succeeds. Growing the delay only lengthens the wait for
# the next good window, so the retry delay stays short and constant. A failing
# request already takes the server ~8s to answer, which is the real pacing.
RETRY_DELAY = 2.0
# Courtesy gap between successful requests.
GAP = 0.5
USER_AGENT = "occam-research/0.1 (+https://github.com/; ChEMBL REST client)"


def get_json(url, tries=MAX_TRIES):
    """GETs a URL, retrying the API's intermittent 500s."""
    last = None
    for attempt in range(1, tries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            code = getattr(e, "code", None)
            if code in (400, 404):
                raise
            if attempt < tries:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"{url}: giving up after {tries} tries ({last})")


def list_assays(src_id):
    """Every assay of one source, with its strain and description."""
    assays, offset = [], 0
    while True:
        url = f"{BASE}/assay.json?src_id={src_id}&limit={PAGE}&offset={offset}"
        data = get_json(url)
        assays += data["assays"]
        total = data["page_meta"]["total_count"]
        offset += PAGE
        if offset >= total:
            return assays, total


def fetch_assay(assay_id, out_path, expected=None):
    """Writes one assay's activities as JSON lines; returns the row count."""
    if out_path.exists():
        with out_path.open() as fh:
            have = sum(1 for _ in fh)
        if expected is None or have == expected:
            return have, True
        # A partial file from an interrupted run: start it over.
        out_path.unlink()

    tmp = out_path.with_suffix(".tmp")
    rows, offset, total = 0, 0, None
    with tmp.open("w") as fh:
        while total is None or offset < total:
            url = f"{BASE}/activity.json?assay_chembl_id={assay_id}&limit={PAGE}&offset={offset}"
            data = get_json(url)
            total = data["page_meta"]["total_count"]
            for activity in data["activities"]:
                fh.write(json.dumps(activity, separators=(",", ":")) + "\n")
                rows += 1
            offset += PAGE
            print(f"    {rows}/{total}", end="\r", file=sys.stderr, flush=True)
            time.sleep(GAP)
    tmp.replace(out_path)
    return rows, False


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src-id", type=int, required=True, help="ChEMBL source id (CO-ADD is 40)")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--assays", nargs="+", default=None,
        help="only these assay_chembl_ids, instead of every assay of the source",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or RAW_DIR / f"chembl_src{args.src_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"listing assays for src_id={args.src_id} ...", file=sys.stderr)
    assays, total_assays = list_assays(args.src_id)
    if args.assays:
        wanted = set(args.assays)
        assays = [a for a in assays if a["assay_chembl_id"] in wanted]
    print(f"{len(assays)} assays ({total_assays} in source)", file=sys.stderr)

    manifest = {
        "source": f"{BASE} src_id={args.src_id}",
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "license": "CC BY-SA 3.0; cite Mendez et al. Nucleic Acids Res 2019 47(D1):D930-D940",
        "assays": [],
    }
    grand_total = 0
    for i, assay in enumerate(assays, 1):
        assay_id = assay["assay_chembl_id"]
        path = out_dir / f"{assay_id}.jsonl"
        print(f"[{i}/{len(assays)}] {assay_id} {assay.get('assay_strain')}", file=sys.stderr)
        rows, cached = fetch_assay(assay_id, path)
        grand_total += rows
        print(f"    {rows} rows{' (cached)' if cached else ''}        ", file=sys.stderr)
        manifest["assays"].append(
            {
                "assay_chembl_id": assay_id,
                "strain": assay.get("assay_strain"),
                "organism": assay.get("assay_organism"),
                "description": assay.get("description"),
                "rows": rows,
                "file": path.name,
            }
        )

    manifest["total_rows"] = grand_total
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n{grand_total} activities -> {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
