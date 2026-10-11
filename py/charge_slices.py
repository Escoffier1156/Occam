"""Formal charge as drawn, per ZINC22 charge slice. The measurement behind the
retirement of the charge knob.

    python3 py/charge_slices.py --out data/work/charge_slices.json

ZINC22 splits each tranche by the protonation state it PREDICTS at pH 7.4 and
records that in the filename suffix: -N neutral, -O +1, -M -1, -P +2 or more.
The retired charge knob read something else -- the formal charge of the
structure as it is written in the SMILES. This script measures the second
quantity on each slice of the first, which is how the two were shown to be
unrelated.

The criterion was declared before the first run: a setting that keeps only
formal charge 0 is inert on a slice if at least 99.9% of that slice is drawn
with formal charge 0, i.e. applying it removes at most one molecule in a
thousand. The result did not meet it on -O, so the knob was not retired for
being inert. It was retired because the -M sample -- the slice whose molecules
are predicted to carry -1 -- held no molecule drawn with a negative charge at
all, so the knob cannot reach the population it appears to name.

WHAT A COUNT OF ZERO HERE DOES NOT MEAN

Three slices came back with no non-zero charge in the sample. That is an
observation of none in 20,000, which bounds the share at roughly 0.015% at 95%
and no lower. It is not a demonstration that none exists, and the record says
so in the same breath as the number.
"""

import argparse
import glob
import gzip
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "work"

SAMPLE = 20000
SEED = 0
# Declared before the first run, not fitted to it.
INERT_AT = 0.999

# The ingested slices of zinc-22x H29P240. A slice that is not on this disk is
# reported as not measured rather than skipped, because "we did not look" and
# "there was nothing" are the two readings this whole module exists to keep
# apart.
SLICE_MEANS = {
    "-N": "predicted neutral at pH 7.4",
    "-O": "predicted +1 at pH 7.4",
    "-M": "predicted -1 at pH 7.4",
    "-P": "predicted +2 or more at pH 7.4",
}

SLICES = [
    {"slice": "-N", "kind": "parquet", "path": WORK / "zinc"},
    {"slice": "-O", "kind": "parquet", "path": WORK / "zinc_O"},
    {"slice": "-P", "kind": "parquet", "path": WORK / "zinc_P"},
]


def _charges(smiles):
    counts, unparsed = Counter(), 0
    for text in smiles:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            unparsed += 1
            continue
        counts[Chem.GetFormalCharge(mol)] += 1
    return counts, unparsed


def _from_parquet(part_dir, sample):
    """A seeded sample. Parts are read until there is a pool to draw from, so
    the pool is a prefix of the library and the record says so."""
    pool = []
    for path in sorted(glob.glob(f"{part_dir}/part-*.parquet")):
        pool.extend(pq.read_table(path, columns=["smiles"]).column(0).to_pylist())
        if len(pool) >= sample * 4:
            break
    random.Random(SEED).shuffle(pool)
    return pool[:sample], len(pool)


def _from_smi_gz(path, sample):
    pool = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            field = line.split()
            if field and not field[0].lower().startswith("smiles"):
                pool.append(field[0])
            if len(pool) >= sample * 4:
                break
    random.Random(SEED).shuffle(pool)
    return pool[:sample], len(pool)


def measure(entry, sample=SAMPLE):
    path = Path(entry["path"])
    means = SLICE_MEANS.get(entry["slice"], "slice letter not recognised")
    if not path.exists():
        return {"slice": entry["slice"], "means": means, "measured": False,
                "why": f"{path} is not on this disk"}
    smiles, pool = (_from_parquet(path, sample) if entry["kind"] == "parquet"
                    else _from_smi_gz(path, sample))
    counts, unparsed = _charges(smiles)
    total = sum(counts.values())
    zero = counts.get(0, 0)
    share = zero / total if total else 0.0
    return {
        "slice": entry["slice"],
        "means": means,
        "measured": True,
        "source": str(path),
        "partial_download": bool(entry.get("partial")),
        "pool": pool,
        "sampled": total,
        "unparsed": unparsed,
        "formal_charge_zero": zero,
        "share_zero": round(share, 6),
        "other_charges": {f"{k:+d}": v for k, v in sorted(counts.items())
                          if k != 0},
        "meets_inert_criterion": share >= INERT_AT,
    }


def report(extra=None, sample=SAMPLE):
    rows = [measure(entry, sample) for entry in SLICES + list(extra or [])]
    done = [r for r in rows if r["measured"]]
    return {
        "what_this_is": "a measurement of these samples, not a property of "
                        "ZINC22",
        "tranche": "zinc-22x H29P240",
        "sample": sample,
        "seed": SEED,
        "inert_criterion": INERT_AT,
        "criterion_declared": "before the first run, 2026-10-11",
        "slices": rows,
        "meets_inert_criterion_everywhere": bool(done) and all(
            r["meets_inert_criterion"] for r in done),
        "a_count_of_zero_means": "none was observed in the sample. With 20,000 "
                                 "drawn that bounds the share at roughly "
                                 "0.015% at 95% and no lower. It is not an "
                                 "absence.",
    }


def show(body):
    print(f"formal charge as drawn, {body['tranche']}, "
          f"sample {body['sample']:,}, seed {body['seed']}")
    print(f"  inert criterion, declared {body['criterion_declared']}: "
          f"at least {body['inert_criterion']:.1%} drawn with charge 0")
    print()
    for row in body["slices"]:
        if not row["measured"]:
            print(f"  {row['slice']}  not measured: {row['why']}")
            continue
        verdict = ("meets it" if row["meets_inert_criterion"]
                   else "DOES NOT meet it")
        print(f"  {row['slice']}  {row['means']}")
        print(f"      charge 0 in {row['formal_charge_zero']:,} of "
              f"{row['sampled']:,} = {row['share_zero']:.4%}  {verdict}")
        if row["other_charges"]:
            print(f"      drawn with a charge: {row['other_charges']}")
        if row["partial_download"]:
            print(f"      the pool is a PARTIAL download: {row['pool']:,} "
                  f"molecules, a prefix of the slice rather than all of it")
    print()
    print("a count of 0 is an observation of none in the sample, not an "
          "absence.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=WORK / "charge_slices.json")
    parser.add_argument("--sample", type=int, default=SAMPLE)
    parser.add_argument("--also", type=Path, action="append", default=[],
                        help="an extra .smi.gz slice to measure, e.g. a "
                             "partial download of -M")
    parser.add_argument("--also-slice", action="append", default=[],
                        help="the slice letter for each --also, in order")
    args = parser.parse_args()

    extra = []
    for index, path in enumerate(args.also):
        letter = (args.also_slice[index] if index < len(args.also_slice)
                  else "?")
        extra.append({"slice": letter, "kind": "smi", "path": path,
                      "partial": True})

    body = report(extra, args.sample)
    show(body)
    args.out.write_text(json.dumps(body, indent=2) + "\n")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
