"""SMILES -> descriptor columns and a Bemis-Murcko scaffold, computed once.

Writes <out>/features.parquet with the nine descriptors, the binary label and
the scaffold each molecule belongs to. Splitting is py/folds.py's job: the
descriptors do not depend on the split, so computing them once and reusing
them across every fold and repeat avoids running RDKit 25 times over the same
molecules.

Work this module does that the engine must not: it reads SMILES and computes
descriptors over the whole file. The memory invariants constrain the engine,
not this preprocessing. Even so, nothing here holds the dataset: molecules are
streamed in batches and rows are written out in fixed-size chunks.
"""

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import smiles as smiles_mod  # noqa: E402

WORK_DIR = ROOT / "data" / "work"
PART_ROWS = 8192

# Nine descriptors, fixed at this order: engine/src/carry.rs mirrors it as F.
FEATURES = ("mw", "logp", "tpsa", "hbd", "hba", "rotb", "aromatic_rings", "basic_nitrogen", "fsp3")

# A nitrogen that is protonated around pH 7.4: not amide, aromatic, nitro,
# nitrile or attached to a heteroatom. Formal charge was tried first and
# turned out to be identically 0 across CHAMP, carrying no signal at all.
# eNTRy's own primary-amine pattern is also 0 across CHAMP, so this broader
# pattern is the closest usable stand-in for "ionisable nitrogen".
BASIC_NITROGEN = Chem.MolFromSmarts(
    "[$([NX3;H0,H1,H2;!$(N[!#6]);!$(N*=[O,N,S]);!$(N#*);!$([N][a])]),$([NX4+;!$(N*=[O,N,S])])]"
)
SPLITS = ("train", "valid", "test")
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
# Above this share of unparseable SMILES the pipeline is suspect, not the data.
MAX_PARSE_FAILURE_RATE = 0.01
# Same limit for molecules left holding unpaired electrons. This is the half of
# a dialect mismatch that raises no error: `[N+]` reads as a zero-hydrogen
# cation, misses the primary-amine pattern, and only shows up as a descriptor
# that is quietly wrong. Measured with py/smiles.py in place: CHAMP, IMI and
# GramNegAccum sit at 0.000% and B3DB at 0.013%, against 38.2% for
# GramNegAccum parsed without it. 1% separates the two cases with room to spare.
MAX_RADICAL_RATE = 0.01

RDLogger.DisableLog("rdApp.*")


def descriptors(mol):
    """The nine features, in FEATURES order."""
    return (
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        float(rdMolDescriptors.CalcNumHBD(mol)),
        float(rdMolDescriptors.CalcNumHBA(mol)),
        float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        float(len(mol.GetSubstructMatches(BASIC_NITROGEN))),
        rdMolDescriptors.CalcFractionCSP3(mol),
    )


def scaffold_of(mol):
    """Bemis-Murcko scaffold. Molecules without a ring scaffold share ""."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except ValueError:
        return ""


def iter_records(path, smiles_column, label_column, batch_rows=PART_ROWS, keep=None, notes=None):
    """Streams (scaffold, features, label) for every parseable row.

    `keep` is an optional set of label values; rows with any other label are
    skipped as out of scope, separately from rows dropped as unusable. The
    middle tertile of a continuous measurement is excluded this way, so it is
    never forced to a side the measurement does not support.

    `notes` is an optional Counter for rows that were kept but needed work --
    a repaired hydrogen count, a rewritten bracket, a leftover radical. It is
    separate from `dropped` because nothing in it was dropped, and because the
    totals `dropped` feeds would be wrong if the two were mixed.

    Rows are yielded in file order. Unparseable SMILES and unusable labels are
    skipped and counted; the counter is the second element of the final tuple.
    """
    dropped = Counter()
    if notes is None:
        notes = Counter()
    source = pq.ParquetFile(path)
    for batch in source.iter_batches(batch_size=batch_rows, columns=[smiles_column, label_column]):
        smiles_list = batch.column(0).to_pylist()
        labels = batch.column(1).to_pylist()
        for smiles, label in zip(smiles_list, labels):
            if keep is not None and label not in keep:
                dropped["out_of_scope"] += 1
                continue
            if smiles is None:
                dropped["smiles_missing"] += 1
                continue
            mol, repaired, reordered = smiles_mod.parse_report(smiles)
            if mol is None:
                dropped["smiles_unparseable"] += 1
                continue
            if repaired:
                notes["hydrogens_recounted"] += 1
            if reordered:
                notes["brackets_reordered"] += 1
            if smiles_mod.radical_atoms(mol):
                notes["radical_remaining"] += 1
            if label is None:
                dropped["label_missing"] += 1
                continue
            try:
                values = descriptors(mol)
            except (ValueError, RuntimeError):
                dropped["descriptor_error"] += 1
                continue
            if any(math.isnan(v) or math.isinf(v) for v in values):
                dropped["descriptor_not_finite"] += 1
                continue
            yield scaffold_of(mol), values, label, smiles
    yield None, dropped, None, None


def run(src, out_dir, smiles_column, label_column, positive_label, keep=None):
    """Computes descriptors for every usable row; returns (rows, dropped)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "features.parquet"

    schema = pa.schema(
        [(name, pa.float32()) for name in FEATURES]
        + [("label", pa.int8()), ("scaffold", pa.string()), ("smiles", pa.string())]
    )
    columns = [[] for _ in schema]
    rows = 0
    writer = pq.ParquetWriter(out, schema)

    def flush():
        nonlocal columns
        if columns[0]:
            writer.write_table(pa.table(columns, schema=schema))
            columns = [[] for _ in schema]

    label_counts = Counter()
    notes = Counter()
    for scaffold, values, label, smiles in iter_records(
        src, smiles_column, label_column, keep=keep, notes=notes
    ):
        if scaffold is None:
            dropped = values
            break
        y = 1 if label == positive_label else 0
        for col, v in zip(columns, values):
            col.append(v)
        columns[-3].append(y)
        columns[-2].append(scaffold)
        columns[-1].append(smiles)
        label_counts[y] += 1
        rows += 1
        if len(columns[0]) >= PART_ROWS:
            flush()
    flush()
    writer.close()

    # [LOCK: ast name="parse_failure_limit"]
    # Stops rather than continues when most of the input will not parse. This
    # fired on GramNegAccum at 45.5% and the cause was a notation dialect, not
    # broken chemistry; without it the dataset would have silently shrunk.
    failures = dropped["smiles_unparseable"] + dropped["smiles_missing"]
    total = rows + sum(v for k, v in dropped.items() if k != "out_of_scope")
    if total and failures / total > MAX_PARSE_FAILURE_RATE:
        sys.exit(
            f"{src}: {failures}/{total} SMILES unparseable ({failures/total:.2%}), "
            f"above the {MAX_PARSE_FAILURE_RATE:.0%} limit. The descriptor pipeline is "
            f"more likely at fault than the data; stopping instead of continuing."
        )
    # [/LOCK]

    # The quiet half of the same problem. A dialect that parses but drops
    # hydrogen counts leaves radicals behind and nothing else complains, so
    # this is checked on the rows that were kept rather than on the failures.
    if rows and notes["radical_remaining"] / rows > MAX_RADICAL_RATE:
        sys.exit(
            f"{src}: {notes['radical_remaining']}/{rows} molecules still carry unpaired "
            f"electrons ({notes['radical_remaining']/rows:.2%}), above the "
            f"{MAX_RADICAL_RATE:.0%} limit. These parse without error but their "
            f"descriptors are wrong; py/smiles.py does not understand this dialect yet."
        )

    summary = {
        "source": str(src),
        "features": list(FEATURES),
        "rows": rows,
        "repairs": dict(notes),
        "positive_label": positive_label,
        "labels": {"positive": label_counts[1], "negative": label_counts[0]},
        "distinct_scaffolds": None,
        "dropped": dict(dropped),
    }
    table = pq.read_table(out, columns=["scaffold"])
    summary["distinct_scaffolds"] = len(set(table.column(0).to_pylist()))
    del table
    (out_dir / "features.json").write_text(json.dumps(summary, indent=2) + "\n")
    return rows, dropped, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="parquet file produced by ingest")
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--positive-label", required=True, help="label value counted as class 1")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--keep-labels", nargs="+", default=None, metavar="VALUE",
        help="only these label values; others are skipped (e.g. high low, dropping mid)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or WORK_DIR / f"{args.source.stem}_desc"
    rows, dropped, summary = run(
        args.source, out_dir, args.smiles_column, args.label_column, args.positive_label,
        set(args.keep_labels) if args.keep_labels else None,
    )
    reasons = " ".join(f"{k}={v}" for k, v in sorted(dropped.items())) or "none"
    print(f"{out_dir}/features.parquet\trows={rows}\tdropped={sum(dropped.values())} ({reasons})")
    print(f"  labels: {summary['labels']}  scaffolds: {summary['distinct_scaffolds']}")
    repairs = " ".join(f"{k}={v}" for k, v in sorted(summary["repairs"].items())) or "none"
    print(f"  repairs: {repairs}")


if __name__ == "__main__":
    main()
