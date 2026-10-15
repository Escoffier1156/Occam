"""Evaluates many screening specifications in one pass over the library.

This is the step that makes a proposal possible. Judging whether a different
specification would have been better means measuring it, and measuring twenty
specifications the obvious way means twenty passes over a million molecules --
twenty times twenty-eight minutes. Here the molecule is read once, its
properties computed once, and every variant is tested against those properties
while it is in hand.

What makes that affordable is the same constraint the engine is built around:
the state held per variant is fixed. One counter, and a bounded sketch for
counting distinct scaffolds.

THE SCAFFOLD PROBLEM, AND WHY A SKETCH

Diversity is the reason a filter can be too strict: 254 survivors cannot fill a
request for 256 distinct scaffolds. So each variant needs a distinct count --
and an exact distinct count means holding a set of up to a million strings per
variant, which is precisely the dependence on data volume this project refuses.

HyperLogLog gives the count from a fixed array of registers. With 2^12
registers the standard error is about 1.6%, and the memory is 4 KiB per
variant whatever the library's size. The estimate is checked against an exact
count on a run small enough to hold one, so the error is measured here rather
than quoted.

PER-VARIANT STATE

  survivors        8 bytes
  scaffold sketch  4096 bytes
  weight/rotatable running mean and variance, 48 bytes

Twenty variants is therefore about 82 KiB, and a thousand would be 4 MiB.
Neither depends on the number of molecules.
"""

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import knobs as knobs_mod  # noqa: E402
import smiles as smiles_mod  # noqa: E402

from rdkit import RDLogger  # noqa: E402
from rdkit.Chem import Descriptors, rdMolDescriptors  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

RDLogger.DisableLog("rdApp.*")

WORK = ROOT / "data" / "work"
HLL_P = 12
HLL_M = 1 << HLL_P
# Bias correction for m registers, from the standard construction.
HLL_ALPHA = 0.7213 / (1.0 + 1.079 / HLL_M)


class Sketch:
    """HyperLogLog: distinct count from a fixed array of registers."""

    __slots__ = ("registers",)

    def __init__(self):
        self.registers = bytearray(HLL_M)

    def add(self, text):
        if not text:
            return
        digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value >> (64 - HLL_P)
        rest = (value << HLL_P) & ((1 << 64) - 1)
        rank = 1 if rest == 0 else 64 - rest.bit_length() + 1
        if rank > self.registers[index]:
            self.registers[index] = rank

    def estimate(self):
        raw = HLL_ALPHA * HLL_M * HLL_M / sum(2.0 ** -r for r in self.registers)
        zeros = self.registers.count(0)
        # Linear counting is the accurate estimator while registers are empty.
        if raw <= 2.5 * HLL_M and zeros:
            return HLL_M * (len(self.registers) and 1) * __import__("math").log(HLL_M / zeros)
        return raw

    def nbytes(self):
        return len(self.registers)


class Running:
    """Mean and variance without keeping the values (Welford)."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def sd(self):
        return (self.m2 / (self.n - 1)) ** 0.5 if self.n > 1 else 0.0


class Variant:
    """One specification: a setting per knob, plus its accumulators."""

    def __init__(self, choices):
        self.choices = choices
        self.survivors = 0
        self.scaffolds = Sketch()
        self.weight = Running()
        self.rotatable = Running()
        self.nitrogen = knobs_mod.setting("nitrogen", choices["nitrogen"])
        self.rot_limit = knobs_mod.setting("rotatable", choices["rotatable"])["limit"]
        self.mw_limit = knobs_mod.setting("weight", choices["weight"])["limit"]

    @property
    def name(self):
        # The registry's order, not a tuple written out here. A knob retired
        # from the registry used to leave four of these behind.
        return " + ".join(self.choices[k] for k in knobs_mod.ORDER
                          if k in self.choices)

    def provenance(self):
        kinds = {k: knobs_mod.setting(k, v)["provenance"] for k, v in self.choices.items()}
        worst = ("unjustified" if "unjustified" in kinds.values()
                 else "relaxation" if "relaxation" in kinds.values()
                 else "published")
        return {"per_knob": kinds, "weakest": worst}

    def distance_from(self, baseline):
        """How many steps looser than a baseline specification, per knob."""
        return {k: knobs_mod.strictness(k, self.choices[k])
                   - knobs_mod.strictness(k, baseline[k])
                for k in self.choices}

    def consider(self, mol, weight, rotatable, scaffold, nitrogen_hits):
        """`scaffold` may be a callable, so only a survivor pays for it."""
        if weight >= self.mw_limit:
            return
        if rotatable > self.rot_limit:
            return
        if not nitrogen_hits[self.choices["nitrogen"]]:
            return
        self.survivors += 1
        self.scaffolds.add(scaffold() if callable(scaffold) else scaffold)
        self.weight.add(weight)
        self.rotatable.add(rotatable)

    def report(self, evaluated):
        return {
            "specification": self.name,
            "choices": self.choices,
            "survivors": self.survivors,
            "share": self.survivors / evaluated if evaluated else 0.0,
            "distinct_scaffolds_estimated": int(round(self.scaffolds.estimate()))
                                            if self.survivors else 0,
            "mw_mean": round(self.weight.mean, 1),
            "mw_sd": round(self.weight.sd(), 1),
            "rotatable_mean": round(self.rotatable.mean, 2),
            "provenance": self.provenance(),
        }


def enumerate_variants(baseline, only_looser=True):
    """Every combination of settings, optionally no stricter than the baseline."""
    order = knobs_mod.ORDER
    options = [[s["name"] for s in knobs_mod.settings_of(k)] for k in order]
    out = []
    for combination in itertools.product(*options):
        choices = dict(zip(order, combination))
        if only_looser and any(
            knobs_mod.strictness(k, choices[k]) < knobs_mod.strictness(k, baseline[k])
            for k in order
        ):
            continue
        out.append(Variant(choices))
    return out


def library_parts(part_dir):
    """The parquet files a library is held in, in order.

    TWO SHAPES, AND ONLY ONE WAS READ
    --------------------------------------------------------------------
    This globbed `part-*.parquet` inside a directory, which is how the ZINC
    slices are written. The other four libraries the request vocabulary names
    -- CHAMP, IMI, GramNegAccum, B3DB -- are single parquet files, and
    globbing inside a FILE path matches nothing and raises nothing. A request
    for CHAMP read zero molecules, finished normally, and reported that no
    setting of any knob could fill it.

    The glob returning nothing is not the same as a library being empty, and
    neither of them is an answer. run() refuses to produce a verdict from
    either; this function's job is only to say which files to read.
    """
    path = Path(part_dir)
    if path.is_file():
        return [path]
    return sorted(path.glob("part-*.parquet"))


def smiles_column(path):
    """Whichever spelling this file uses. B3DB writes SMILES, the rest smiles."""
    names = pq.read_schema(path).names
    for candidate in ("smiles", "SMILES"):
        if candidate in names:
            return candidate
    raise ValueError(
        f"{path} has no SMILES column; its columns are {names}. A library "
        f"this cannot read is a library this cannot screen.")


def iter_library(part_dir, limit=None):
    seen = 0
    for path in library_parts(part_dir):
        column = smiles_column(path)
        table = pq.read_table(path, columns=[column])
        values = table.column(0).to_pylist()
        del table
        for text in values:
            yield text
            seen += 1
            if limit and seen >= limit:
                return


def run(part_dir, variants, limit=None, exact_check=False):
    """One pass. Every variant is tested against each molecule's properties."""
    nitrogen_settings = {s["name"]: s for s in knobs_mod.settings_of("nitrogen")}
    evaluated = unreadable = 0
    exact = set() if exact_check else None
    started = time.monotonic()

    for text in iter_library(part_dir, limit):
        mol = smiles_mod.parse(text)
        if mol is None:
            unreadable += 1
            continue
        evaluated += 1
        weight = Descriptors.MolWt(mol)
        rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
        # Computed once per molecule, reused by every variant. This is the
        # saving: K specifications, one RDKit pass.
        hits = {name: bool(body["test"](mol)) for name, body in nitrogen_settings.items()}

        # Profiled on the -P slice, 17,775 molecules, 2026-10-16:
        #
        #     smiles.parse            10.28 s
        #     MurckoScaffoldSmiles     5.24 s   27.8% of the total
        #     CalcNumRotatableBonds    2.36 s
        #     nitrogen tests (3)       0.86 s
        #     MolWt                    0.07 s
        #                             ------
        #                             18.81 s
        #
        # The figure had been carried as "27.6%" with no record behind it,
        # and re-measuring put it at 27.8%. It was right; it just could not
        # be checked, which is the same position as being wrong until someone
        # measures.
        # Only a survivor needs one, and survivors were 2.5% of the slice
        # measured, so computing it for every molecule spent a quarter of the
        # run on rows no specification kept. Deferred to the first variant
        # that asks for it.
        holder = []

        def scaffold_of():
            if not holder:
                try:
                    holder.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "")
                except ValueError:
                    holder.append("")
            return holder[0]

        for variant in variants:
            variant.consider(mol, weight, rotatable, scaffold_of, hits)
        if exact is not None and holder:
            exact.add(holder[0])

    # A SWEEP THAT READ NOTHING HAS NOT MEASURED ANYTHING
    #
    # Zero molecules produced a full set of verdicts: "no setting of these
    # knobs can fill the request", "the best reachable is 0 distinct
    # scaffolds", and -- worst -- two knobs reported as having "filtered
    # nothing on this library, in all 3 comparisons", which is a claim to have
    # compared something. Every pair of empty counts is equal, so the check
    # that exists to stop a knob being reported as a constraint it never
    # applied became a claim that measurements were taken.
    #
    # The whole product is the difference between "the answer is no" and "this
    # cannot be answered". An empty population supports neither. It stops
    # here, where the population is known, rather than at any of the places
    # downstream that would each have to remember to ask.
    if evaluated == 0:
        parts = library_parts(part_dir)
        raise ValueError(
            f"read 0 molecules from {part_dir}: "
            + (f"no parquet file was found there"
               if not parts else
               f"{len(parts)} file(s) were read and held no rows")
            + ". A verdict needs a population to be about; this one would be "
              "about nothing, and would say so in the same words it uses for "
              "a library that was read and did not reach.")

    seconds = time.monotonic() - started
    state = (8 + variants[0].scaffolds.nbytes() + 48) * len(variants) if variants else 0
    return {
        "library": str(part_dir),
        # Recorded so that a later pass cannot be run over a different amount
        # of the library and have its numbers put in the same row as these.
        "limit": limit,
        "evaluated": evaluated,
        "unreadable": unreadable,
        "variants": len(variants),
        "seconds": round(seconds, 1),
        "passes_over_the_library": 1,
        "passes_if_run_separately": len(variants),
        "state_bytes_total": state,
        "state_bytes_per_variant": state // len(variants) if variants else 0,
        "state_depends_on_rows": False,
        "exact_distinct_scaffolds": len(exact) if exact is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", type=Path, default=WORK / "zinc")
    parser.add_argument("--baseline", nargs=4,
                        default=["primary_amine", "rotatable_le_5",
                                 "mw_under_450", "neutral_only"],
                        metavar=("NITROGEN", "ROTATABLE", "WEIGHT", "CHARGE"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--all-combinations", action="store_true",
                        help="include specifications stricter than the baseline too")
    parser.add_argument("--exact-check", action="store_true",
                        help="also hold an exact scaffold set, to measure the "
                             "sketch's error. Only for runs small enough to afford it.")
    parser.add_argument("--out", type=Path, default=WORK / "sweep.json")
    args = parser.parse_args()

    baseline = dict(zip(knobs_mod.ORDER, args.baseline))
    variants = enumerate_variants(baseline, only_looser=not args.all_combinations)
    print(f"baseline: {' + '.join(args.baseline)}")
    print(f"{len(variants)} specifications, one pass\n")

    summary = run(args.parts, variants, args.limit or None, args.exact_check)
    rows = [v.report(summary["evaluated"]) for v in variants]
    for row in rows:
        row["steps_looser_than_baseline"] = variants[rows.index(row)].distance_from(baseline)
    rows.sort(key=lambda r: -r["survivors"])

    print(f"{'specification':70} {'survivors':>10} {'share':>7} "
          f"{'scaffolds~':>11}  provenance")
    for row in rows:
        print(f"{row['specification']:70} {row['survivors']:>10,} "
              f"{row['share']:>6.2%} {row['distinct_scaffolds_estimated']:>11,}  "
              f"{row['provenance']['weakest']}")

    if summary["exact_distinct_scaffolds"] is not None:
        est = next((r["distinct_scaffolds_estimated"] for r in rows
                    if r["choices"] == baseline), None)
        print(f"\nsketch check: exact distinct scaffolds over the whole sample = "
              f"{summary['exact_distinct_scaffolds']:,}")

    print(f"\none pass instead of {summary['passes_if_run_separately']}: "
          f"{summary['seconds']}s")
    print(f"state held: {summary['state_bytes_total']:,} B total, "
          f"{summary['state_bytes_per_variant']:,} B per specification, "
          f"independent of the row count")

    (args.out).write_text(json.dumps({"baseline": baseline, "summary": summary,
                                      "specifications": rows}, indent=2) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()


def eliminating_conditions(part_dir, choices, limit=None):
    """When nothing survives, which condition removed everything?

    WHY THIS EXISTS, AND WHY IT RUNS ONLY ON A ZERO
    --------------------------------------------------------------------
    inert_knobs() says which conditions filtered NOTHING, so a reader is not
    told a constraint was applied when it did nothing. Its opposite was never
    said. A request for CHAMP with a strict primary amine returned "no setting
    of these knobs can fill the request" and "the best reachable is 0" -- true,
    and silent about the fact that the library holds no primary amine at all
    while the weight and flexibility bounds removed four molecules out of
    1,132.

    The sweep cannot answer this on its own: a condition held as asked is
    never varied, so its effect never appears as a difference between two
    specifications. Measuring it needs each condition applied ALONE, which is
    one more pass over the library. That is why this runs only when the sweep
    has already returned zero everywhere -- at that point one extra pass buys
    the only thing a reader wants, and there is no answer to spend it on.

    Returns one row per condition with how many molecules it keeps by itself,
    in the order the registry lists them.
    """
    import knobs as knobs_mod

    variant = Variant(choices)
    nitrogen = {s["name"]: s for s in knobs_mod.settings_of("nitrogen")}
    counts = {"nitrogen": 0, "rotatable": 0, "weight": 0}
    read = 0
    for text in iter_library(part_dir, limit):
        mol = smiles_mod.parse(text)
        if mol is None:
            continue
        read += 1
        if nitrogen[choices["nitrogen"]]["test"](mol):
            counts["nitrogen"] += 1
        if rdMolDescriptors.CalcNumRotatableBonds(mol) <= variant.rot_limit:
            counts["rotatable"] += 1
        if Descriptors.MolWt(mol) < variant.mw_limit:
            counts["weight"] += 1
    return {
        "read": read,
        "alone": [{"knob": knob,
                   "setting": choices[knob],
                   "keeps": counts[knob],
                   "share": round(counts[knob] / read, 4) if read else None}
                  for knob in knobs_mod.ORDER if knob in counts],
        "measured": "each condition applied by itself, over one pass of the "
                    "library. A condition keeping 0 on its own removes "
                    "everything whatever the others do.",
    }
