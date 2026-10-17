"""
NOT THE PRODUCT.

This emits a list of molecules. The product is the assist: py/occam.py takes a
request and returns what a screen of it would and would not establish, and
emits nothing. Choosing which molecules to make is the researcher's.

It is kept, rather than deleted, because it produced the confirmed numbers in
section 4 of HANDOFF.md and because py/candidates.py carries a KeepOut
lock. Deleting it would make those numbers unverifiable.

It is guarded, rather than left alone, because on 2026-10-10 the assistant ran
it directly, twice, without being asked -- a candidate list looks like a
deliverable in a way that a verdict does not. Emitting molecules now needs
--emit-molecules, so reaching for it has to be deliberate.
Rank the library and emit candidates, with the grounds for each one.

Three filters run over the same pool and the same binding scores, so the
output shows what each way of narrowing gives up:

  A  amine + rotatable bonds <= 5   the two reproducible eNTRy conditions
  B  basic nitrogen + rotatable bonds <= 5   a looser stand-in
  C  no filter at all              binding score alone

A concentrates hardest and discards the most: on the evaluation pool it keeps
4.5% of the library at 2.79x the base hit rate, but 87% of the true hits are
in what it throws away. Tetracyclines, chloramphenicol and some
fluoroquinolones carry no primary amine and are removed as a class. Printing
all three lets the reader judge which cut is reasonable, which is the question
worth asking them.

No learned permeability model is involved. One was built and does transfer to
a neighbouring dataset, but its applicability domain does not reach a general
medicinal-chemistry library: 66% of this pool sits outside it. The filters
here are rules computed from the molecule, so they extrapolate nowhere and
apply unchanged to any library.

The row order of IMI-COMBINE is itself enriched -- its first 1,000 rows carry
2.27x the overall hit rate -- and a ranking that cannot separate its rows
falls back on that order. Every run therefore repeats itself over a shuffled
pool and reports how much of the top N survives; anything short of a near
complete overlap means the ranking is reading file position.
"""

import argparse
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import folds as folds_mod  # noqa: E402
import smiles as smiles_mod  # noqa: E402
from m3_order import assert_ranking_is_usable, sh, stats_from, write_parts  # noqa: E402

ENGINE_FP = ROOT / "engine" / "target-fp" / "release" / "engine"
WORK = ROOT / "data" / "work"
RDLogger.DisableLog("rdApp.*")

# The rule's own pattern, from Richter et al., Nature 2017 (n = 68).
ENTRY_AMINE = Chem.MolFromSmarts("[$([N;H2;X3][CX4]),$([N;H3;X4+][CX4])]")
# Broader: any nitrogen that is basic around pH 7.4.
BASIC_N = Chem.MolFromSmarts(
    "[$([NX3;H0,H1,H2;!$(N[!#6]);!$(N*=[O,N,S]);!$(N#*);!$([N][a])]),$([NX4+;!$(N*=[O,N,S])])]"
)
MAX_ROTATABLE = 5

# Added here, not part of the published rule.
#
# Richter et al. set no upper bound on size, because the 68 molecules behind
# the rule are small and none of them is a glycopeptide. Applied without one,
# the two conditions admit a teicoplanin analogue of 1,281 Da -- rigid enough
# to pass the rotatable-bond count and carrying an amine, but far too large to
# cross an outer membrane. A rule has an applicability domain just as a fitted
# model does, and this is the edge of it.
#
# The bound comes from the library: the third quartile of molecular weight
# among true hits is 568, and 66.7% of them fall between 344 and 600. The
# 21.6% above it are macrolides and glycopeptides, which do not enter through
# porins at all, so a porin-oriented filter should not be ranking them.
MAX_MOLECULAR_WEIGHT = 600.0

# Not three answers to one question. A primary amine splits the library into
# two populations that enter cells by different routes, so the groups name
# which route a candidate belongs to.
FILTERS = {
    "A": ("porin route: amine + rotb<=5 + MW<=600",
          lambda p: p["amine"] and p["rotb_ok"] and p["mw_ok"]),
    "B": ("intermediate: basic N + rotb<=5 + MW<=600",
          lambda p: p["basic_n"] and p["rotb_ok"] and p["mw_ok"]),
    "C": ("unfiltered: binding score alone, any size or route",
          lambda p: True),
}


def inchikeys(mol):
    """Full key and its first block.

    The first block encodes the skeleton alone: stereoisomers, salts and
    solvates of one compound share it, while the full key separates them. A
    candidate list that counts the same molecule three times is not 25
    candidates, so both are reported and the stricter one is used only when
    the two disagree by little.
    """
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None, None
    return (key, key.split("-")[0]) if key else (None, None)


def properties(smiles):
    mol = smiles_mod.parse(smiles)
    if mol is None:
        return None
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    mw = Descriptors.MolWt(mol)
    key, skeleton = inchikeys(mol)
    return {
        "mol": mol,
        "inchikey": key,
        "skeleton": skeleton,
        "mw": mw,
        "mw_ok": mw <= MAX_MOLECULAR_WEIGHT,
        "rotb": rotb,
        "rotb_ok": rotb <= MAX_ROTATABLE,
        "amine": mol.HasSubstructMatch(ENTRY_AMINE),
        "basic_n": mol.HasSubstructMatch(BASIC_N),
    }


def known_antibiotics():
    """Named compounds from GramNegAccum, for a nearest-neighbour reference.

    These tables are the reason py/smiles.py exists. This function used to
    reorder the brackets itself and stop there, which parses the molecule but
    still reads `[N+]` as carrying no hydrogen, so the reference structures
    were subtly wrong.
    """
    import csv
    import io

    out = {}
    for n in (1, 5, 6):
        path = ROOT / "data" / "raw" / "gramnegaccum" / f"table{n}.csv"
        raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        for r in csv.DictReader(io.StringIO(raw, newline="")):
            name = (r.get("name") or "").strip()
            smiles = (r.get("smiles") or "").strip()
            if not name or not smiles:
                continue
            mol = smiles_mod.parse(smiles)
            if mol is not None:
                out[name] = mol
    return out


# [LOCK: ast name="deterministic_tiebreak"]
# Every "take the first of the tied ones" is a way for input order to decide
# the output. ECFP ignores stereochemistry, so stereoisomers score identically,
# and deduplication silently became selection by file position until this was
# added. The shuffle control went from 80% back to 100%.
def rank_key(i, scores, props):
    """Sort key: score first, then a property of the molecule itself.

    Scores alone are not enough to order the list. ECFP does not encode
    stereochemistry, so stereoisomers receive identical scores, and any
    "take the first" step over them -- choosing a representative, cutting the
    top N -- would be settled by the order the rows happened to arrive in.
    That order is not neutral here: the library's leading rows carry 2.27x the
    base hit rate. Breaking ties on the InChIKey removes the question.
    """
    return (-scores[i], props[i]["inchikey"] or "", i)
# [/LOCK]


def dedup(ordered, props):
    """Keeps the first row per skeleton, preserving the given order."""
    seen, out = set(), []
    for i in ordered:
        key = props[i]["skeleton"] or props[i]["inchikey"] or i
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
    return out


def score_pool(table, features, is_eval, epochs, learning_rate, tmp):
    """Trains the binding model on the held-out half and scores the other."""
    splits = [folds_mod.TEST if e else folds_mod.TRAIN for e in is_eval]
    part = tmp / "bind"
    write_parts(table, features, splits, part)
    stats = part / "stats.json"
    stats.write_text(json.dumps(stats_from(table, features, [not e for e in is_eval])) + "\n")
    weights = tmp / "bind.txt"
    sh(ENGINE_FP, part, "--stats", stats, "--evaluate", "test",
       "--epochs", epochs, "--learning-rate", learning_rate, "--save-weights", weights)
    dump = tmp / "bind.tsv"
    sh(ENGINE_FP, part, "--stats", stats, "--evaluate", "test",
       "--load-weights", weights, "--dump-scores", dump)
    logits = []
    with open(dump) as fh:
        next(fh)
        for line in fh:
            _, _, logit = line.split("\t")
            logits.append(float(logit))
    assert_ranking_is_usable(logits, "binding")
    rows = [i for i, e in enumerate(is_eval) if e]
    return dict(zip(rows, logits))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", type=Path, default=WORK / "imi_clean.parquet")
    parser.add_argument("--features", type=Path, default=WORK / "imi_fp" / "features.parquet")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=WORK / "candidates.json")
    parser.add_argument("--emit-molecules", action="store_true",
                        help="actually write the molecule list. Without this the "
                             "run stops after saying what it would have emitted.")
    args = parser.parse_args()

    print(__doc__[:__doc__.index("NOT THE PRODUCT") + 400].strip()
          if "NOT THE PRODUCT" in __doc__ else "", file=sys.stderr)
    if not args.emit_molecules:
        print(f"\nstopping: this writes a list of molecules, which is not the "
              f"assist output. Pass --emit-molecules if that is really what is "
              f"wanted, or use py/occam.py for a verdict on a specification.",
              file=sys.stderr)
        return 1

    table = pq.read_table(args.features)
    features = [n for n in table.column_names if n not in ("label", "scaffold", "smiles")]
    smiles = table.column("smiles").to_pylist()
    groups = table.column("scaffold").to_pylist()
    assign, _ = folds_mod.assign_folds(Counter(groups), 2, 0)
    is_eval = [assign[g] == 1 for g in groups]

    truth_table = pq.read_table(args.library, columns=["smiles", "true_hit"])
    truth_of = dict(zip(truth_table.column(0).to_pylist(),
                        [1 if v == "active" else 0 for v in truth_table.column(1).to_pylist()]))

    with tempfile.TemporaryDirectory() as tmp:
        scores = score_pool(table, features, is_eval, args.epochs, args.learning_rate, Path(tmp))

    pool = sorted(scores)
    props = {i: properties(smiles[i]) for i in pool}
    pool = [i for i in pool if props[i] is not None]
    base = sum(truth_of[smiles[i]] for i in pool) / len(pool)
    print(f"pool {len(pool)} molecules, true-hit rate {base:.1%}, "
          f"binding model trained on {len(is_eval) - sum(is_eval)} held-out rows\n")

    # How much of the pool is the same compound written more than once?
    full = Counter(props[i]["inchikey"] for i in pool if props[i]["inchikey"])
    skel = Counter(props[i]["skeleton"] for i in pool if props[i]["skeleton"])
    print(f"identity: {len(full)} distinct InChIKeys, {len(skel)} distinct skeletons "
          f"(first block) over {len(pool)} rows")
    print(f"  rows that share a skeleton with another: "
          f"{sum(n for n in skel.values() if n > 1)} "
          f"({sum(n for n in skel.values() if n > 1)/len(pool):.1%})\n")

    # Shuffle control: the ranking must not depend on the order rows arrive in.
    shuffled = list(pool)
    random.Random(0).shuffle(shuffled)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    known = known_antibiotics()
    known_fps = {n: generator.GetFingerprint(m) for n, m in known.items()}

    report = {"pool": len(pool), "base_hit_rate": base, "top": args.top, "groups": {}}
    chosen = {}
    for key, (label, test) in FILTERS.items():
        kept = [i for i in pool if test(props[i])]
        ranked = dedup(sorted(kept, key=lambda i: rank_key(i, scores, props)), props)[: args.top]
        kept_shuffled = [i for i in shuffled if test(props[i])]
        ranked_shuffled = dedup(
            sorted(kept_shuffled, key=lambda i: rank_key(i, scores, props)), props
        )[: args.top]
        # The guard runs per group as well: a score array can separate the
        # library as a whole and still be flat across whatever one filter keeps.
        if kept:
            assert_ranking_is_usable([scores[i] for i in kept], f"group {key}")
        overlap = len(set(ranked) & set(ranked_shuffled)) / max(1, len(ranked))
        kept_unique = dedup(sorted(kept, key=lambda i: rank_key(i, scores, props)), props)
        hit_rate = (sum(truth_of[smiles[i]] for i in kept) / len(kept)) if kept else 0.0
        chosen[key] = ranked
        report["groups"][key] = {
            "filter": label, "kept": len(kept), "kept_unique": len(kept_unique),
            "kept_share": len(kept) / len(pool),
            "hit_rate_of_kept": hit_rate, "lift": hit_rate / base if base else None,
            "shuffle_overlap": overlap,
            "top_hit_rate": sum(truth_of[smiles[i]] for i in ranked) / len(ranked),
        }
        print(f"group {key} ({label}): kept {len(kept)} ({len(kept)/len(pool):.1%}), "
              f"unique {len(kept_unique)}, "
              f"lift {hit_rate/base:.2f}, top-{args.top} hit rate "
              f"{sum(truth_of[smiles[i]] for i in ranked)/len(ranked):.0%}, "
              f"shuffle overlap {overlap:.0%}")

    everywhere = set.intersection(*(set(v) for v in chosen.values()))
    print(f"\nin all three groups: {len(everywhere)} molecules")

    report["candidates"] = {}
    for key, ranked in chosen.items():
        rows = []
        for rank, i in enumerate(ranked, 1):
            p = props[i]
            fp = generator.GetFingerprint(p["mol"])
            best, best_sim = None, 0.0
            for name, kfp in known_fps.items():
                sim = DataStructs.TanimotoSimilarity(fp, kfp)
                if sim > best_sim:
                    best, best_sim = name, sim
            rows.append({
                "rank": rank, "smiles": smiles[i],
                "inchikey": p["inchikey"],
                "molecular_weight": round(p["mw"], 1),
                "binding_logit": round(scores[i], 4),
                "rotatable_bonds": p["rotb"],
                "within_molecular_weight_bound": p["mw_ok"],
                "has_primary_amine": p["amine"],
                "has_basic_nitrogen": p["basic_n"],
                "nearest_known_antibiotic": best,
                "tanimoto_to_it": round(best_sim, 3),
                "in_all_three_groups": i in everywhere,
                "gram_negative_active": bool(truth_of[smiles[i]]),
            })
        report["candidates"][key] = rows

    # Are the top candidates a finding, or just the best-measured series?
    print("\npool composition by nearest known antibiotic (Tanimoto >= 0.4):")
    near = Counter()
    for i in pool:
        fp = generator.GetFingerprint(props[i]["mol"])
        best, best_sim = None, 0.0
        for name, kfp in known_fps.items():
            sim = DataStructs.TanimotoSimilarity(fp, kfp)
            if sim > best_sim:
                best, best_sim = name, sim
        near[best if best_sim >= 0.4 else "(none)"] += 1
    report["pool_composition"] = {k: v for k, v in near.most_common(12)}
    for name, n in near.most_common(8):
        print(f"  {str(name)[:28]:28} {n:>6} ({n/len(pool):>5.1%})")

    report["caveats"] = [
        "Activity is for Gram-negative bacteria as a group, not a named strain: "
        "the training data aggregates pMIC over 2052 organisms into four groups, "
        "so strain-level output needs the source data re-fetched.",
        "No learned permeability model is used. The amine and rotatable-bond "
        "conditions are from Richter et al., Nature 2017 (n = 68); the molecular "
        "weight bound of 600 is added here and is not part of that rule. The "
        "learned permeability model's applicability domain covers only 34% of "
        "this library, which is why it is not used.",
        "A rule has an applicability domain too. Without the weight bound the "
        "published conditions rank a 1,281 Da glycopeptide first, because the 68 "
        "molecules they were derived from contain nothing of that size.",
        "The groups are routes, not rival answers: a primary amine separates the "
        "library into populations that enter by different mechanisms, and the "
        "three sets barely overlap. Group A removes every molecule without a "
        "primary amine, excluding tetracyclines and chloramphenicol as classes.",
        "Filtering concentrates the whole set but does not improve the very top "
        "of it: group C's top 25 already run at 92% because the binding model "
        "puts the right series there. A permeability filter earns its place "
        "where the binding evaluation cannot be afforded for every molecule, "
        "which is not the case at 11,025.",
        "These molecules are drawn from a library of known compounds, so they are "
        "a test of the ranking rather than new chemical matter.",
        "ECFP8 does not encode stereochemistry, so stereoisomers receive the same "
        "binding score. In series where the isomers differ in activity -- the "
        "fluoroquinolones among them, which is 18% of group A before "
        "deduplication -- the model cannot tell them apart. These are candidates "
        "at the level of the skeleton; choosing the isomer is a separate step.",
    ]
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n-> {args.out}")

    for key in FILTERS:
        print(f"\n--- group {key}: {FILTERS[key][0]}, top 8")
        print(f"{'#':>2} {'MW':>7} {'logit':>8} {'rotb':>5} {'amine':>6} "
              f"{'nearest known':>22} {'sim':>5} {'active':>7}")
        for r in report["candidates"][key][:8]:
            print(f"{r['rank']:>2} {r['molecular_weight']:>7.1f} {r['binding_logit']:>8.2f} "
                  f"{r['rotatable_bonds']:>5} {str(r['has_primary_amine']):>6} "
                  f"{str(r['nearest_known_antibiotic'])[:22]:>22} {r['tanimoto_to_it']:>5.2f} "
                  f"{str(r['gram_negative_active']):>7}")


if __name__ == "__main__":
    main()
