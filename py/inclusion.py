"""Does the inclusion relation predict which direction of transfer fails?

The claim in M4 section 3 came from one pair of datasets: GramNegAccum ->
CHAMP holds and CHAMP -> GramNegAccum fails, and the mutual coverage (53.2%
against 0.8%) points the same way. One pair is one observation. This splits
B3DB into disjoint pairs and asks whether the coverage, computed without any
label and without fitting anything, says in advance which direction will fail.

B3DB is used because it is the one dataset here whose numbers are independently
reproduced (0.8382 +- 0.0512 over 21 seeds).

PREDICTION RULE, FIXED BEFORE ANY TRANSFER WAS RUN

  For a pair (A, B), cov(A->B) is the share of B inside A's applicability
  domain, by the `covered` definition in py/coverage.py: inside the 1st-99th
  percentile box and within the 95th-percentile internal nearest-neighbour
  distance. Neither quantity uses a label or a fitted model.

  primary, a sign test over the pairs
      the direction with the lower coverage has the lower 95% CI lower bound.
      Scored on every pair whose two coverages differ at all. Under the null
      that coverage says nothing about direction, each pair is a coin flip.

  secondary, a verdict test
      where the coverages differ by a factor of 3 or more, the lower direction
      fails (its CI contains 0.5) and the higher direction holds. Pairs inside
      that factor make no verdict prediction: the rule is not claimed to
      resolve near-symmetric cases.

  The factor of 3 is arbitrary. It is written here rather than chosen later.

The pairs are disjoint by construction, so no molecule is ever in both the
training and the evaluation set. One pair is a control: two random halves,
whose coverage should be symmetric and high.

Run with --phase predict to write the predictions, then --phase test to score
them. The default runs both in order, and the predictions are written to disk
before the engine is invoked even once.
"""

import argparse
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import coverage as coverage_mod  # noqa: E402
import transfer as transfer_mod  # noqa: E402

WORK = ROOT / "data" / "work"
DEFAULT_SOURCE = WORK / "B3DB_classification_feat" / "features.parquet"
ASYMMETRY_FACTOR = 3.0
SPLIT_SEED = 0
# Each side of a pair needs enough molecules for a domain and both classes.
MIN_SUBSET = 150


def structural_flags(table):
    """Per-molecule halogen and stereocentre counts, read from the SMILES.

    Neither is one of the nine descriptors, so a split on them does not force
    the coverage it is about to be judged by.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    sys.path.insert(0, str(ROOT / "py"))
    import smiles as smiles_mod

    halogens, stereo = [], []
    for s in table.column("smiles").to_pylist():
        mol = smiles_mod.parse(s)
        if mol is None:
            halogens.append(0)
            stereo.append(0)
            continue
        halogens.append(sum(1 for a in mol.GetAtoms()
                            if a.GetAtomicNum() in (9, 17, 35, 53)))
        stereo.append(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                                    useLegacyImplementation=False)))
    return halogens, stereo


def subsets(table, names):
    """Disjoint (A, B) index pairs, each with a name and how it was cut.

    Every rule partitions on structure alone -- never on the label -- so the
    cut cannot smuggle in the thing the transfer is being asked to predict.

    The last element of each tuple says whether the cut is made on one of the
    nine descriptors the coverage is computed from. Where it is, a coverage of
    0% is forced by construction rather than measured, and the pair carries
    much less weight: the transfer verdict is still a real prediction, but the
    coverage that predicts it is a tautology. Pairs cut on something outside
    the descriptor set are the ones that test the rule.
    """
    raw = coverage_mod.matrix(table, names)
    n = len(raw)
    column = {name: raw[:, i] for i, name in enumerate(names)}
    scaffolds = table.column("scaffold").to_pylist()
    standardised = (raw - raw.mean(axis=0)) / np.where(raw.std(axis=0) > 0, raw.std(axis=0), 1)
    from_centre = np.linalg.norm(standardised, axis=1)
    counts = Counter(scaffolds)

    rng = random.Random(SPLIT_SEED)
    shuffled = list(range(n))
    rng.shuffle(shuffled)
    half = len(shuffled) // 2

    def band(values, lo, hi):
        low, high = np.percentile(values, lo), np.percentile(values, hi)
        inside = [i for i in range(n) if low <= values[i] <= high]
        outside = [i for i in range(n) if not (low <= values[i] <= high)]
        return inside, outside

    mw_in, mw_out = band(column["mw"], 30, 55)
    tpsa_in, tpsa_out = band(column["tpsa"], 35, 60)
    near = sorted(range(n), key=lambda i: from_centre[i])[: int(0.30 * n)]
    far = sorted(range(n), key=lambda i: -from_centre[i])[: int(0.30 * n)]

    halogens, stereo = structural_flags(table)

    return [
        ("random_halves", "control: two random disjoint halves",
         shuffled[:half], shuffled[half:], "A", "B", False),
        ("scaffold_frequency", "A: scaffolds seen 5+ times; B: scaffolds seen once",
         [i for i in range(n) if counts[scaffolds[i]] >= 5],
         [i for i in range(n) if counts[scaffolds[i]] == 1], "common", "rare", False),
        ("halogen", "A: no halogen; B: one or more",
         [i for i in range(n) if halogens[i] == 0],
         [i for i in range(n) if halogens[i] > 0], "none", "halogenated", False),
        ("stereocentres", "A: no stereocentre; B: two or more",
         [i for i in range(n) if stereo[i] == 0],
         [i for i in range(n) if stereo[i] >= 2], "flat", "stereo-rich", False),
        ("mw_band", "A: molecular weight in its 30th-55th percentile band; B: the rest",
         mw_in, mw_out, "narrow", "rest", True),
        ("tpsa_band", "A: TPSA in its 35th-60th percentile band; B: the rest",
         tpsa_in, tpsa_out, "narrow", "rest", True),
        ("core_shell", "A: nearest 30% to the centroid; B: farthest 30%",
         near, far, "core", "shell", True),
    ]


def write_subset(table, rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(rows), path, row_group_size=8192)


def predict(source, out_dir, neighbour_percentile):
    table = pq.read_table(source)
    names = coverage_mod.feature_names(table)
    labels = np.asarray(table.column("label").to_pylist())
    predictions = []

    print(__doc__[__doc__.index("PREDICTION RULE"):__doc__.index("The pairs are")].strip())
    print(f"\nsource {source}, {table.num_rows} molecules, "
          f"{int(labels.sum())} positive\n")
    header = (f"{'pair':20} {'A':>6} {'B':>6} {'cov(A->B)':>10} {'cov(B->A)':>10} "
              f"{'factor':>7}  prediction")
    print(header)

    for key, how, a_rows, b_rows, a_name, b_name, forced in subsets(table, names):
        if len(a_rows) < MIN_SUBSET or len(b_rows) < MIN_SUBSET:
            print(f"{key:20} skipped: {len(a_rows)}/{len(b_rows)} below {MIN_SUBSET}")
            continue
        for rows, side in ((a_rows, "a"), (b_rows, "b")):
            if len(set(labels[rows])) < 2:
                sys.exit(f"{key}: side {side} has one class only; the cut is degenerate")
        a_path = out_dir / f"{key}_a" / "features.parquet"
        b_path = out_dir / f"{key}_b" / "features.parquet"
        write_subset(table, a_rows, a_path)
        write_subset(table, b_rows, b_path)

        a_domain = coverage_mod.Domain(a_name, pq.read_table(a_path), names, neighbour_percentile)
        b_domain = coverage_mod.Domain(b_name, pq.read_table(b_path), names, neighbour_percentile)
        a_raw = coverage_mod.matrix(table, names)[a_rows]
        b_raw = coverage_mod.matrix(table, names)[b_rows]

        def covered(domain, raw):
            in_box, near, _, _ = domain.test(raw)
            return float((in_box & near).mean())

        a_to_b = covered(a_domain, b_raw)
        b_to_a = covered(b_domain, a_raw)
        lower = "a_to_b" if a_to_b < b_to_a else "b_to_a"
        factor = (max(a_to_b, b_to_a) / min(a_to_b, b_to_a)
                  if min(a_to_b, b_to_a) > 0 else float("inf"))
        verdict = ("lower direction fails, higher holds"
                   if factor >= ASYMMETRY_FACTOR else "no verdict predicted")
        mark = " [forced]" if forced else ""
        print(f"{key:20} {len(a_rows):>6} {len(b_rows):>6} {a_to_b:>9.1%} {b_to_a:>9.1%} "
              f"{factor:>7.1f}  lower={lower}, {verdict}{mark}")
        predictions.append({
            "pair": key, "how": how,
            "a": {"name": a_name, "n": len(a_rows), "path": str(a_path)},
            "b": {"name": b_name, "n": len(b_rows), "path": str(b_path)},
            "coverage": {"a_to_b": a_to_b, "b_to_a": b_to_a},
            "factor": factor if factor != float("inf") else None,
            "lower_direction": lower,
            "predicts_verdicts": factor >= ASYMMETRY_FACTOR,
            "coverage_forced_by_construction": forced,
            "neighbour_percentile": neighbour_percentile,
        })
    return predictions


def test(predictions, epochs, learning_rate, bootstrap_rounds):
    # A precision parameter, not part of the declared criterion: the CI is
    # computed the same way, with fewer resamples because these evaluation sets
    # are thousands of rows rather than the 129 the default was chosen for.
    transfer_mod.BOOTSTRAP_ROUNDS = bootstrap_rounds
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for p in predictions:
            a, b = Path(p["a"]["path"]), Path(p["b"]["path"])
            forward = transfer_mod.run(f"{p['pair']}: {p['a']['name']} -> {p['b']['name']}",
                                       a, b, epochs, learning_rate, tmp)
            backward = transfer_mod.run(f"{p['pair']}: {p['b']['name']} -> {p['a']['name']}",
                                        b, a, epochs, learning_rate, tmp)
            results.append({**p, "a_to_b": forward, "b_to_a": backward})
    return results


CONTROL_PAIR = "random_halves"
# A CI lower bound this close to 0.5 makes the verdict sensitive to the number
# of bootstrap resamples, so those directions are re-measured at the full count.
BOUNDARY_MARGIN = 0.05


def label_balance(path):
    """(rows, positive share) for one side of a pair."""
    column = pq.read_table(path, columns=["label"]).column(0).to_pylist()
    return len(column), sum(column) / len(column)


def directions(results):
    """One row per transfer direction, with its coverage and its confounders."""
    rows = []
    for r in results:
        for key, src, tgt in (("a_to_b", "a", "b"), ("b_to_a", "b", "a")):
            n_train, pos_train = label_balance(Path(r[src]["path"]))
            n_eval, pos_eval = label_balance(Path(r[tgt]["path"]))
            rows.append({
                "pair": r["pair"], "direction": key,
                "label": f"{r['pair']}: {r[src]['name']} -> {r[tgt]['name']}",
                "coverage": r["coverage"][key],
                "ci_lo": r[key]["ci95"][0], "auc": r[key]["auc"],
                "verdict": r[key]["verdict"],
                "n_train": n_train, "n_eval": n_eval,
                "pos_train": pos_train, "pos_eval": pos_eval,
                "forced": r.get("coverage_forced_by_construction", False),
                "control": r["pair"] == CONTROL_PAIR,
            })
    return rows


def spearman(xs, ys):
    """Rank correlation, with an exact permutation p-value for small n."""
    from itertools import permutations
    from math import comb  # noqa: F401  (kept beside binomial_tail's import site)

    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    def pearson(a, b):
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        va = sum((x - ma) ** 2 for x in a) ** 0.5
        vb = sum((y - mb) ** 2 for y in b) ** 0.5
        return cov / (va * vb) if va and vb else 0.0

    rx, ry = rank(xs), rank(ys)
    rho = pearson(rx, ry)
    if len(xs) > 9:
        return rho, None
    # One-sided: how often does a relabelling reach at least this rho?
    at_least = total = 0
    for perm in permutations(ry):
        total += 1
        at_least += pearson(rx, list(perm)) >= rho - 1e-12
    return rho, at_least / total


def score(results):
    print("\n\n================ SCORING AGAINST THE FIXED PREDICTIONS ================\n")
    rows = directions(results)

    print("every direction, with the confounders the rule has to survive:\n")
    print(f"{'direction':46} {'cov':>7} {'CI lo':>7} {'AUC':>7} {'train':>6} {'eval':>6} "
          f"{'pos tr':>7} {'pos ev':>7}  flags")
    for d in sorted(rows, key=lambda d: d["coverage"]):
        flags = " ".join(f for f, on in (("forced", d["forced"]),
                                        ("control", d["control"]),
                                        ("near-0.5", abs(d["ci_lo"] - 0.5) < BOUNDARY_MARGIN))
                         if on) or "-"
        print(f"{d['label']:46} {d['coverage']:>6.1%} {d['ci_lo']:>7.4f} {d['auc']:>7.4f} "
              f"{d['n_train']:>6} {d['n_eval']:>6} {d['pos_train']:>6.1%} "
              f"{d['pos_eval']:>6.1%}  {flags}")

    # Pre-registered test. The control is excluded: its two coverages differ by
    # 1.5 points, so calling the direction there is a coin flip dressed up as a
    # prediction, and counting it would inflate the hit rate.
    print("\n--- primary, pre-registered: sign test on the asymmetric pairs")
    print(f"{'pair':20} {'cov lower':>10} {'CI lo A->B':>11} {'CI lo B->A':>11} "
          f"{'CI lower in':>12}  sign")
    hits = {"unforced": [0, 0], "forced": [0, 0]}
    verdict_hits = verdict_total = 0
    for r in results:
        if r["pair"] == CONTROL_PAIR:
            continue
        lo_ab, lo_ba = r["a_to_b"]["ci95"][0], r["b_to_a"]["ci95"][0]
        ci_lower_in = "a_to_b" if lo_ab < lo_ba else "b_to_a"
        hit = ci_lower_in == r["lower_direction"]
        bucket = "forced" if r.get("coverage_forced_by_construction") else "unforced"
        hits[bucket][0] += hit
        hits[bucket][1] += 1
        print(f"{r['pair']:20} {r['lower_direction']:>10} {lo_ab:>11.4f} {lo_ba:>11.4f} "
              f"{ci_lower_in:>12}  {'HIT' if hit else 'MISS'}"
              f"{'  [forced]' if bucket == 'forced' else ''}")
        if r["predicts_verdicts"]:
            low = r["a_to_b"] if r["lower_direction"] == "a_to_b" else r["b_to_a"]
            high = r["b_to_a"] if r["lower_direction"] == "a_to_b" else r["a_to_b"]
            verdict_total += 1
            verdict_hits += (low["verdict"] != "transfer holds"
                             and high["verdict"] == "transfer holds")
    u_hit, u_tot = hits["unforced"]
    f_hit, f_tot = hits["forced"]
    print(f"\n  pairs that test the rule (cut outside the nine descriptors): "
          f"{u_hit}/{u_tot}"
          + (f", P(>= {u_hit}) = {binomial_tail(u_hit, u_tot):.4f}" if u_tot else ""))
    print(f"  pairs whose coverage is forced by the cut (weak evidence): {f_hit}/{f_tot}")
    print(f"  secondary verdict test, pairs differing by {ASYMMETRY_FACTOR:g}x or more: "
          f"{verdict_hits}/{verdict_total}")

    # Secondary, added after 4 of the 14 directions had already been seen, so
    # it is an analysis of this data rather than a test registered against it.
    print("\n--- secondary, NOT pre-registered: does lower coverage mean lower transfer?")
    free = [d for d in rows if not d["forced"]]
    rho, pval = spearman([d["coverage"] for d in free], [d["ci_lo"] for d in free])
    print(f"  Spearman over the {len(free)} unforced directions: rho = {rho:+.3f}"
          + (f", exact one-sided p = {pval:.4f}" if pval is not None else ""))
    allrows = rows
    rho_all, p_all = spearman([d["coverage"] for d in allrows],
                              [d["ci_lo"] for d in allrows])
    print(f"  including the {len(allrows) - len(free)} forced directions: "
          f"rho = {rho_all:+.3f}"
          + (f", exact one-sided p = {p_all:.4f}" if p_all is not None else ""))

    # The confounders the rule would be mistaken for.
    print("\n--- confounders")
    worse = [d for d in rows if not d["forced"] and not d["control"]]
    by_cov = sorted(worse, key=lambda d: d["coverage"])
    half = len(by_cov) // 2
    low_half, high_half = by_cov[:half], by_cov[half:]
    for name, group in (("lower-coverage half", low_half), ("higher-coverage half", high_half)):
        if not group:
            continue
        print(f"  {name}: median training rows "
              f"{sorted(d['n_train'] for d in group)[len(group) // 2]}, "
              f"median training positive share "
              f"{sorted(d['pos_train'] for d in group)[len(group) // 2]:.1%}")
    print("  If the lower-coverage half were systematically the smaller or the more")
    print("  imbalanced training set, the rule would be explained by that instead.")

    near = [d["label"] for d in rows if abs(d["ci_lo"] - 0.5) < BOUNDARY_MARGIN]
    print(f"\ndirections within {BOUNDARY_MARGIN} of the 0.5 verdict boundary: "
          f"{len(near)}")
    for label in near:
        print(f"  {label}  <- re-measure at the full bootstrap count")

    return {
        "unforced_sign": [u_hit, u_tot],
        "forced_sign": [f_hit, f_tot],
        "verdict": [verdict_hits, verdict_total],
        "spearman_unforced": {"rho": rho, "p_one_sided": pval, "n": len(free)},
        "spearman_all": {"rho": rho_all, "p_one_sided": p_all, "n": len(allrows)},
        "near_boundary": near,
        "directions": rows,
    }


def binomial_tail(hits, total):
    """P(X >= hits) for X ~ Binomial(total, 0.5)."""
    from math import comb
    return sum(comb(total, k) for k in range(hits, total + 1)) / 2 ** total


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=WORK / "inclusion")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--phase", choices=["predict", "test", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--neighbour-percentile", type=float,
                        default=coverage_mod.NEIGHBOUR_PERCENTILE)
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.engine:
        transfer_mod.ENGINE = args.engine
    args.out_dir.mkdir(parents=True, exist_ok=True)
    record = args.predictions or args.out_dir / "predictions.json"

    if args.phase in ("predict", "both"):
        predictions = predict(args.source, args.out_dir, args.neighbour_percentile)
        # Written before the engine runs, so the prediction cannot be edited
        # after seeing a result.
        record.write_text(json.dumps(predictions, indent=2) + "\n")
        print(f"\npredictions fixed in {record}")
    if args.phase == "predict":
        return

    predictions = json.loads(record.read_text())
    results = test(predictions, args.epochs, args.learning_rate, args.bootstrap)
    summary = score(results)
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "pairs": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
