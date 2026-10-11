"""M3: does filtering on permeability before the expensive step pay off?

The library is IMI-COMBINE rows carrying both labels. Gram-positive bacteria
have no outer membrane, so activity there stands in for target binding; a
Gram-negative active needs binding *and* entry, which makes it the ground
truth for a real hit.

Three routes spend the budget differently:

  A  permeability first  rank every molecule by the cheap filter, spend the
                         whole budget on the most permeable
  B  binding first       spend the budget on molecules chosen without
                         permeability information, then filter survivors
  C  upper bound         no budget at all: evaluate binding on everything

C turns the other two into percentages, so the numbers keep their meaning when
the library or the budget changes.

One assumption, stated rather than hidden: the binding evaluation is expensive
and the permeability filter is cheap. In practice binding needs a protein
structure and a docking run while permeability needs only the molecule, a gap
of several orders of magnitude; here both happen to be nine-descriptor models,
so the asymmetry is imposed by the budget rather than measured.

Everything else is measured. In particular the target-count axis needs no
assumption at all: a permeability score depends on the molecule alone and is
computed once for any number of targets, while binding must be redone per
target, so the evaluation counts diverge as targets are added.

Leakage: the binding model is trained on rows held out from the library by
Bemis-Murcko scaffold, so no molecule it scores shared a scaffold with its
training set. The permeability model is trained on CHAMP, an unrelated
dataset, and never sees this library at all.
"""

import argparse
import json
import random
import statistics as st
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import folds as folds_mod  # noqa: E402

# Two builds: nine physicochemical numbers for permeability, a 2048-bit
# fingerprint for binding. Binding depends on which substructures a molecule
# carries, which no amount of molecular weight expresses: the physicochemical
# build scores 0.72 on this task against the fingerprint build's 0.82.
ENGINE = ROOT / "engine" / "target" / "release" / "engine"
ENGINE_FP = ROOT / "engine" / "target-fp" / "release" / "engine"
WORK = ROOT / "data" / "work"
BUDGETS = (100, 500, 1000, 5000, 10000)
TARGET_COUNTS = (1, 5, 20)
RANDOM_REPEATS = 20
# Share of binding-evaluated molecules that pass the binding step.
BINDING_KEEP = 0.25


def sh(*cmd):
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


PART_ROWS = 8192


def write_parts(table, features, split_of_row, out_dir):
    """Writes engine-readable parts; stats must be supplied separately.

    One part holds one row group: a parquet footer grows with the row group
    count, which would make the engine's memory depend on the data volume, so
    the engine refuses a file holding more than one.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("part-*.parquet"):
        stale.unlink()
    schema = pa.schema(
        [(n, pa.float32()) for n in features] + [("label", pa.int8()), ("split", pa.int8())]
    )
    columns = [table.column(n).to_pylist() for n in features]
    labels = table.column("label").to_pylist()
    rows = [list(c) for c in columns] + [labels, list(split_of_row)]
    for part, start in enumerate(range(0, len(labels), PART_ROWS)):
        chunk = [col[start:start + PART_ROWS] for col in rows]
        pq.write_table(pa.table(chunk, schema=schema),
                       out_dir / f"part-{part:08d}.parquet", row_group_size=PART_ROWS)


def stats_from(table, features, mask):
    """Mean and variance over the rows selected by `mask`."""
    accumulators = [folds_mod.Welford() for _ in features]
    for name, acc in zip(features, accumulators):
        for v, keep in zip(table.column(name).to_pylist(), mask):
            if keep:
                acc.add(v)
    return {
        "features": list(features),
        "mean": [a.mean for a in accumulators],
        "variance": [a.variance for a in accumulators],
        "train_rows": accumulators[0].n,
    }


def score_with(part_dir, stats_path, weights, tmp, tag, split_of_row, engine=None):
    """Scores the test rows of part_dir; returns {row index: score}.

    The dump holds test rows only, in file order, so the caller has to know
    which rows those were. Returning a plain list makes it easy to zip the
    scores against every label instead -- a mistake already made twice, both
    times collapsing the measured AUC towards chance without failing. A
    mapping keyed by position in the full table removes the choice: indexing
    it with a row that was not scored raises instead of lining up silently.
    """
    dump = tmp / f"{tag}.tsv"
    sh(engine or ENGINE, part_dir, "--stats", stats_path, "--evaluate", "test",
       "--load-weights", weights, "--dump-scores", dump)
    scored = []
    with open(dump) as fh:
        next(fh)
        for line in fh:
            label, score, logit = line.split("\t")
            # Rank on the logit. The probability saturates to exactly 1.0 in
            # f32 on data far from the training distribution, and sorting
            # saturated values keeps input order instead of score order.
            scored.append(float(logit))

    test_rows = [i for i, split in enumerate(split_of_row) if split == folds_mod.TEST]
    if len(test_rows) != len(scored):
        raise RuntimeError(
            f"{tag}: {len(scored)} scores for {len(test_rows)} test rows"
        )
    assert_ranking_is_usable(scored, tag)
    return dict(zip(test_rows, scored))


# A ranking is meaningless once the scores stop separating the rows: whatever
# the sort cannot distinguish it leaves in file order, so selection quietly
# becomes selection by file position. That happened once here, and the file
# order was itself enriched, so it looked like a real effect.
#
# Two ways the separation fails, and both are checked. Many rows can share one
# value, or the scores can take too few distinct values overall -- the second
# passes a check on the largest tie while still deciding most comparisons by
# position.
MAX_TIE_SHARE = 0.05
MIN_DISTINCT_SHARE = 0.20


# [LOCK: ast name="ranking_guard"]
# This caught a real failure: 78% of a library shared one score because the
# logistic function had saturated in f32, the sort fell back on file order,
# and the file order was itself enriched for hits. The result looked like a
# permeability effect and was reported as one.
def assert_ranking_is_usable(scores, tag, max_tie=MAX_TIE_SHARE,
                             min_distinct=MIN_DISTINCT_SHARE):
    """Refuses a score array that cannot order its rows; returns the diagnostics."""
    counts = Counter(scores)
    value, n = counts.most_common(1)[0]
    tie_share = n / len(scores)
    distinct_share = len(counts) / len(scores)
    if tie_share > max_tie:
        raise RuntimeError(
            f"{tag}: {n}/{len(scores)} rows ({tie_share:.1%}) share the score "
            f"{value!r}, above the {max_tie:.0%} limit. Ranking them would order "
            f"by file position, not by the model. Rank on the logit, and check "
            f"how far this data sits from what the model was fitted to."
        )
    if distinct_share < min_distinct:
        raise RuntimeError(
            f"{tag}: only {len(counts)}/{len(scores)} distinct scores "
            f"({distinct_share:.1%}), below the {min_distinct:.0%} limit. Most "
            f"comparisons would be decided by file position rather than by score."
        )
    return {"tie_share": tie_share, "distinct_share": distinct_share}
# [/LOCK]


def bootstrap_ci(values, rounds=20000, seed=0):
    """Percentile interval for the mean, resampling the draws themselves."""
    rng = random.Random(seed)
    boot = sorted(st.fmean(rng.choices(values, k=len(values))) for _ in range(rounds))
    return boot[int(0.025 * rounds)], boot[int(0.975 * rounds)]


def top_m(indices, score, m):
    """The m best of `indices` by `score`, or all of them if fewer."""
    return sorted(indices, key=lambda i: -score[i])[:m]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", type=Path, default=WORK / "imi_clean.parquet")
    parser.add_argument("--champ-features", type=Path,
                        default=WORK / "champ_feat" / "features.parquet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=RANDOM_REPEATS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Two descriptions of the same library: fingerprints for binding,
        # physicochemical descriptors for permeability.
        fp_dir = tmp / "lib_fp"
        sh(sys.executable, ROOT / "py" / "fingerprints.py", args.library,
           "--smiles-column", "smiles", "--label-column", "binding",
           "--positive-label", "active", "--out-dir", fp_dir)
        table = pq.read_table(fp_dir / "features.parquet")
        features = [n for n in table.column_names if n not in ("label", "scaffold", "smiles")]

        desc_dir = tmp / "lib_desc"
        sh(sys.executable, ROOT / "py" / "descriptors.py", args.library,
           "--smiles-column", "smiles", "--label-column", "binding",
           "--positive-label", "active", "--out-dir", desc_dir)
        desc_table = pq.read_table(desc_dir / "features.parquet")
        desc_features = [n for n in desc_table.column_names
                         if n not in ("label", "scaffold", "smiles")]
        if desc_table.num_rows != table.num_rows:
            sys.exit("the two descriptions cover different rows")

        # The ground truth travels separately: descriptors.py only carries one
        # label, and the binding proxy is the one the model is allowed to see.
        truth_table = pq.read_table(args.library, columns=["smiles", "true_hit"])
        truth_of = dict(zip(truth_table.column(0).to_pylist(),
                            [1 if v == "active" else 0 for v in truth_table.column(1).to_pylist()]))
        smiles = table.column("smiles").to_pylist()
        truth = [truth_of[s] for s in smiles]
        binding_label = table.column("label").to_pylist()

        # Split by scaffold so the binding model never scores a molecule from a
        # scaffold it trained on. Scaffolds are the right grouping here, unlike
        # on CHAMP: this library has 7,394 of them and its two largest cover
        # 3.2% of the rows, against 52.7% on CHAMP. Butina would also need the
        # full distance matrix, which 22k molecules do not fit in memory.
        groups = table.column("scaffold").to_pylist()
        assignment, filled = folds_mod.assign_folds(Counter(groups), 2, 0)
        is_eval = [assignment[g] == 1 for g in groups]
        n_eval = sum(is_eval)
        print(f"library {len(smiles)} rows -> binding-train {len(smiles)-n_eval}, "
              f"evaluation {n_eval} (disjoint scaffolds)")

        # Binding model: train on the held-out half, score the evaluation half.
        bind_dir = tmp / "bind"
        write_parts(table, features,
                    [folds_mod.TRAIN if not e else folds_mod.TEST for e in is_eval], bind_dir)
        bind_stats = bind_dir / "stats.json"
        bind_stats.write_text(json.dumps(stats_from(table, features, [not e for e in is_eval])) + "\n")
        bind_splits = [folds_mod.TRAIN if not e else folds_mod.TEST for e in is_eval]
        bind_weights = tmp / "bind.txt"
        sh(ENGINE_FP, bind_dir, "--stats", bind_stats, "--evaluate", "test",
           "--epochs", args.epochs, "--learning-rate", args.learning_rate,
           "--save-weights", bind_weights)
        binding_all = score_with(bind_dir, bind_stats, bind_weights, tmp, "bind",
                                 bind_splits, engine=ENGINE_FP)

        # Permeability model: trained on CHAMP, applied with CHAMP's statistics.
        champ_parts = tmp / "champ"
        champ = pq.read_table(args.champ_features)
        champ_features = [n for n in champ.column_names if n not in ("label", "scaffold", "smiles")]
        if champ_features != desc_features:
            sys.exit(f"feature mismatch: {champ_features} vs {desc_features}")
        write_parts(champ, champ_features, [folds_mod.TRAIN] * champ.num_rows, champ_parts)
        champ_stats = champ_parts / "stats.json"
        champ_stats.write_text(json.dumps(stats_from(champ, champ_features,
                                                     [True] * champ.num_rows)) + "\n")
        perm_weights = tmp / "perm.txt"
        sh(ENGINE, champ_parts, "--stats", champ_stats, "--evaluate", "train",
           "--epochs", args.epochs, "--learning-rate", args.learning_rate,
           "--save-weights", perm_weights)
        perm_dir = tmp / "perm_apply"
        write_parts(desc_table, desc_features, [folds_mod.TEST] * desc_table.num_rows, perm_dir)
        perm_all = score_with(perm_dir, champ_stats, perm_weights, tmp, "perm",
                              [folds_mod.TEST] * desc_table.num_rows)

        # Only the evaluation half takes part in the comparison.
        pool = [i for i, e in enumerate(is_eval) if e]
        total_hits = sum(truth[i] for i in pool)
        print(f"evaluation pool: {len(pool)} molecules, {total_hits} true hits "
              f"({total_hits/len(pool):.1%})")

        # Every route spends its budget on binding and then reports the same
        # number of molecules, so the comparison is of which molecules were
        # chosen, not of how many were returned.

        def route_a(budget, m, perm):
            """Permeability first: the whole budget goes to the most permeable."""
            evaluated = top_m(pool, perm, budget)
            return evaluated, top_m(evaluated, binding_all, m)

        def route_b(budget, m, perm, sample_rng):
            """Binding first: molecules chosen without permeability, filtered after.

            The filter is the same operation as route A's, at the same
            selection ratio: route A keeps the budget most permeable of the
            pool, so the cut-off is that molecule's score, and route B drops
            whatever falls below it.
            """
            cutoff = perm[top_m(pool, perm, budget)[-1]]
            evaluated = sample_rng.sample(pool, budget)
            permeable = [i for i in evaluated if perm[i] >= cutoff]
            return evaluated, top_m(permeable, binding_all, m)

        def binding_only(m):
            """Binding alone, scoring the whole pool: no budget, no filter.

            This pays for a binding evaluation on every molecule, so it is not
            comparable with a route that pays for `budget` of them. It is a
            reference point, not the contrast: the matched comparison is A
            against B, which spend the same budget and differ only in the
            order the two filters are applied.
            """
            return pool, top_m(pool, binding_all, m)

        def oracle(m):
            """The best any ranking could do: the true hits, up to m of them.

            Using the true labels keeps the denominator free of modelling
            choices. Any model-based ceiling would need a rule for combining
            the two scores, and that rule would itself need justifying.
            """
            return min(m, total_hits)

        def hits(selection):
            return sum(1 for i in selection if truth[i] == 1)

        # Check 1: with the budget set to the whole pool, binding-first has to
        # reproduce the unbudgeted route exactly. If it does not, route B is wrong.
        _, b_full = route_b(len(pool), 250, perm_all, random.Random(1))
        _, ref_full = binding_only(250)
        identical = sorted(b_full) == sorted(ref_full)
        print(f"\ncheck: at budget = N, route B matches the binding-only reference: "
              f"{'YES' if identical else 'NO -- route B is wrong'}")
        if not identical:
            sys.exit("route B does not reduce to the binding-only reference at full budget")

        def null_scores(seed):
            """A filter that knows nothing, to separate the structure from the signal."""
            r = random.Random(seed)
            return {i: r.random() for i in range(table.num_rows)}

        results = []
        print(f"\n{'budget':>7} {'out':>5} {'A':>5} {'B(matched)':>12} "
              f"{'A-B (95% CI)':>22} {'ratio':>6} {'A/oracle':>9} {'bind-only*':>11}")
        for budget in list(BUDGETS) + [len(pool)]:
            if budget > len(pool):
                continue
            m = max(1, round(budget * BINDING_KEEP))
            _, a_out = route_a(budget, m, perm_all)
            a_hits = hits(a_out)
            oracle_hits = oracle(m)
            _, ref_out = binding_only(m)
            ref_hits = hits(ref_out)

            b_runs, null_runs, b_sizes = [], [], []
            for r in range(args.repeats):
                _, b_out = route_b(budget, m, perm_all, random.Random(1000 + r))
                b_runs.append(hits(b_out))
                b_sizes.append(len(b_out))
                # Check 4: the same pipeline with a filter that knows nothing.
                # A - null is what the permeability signal is worth; the rest
                # of A's lead over B is the shape of the pipeline.
                _, n_out = route_a(budget, m, null_scores(2000 + r))
                null_runs.append(hits(n_out))
            b_mean, b_sd = st.fmean(b_runs), (st.stdev(b_runs) if len(b_runs) > 1 else 0.0)
            null_mean = st.fmean(null_runs)
            null_sd = st.stdev(null_runs) if len(null_runs) > 1 else 0.0
            # The matched contrast: same budget, opposite order.
            over_b = [a_hits - v for v in b_runs]
            lo, hi = bootstrap_ci(over_b) if len(over_b) > 1 else (over_b[0], over_b[0])
            gains = [a_hits - n for n in null_runs]
            null_lo, null_hi = bootstrap_ci(gains) if len(gains) > 1 else (gains[0], gains[0])

            results.append({
                "budget": budget, "output_size": m,
                "oracle_hits": oracle_hits,
                "a_hits": a_hits, "a_share_of_oracle": a_hits / oracle_hits,
                "b_hits_mean": b_mean, "b_hits_sd": b_sd,
                "b_output_size_mean": st.fmean(b_sizes),
                "null_hits_mean": null_mean, "null_hits_sd": null_sd,
                "gain_over_b": {
                    "mean": st.fmean(over_b), "ci95": [lo, hi],
                    "ratio": a_hits / b_mean if b_mean else None,
                    "per_output_slot": st.fmean(over_b) / m,
                },
                "gain_over_null": {"mean": st.fmean(gains), "ci95": [null_lo, null_hi],
                                   "ratio": a_hits / null_mean if null_mean else None},
                "binding_only_hits": ref_hits,
                "binding_only_is_unbudgeted": True,
                "binding_evaluations": {"A": budget, "B": budget, "binding_only": len(pool)},
            })
            ratio = a_hits / b_mean if b_mean else float("nan")
            print(f"{budget:>7} {m:>5} {a_hits:>5} {b_mean:>7.1f}+-{b_sd:<4.1f} "
                  f"{st.fmean(over_b):>+8.1f} [{lo:>+6.1f},{hi:>+6.1f}] "
                  f"{ratio:>6.2f} {a_hits/oracle_hits:>8.0%} {ref_hits:>11}")
        print("\n  A and B spend the same budget and differ only in the order of the"
              " two filters.")
        print("  * binding-only scores the whole pool, so it pays for "
              f"{len(pool)} evaluations at every row: a reference, not a contrast.")
        print("  oracle = the true hits a perfect ranking would return in M slots.")
        print("\n  gain per output slot ((A-B)/M), which removes the growth of M:")
        for row in results:
            print(f"    budget {row['budget']:>6}: {row['gain_over_b']['per_output_slot']:+.4f}")

        # Target axis: counts only, no cost assumption.
        target_axis = []
        for t in TARGET_COUNTS:
            budget = BUDGETS[2]
            survivors = max(1, round(budget * BINDING_KEEP))
            target_axis.append({
                "targets": t,
                "permeability_evaluations": {"A": len(pool), "B": survivors},
                "binding_evaluations": {"A": budget * t, "B": len(pool) * t},
            })
        print(f"\n{'targets':>8} {'A: perm + bind':>22} {'B: perm + bind':>24}")
        for row in target_axis:
            a = row["permeability_evaluations"]["A"], row["binding_evaluations"]["A"]
            b = row["permeability_evaluations"]["B"], row["binding_evaluations"]["B"]
            print(f"{row['targets']:>8} {a[0]:>10} + {a[1]:<9} {b[0]:>12} + {b[1]:<9}")

        summary = {
            "library": str(args.library),
            "library_rows": len(smiles),
            "evaluation_pool": len(pool),
            "true_hits_in_pool": total_hits,
            "binding_train_rows": len(smiles) - n_eval,
            "output_share_of_budget": BINDING_KEEP,
            "random_repeats": args.repeats,
            "budgets": results,
            "target_axis": target_axis,
            "assumption": "binding evaluation is expensive, permeability is cheap",
        }
        if args.out:
            args.out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
