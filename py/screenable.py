"""What a screen of this specification would and would not establish.

Two measurements that used to sit inside the candidate pipeline, where they
were gates on a list of molecules. They are verdicts, not gates: a researcher
deciding whether to run this screen needs them before the screen, and the list
of molecules is not the product.

  rankability    can the activity model put this set in an order at all? If
                 most scores are duplicates, the top N is settled by the
                 tie-break rather than by the model, and "the highest scoring
                 256" is not a true description of the output. Measured on the
                 -O slice at 47.6% distinct against a 50% floor declared
                 before the run.

  applicability  is the set inside the range the activity model was fitted on?
                 Measured by each molecule's greatest Tanimoto to a training
                 structure. On the earlier candidate list the median was 0.165
                 and the maximum 0.285, so every molecule was outside.

Both run on a sample, which is said in the output. Deciding that a model cannot
order a set does not need every row: a thousand rows showing 47% distinct
settles it, and fingerprinting four thousand took twelve minutes.

That licence is one-directional, and this sentence used to be read as though it
went both ways. The distinct share rises as fewer molecules are measured, so a
sample can put a set below the floor -- the whole set will be lower still --
and cannot put one above it. A sampled figure that clears the floor settles
nothing, and rankability() now says so instead of reporting a pass. The
asymmetry is written out at the verdict, with what it rests on.

Each verdict that comes back negative carries a revision: what to put in the
request instead so that the same question, or the nearest question the evidence
can answer, becomes answerable. The revisions are measured here and not
invented -- a target with no label is answered by naming the resolution the
evidence does have, an unsettled ranking by measuring the rest of the set, an
extrapolating set by measuring which training set it is actually nearest to.
Nothing proposes chemistry.

A third thing is reported rather than measured: which label the activity model
was fitted on. The IMI-COMBINE data carries four organism classes, and the
model in use was fitted on the Gram-positive one. That is a fact a researcher
has to know and not a verdict this module should reach.
"""

import argparse
import json
import random
import statistics as st
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import fingerprints as fingerprints_mod  # noqa: E402
import folds as folds_mod  # noqa: E402
import render as render_mod  # noqa: E402
import smiles as smiles_mod  # noqa: E402
from m3_order import sh, stats_from, write_parts  # noqa: E402

from rdkit import DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

RDLogger.DisableLog("rdApp.*")

WORK = ROOT / "data" / "work"
ENGINE_FP = ROOT / "engine" / "target-fp" / "release" / "engine"

# Declared before any of this was measured, and the same numbers the candidate
# pipeline used as gates.
#
# WHERE 50% COMES FROM: NOWHERE PUBLISHED. IT WAS CHOSEN.
#
# Asked for its basis on the first day and it did not have one. It still does
# not. No paper states a distinct-score share below which a ranking stops
# meaning anything, and none was found. The reasoning behind the number is
# only this: at half the scores duplicated, at least half of any top N is
# placed by whatever breaks the tie, and calling that output "the highest
# scoring N" is no longer a true description. That argues for a floor
# somewhere in the middle of the range. It does not single out 0.50 over 0.45
# or 0.55.
#
# What the two calibration curves below settled is narrower, and it is not the
# floor's justification: they show that the SHARE moves with how many
# molecules were measured, which is why a sampled figure cannot clear the
# floor (see the asymmetry in rankability()). A threshold crossing a
# sample-dependent statistic is a declared convention, and the output says so
# rather than implying a derivation.
FLOOR_BASIS = ("declared, not derived: no published source states a "
               "distinct-share floor. The reasoning is that at half the "
               "scores duplicated, half of any top N is placed by the "
               "tie-break rather than by the model, which argues for a floor "
               "near the middle of the range and does not single out this "
               "value over 0.45 or 0.55. Read a verdict near the floor as "
               "near a convention")
MIN_DISTINCT_SHARE = 0.50
MAX_TIE_SHARE = 0.05
# WHY THE DISTINCT SHARE IS NOT COMPARABLE BETWEEN SPECIFICATIONS
#
# It is a share: draw fewer molecules from the same distribution and
# collisions get rarer, so the figure rises with nothing about the model
# having changed. Measured at 1,200 of 1,820 for one specification and at 430
# of 430 for another, the two are not the same measurement -- and the only
# specification that cleared the floor was the only one that had been sampled.
#
# Fixing a common sample size was tried and dropped. The size that made the
# runs comparable was chosen to fit the smallest of them, which is no basis
# for a floor, and moving every measurement onto it would have raised every
# share at once: the tool would have stopped saying "cannot be ordered", which
# is the thing it is for.
#
# So the figure is reported with the number it was measured on, and with the
# statement that it does not compare across specifications. A tool whose
# product is saying what cannot be concluded can say that about its own
# numbers.
# The applicability-domain threshold. Not derived here: nearest-neighbour
# Tanimoto cut-offs in the 0.4 to 0.6 range are the common practice, and 0.4 is
# the bottom of that band, chosen so the verdict errs towards saying a set is
# inside rather than outside. Written down rather than tuned.
EXTRAPOLATION_BELOW = 0.40

# Positive controls, measured 2026-10-10. A verdict of INERT or out-of-domain
# looks identical to a broken calculation, so neither is issued without these.
CONTROLS = {
    "tanimoto_pipeline": {
        "test": "hold 300 rows out of IMI-COMBINE and measure each one's "
                "greatest Tanimoto to the remaining 21,752",
        "expected": "high, because the molecules come from the same collection",
        "measured": {"median": 0.747, "p5": 0.358, "max": 1.000,
                     "share_at_or_above_0_4": 0.940},
        "verdict": "pass",
        "what_it_rules_out": "a fingerprint radius or bit-length mismatch, or "
                             "salts and stereochemistry left on one side. Any of "
                             "those would depress the control too.",
    },
    "globularity": {
        "test": "compute globularity for the 68 compounds of GramNegAccum "
                "table4 and compare against the `glob` column its authors "
                "computed",
        "expected": "correlated, and spanning a similar range",
        "measured": {"published_median": 0.181, "measured_median": 0.156,
                     "published_range": [0.046, 0.450],
                     "measured_range": [0.037, 0.343],
                     "pearson_r": 0.529,
                     "median_absolute_difference": 0.043,
                     "threshold_agreement": 0.824},
        "verdict": "pass for the range, weak per molecule",
        "what_it_rules_out": "the failure mode where no conformer is embedded "
                             "and every molecule reads as flat: that would pin "
                             "the measured values near 0, and they reach 0.343.",
        "caveat": "r = 0.529 and 12 of 68 disagree on the 0.25 threshold, so an "
                  "individual molecule's value is not reliable. The conformer "
                  "choice moves it by up to 0.236. Use it for a distribution, "
                  "not for a single compound.",
        "bears_on": "the published `glob` column itself has 55 of 68 compounds "
                    "at or below 0.25, so the condition passes 81% of the set it "
                    "was derived from. The INERT finding on ZINC is the same "
                    "direction, more extreme.",
    },
}
# The fallback count, used only when the whole set is too large to read inside
# the budget below. Everything at or under the ceiling is measured in full.
SAMPLE = 1200
SAMPLE_SEED = 0

# WHEN TO MEASURE EVERY SURVIVOR
#
# A sample cannot clear the floor (see rank_verdict), so a tool that always
# sampled would answer "not settled" for every set larger than SAMPLE and the
# ranking verdict would be decorative. The obvious counterweight is cost, so
# the cost was measured. It is not there.
#
#   TIMED 2026-10-13, ONE MACHINE, rankability() WALL CLOCK
#
#     n=   200   221.89  220.56  196.36
#     n=   600   254.78
#     n=  1200   262.03
#     n=  1820   242.92  236.77
#     n= 13946   248.06
#
#   Reading 70x more molecules does not cost more. The work is the fit --
#   22,051 training rows, 10 epochs, once per measurement -- and the survivors
#   are scored afterwards in a pass that disappears into it.
#
# WHAT WENT WRONG IN MEASURING IT, TWICE
#
# The first four points went in ascending n and declined, 273 -> 243. Fitting
# them gave -14.9 ms per molecule, significant at two standard errors: a
# NEGATIVE cost per molecule, which cannot be. n and time-of-run were
# confounded, the first run in a batch pays a cold start, and the fit was
# measuring the batch and not the count. Re-run in descending order the same
# 1,820 took 299.84 as the first point against 242.92 as the fourth.
#
# The second attempt compared warm points only and separated 200 from 1,820 by
# 18.6 s against a 6.2 s within-count spread -- but both 200s came from the
# faster batch, so count and batch were correlated again. Bracketing one 1,820
# between two 200s inside ONE batch gave 9.6 ms per molecule, which predicted
# 349 s for 13,946. It came in at 248.06.
#
# Three estimates of the per-molecule cost, all withdrawn: negative, then
# noise dressed as a fit, then an extrapolation the direct measurement broke.
# What survives is the only claim the data makes without a model: from 200 to
# 13,946 survivors, no per-molecule cost is measurable.
#
# SO THE CEILING IS NOT A TIME BUDGET
#
# A budget was declared first -- 300 s of wall clock, with the ceiling derived
# from it -- and that design is withdrawn with the slope it depended on. There
# is nothing to divide a budget by. The ceiling is now the edge of the range
# the measurement covers: beyond it the cost is unmeasured, not large.
#
# Above the ceiling the verdict is not_settled_at_this_sample, with --full to
# measure anyway, so "not settled" stays a choice rather than a dead end.
#
# THIS APPLIES TO rankability() AND NOT TO applicability()
#
# Their costs are different in kind. Applicability takes every survivor's
# greatest Tanimoto to 22,051 training structures, which is O(n x 22,051) and
# does scale. Its sampling is also a different question: a sampled MEDIAN does
# not drift with n the way a distinct-SHARE does, so the asymmetry that
# rank_verdict() exists for has no counterpart there. Left as it is, and
# unexamined rather than cleared.
SECONDS_PER_MOLECULE = None  # not measurable from 200 to 13,946; see above
RANKING_FIXED_SECONDS = 240.0  # mean of the warm points, all counts
RANKING_TIMED_SECONDS = {200: [221.89, 220.56, 196.36], 600: [254.78],
                         1200: [262.03], 1820: [242.92, 236.77],
                         13946: [248.06]}
# Both ends of the measured range come from the measurements, not from a
# second copy of them. The repository's duplicated-count guard caught 13,946
# written once in the table and once in the sentence about it, which is how
# the sentence would have gone stale the next time a point was added.
RANKING_MEASURED_FROM = min(RANKING_TIMED_SECONDS)
RANKING_MEASURED_TO = max(RANKING_TIMED_SECONDS)
RANKING_COST = {
    "timed_seconds": RANKING_TIMED_SECONDS,
    "machine": "one machine, 2026-10-13",
    "finding": f"no per-molecule cost is measurable between "
               f"{RANKING_MEASURED_FROM:,} and {RANKING_MEASURED_TO:,} "
               f"survivors; the fit on the training rows is the whole cost",
    "withdrawn": f"three estimates of a per-molecule cost. The first was "
                 f"negative because n was confounded with run order; the "
                 f"second correlated count with batch; the third, 9.6 ms, "
                 f"predicted 349 s for {RANKING_MEASURED_TO:,} survivors "
                 f"against "
                 f"{RANKING_TIMED_SECONDS[RANKING_MEASURED_TO][0]:.0f} "
                 f"measured. A time budget divided by a per-molecule cost was "
                 f"the original design and is withdrawn with it",
}
# The edge of the measured range, rounded up to cover the whole -P slice
# (17,775). An extension past what was measured, and said to be one.
FULL_MEASUREMENT_CEILING = 20000


def generator():
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=fingerprints_mod.FP_RADIUS, fpSize=fingerprints_mod.FP_BITS
    )


def training_label(features):
    """Which label the activity model was fitted on, from its own sidecar."""
    sidecar = Path(features).with_name("features.json")
    if not sidecar.exists():
        return {"known": False,
                "why": f"no features.json beside {features}"}
    record = json.loads(sidecar.read_text())
    source = record.get("source", "")
    # py/ingest_imi.py maps the organism columns onto these two names. The
    # column names themselves are not quoted here: `gram-positive_label` is
    # how the file spells it, and a reader is owed the thing it records.
    mapping = {
        "binding": "the gram-positive label, used as a stand-in for target "
                   "binding without the permeability barrier",
        "binding=active": "the gram-positive label, used as a stand-in for "
                          "target binding without the permeability barrier",
        "true_hit": "the gram-negative label, which is binding AND entry",
        "true_hit=active": "the gram-negative label, which is binding AND entry",
    }
    column = record.get("label_column")
    if column is None:
        # Older fingerprint sets did not record it. Fall back to matching the
        # positive count against the source library's columns rather than
        # guessing from the file name.
        column = _infer_from_counts(record, source)
    return {"known": True, "source": source,
            "rows": record.get("rows"),
            "labels": record.get("labels"),
            "column": column,
            "means": mapping.get(column),
            "note": "IMI-COMBINE carries four organism classes "
                    "(gram-negative, gram-positive, fungi, acid-fast) with pMIC "
                    "values. Which one to fit on is a decision, not a default."}


def _infer_from_counts(record, source):
    """Which column has exactly this many actives? Measured, not guessed.

    A fingerprint set built before the label column was recorded can still be
    identified: the positive count has to match one of the source's columns.
    If two match, or none, the answer is None and the caller says so.
    """
    labels = record.get("labels") or {}
    positives = labels.get("positive")
    if positives is None or not Path(source).exists():
        return None
    try:
        table = pq.read_table(source)
    except Exception:
        return None
    matches = []
    for name in table.column_names:
        if name == "smiles":
            continue
        values = table.column(name).to_pylist()
        for value in set(values):
            if sum(1 for v in values if v == value) == positives:
                matches.append(f"{name}={value}")
    return matches[0] if len(matches) == 1 else None


def measure_count(total, full=False, ceiling=None, fallback=SAMPLE):
    """How many of the survivors the ranking measurement should read.

    Three cases, and the middle one is the point: a set larger than the
    fallback sample but inside the time budget is measured in full, so the
    verdict can be stated rather than deferred.

        total <= ceiling   every survivor, and the verdict is stated
        total >  ceiling   the fallback sample, and the verdict says it is
                           not settled, with what a full pass would cost
        full=True          every survivor regardless, which is what --full is

    The ceiling comes from a declared time budget and not from a count, so
    changing machines changes the number and not the rule.
    """
    if ceiling is None:
        ceiling = FULL_MEASUREMENT_CEILING
    if full or ceiling is None or total <= ceiling:
        return total
    return min(fallback, total)


def full_measurement_seconds(total):
    """Roughly how long reading every survivor would take, for the sentence
    that offers --full.

    Flat in `total` across every count measured, so this returns the fixed
    cost and ignores its argument -- which is the finding, not a shortcut.
    Past the ceiling it is an extrapolation out of the measured range, and the
    caller says so rather than printing a bare number.
    """
    return RANKING_FIXED_SECONDS


def rank_verdict(distinct_share, top_share, sampled):
    """A SAMPLE CAN REFUSE A SET AND CANNOT CLEAR ONE.

    The distinct share rises as fewer molecules are measured: collisions get
    rarer when there are fewer pairs to collide. Measured on two independent
    sets, every step down in count raised it --

        430 survivors:    n=100 0.780  n=200 0.675  n=400 0.4975  n=430 0.486
      1,820 survivors:  n=1200 0.5967  n=1500 0.5253  n=1820 0.4648

    -- so a sampled figure behaves as an UPPER BOUND on the figure the whole
    set would give. That makes the two directions unequal:

      below the floor on a sample -> the full set is also below -> the refusal
        can be stated from the sample. This is what the module was built on:
        deciding that a model CANNOT order a set does not need every row.

      above the floor on a sample -> the full set may be below it, and here
        was: 1,200 of 1,820 gave 0.5967 and passed, and all 1,820 gave 0.4648
        and did not. The pass was the sample, not the model.

    The old code took the docstring's licence for the first case and applied
    it to the second. Of six runs, the one specification whose survivor set
    was larger than the sample was the only one reported rankable.

    Both directions lean on the bound holding, which is OBSERVED IN TWO SETS
    AND NOT PROVEN, so a sampled verdict says so either way. The risks are not
    symmetric -- claiming an order exists is worse than denying one that does
    -- but neither branch is assumption-free, and declining to call the
    monotonicity a theorem means disclosing where it is being relied on.

    Separated from rankability() so the rule can be tested without a model, a
    library and twelve minutes of fingerprinting. It is the whole of the
    decision; rankability() measures the two shares and hands them here.
    """
    clears = distinct_share >= MIN_DISTINCT_SHARE and top_share <= MAX_TIE_SHARE
    if not clears:
        return "not_answerable"
    return "not_settled_at_this_sample" if sampled else "answerable"


def sample_survivors(survivors, size=SAMPLE, seed=SAMPLE_SEED):
    if len(survivors) <= size:
        return list(survivors), len(survivors)
    return random.Random(seed).sample(list(survivors), size), len(survivors)


def rankability(survivors, features, epochs=10, learning_rate=0.5,
                size=SAMPLE):
    """Can the activity model put this set in an order?

    The verdict holds for this set at this size. It does not compare with
    another specification's, and the output says so.
    """
    chosen, total = sample_survivors(survivors, size)
    if len(chosen) < 50:
        return {"verdict": "not_measurable",
                "why": f"only {len(chosen)} molecules survive; too few to say "
                       f"whether an order is meaningful"}

    table = pq.read_table(features)
    names = [n for n in table.column_names if n not in ("label", "scaffold", "smiles")]
    gen = generator()

    import numpy as np
    import pyarrow as pa

    bits = fingerprints_mod.FP_BITS
    block = np.zeros((len(chosen), bits), dtype=np.float32)
    kept = 0
    for text in chosen:
        mol = smiles_mod.parse(text)
        if mol is None:
            continue
        block[kept] = gen.GetFingerprintAsNumPy(mol)
        kept += 1
    schema = pa.schema([(n, pa.float32()) for n in names]
                       + [("label", pa.int8()), ("scaffold", pa.string()),
                          ("smiles", pa.string())])
    columns = [pa.array(block[:kept, i]) for i in range(bits)]
    columns += [pa.array(np.zeros(kept, dtype=np.int8)),
                pa.array([""] * kept), pa.array(chosen[:kept])]
    scored_table = pa.table(columns, schema=schema)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        train = tmp / "train"
        write_parts(table, names, [folds_mod.TRAIN] * table.num_rows, train)
        stats = train / "stats.json"
        stats.write_text(json.dumps(
            stats_from(table, names, [True] * table.num_rows)) + "\n")
        weights = tmp / "w.txt"
        sh(ENGINE_FP, train, "--stats", stats, "--evaluate", "train",
           "--epochs", epochs, "--learning-rate", learning_rate,
           "--save-weights", weights)
        part = tmp / "score"
        write_parts(scored_table, names, [folds_mod.TEST] * kept, part)
        dump = tmp / "score.tsv"
        sh(ENGINE_FP, part, "--stats", stats, "--evaluate", "test",
           "--load-weights", weights, "--dump-scores", dump)
        logits = []
        with open(dump) as handle:
            next(handle)
            for line in handle:
                logits.append(float(line.split("\t")[2]))

    counts = Counter(logits)
    distinct = len(counts) / len(logits)
    top_value, top_count = counts.most_common(1)[0]
    ties = 1.0 - distinct
    sampled = len(logits) < total
    verdict = rank_verdict(distinct, top_count / len(logits), sampled)

    if verdict == "not_settled_at_this_sample":
        means = render_mod.cannot_say(
            "whether the model can order this set",
            f"{distinct:.1%} of scores are distinct, which clears the "
            f"{MIN_DISTINCT_SHARE:.0%} floor, but it was measured on "
            f"{len(logits):,} of {total:,} survivors and this share falls as "
            f"more are measured. Calibrated on two sets, every step up in "
            f"count lowered it, and on one of them a figure above this floor "
            f"crossed below it before the whole set had been read",
            f"the floor is not cleared by anything measured here. Measuring "
            f"all {total:,} would settle it")
    elif verdict == "answerable":
        means = [f"the model can order this set: {distinct:.1%} of scores are "
                 f"distinct against a {MIN_DISTINCT_SHARE:.0%} floor declared "
                 f"before the run, measured on all {total:,} survivors"]
    else:
        means = [f"the model cannot order this set: {distinct:.1%} of scores "
                 f"are distinct against a {MIN_DISTINCT_SHARE:.0%} floor "
                 f"declared before the run, measured on {len(logits):,} "
                 f"molecules. A top N drawn from it would be settled by the "
                 f"tie-break, not by the model."]

    out = {
        "verdict": verdict,
        "sampled": len(logits),
        "of_survivors": total,
        "measured_every_survivor": not sampled,
        "distinct_share": round(distinct, 4),
        "tie_share": round(ties, 4),
        "most_common_value": round(top_value, 6),
        "most_common_count": top_count,
        "floor_distinct": MIN_DISTINCT_SHARE,
        "floor_basis": FLOOR_BASIS,
        "ceiling_single_value": MAX_TIE_SHARE,
        "logit_range": [round(min(logits), 4), round(max(logits), 4)],
        "fitted_on": str(features),
        # The numbers are in the sentence whichever way the verdict went. A
        # passing verdict used to say only "the model can order this set",
        # which is the one a reader cannot check: 50.3% and 71% are not the
        # same answer, and the measurement that exposed the sampling fault
        # above was only visible because a failing verdict printed its share.
        "not_comparable": "this share moves with the number measured, so it "
                          "does not compare with another specification's "
                          "unless both were measured on the same count",
        # Joined for the record; the screen prints the lines.
        "means": " ".join(means),
        "means_lines": means,
    }
    if sampled:
        out["rests_on"] = ("sampled values behaving as an upper bound on the "
                           "full-set value, observed in 2 sets, not proven. "
                           "Measuring every survivor removes the assumption")
    return out


def applicability(survivors, training_smiles, size=SAMPLE):
    """Is this set inside the range the activity model was fitted on?"""
    chosen, total = sample_survivors(survivors, size)
    gen = generator()
    reference = []
    for text in training_smiles:
        mol = smiles_mod.parse(text)
        if mol is not None:
            reference.append(gen.GetFingerprint(mol))
    if not reference:
        return {"verdict": "not_measurable", "why": "no readable training structures"}
    nearest = []
    for text in chosen:
        mol = smiles_mod.parse(text)
        if mol is None:
            continue
        nearest.append(max(DataStructs.BulkTanimotoSimilarity(
            gen.GetFingerprint(mol), reference)))
    if not nearest:
        return {"verdict": "not_measurable", "why": "no readable survivors"}
    median = st.median(nearest)
    inside = sum(1 for v in nearest if v >= EXTRAPOLATION_BELOW) / len(nearest)
    return {
        "verdict": "answerable" if median >= EXTRAPOLATION_BELOW
                   else "answerable_but_extrapolating",
        "sampled": len(nearest),
        "of_survivors": total,
        "training_structures": len(reference),
        "nearest_tanimoto": {"median": round(median, 3),
                             "max": round(max(nearest), 3),
                             "min": round(min(nearest), 3)},
        "share_at_or_above": round(inside, 4),
        "threshold": EXTRAPOLATION_BELOW,
        "threshold_source": "nearest-neighbour Tanimoto cut-offs of 0.4 to 0.6 "
                            "are common applicability-domain practice; 0.4 is "
                            "the lenient end of that band",
        "control": CONTROLS["tanimoto_pipeline"],
        # "every part of this set" was printed beside a maximum of 0.457
        # against a threshold of 0.4, which the same line disproves. The
        # branch read the median and the sentence spoke about every molecule.
        # It now says what was measured: how much of the set is outside, and
        # whether any of it is not.
        "means": ("the set overlaps the training structures at the declared "
                  "threshold"
                  if median >= EXTRAPOLATION_BELOW else
                  (f"{1 - inside:.1%} of this set sits outside the structures "
                   f"the model was fitted on"
                   + (f", and none of it is inside"
                      if max(nearest) < EXTRAPOLATION_BELOW else
                      f", while {inside:.1%} reaches the threshold")
                   + f" (median {median:.3f}, max {max(nearest):.3f} against a "
                     f"{EXTRAPOLATION_BELOW} threshold). An order can be "
                     f"produced; its meaning is not established.")),
    }


def nearest_training_set(survivors, candidates=None, size=300):
    """Which of the available training sets is this set actually nearest to?

    The revision for an extrapolating verdict. Measured over every labelled
    library present, so the answer is "this one, at this distance" rather than
    "get more data".
    """
    # Name and what it measures. The row count used to be written into the
    # name as well, and the same count was then printed twice in one report --
    # once from this literal and once from the sentence that states the scope.
    # Two copies of a count drift apart the moment a set is re-ingested. The
    # count below comes from the rows this comparison actually read.
    candidates = candidates or {
        "IMI-COMBINE": ("activity", WORK / "imi_clean.parquet"),
        "CHAMP": ("permeability", WORK / "champ.parquet"),
        "GramNegAccum": ("permeability", WORK / "gna.parquet"),
        # A different barrier, and listed for exactly that reason. If a set
        # is nearer to the blood-brain barrier data than to anything about
        # Gram-negatives, that is worth a reader seeing: it says the library
        # resembles brain-penetrant chemistry, which is not what was asked
        # for. Naming the barrier beside each set is what makes the row
        # readable rather than a bare distance.
        "B3DB": ("blood-brain barrier",
                 WORK / "B3DB_classification.parquet"),
    }
    chosen, total = sample_survivors(survivors, size)
    gen = generator()
    probes = []
    for text in chosen:
        mol = smiles_mod.parse(text)
        if mol is not None:
            probes.append(gen.GetFingerprint(mol))
    if not probes:
        return []
    out = []
    for name, (measures, path) in candidates.items():
        if not Path(path).exists():
            continue
        try:
            names = pq.read_schema(path).names
            column = "smiles" if "smiles" in names else "SMILES"
            rows = pq.read_table(path, columns=[column]).column(0).to_pylist()
        except Exception:
            continue
        reference = [gen.GetFingerprint(m) for m in
                     (smiles_mod.parse(s) for s in rows) if m is not None]
        if not reference:
            continue
        nearest = [max(DataStructs.BulkTanimotoSimilarity(p, reference))
                   for p in probes]
        out.append({"training_set": name,
                    "measures": measures,
                    "structures": len(reference),
                    "median_nearest": round(st.median(nearest), 3),
                    "max_nearest": round(max(nearest), 3),
                    "inside_threshold": round(
                        sum(1 for v in nearest if v >= EXTRAPOLATION_BELOW)
                        / len(nearest), 4)})
    out.sort(key=lambda r: -r["median_nearest"])
    # J: 0.220 against 0.217 is not "actually nearest". The medians are
    # rounded to three places, so anything inside that is a tie and is said to
    # be one.
    if len(out) > 1:
        top = out[0]["median_nearest"]
        tied = [r["training_set"] for r in out
                if abs(r["median_nearest"] - top) <= 0.005]
        if len(tied) > 1:
            out[0]["tied_with"] = tied[1:]
            out[0]["tie_note"] = (
                f"{' and '.join(tied)} are level within the precision these "
                f"medians are reported to; neither is nearer")
    return out


def answerable_questions(words=None):
    """What resolution does the evidence actually have?

    The revision for a target that has no label. Read from the datasets rather
    than asserted: IMI-COMBINE separates organism classes, so a request for a
    molecular target can be restated as a request about an organism class, and
    the cost of that substitution is stated.
    """
    import inventory as inventory_mod

    manifest = ROOT / "data" / "raw" / "imi_combine" / "manifest.json"
    out = {"resolution_available": [], "resolution_absent": "molecular target",
           "why": "no dataset here records which protein a compound acted on"}
    # H: this sentence existed to stop a reader confusing two scopes of the
    # same label, and every number in it was typed. One of them, 8,601, is one
    # away from what the fitted set holds -- so the note written to prevent a
    # misreading carried one. The figures are read from the records now.
    #
    # The duplicated-count test did not catch it: it looks for the same number
    # twice, and 8,600 appears elsewhere without a thousands separator.
    full = fitted = None
    if manifest.exists():
        body = json.loads(manifest.read_text())
        verified = body.get("verified_contents", {})
        for key, value in verified.items():
            if key.endswith("_label") and isinstance(value, dict):
                out["resolution_available"].append({
                    "label": key,
                    "active": value.get("active"),
                    "inactive": value.get("inactive"),
                })
                if key.startswith("gram-positive"):
                    full = value.get("active")
        if verified.get("continuous_values"):
            out["continuous"] = verified["continuous_values"]
    features = WORK / "imi_fp" / "features.json"
    rows = None
    if features.exists():
        record = json.loads(features.read_text())
        rows = record.get("rows")
        labels = record.get("labels") or {}
        fitted = labels.get("positive")
    if full is not None and fitted is not None and rows:
        out["scope"] = (
            f"the counts below are over the FULL source "
            f"(data/processed/combined_bioassay_data.tsv). The model in use "
            f"was fitted on data/work/imi_clean.parquet, which keeps only the "
            f"{rows:,} rows carrying BOTH the gram-positive and gram-negative "
            f"labels -- so its gram-positive actives number {fitted:,}, not "
            f"{full:,}. The two figures are the same label at different "
            f"scopes.")
    else:
        out["scope"] = (
            "the counts below are over the full source; the model was fitted "
            "on the subset carrying both organism labels, and the same label "
            "has a different count in each. The two records needed to state "
            "both numbers are not both on this disk, so neither is given here "
            "rather than one being typed in.")
    out["substitution"] = (
        "a request naming a molecular target can be restated as a request about "
        "an organism class, which the evidence does resolve. That answers a "
        "different question: whether a compound is active against the organism, "
        "not whether it inhibits the named protein."
    )
    return out


# REMOVED: rankable_alternatives()
#
# It measured each frontier member to name which looser specification the
# model could order. Nothing called it -- the revision for an unrankable
# verdict states the action in words and says the measurement was not run.
#
# It is not left in place because of its default, `size=400`: the sample
# count this module retracted. The distinct share rises as fewer molecules
# are measured, so a search that stops at the first specification reading
# `answerable` on 400 of them would have returned a pass for almost any
# frontier. Dead code holding the retracted number is how the retraction
# gets undone by whoever wires it up next. The measurement is worth having;
# it has to be written against every survivor, not against a sample.

def report(survivors, features=WORK / "imi_fp" / "features.parquet",
           training_library=WORK / "imi_clean.parquet", size=SAMPLE,
           epochs=10, learning_rate=0.5, full=False):
    """All three, for one survivor set. Molecules are never returned.

    `size` is the applicability sample, whose cost scales. The ranking count
    is chosen by measure_count() instead, because its cost does not: sampling
    there bought no time and cost a verdict.
    """
    training_smiles = pq.read_table(training_library,
                                    columns=["smiles"]).column(0).to_pylist()
    rank = rankability(survivors, features, epochs, learning_rate,
                       measure_count(len(survivors), full=full,
                                     fallback=size))
    apply_ = applicability(survivors, training_smiles, size)
    model = training_label(features)

    # The third pillar. A negative verdict without a revision is half an answer.
    revisions = {}
    if apply_["verdict"] == "answerable_but_extrapolating":
        revisions["activity prediction"] = {
            "problem": "the set sits outside the structures the model was fitted on",
            "measured_alternatives": nearest_training_set(survivors),
            "what_this_means": "the training set listed first is the one this "
                               "library is actually nearest to. If none reaches "
                               "the threshold, no model fitted on what is here "
                               "can speak about this library, and the revision "
                               "is data rather than a setting.",
        }
    if rank["verdict"] == "not_answerable":
        revisions["ranking"] = {
            "problem": f"only {rank.get('distinct_share', 0):.1%} of scores are "
                       f"distinct against a {MIN_DISTINCT_SHARE:.0%} floor",
            # Named a function, which is a line of this codebase and not an
            # answer. A reader deciding what to do next needs the action, not
            # the call that performs it.
            "measured_alternatives": "a looser specification spreads the "
                                     "scores further, so one of the "
                                     "alternatives on the frontier may clear "
                                     "the floor. Which of them does is "
                                     "measurable and has not been measured "
                                     "here",
            "what_this_means": "the set can still be screened; what cannot be "
                               "done is to take the top N from it, because "
                               "the tie-break rather than the model would "
                               "choose them",
        }
    # A verdict the sample could not reach has a revision too, and it is the
    # cheapest one in here: measure the rest of the set. Leaving it out would
    # make the one unresolved verdict the only one with nothing to do about it.
    if rank["verdict"] == "not_settled_at_this_sample":
        revisions["ranking"] = {
            "problem": f"{rank.get('distinct_share', 0):.1%} of scores are "
                       f"distinct, above the {MIN_DISTINCT_SHARE:.0%} floor, "
                       f"but measured on {rank.get('sampled', 0):,} of "
                       f"{rank.get('of_survivors', 0):,} survivors, and this "
                       f"share falls as more are measured",
            "measured_alternatives": (
                f"run it again with --full, which measures all "
                f"{rank.get('of_survivors', 0):,} survivors. No setting has "
                f"to change and no new data is needed, only the rest of this "
                f"set. Expect about "
                f"{full_measurement_seconds(rank.get('of_survivors', 0)) / 60:.0f}"
                f" minutes: the cost is the fit on the training rows and did "
                f"not grow with the number of survivors anywhere it was "
                f"measured, though this set is past the range that was "
                f"measured, which is why the verdict waits for you to ask"),
            "what_this_means": "nothing is established about ordering this "
                               "set either way. A top N must not be drawn "
                               "from it on the strength of this figure",
        }

    return {
        "survivors": len(survivors),
        "sample_size": size,
        "sample_seed": SAMPLE_SEED,
        "rankability": rank,
        "applicability": apply_,
        "activity_model": model,
        "suggested_revisions": revisions,
        "molecules_emitted": False,
        "note": "this module answers what a screen would establish. It does not "
                "return molecules; choosing which to make is the researcher's.",
    }


def show(body, cost=None):
    """The two verdicts, said as claims about the screen rather than as fields.

    A verdict is the one line a reader carries away, so none of it is printed
    as a field value: `not_measurable` in square brackets is legible but it is
    not a sentence, and a reader who is a chemist and not a programmer should
    not have to decide whether the brackets mean anything.

    The control stays beside the applicability verdict and is never printed
    without it. An out-of-domain verdict with no control is the failure this
    function exists to make visible: the same measurement run inside the
    training set has to show the threshold can be met at all.
    """
    print("what a screen of this specification would establish:")

    r = body["rankability"]
    print(f"\n  ordering the survivors: {render_mod.verdict(r['verdict'])}")
    for line in r.get("means_lines") or ([r["means"]] if r.get("means") else []):
        for out_line in render_mod.wrap(line, indent=6, hanging=8):
            print(out_line)
    if r.get("sampled"):
        print(f"      measured on {r['sampled']:,} of {r['of_survivors']:,} "
              f"survivors (sample seed {body['sample_seed']})")
    # The floor is a convention and the sampled reading leans on a bound that
    # was observed rather than proved. Both are the reader's to weigh, so
    # neither stays in the record only.
    if r.get("floor_basis"):
        for line in render_mod.wrap(f"the {r['floor_distinct']:.0%} floor is "
                                    f"{r['floor_basis']}", indent=6, hanging=8):
            print(line)
    if r.get("rests_on"):
        for line in render_mod.wrap(f"this rests on {r['rests_on']}",
                                    indent=6, hanging=8):
            print(line)

    a = body["applicability"]
    print(f"\n  predicting activity: {render_mod.verdict(a['verdict'])}")
    if a.get("means"):
        for line in render_mod.wrap(a["means"], indent=6, hanging=8):
            print(line)
    if a.get("sampled"):
        print(f"      measured on {a['sampled']:,} of {a['of_survivors']:,} "
              f"survivors against {a['training_structures']:,} training "
              f"structures")
    if a.get("nearest_tanimoto"):
        n = a["nearest_tanimoto"]
        # Three decimals throughout: the same median appears again under the
        # alternatives below, formatted there, and 0.2 beside 0.200 reads as
        # two different measurements.
        print(f"      nearest training structure, Tanimoto: median "
              f"{n['median']:.3f}, best {n['max']:.3f}, "
              f"threshold {a['threshold']}")
        for line in render_mod.wrap(f"the threshold comes from: "
                                    f"{a['threshold_source']}",
                                    indent=6, hanging=8):
            print(line)
    control = a.get("control")
    if control:
        m = control["measured"]
        print(f"      positive control: "
              f"{render_mod.control_verdict(control['verdict'])}")
        for line in render_mod.wrap(
                f"the same measurement run inside the training set gives "
                f"median {m['median']}, with "
                f"{m['share_at_or_above_0_4']:.1%} at or above the threshold, "
                f"so the measurement can clear it",
                indent=8, hanging=10):
            print(line)
        for line in render_mod.wrap(f"this rules out: "
                                    f"{control['what_it_rules_out']}",
                                    indent=8, hanging=10):
            print(line)

    # The count of what would close this verdict belongs to the verdict, not
    # to a second document printed after it. py/what_to_measure.py computes
    # it; this prints it where the reader is already standing.
    if cost and cost.get("measurable"):
        sizes = cost["neighbourhood_sizes"]
        band = ", ".join(f"{row['covered_at']:.2f}: {row['compounds']:,}"
                         for row in cost["sensitivity"])
        print(f"      what would close it: "
              f"{cost['compounds_to_measure']:,} compounds measured, one per "
              f"neighbourhood")
        print(f"        at Tanimoto {cost['covered_at']}. "
              f"{cost['for_half_the_set']:,} of them cover half the set, "
              f"{cost['for_four_fifths']:,} cover four fifths")
        print(f"        {sizes['singletons']:,} neighbourhoods hold one "
              f"compound, so the tail is one purchase each")
        for line in render_mod.wrap(
                f"the count is a property of the threshold: {band}, moving "
                f"{cost['sensitivity_spread']}x across the band",
                indent=8, hanging=10):
            print(line)
        for line in render_mod.wrap(
                "this is coverage, not accuracy: it leaves no part of the set "
                "further than the threshold from something measured, and "
                "promises nothing about how a model fitted on the results "
                "ranks the rest. No compound is emitted.", indent=8,
                hanging=10):
            print(line)

    m = body["activity_model"]
    if m.get("known"):
        label = (str(m["column"]).replace("_", " ") if m.get("column")
                 else "not recorded; could not be identified")
        print(f"\n  the activity model was trained on: {label}")
        if m.get("means"):
            for line in render_mod.wrap(m["means"], indent=6, hanging=8):
                print(line)
        print(f"      {render_mod.counts_phrase(m['labels'])}, "
              f"over {m['rows']:,} rows")
        for line in render_mod.wrap(m["note"], indent=6, hanging=8):
            print(line)

    revisions = body.get("suggested_revisions") or {}
    if revisions:
        # Two blocks answer "what instead", and they answer different
        # questions: this one asks which measured set is nearest, and
        # py/what_to_measure.py asks how much new measuring would close the
        # gap. The headings name the question so the pair does not read as one
        # answer split in half.
        print("\nwhich measured set is nearest "
              "(measured, not a chemical suggestion):")
        for name, body_ in revisions.items():
            print()
            for line in render_mod.wrap(
                    f"{name.replace('_', ' ')}: {body_['problem']}",
                    indent=2, hanging=4):
                print(line)
            alternatives = body_.get("measured_alternatives")
            if isinstance(alternatives, list):
                for row in alternatives:
                    # The tick marks the training sets this library is inside.
                    # It is the column a reader scans, so it earns the mark.
                    mark = "*" if row["inside_threshold"] >= 0.5 else " "
                    label = render_mod.training_set_phrase(row)
                    if row.get("tie_note"):
                        for line in render_mod.wrap(row["tie_note"],
                                                    indent=4, hanging=6):
                            print(line)
                    # J, second half: a set measuring another barrier sitting
                    # at the top is worth saying out loud. The same run warns
                    # that these conditions are about the Gram-negative
                    # membrane and not the blood-brain barrier.
                    if row.get("measures") and "blood-brain" in row["measures"]:
                        for line in render_mod.wrap(
                                "that set measures a different barrier from "
                                "the one these conditions are about; its "
                                "distance says what the structures resemble, "
                                "not what would accumulate",
                                indent=4, hanging=6):
                            print(line)
                    print(f"    {mark} {label:38} "
                          f"median nearest {row['median_nearest']:.3f}, "
                          f"{row['inside_threshold']:.1%} at or above "
                          f"Tanimoto {EXTRAPOLATION_BELOW}")
            elif alternatives:
                for line in render_mod.wrap(str(alternatives), indent=4,
                                            hanging=6):
                    print(line)
            if body_.get("what_this_means"):
                for line in render_mod.wrap(body_["what_this_means"],
                                            indent=4, hanging=6):
                    print(line)

    print()
    for line in render_mod.wrap(
            "No molecules are emitted. The verdicts above state what a screen "
            "of this specification would and would not establish; choosing "
            "what to make from it is the chemist's call, not this tool's.",
            hanging=2):
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--survivors", type=Path, required=True,
                        help="JSON list of SMILES, or a parquet with a smiles column")
    parser.add_argument("--features", type=Path,
                        default=WORK / "imi_fp" / "features.parquet")
    parser.add_argument("--training-library", type=Path,
                        default=WORK / "imi_clean.parquet")
    parser.add_argument("--sample", type=int, default=SAMPLE,
                        help="the applicability sample, whose cost "
                             "scales. The ranking count is chosen from "
                             "the survivor total, because its cost does "
                             "not")
    parser.add_argument("--full", action="store_true",
                        help="measure the ranking on every survivor even "
                             "past the ceiling, instead of returning a "
                             "verdict the sample cannot settle")
    parser.add_argument("--out", type=Path, default=WORK / "screenable.json")
    args = parser.parse_args()

    path = args.survivors
    if path.suffix == ".json":
        body = json.loads(path.read_text())
        survivors = body if isinstance(body, list) else body.get("smiles", [])
    else:
        survivors = pq.read_table(path, columns=["smiles"]).column(0).to_pylist()

    result = report(survivors, args.features, args.training_library,
                    args.sample, full=args.full)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    show(result)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
