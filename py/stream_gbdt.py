"""Histogram gradient boosting that never holds the dataset.

Why this exists: Occam's 54-dimension linear model sits 0.0540 AUC below
sklearn's HistGradientBoostingClassifier on CHAMP under matched folds. That
gap has two possible causes and HANDOFF.md has carried it as unresolved:
the linear model class, or the O(1) memory constraint itself. The baseline is
already histogram-based, so binning is held constant and the question is sharp
-- a histogram learner can in principle run in bounded memory, so if it does
and still reaches the baseline, the memory constraint is not what costs the
0.0540.

The prediction was written down before this ran; the file it was written
to is gone, and HANDOFF.md section 11 records the timestamps (21:53 against
a result at 22:03) that made it checkable.

WHAT IS HELD, AND WHY IT IS FIXED LENGTH

  bin edges        F x (B+1) floats, from one streaming pass over coarse
                   pre-bins. Independent of the row count.
  leaf histograms  L x F x B x 2 floats, gradient and hessian sums per bin.
                   One level of the tree at a time, so L is the number of
                   leaves at that level, not the number of rows.
  trees            T x nodes x a handful of numbers. Set by the
                   hyperparameters, not by the data.

Nothing above grows with the number of rows. What does grow is the number of
passes: one per tree level, so T x depth passes in total. That is allowed --
the invariant is on memory, not on I/O, and multi-pass learning is already
how the engine's own numbers were measured.

WHAT THIS IS NOT

At 755 rows no memory limit binds, so matching the baseline here shows that a
bounded-memory tree learner is *algorithmically* able to reach it. It does not
show what a memory limit costs at a scale where it bites. Saying more than
that would be the same overreach this project has retracted three times.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
import folds as folds_mod  # noqa: E402

# Matching sklearn's HistGradientBoostingClassifier defaults, so the
# comparison is between memory regimes and not between hyperparameters.
BINS = 255
TREES = 100
MAX_DEPTH = None          # sklearn grows by leaves; this grows by depth
MAX_LEAVES = 31
LEARNING_RATE = 0.1
MIN_SAMPLES_LEAF = 20
L2 = 0.0
# Coarse pre-bins for the quantile pass. Fixed length, so the edges can be
# derived from a stream without keeping the values.
PRE_BINS = 4096


def iter_batches(part_dir, columns, batch_rows=8192):
    """Streams one part at a time. Taboo 3: never two row groups at once."""
    for path in sorted(part_dir.glob("part-*.parquet")):
        handle = pq.ParquetFile(path)
        for batch in handle.iter_batches(batch_size=batch_rows, columns=columns):
            yield batch
        del handle


def feature_names(part_dir):
    first = sorted(part_dir.glob("part-*.parquet"))[0]
    names = pq.ParquetFile(first).schema_arrow.names
    return [n for n in names if n not in ("label", "split", "smiles", "scaffold")]


def bin_edges(part_dir, names, train_only=True):
    """Quantile bin edges per feature, from a bounded streaming histogram.

    Two passes: one for the range, one for a coarse histogram inside it. The
    coarse counts are then walked to find quantile boundaries. Memory is
    F x PRE_BINS counts, which does not depend on the row count.
    """
    columns = names + ["split"]
    low = np.full(len(names), np.inf)
    high = np.full(len(names), -np.inf)
    for batch in iter_batches(part_dir, columns):
        split = np.asarray(batch.column(len(names)).to_pylist())
        keep = split != folds_mod.TEST if train_only else np.ones(len(split), bool)
        if not keep.any():
            continue
        block = np.column_stack([np.asarray(batch.column(i).to_pylist(), dtype=np.float64)
                                 for i in range(len(names))])[keep]
        low = np.minimum(low, block.min(axis=0))
        high = np.maximum(high, block.max(axis=0))

    span = np.where(high > low, high - low, 1.0)
    counts = np.zeros((len(names), PRE_BINS), dtype=np.int64)
    total = 0
    for batch in iter_batches(part_dir, columns):
        split = np.asarray(batch.column(len(names)).to_pylist())
        keep = split != folds_mod.TEST if train_only else np.ones(len(split), bool)
        if not keep.any():
            continue
        block = np.column_stack([np.asarray(batch.column(i).to_pylist(), dtype=np.float64)
                                 for i in range(len(names))])[keep]
        total += len(block)
        scaled = np.clip(((block - low) / span * PRE_BINS).astype(np.int64),
                         0, PRE_BINS - 1)
        for f in range(len(names)):
            np.add.at(counts[f], scaled[:, f], 1)

    edges = []
    for f in range(len(names)):
        cumulative = np.cumsum(counts[f])
        wanted = np.linspace(0, total, BINS + 1)[1:-1]
        positions = np.searchsorted(cumulative, wanted)
        inner = low[f] + positions / PRE_BINS * span[f]
        edges.append(np.unique(np.concatenate(([-np.inf], inner, [np.inf]))))
    return edges, low, high, total


def binned(block, edges):
    """Bin indices for one block of rows."""
    out = np.empty(block.shape, dtype=np.int16)
    for f in range(block.shape[1]):
        out[:, f] = np.searchsorted(edges[f][1:-1], block[:, f], side="right")
    return out


class Tree:
    """One boosted tree: split features, thresholds as bin indices, leaf values."""

    def __init__(self):
        self.feature = []
        self.bin_threshold = []
        self.left = []
        self.right = []
        self.value = []

    def add_leaf(self, value):
        self.feature.append(-1)
        self.bin_threshold.append(0)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(value)
        return len(self.value) - 1

    def make_split(self, node, feature, threshold, left, right):
        self.feature[node] = feature
        self.bin_threshold[node] = threshold
        self.left[node] = left
        self.right[node] = right
        self.value[node] = 0.0

    def route(self, codes):
        """Leaf index for every row in a binned block."""
        node = np.zeros(len(codes), dtype=np.int32)
        active = np.ones(len(codes), dtype=bool)
        while True:
            internal = active & (np.asarray(self.feature)[node] >= 0)
            if not internal.any():
                break
            idx = np.where(internal)[0]
            f = np.asarray(self.feature)[node[idx]]
            t = np.asarray(self.bin_threshold)[node[idx]]
            go_left = codes[idx, f] <= t
            node[idx] = np.where(go_left,
                                 np.asarray(self.left)[node[idx]],
                                 np.asarray(self.right)[node[idx]])
        return node

    def predict(self, codes):
        return np.asarray(self.value)[self.route(codes)]


def leaf_value(gradient, hessian):
    return -gradient / (hessian + L2) if hessian + L2 > 0 else 0.0


def gain(g_left, h_left, g_right, h_right, g_parent, h_parent):
    def score(g, h):
        return g * g / (h + L2) if h + L2 > 0 else 0.0
    return score(g_left, h_left) + score(g_right, h_right) - score(g_parent, h_parent)


def fit(part_dir, names, edges, trees=TREES, max_leaves=MAX_LEAVES,
        learning_rate=LEARNING_RATE, min_samples_leaf=MIN_SAMPLES_LEAF):
    """Boosts `trees` trees, holding only histograms and the trees themselves."""
    columns = names + ["label", "split"]
    label_at, split_at = len(names), len(names) + 1
    forest = []

    # The base score: log-odds of the training labels, from one streaming count.
    positives = negatives = 0
    for batch in iter_batches(part_dir, columns):
        split = np.asarray(batch.column(split_at).to_pylist())
        label = np.asarray(batch.column(label_at).to_pylist())
        keep = split != folds_mod.TEST
        positives += int(label[keep].sum())
        negatives += int(keep.sum() - label[keep].sum())
    if not positives or not negatives:
        raise ValueError("one class only in the training folds")
    base = math.log(positives / negatives)

    def stream_train():
        for batch in iter_batches(part_dir, columns):
            split = np.asarray(batch.column(split_at).to_pylist())
            keep = split != folds_mod.TEST
            if not keep.any():
                continue
            block = np.column_stack(
                [np.asarray(batch.column(i).to_pylist(), dtype=np.float64)
                 for i in range(len(names))])[keep]
            label = np.asarray(batch.column(label_at).to_pylist(), dtype=np.float64)[keep]
            yield binned(block, edges), label

    for _ in range(trees):
        tree = Tree()
        root = tree.add_leaf(0.0)
        frontier = [root]
        # Grow one level at a time. Each level costs one pass; the histograms
        # held are L x F x B x 2, never anything per row.
        while frontier and len(tree.value) < 2 * max_leaves - 1:
            slots = {node: i for i, node in enumerate(frontier)}
            hist_g = np.zeros((len(frontier), len(names), BINS + 1))
            hist_h = np.zeros((len(frontier), len(names), BINS + 1))
            counts = np.zeros((len(frontier), len(names), BINS + 1), dtype=np.int64)
            for codes, label in stream_train():
                raw = base + np.zeros(len(label))
                for done in forest:
                    raw += learning_rate * done.predict(codes)
                raw += learning_rate * tree.predict(codes)
                p = 1.0 / (1.0 + np.exp(-raw))
                g, h = p - label, np.maximum(p * (1.0 - p), 1e-12)
                leaves = tree.route(codes)
                for node, slot in slots.items():
                    rows = leaves == node
                    if not rows.any():
                        continue
                    for f in range(len(names)):
                        np.add.at(hist_g[slot, f], codes[rows, f], g[rows])
                        np.add.at(hist_h[slot, f], codes[rows, f], h[rows])
                        np.add.at(counts[slot, f], codes[rows, f], 1)
            next_frontier = []
            for node, slot in slots.items():
                g_parent = hist_g[slot, 0].sum()
                h_parent = hist_h[slot, 0].sum()
                n_parent = counts[slot, 0].sum()
                best = (0.0, None, None)
                for f in range(len(names)):
                    cg = np.cumsum(hist_g[slot, f])
                    ch = np.cumsum(hist_h[slot, f])
                    cn = np.cumsum(counts[slot, f])
                    for b in range(BINS):
                        if cn[b] < min_samples_leaf or n_parent - cn[b] < min_samples_leaf:
                            continue
                        value = gain(cg[b], ch[b], g_parent - cg[b], h_parent - ch[b],
                                     g_parent, h_parent)
                        if value > best[0]:
                            best = (value, f, b)
                if best[1] is None or len(tree.value) >= 2 * max_leaves - 1:
                    tree.value[node] = leaf_value(g_parent, h_parent)
                    continue
                _, f, b = best
                cg = np.cumsum(hist_g[slot, f])
                ch = np.cumsum(hist_h[slot, f])
                left = tree.add_leaf(leaf_value(cg[b], ch[b]))
                right = tree.add_leaf(leaf_value(g_parent - cg[b], h_parent - ch[b]))
                tree.make_split(node, f, b, left, right)
                next_frontier.extend([left, right])
            frontier = next_frontier
        forest.append(tree)
    return forest, base


def predict_test(part_dir, names, edges, forest, base, learning_rate=LEARNING_RATE):
    """Scores the test rows, streaming. Returns (score, label) pairs."""
    columns = names + ["label", "split"]
    label_at, split_at = len(names), len(names) + 1
    out = []
    for batch in iter_batches(part_dir, columns):
        split = np.asarray(batch.column(split_at).to_pylist())
        keep = split == folds_mod.TEST
        if not keep.any():
            continue
        block = np.column_stack([np.asarray(batch.column(i).to_pylist(), dtype=np.float64)
                                 for i in range(len(names))])[keep]
        label = np.asarray(batch.column(label_at).to_pylist())[keep]
        codes = binned(block, edges)
        raw = base + np.zeros(len(label))
        for tree in forest:
            raw += learning_rate * tree.predict(codes)
        out.extend(zip((1.0 / (1.0 + np.exp(-raw))).tolist(),
                       [int(v) for v in label]))
    return out


def state_bytes(n_features, n_leaves):
    """The fixed-length state, so the claim can be checked rather than asserted."""
    histograms = n_leaves * n_features * (BINS + 1) * 3 * 8
    edges = n_features * (BINS + 1) * 8
    prebins = n_features * PRE_BINS * 8
    return {"leaf_histograms": histograms, "bin_edges": edges,
            "quantile_prebins": prebins,
            "total": histograms + edges + prebins,
            "depends_on_rows": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parts", type=Path, help="a fold directory with part-*.parquet")
    parser.add_argument("--trees", type=int, default=TREES)
    parser.add_argument("--max-leaves", type=int, default=MAX_LEAVES)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    names = feature_names(args.parts)
    started = time.monotonic()
    edges, low, high, rows = bin_edges(args.parts, names)
    forest, base = fit(args.parts, names, edges, args.trees, args.max_leaves,
                       args.learning_rate)
    scored = predict_test(args.parts, names, edges, forest, base, args.learning_rate)
    seconds = time.monotonic() - started

    report = {"parts": str(args.parts), "features": names,
              "training_rows": rows, "test_rows": len(scored),
              "trees": args.trees, "max_leaves": args.max_leaves,
              "bins": BINS, "seconds": round(seconds, 1),
              "fixed_state_bytes": state_bytes(len(names), args.max_leaves)}
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps({**report, "scored": scored}, indent=2) + "\n")


if __name__ == "__main__":
    main()
