//! Fixed-size carry state.
//!
//! Carry is the only value that survives between operators.
//! Its size must not depend on bulk size or total data volume.

use std::io::{self, Write};
use std::mem::size_of;

/// Number of candidate slots. Fixed at compile time.
pub const K: usize = 256;

/// Number of input features.
///
/// Two builds exist because the two judgements need different descriptions of
/// a molecule. Permeability follows from bulk, polarity and lipophilicity, so
/// nine physicochemical numbers describe it. Binding follows from whether a
/// particular substructure fits a particular pocket, which no amount of
/// molecular weight can express, so that build reads a fingerprint instead.
///
/// Either way the count is fixed at compile time: it is a property of the
/// model, never of the data volume, so the carry stays a constant size.
#[cfg(not(feature = "fingerprint"))]
pub const F: usize = 9;
#[cfg(feature = "fingerprint")]
pub const F: usize = 2048;

/// Whether to centre and scale the inputs before using them.
///
/// Continuous descriptors live on unrelated scales -- molecular weight in the
/// hundreds, fraction sp3 below one -- so they have to be standardised before
/// a single learning rate can suit all of them.
///
/// Sparse binary features must not be. A bit set in a fraction p of molecules
/// has variance p(1-p), so dividing by its standard deviation multiplies a
/// rare bit by 1/sqrt(p(1-p)): at p = 0.001 a set bit becomes +31 while every
/// clear bit sits near zero, and the dot product ends up decided by a handful
/// of rare bits. Subtracting the mean also destroys sparsity, turning every
/// clear bit into -p and making the vector dense. Raw 0/1 keeps both the
/// geometry and the scale the model was designed for.
pub const STANDARDIZE: bool = !cfg!(feature = "fingerprint");

/// Whether to add squares and pairwise products of the inputs.
///
/// A linear model cannot express "high logP *and* low TPSA"; handing it the
/// products lets it, without changing anything about how data flows. For a
/// fingerprint the inputs are already indicator bits and the pairwise terms
/// would number two million, so that build keeps the features as they are.
pub const QUADRATIC: bool = !cfg!(feature = "fingerprint") && !cfg!(feature = "linear");

/// Model dimension: the inputs, and with QUADRATIC their squares and products.
pub const W: usize = if QUADRATIC { F + F + F * (F - 1) / 2 } else { F };

/// Score histogram bins per class. Scores are squashed to [0, 1], so the
/// bins cover the whole range and the AUC comes out of a fixed-size state.
pub const BINS: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub id: u64,
    pub score: f32,
    pub support: u32,
}

impl Candidate {
    /// An unused slot. Real candidates always have support >= 1.
    pub const EMPTY: Candidate = Candidate { id: 0, score: f32::NEG_INFINITY, support: 0 };

    pub fn is_empty(&self) -> bool {
        self.support == 0
    }

    /// True if `self` should be evicted before `other`.
    fn weaker_than(&self, other: &Candidate) -> bool {
        match (self.is_empty(), other.is_empty()) {
            (true, false) => true,
            (false, true) => false,
            _ => self.score < other.score,
        }
    }
}

/// `Copy` rules out every heap-owning field (Vec, HashMap, Box, String),
/// so the whole state is inline and its size is a compile-time constant.
// [LOCK: ast name="carry_is_fixed_size"]
// Invariant 1: R is a fixed length, independent of bulk size and data volume.
// `Copy` is what enforces it -- it rules out Vec, HashMap, Box and String, so
// every field is inline and size_of::<Carry>() is a compile-time constant.
// Adding a heap-owning field would not fail to compile here; it would fail the
// derive, and the const assert below would stop matching. Do not weaken either.
#[derive(Clone, Copy)]
pub struct Carry {
    pub top: [Candidate; K],
    pub bulks_seen: u64,
    /// Logistic regression weights over the expanded row, learned in pass 1.
    pub weights: [f32; W],
    pub bias: f32,
    /// Standardization constants from py/descriptors.py, not learned here.
    pub mean: [f32; F],
    pub inv_std: [f32; F],
    /// Training rows absorbed so far; drives the learning rate schedule.
    pub seen: u64,
    /// Scores of held-out rows, binned by class: [negative, positive].
    /// This is what makes a global AUC come out of a fixed-size carry.
    pub hist: [[u32; BINS]; 2],
    /// Same, but filled during pass 1 from each row's score *before* it is
    /// learned from. Reported as a secondary number, not used for the verdict.
    pub prequential: [[u32; BINS]; 2],
}

/// R in M ~ D / N + R, in bytes.
pub const CARRY_BYTES: usize = size_of::<Carry>();

// Breaks the build if Carry gains anything beyond its fixed fields.
// Rust may reorder fields, so the bound is "no padding beyond alignment"
// rather than an exact sum: what matters is that the size is a constant and
// that nothing heap-owning can appear (Copy already rules that out).
const PAYLOAD_BYTES: usize = K * size_of::<Candidate>()
    + 2 * size_of::<u64>()
    + (W + 2 * F + 1) * size_of::<f32>()
    + 4 * BINS * size_of::<u32>();
const _: () = assert!(CARRY_BYTES >= PAYLOAD_BYTES - 8 && CARRY_BYTES <= PAYLOAD_BYTES + 8);
// [/LOCK]

impl Carry {
    pub fn empty() -> Self {
        Carry {
            top: [Candidate::EMPTY; K],
            bulks_seen: 0,
            weights: [0.0; W],
            bias: 0.0,
            mean: [0.0; F],
            inv_std: [1.0; F],
            seen: 0,
            hist: [[0; BINS]; 2],
            prequential: [[0; BINS]; 2],
        }
    }

    /// Puts one row on the scale the model expects.
    ///
    /// For continuous features that means centring and scaling; for sparse
    /// binary ones it means leaving them exactly as they are. See STANDARDIZE.
    pub fn standardize(&self, raw: &[f32; F]) -> [f32; F] {
        if !STANDARDIZE {
            return *raw;
        }
        let mut x = [0.0; F];
        for i in 0..F {
            x[i] = (raw[i] - self.mean[i]) * self.inv_std[i];
        }
        x
    }

    /// Expands a standardized row into the model's feature space.
    ///
    /// Products of standardized values are already near unit scale, so no
    /// second set of statistics is needed for the derived terms.
    pub fn expand(x: &[f32; F]) -> [f32; W] {
        let mut out = [0.0; W];
        let mut k = 0;
        for i in 0..F {
            out[k] = x[i];
            k += 1;
        }
        if QUADRATIC {
            for i in 0..F {
                out[k] = x[i] * x[i];
                k += 1;
            }
            for i in 0..F {
                for j in (i + 1)..F {
                    out[k] = x[i] * x[j];
                    k += 1;
                }
            }
        }
        debug_assert_eq!(k, W);
        out
    }

    /// The linear score before the logistic squash.
    ///
    /// Ranking must use this, not the probability. The logistic function is
    /// monotone, so it preserves order in exact arithmetic, but in f32 it
    /// saturates: on data far from what the model was fitted to, z runs large
    /// enough that sigmoid(z) rounds to exactly 1.0 for a large share of rows.
    /// The ordering among those rows is then gone, and a stable sort quietly
    /// falls back to input order -- selecting by file position while appearing
    /// to select by score.
    pub fn logit(&self, x: &[f32; W]) -> f32 {
        let mut z = self.bias;
        for i in 0..W {
            z += self.weights[i] * x[i];
        }
        z
    }

    /// Probability that an expanded row is positive.
    pub fn predict(&self, x: &[f32; W]) -> f32 {
        sigmoid(self.logit(x))
    }

    /// Records a score in one of the histograms.
    pub fn record(hist: &mut [[u32; BINS]; 2], score: f32, label: u8) {
        let bin = ((score * BINS as f32) as usize).min(BINS - 1);
        hist[(label != 0) as usize][bin] += 1;
    }
}

/// Numerically stable logistic function.
pub fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Area under the ROC curve from a binned score histogram.
///
/// Ties inside a bin are split evenly, which is what the trapezoid rule does
/// for a tied block; the only loss against exact AUC is that two scores in
/// one bin are treated as equal.
pub fn auc(hist: &[[u32; BINS]; 2]) -> Option<f64> {
    let neg: u64 = hist[0].iter().map(|&c| c as u64).sum();
    let pos: u64 = hist[1].iter().map(|&c| c as u64).sum();
    if neg == 0 || pos == 0 {
        return None;
    }
    // Walk from the lowest bin up, counting negatives already passed.
    let mut seen_neg = 0u64;
    let mut concordant = 0f64;
    for bin in 0..BINS {
        let n = hist[0][bin] as u64;
        let p = hist[1][bin] as u64;
        // Positives in this bin beat every negative below, and tie with n.
        concordant += p as f64 * (seen_neg as f64 + n as f64 / 2.0);
        seen_neg += n;
    }
    Some(concordant / (pos as f64 * neg as f64))
}

impl Carry {

    /// Records one vote for `id`. A known id gains support; an unknown id
    /// takes a free slot or replaces the weakest one if it scores higher.
    pub fn credit(&mut self, id: u64, score: f32) {
        let mut weakest = 0;
        for i in 0..K {
            let slot = &mut self.top[i];
            if !slot.is_empty() && slot.id == id {
                slot.support = slot.support.saturating_add(1);
                slot.score = slot.score.max(score);
                return;
            }
            if self.top[i].weaker_than(&self.top[weakest]) {
                weakest = i;
            }
        }
        let slot = &mut self.top[weakest];
        if slot.is_empty() || score > slot.score {
            *slot = Candidate { id, score, support: 1 };
        }
    }

    /// Occupied slots.
    pub fn candidates(&self) -> impl Iterator<Item = &Candidate> {
        self.top.iter().filter(|c| !c.is_empty())
    }

    /// Candidates and their weights (share of bulks that supported them),
    /// best score first.
    pub fn write_tsv<W: Write>(&self, out: &mut W) -> io::Result<()> {
        let mut ranked = self.top;
        ranked.sort_by(|a, b| b.is_empty().cmp(&a.is_empty()).reverse().then(b.score.total_cmp(&a.score)));
        writeln!(out, "rank\tid\tscore\tsupport\tweight")?;
        for (rank, c) in ranked.iter().take_while(|c| !c.is_empty()).enumerate() {
            let weight = c.support as f64 / self.bulks_seen.max(1) as f64;
            writeln!(out, "{}\t{:016x}\t{:.6}\t{}\t{:.6}", rank + 1, c.id, c.score, c.support, weight)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_is_fixed() {
        assert_eq!(size_of::<Candidate>(), 16);
        // K candidates + 2 u64 + (weights over W, mean and inv_std over F, bias)
        // + 2 histograms of 2 classes x BINS u32.
        if QUADRATIC {
            assert_eq!(F, 9);
            assert_eq!(W, 54);
            assert_eq!(CARRY_BYTES, 20_792);
        } else {
            assert_eq!(W, F);
        }
    }

    #[test]
    fn binary_features_are_left_alone() {
        // Standardising a sparse bit would blow a rare one up to about 1/sqrt(p);
        // the fingerprint build must hand the raw 0/1 through untouched.
        let mut carry = Carry::empty();
        carry.mean[0] = 0.001;
        carry.inv_std[0] = 31.6;
        let mut raw = [0.0f32; F];
        raw[0] = 1.0;
        let x = carry.standardize(&raw);
        if STANDARDIZE {
            assert!(x[0] > 30.0, "continuous features should still be scaled");
        } else {
            assert_eq!(x, raw, "binary features must pass through unchanged");
        }
    }

    #[test]
    fn expand_lays_out_every_term_once() {
        let mut x = [0.0f32; F];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i + 1) as f32;
        }
        let e = Carry::expand(&x);
        assert_eq!(&e[..F], &x[..]);
        if !QUADRATIC {
            assert_eq!(W, F);
            return;
        }
        for i in 0..F {
            assert_eq!(e[F + i], x[i] * x[i]);
        }
        // Every unordered pair appears exactly once after the squares.
        let mut seen = std::collections::HashSet::new();
        let mut k = 2 * F;
        for i in 0..F {
            for j in (i + 1)..F {
                assert_eq!(e[k], x[i] * x[j]);
                assert!(seen.insert((i, j)));
                k += 1;
            }
        }
        assert_eq!(k, W);
        assert_eq!(seen.len(), F * (F - 1) / 2);
    }

    #[test]
    fn auc_matches_exact_computation() {
        // Scores far apart in [0, 1] so binning cannot change the order.
        let cases: &[(&[f32], &[f32])] = &[
            (&[0.1, 0.2, 0.3], &[0.7, 0.8, 0.9]),       // perfectly separated
            (&[0.7, 0.8, 0.9], &[0.1, 0.2, 0.3]),       // perfectly inverted
            (&[0.2, 0.6], &[0.4, 0.8]),                 // interleaved
            (&[0.5, 0.5], &[0.5, 0.5]),                 // all tied
        ];
        for (neg, pos) in cases {
            let mut hist = [[0u32; BINS]; 2];
            for &s in *neg {
                Carry::record(&mut hist, s, 0);
            }
            for &s in *pos {
                Carry::record(&mut hist, s, 1);
            }
            // Exact AUC: the Mann-Whitney statistic over every pair.
            let mut concordant = 0.0;
            for &p in *pos {
                for &n in *neg {
                    concordant += if p > n {
                        1.0
                    } else if p == n {
                        0.5
                    } else {
                        0.0
                    };
                }
            }
            let expected = concordant / (pos.len() * neg.len()) as f64;
            assert!((auc(&hist).unwrap() - expected).abs() < 1e-9, "{neg:?} {pos:?}");
        }
        // Undefined without both classes.
        assert_eq!(auc(&[[0; BINS]; 2]), None);
    }

    #[test]
    fn sigmoid_is_bounded() {
        for z in [-1e30, -100.0, -1.0, 0.0, 1.0, 100.0, 1e30] {
            let p = sigmoid(z);
            assert!((0.0..=1.0).contains(&p), "{z} -> {p}");
        }
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn scores_outside_range_stay_in_bounds() {
        // record must not index out of the histogram even if a score is 1.0.
        let mut hist = [[0u32; BINS]; 2];
        for s in [0.0, 1.0, 0.999_999_9] {
            Carry::record(&mut hist, s, 1);
        }
        assert_eq!(hist[1].iter().map(|&c| c as u64).sum::<u64>(), 3);
    }

    #[test]
    fn known_id_gains_support() {
        let mut c = Carry::empty();
        c.credit(7, 0.5);
        c.credit(7, 0.25);
        let got: Vec<_> = c.candidates().copied().collect();
        assert_eq!(got, vec![Candidate { id: 7, score: 0.5, support: 2 }]);
    }

    #[test]
    fn full_carry_keeps_top_k() {
        let mut c = Carry::empty();
        for id in 0..(4 * K as u64) {
            c.credit(id, id as f32);
        }
        assert_eq!(c.candidates().count(), K);
        let min = c.candidates().map(|c| c.id).min().unwrap();
        assert_eq!(min, 3 * K as u64);
    }
}
