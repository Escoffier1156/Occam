//! Per-bulk processing: one pass of online logistic regression.
//!
//! The operator sees one bulk, updates the weights it inherited, and hands
//! them on. Nothing about a previous bulk is reachable from here: the carry
//! holds the model, not the data it came from.
//!
//! Two passes are needed for a comparable AUC. Pass 1 learns; pass 2 freezes
//! the weights and scores held-out rows into the carry's histogram. Scoring
//! while learning would mix one model's scores with another's, and the
//! resulting number would not belong to any single model.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::bulk::{Bulk, Sample, TEST, TRAIN, VALID};
use crate::carry::Carry;

pub trait Operator {
    fn absorb(&mut self, bulk: &Bulk, carry: Carry) -> Carry;
}

static LIVE_OPERATORS: AtomicUsize = AtomicUsize::new(0);

/// What an operator does with the bulk it is given.
#[derive(Clone, Copy, PartialEq)]
pub enum Pass {
    /// Update the weights from training rows; also fill the prequential
    /// histogram, scoring each row before learning from it.
    Learn,
    /// Weights are frozen; score rows of `evaluate` into the carry histogram.
    Score { evaluate: i8 },
}

/// Step size at the first sample, unless --learning-rate says otherwise.
/// It decays as 1/sqrt(seen). Chosen on the validation split, never on test.
pub const DEFAULT_LEARNING_RATE: f32 = 0.5;
/// L2 penalty, applied per sample.
const L2: f32 = 1e-5;

/// A short-lived operator. It sees one bulk, never a previous one.
pub struct DomainOperator {
    pass: Pass,
    learning_rate: f32,
}

impl DomainOperator {
    // [LOCK: ast name="one_operator_alive"]
    // Invariant 3, the operator half. Rebinding `op` rather than scoping it
    // builds the next operator before dropping the previous one, and this
    // assert is what catches that.
    /// Panics if another operator is still alive.
    pub fn new(pass: Pass, learning_rate: f32) -> Self {
        let claimed = LIVE_OPERATORS.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst);
        assert!(claimed.is_ok(), "two operators alive at once");
        DomainOperator { pass, learning_rate }
    }
    // [/LOCK]

    /// One SGD step on the log-loss gradient.
    fn learn_from(&self, carry: &mut Carry, sample: &Sample) {
        let x = Carry::expand(&carry.standardize(&sample.features));
        let p = carry.predict(&x);
        // Score before the update: this row is unseen at this moment.
        Carry::record(&mut carry.prequential, p, sample.label);

        carry.seen += 1;
        let rate = self.learning_rate / (carry.seen as f32).sqrt();
        let error = p - sample.label as f32;
        for i in 0..x.len() {
            let gradient = error * x[i] + L2 * carry.weights[i];
            carry.weights[i] -= rate * gradient;
        }
        carry.bias -= rate * error;
    }
}

impl Drop for DomainOperator {
    fn drop(&mut self) {
        LIVE_OPERATORS.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Operator for DomainOperator {
    fn absorb(&mut self, bulk: &Bulk, mut carry: Carry) -> Carry {
        match self.pass {
            Pass::Learn => {
                for sample in bulk.rows() {
                    if sample.split == TRAIN {
                        self.learn_from(&mut carry, sample);
                    }
                }
            }
            Pass::Score { evaluate } => {
                for sample in bulk.rows() {
                    if sample.split == evaluate {
                        let x = Carry::expand(&carry.standardize(&sample.features));
                        let p = carry.predict(&x);
                        Carry::record(&mut carry.hist, p, sample.label);
                    }
                }
            }
        }
        carry.bulks_seen += 1;
        carry
    }
}

/// Parses a split name as used on the command line.
pub fn split_code(name: &str) -> Option<i8> {
    match name {
        "train" => Some(TRAIN),
        "valid" => Some(VALID),
        "test" => Some(TEST),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bulk::Bulk;
    use crate::carry::{auc, F};

    fn sample(x: f32, label: u8, split: i8) -> Sample {
        let mut features = [0.0; F];
        features[0] = x;
        Sample { features, label, split }
    }

    // One test on purpose: LIVE_OPERATORS is process-wide and tests run in
    // parallel, so separate tests would see each other's operators.
    #[test]
    fn learning_and_scoring() {
        // Feature 0 alone separates the classes; the weight must follow it.
        let mut carry = Carry::empty();
        let rows: Vec<Sample> = (0..200)
            .map(|i| if i % 2 == 0 { sample(2.0, 1, TRAIN) } else { sample(-2.0, 0, TRAIN) })
            .collect();
        for _ in 0..5 {
            let mut op = DomainOperator::new(Pass::Learn, DEFAULT_LEARNING_RATE);
            carry = op.absorb(&Bulk::for_test(rows.clone()), carry);
            drop(op);
        }
        assert!(carry.weights[0] > 0.5, "weight did not follow the label: {}", carry.weights[0]);
        assert_eq!(carry.seen, 1000);

        // Scoring those rows must separate them completely.
        {
            let mut op = DomainOperator::new(Pass::Score { evaluate: TRAIN }, DEFAULT_LEARNING_RATE);
            carry = op.absorb(&Bulk::for_test(rows), carry);
        }
        assert_eq!(auc(&carry.hist), Some(1.0));

        // A scoring pass must not move the model.
        let before = (carry.weights, carry.bias, carry.seen);
        let held_out = vec![sample(1.0, 1, TEST), sample(-1.0, 0, TEST)];
        {
            let mut op = DomainOperator::new(Pass::Score { evaluate: TEST }, DEFAULT_LEARNING_RATE);
            carry = op.absorb(&Bulk::for_test(held_out.clone()), carry);
        }
        assert_eq!((carry.weights, carry.bias, carry.seen), before);

        // A learning pass must ignore held-out rows.
        {
            let mut op = DomainOperator::new(Pass::Learn, DEFAULT_LEARNING_RATE);
            carry = op.absorb(&Bulk::for_test(held_out), carry);
        }
        assert_eq!((carry.weights, carry.bias, carry.seen), before);
    }

    #[test]
    fn standardization_uses_the_given_constants() {
        let mut carry = Carry::empty();
        carry.mean[0] = 10.0;
        carry.inv_std[0] = 0.5;
        let x = carry.standardize(&sample(14.0, 1, TRAIN).features);
        // The fingerprint build deliberately leaves its inputs untouched.
        let expected = if crate::carry::STANDARDIZE { 2.0 } else { 14.0 };
        assert!((x[0] - expected).abs() < 1e-6, "{}", x[0]);
    }
}
