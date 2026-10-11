//! The loop, nothing else.
//!
//! usage: engine <descriptor-dir> [--stats stats.json] [--bulk-size N]
//!               [--evaluate train|valid|test] [--epochs N] [--learning-rate R]
//!               [--dump-scores FILE] [--save-weights FILE] [--load-weights FILE]
//!
//! Pass 1 learns from the training rows; pass 2 freezes the weights and scores
//! the held-out split. Between passes only the carry survives.

mod bulk;
mod carry;
mod metrics;
mod process;

use std::error::Error;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::mem::size_of;
use std::path::PathBuf;

use bulk::BulkReader;
use carry::{auc, Carry};
use process::{split_code, DomainOperator, Operator, Pass, DEFAULT_LEARNING_RATE};

const DEFAULT_BULK_SIZE: usize = 1024;

struct Args {
    path: PathBuf,
    stats: Option<PathBuf>,
    bulk_size: usize,
    evaluate: i8,
    epochs: usize,
    dump_scores: Option<PathBuf>,
    learning_rate: f32,
    save_weights: Option<PathBuf>,
    load_weights: Option<PathBuf>,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    const USAGE: &str = "usage: engine <descriptor-dir> [--stats FILE] [--bulk-size N] \
                         [--evaluate train|valid|test] [--epochs N] [--learning-rate R]
//!               [--dump-scores FILE] [--save-weights FILE] [--load-weights FILE]";
    let mut args = std::env::args().skip(1);
    let path: PathBuf = args.next().ok_or(USAGE)?.into();
    let mut out = Args {
        path,
        stats: None,
        bulk_size: DEFAULT_BULK_SIZE,
        evaluate: bulk::TEST,
        epochs: 1,
        dump_scores: None,
        learning_rate: DEFAULT_LEARNING_RATE,
        save_weights: None,
        load_weights: None,
    };
    while let Some(flag) = args.next() {
        let value = || args_value(&flag);
        match flag.as_str() {
            "--stats" => out.stats = Some(next(&mut args, &flag)?.into()),
            "--bulk-size" => out.bulk_size = next(&mut args, &flag)?.parse()?,
            "--epochs" => out.epochs = next(&mut args, &flag)?.parse()?,
            "--learning-rate" => out.learning_rate = next(&mut args, &flag)?.parse()?,
            "--dump-scores" => out.dump_scores = Some(next(&mut args, &flag)?.into()),
            "--save-weights" => out.save_weights = Some(next(&mut args, &flag)?.into()),
            "--load-weights" => out.load_weights = Some(next(&mut args, &flag)?.into()),
            "--evaluate" => {
                let name = next(&mut args, &flag)?;
                out.evaluate = split_code(&name).ok_or(format!("unknown split: {name}"))?;
            }
            _ => {
                let _ = value;
                return Err(format!("unknown flag: {flag}\n{USAGE}").into());
            }
        }
    }
    Ok(out)
}

fn next(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    args.next().ok_or_else(|| format!("{flag} needs a value").into())
}

fn args_value(_: &str) {}

/// Reads mean and variance written by py/descriptors.py into the carry.
///
/// Deliberately minimal: the two arrays are read by name from a flat JSON
/// object, so the engine takes no JSON dependency for a file it only reads.
fn load_stats(carry: &mut Carry, path: &PathBuf) -> Result<(), Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    for (key, target) in [("mean", 0), ("variance", 1)] {
        let start = text
            .find(&format!("\"{key}\""))
            .ok_or_else(|| format!("{}: no \"{key}\" field", path.display()))?;
        let open = text[start..].find('[').ok_or("malformed stats")? + start;
        let close = text[open..].find(']').ok_or("malformed stats")? + open;
        let values: Vec<f32> = text[open + 1..close]
            .split(',')
            .map(|v| v.trim().parse())
            .collect::<Result<_, _>>()?;
        if values.len() != carry::F {
            return Err(format!("{key} has {} values, expected {}", values.len(), carry::F).into());
        }
        for (i, v) in values.into_iter().enumerate() {
            match target {
                0 => carry.mean[i] = v,
                // A constant feature has zero variance; leave it at scale 1 so
                // standardizing yields 0 instead of a non-finite number.
                _ => carry.inv_std[i] = if v > 0.0 { 1.0 / v.sqrt() } else { 1.0 },
            }
        }
    }
    Ok(())
}

/// Runs one pass over the whole input. The reader and every bulk it yields are
/// dropped before this returns.
fn run_pass(args: &Args, pass: Pass, mut carry: Carry) -> Result<(Carry, u64), Box<dyn Error>> {
    let mut bulks = BulkReader::open(&args.path, args.bulk_size)?;
    // [LOCK: ast name="the_loop"]
    // The whole of the mechanism. One bulk is read, one operator absorbs it,
    // both die at the end of the iteration and only the carry goes forward.
    // No lookahead, no buffering, nothing accumulated across iterations.
    for bulk in &mut bulks {
        let bulk = bulk?;
        let mut op = DomainOperator::new(pass, args.learning_rate);
        carry = op.absorb(&bulk, carry);
        // op and bulk are dropped here, before the next iteration
    }
    // [/LOCK]
    Ok((carry, bulks.stats().rows))
}

/// Writes every held-out score to a file, for checking the binned AUC against
/// an exact one. This output is O(D) on disk and is not part of the engine's
/// result; the memory invariants are unaffected.
fn dump_scores(args: &Args, carry: &Carry, path: &PathBuf) -> Result<(), Box<dyn Error>> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "label\tscore\tlogit")?;
    let mut bulks = BulkReader::open(&args.path, args.bulk_size)?;
    for bulk in &mut bulks {
        let bulk = bulk?;
        for sample in bulk.rows() {
            if sample.split == args.evaluate {
                let x = Carry::expand(&carry.standardize(&sample.features));
                // Both: the probability for reading, the logit for ranking.
                writeln!(out, "{}\t{:.9}\t{:.9}", sample.label, carry.predict(&x), carry.logit(&x))?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

/// Writes bias then every weight, one per line.
///
/// A model trained on one dataset has to be applied to another for the order
/// experiment, and re-deriving the feature expansion outside the engine would
/// mean two implementations that can drift apart. Saving and loading keeps one.
fn save_weights(carry: &Carry, path: &PathBuf) -> Result<(), Box<dyn Error>> {
    let mut out = String::new();
    out.push_str(&format!("{}\n", carry.bias));
    for w in carry.weights.iter() {
        out.push_str(&format!("{w}\n"));
    }
    std::fs::write(path, out)?;
    Ok(())
}

fn load_weights(carry: &mut Carry, path: &PathBuf) -> Result<(), Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    let values: Vec<f32> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse())
        .collect::<Result<_, _>>()?;
    if values.len() != carry::W + 1 {
        return Err(format!(
            "{}: {} values, expected {} (bias + {} weights)",
            path.display(), values.len(), carry::W + 1, carry::W
        ).into());
    }
    carry.bias = values[0];
    carry.weights.copy_from_slice(&values[1..]);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let mut carry = Carry::empty();
    if let Some(stats) = &args.stats {
        load_stats(&mut carry, stats)?;
    }

    let mut rows = 0;
    if let Some(path) = &args.load_weights {
        // Scoring with a model trained elsewhere: no learning pass at all.
        load_weights(&mut carry, path)?;
    } else {
        // Pass 1: learn. Repeating it is still one pass per epoch over the
        // file; nothing from a previous epoch is held but the carry.
        for _ in 0..args.epochs {
            let (updated, seen) = run_pass(&args, Pass::Learn, carry)?;
            carry = updated;
            rows = seen;
        }
    }
    if let Some(path) = &args.save_weights {
        save_weights(&carry, path)?;
    }

    // Pass 2: weights frozen, score the held-out split.
    let (mut carry, _) = run_pass(&args, Pass::Score { evaluate: args.evaluate }, carry)?;

    if let Some(path) = &args.dump_scores {
        dump_scores(&args, &carry, path)?;
    }

    let mut out = io::stdout().lock();
    writeln!(out, "auc\t{}", auc(&carry.hist).map_or("undefined".into(), |v| format!("{v:.6}")))?;
    writeln!(
        out,
        "prequential_auc\t{}",
        auc(&carry.prequential).map_or("undefined".into(), |v| format!("{v:.6}"))
    )?;
    writeln!(out, "train_rows\t{}", carry.seen)?;
    for (i, w) in carry.weights.iter().enumerate() {
        writeln!(out, "weight[{i}]\t{w:.6}")?;
    }
    writeln!(out, "bias\t{:.6}", carry.bias)?;
    writeln!(out, "learning_rate\t{:.6}", args.learning_rate)?;
    out.flush()?;

    metrics::report(
        &bulk::ReadStats {
            bulks: carry.bulks_seen,
            rows,
            payload_bytes: rows * size_of::<bulk::Sample>() as u64,
        },
        args.bulk_size,
        &carry,
    );
    Ok(())
}
