//! Bulk reader: yields one bulk at a time and never holds two.
//!
//! A bulk is a block of rows from a descriptor part written by
//! py/descriptors.py: F float features, a label and a split code. The reader
//! keeps no reference to a bulk it has returned and reads nothing ahead, so
//! the caller's drop is the only release.

use std::error::Error;
use std::fs::File;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use parquet::errors::ParquetError;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::reader::RowIter;
use parquet::record::{Field, Row};

use crate::carry::F;

static LIVE_BULKS: AtomicUsize = AtomicUsize::new(0);

/// Split codes, matching SPLITS in py/descriptors.py.
pub const TRAIN: i8 = 0;
pub const VALID: i8 = 1;
pub const TEST: i8 = 2;

/// One row of a descriptor part.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub features: [f32; F],
    pub label: u8,
    pub split: i8,
}

pub struct Bulk {
    rows: Vec<Sample>,
}

impl Bulk {
    // [LOCK: ast name="one_bulk_alive"]
    // Invariant 3: one bulk exists at a time. The claim is not that the loop
    // happens to drop it, but that a second one cannot be built while the
    // first lives. compare_exchange, not fetch_add: the check and the claim
    // have to be one step.
    /// Registers a new live bulk. Panics if another one is still alive.
    fn new(rows: Vec<Sample>) -> Self {
        let claimed = LIVE_BULKS.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst);
        assert!(claimed.is_ok(), "two bulks alive at once");
        Bulk { rows }
    }
    // [/LOCK]

    pub fn rows(&self) -> impl Iterator<Item = &Sample> {
        self.rows.iter()
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Builds a bulk directly, for tests that do not go through a file.
    #[cfg(test)]
    pub fn for_test(rows: Vec<Sample>) -> Self {
        Bulk::new(rows)
    }
}

impl Drop for Bulk {
    fn drop(&mut self) {
        LIVE_BULKS.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Totals over everything read so far. Fixed size.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReadStats {
    pub bulks: u64,
    pub rows: u64,
    pub payload_bytes: u64,
}

/// Every parquet file the reader opens may hold at most one row group. The
/// footer of a parquet file grows with its row group count, so a multi-row-group
/// file would make memory grow with data volume. Larger inputs are directories
/// of parts named part-00000000.parquet, part-00000001.parquet, ...
pub struct BulkReader {
    source: Source,
    next_part: u64,
    // Rows of the current part only. At most one part is open at a time.
    rows: Option<RowIter<'static>>,
    bulk_size: usize,
    stats: ReadStats,
}

enum Source {
    File(PathBuf),
    Parts(PathBuf),
}

impl BulkReader {
    /// `path` is a single parquet file or a directory of parts.
    pub fn open(path: impl AsRef<Path>, bulk_size: usize) -> Result<Self, Box<dyn Error>> {
        if bulk_size == 0 {
            return Err("bulk_size must be at least 1".into());
        }
        unmap_large_blocks_on_free();
        let path = path.as_ref().to_path_buf();
        let source = if path.is_dir() { Source::Parts(path) } else { Source::File(path) };
        Ok(BulkReader { source, next_part: 0, rows: None, bulk_size, stats: ReadStats::default() })
    }

    pub fn stats(&self) -> ReadStats {
        self.stats
    }

    fn part_path(&self, index: u64) -> Option<PathBuf> {
        match &self.source {
            Source::File(path) => (index == 0).then(|| path.clone()),
            Source::Parts(dir) => {
                let part = dir.join(format!("part-{index:08}.parquet"));
                part.exists().then_some(part)
            }
        }
    }

    /// Next row, opening parts one at a time. The exhausted part's reader is
    /// dropped before the next one is opened, so decoded pages of two parts
    /// never coexist.
    fn next_row(&mut self) -> Option<Result<Row, ParquetError>> {
        loop {
            if let Some(rows) = self.rows.as_mut() {
                if let Some(row) = rows.next() {
                    return Some(row);
                }
                self.rows = None;
            }
            let path = self.part_path(self.next_part)?;
            self.next_part += 1;
            match open_part(&path) {
                Ok(rows) => self.rows = Some(rows),
                Err(e) => {
                    self.next_part = u64::MAX;
                    return Some(Err(e));
                }
            }
        }
    }
}

impl Iterator for BulkReader {
    type Item = Result<Bulk, ParquetError>;

    fn next(&mut self) -> Option<Self::Item> {
        // The previous bulk must already be dropped before reading the next.
        assert_eq!(LIVE_BULKS.load(Ordering::SeqCst), 0, "previous bulk still alive");

        let mut rows = Vec::with_capacity(self.bulk_size);
        while rows.len() < self.bulk_size {
            match self.next_row() {
                None => break,
                Some(Err(e)) => return Some(Err(e)),
                Some(Ok(row)) => match parse(&row) {
                    Ok(sample) => rows.push(sample),
                    Err(e) => return Some(Err(e)),
                },
            }
        }
        if rows.is_empty() {
            return None;
        }
        self.stats.bulks += 1;
        self.stats.rows += rows.len() as u64;
        self.stats.payload_bytes += (rows.len() * size_of::<Sample>()) as u64;
        Some(Ok(Bulk::new(rows)))
    }
}

/// Serves every allocation of 64 KiB or more from its own mapping, unmapped
/// on free, so a dropped bulk and its decoded pages go back to the OS. glibc
/// otherwise raises this threshold after the first such free and serves later
/// bulks from the heap, where they fragment and the peak drifts with the bulk
/// count and with address layout instead of staying flat.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn unmap_large_blocks_on_free() {
    use std::os::raw::c_int;
    extern "C" {
        fn mallopt(param: c_int, value: c_int) -> c_int;
    }
    const M_MMAP_THRESHOLD: c_int = -3;
    // Setting the threshold explicitly also disables glibc's dynamic adjustment.
    unsafe {
        mallopt(M_MMAP_THRESHOLD, 64 * 1024);
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn unmap_large_blocks_on_free() {}

// [LOCK: ast name="one_row_group_per_file"]
// Invariant 2: a parquet footer grows with its row group count, so a file
// holding many of them makes memory depend on the data volume. This refusal
// is what keeps that out; it fired for real during M3.
fn open_part(path: &Path) -> Result<RowIter<'static>, ParquetError> {
    let file = File::open(path).map_err(|e| ParquetError::External(Box::new(e)))?;
    let reader = SerializedFileReader::new(file)?;
    let groups = reader.num_row_groups();
    if groups > 1 {
        return Err(ParquetError::General(format!(
            "{}: {groups} row groups; at most 1 per file is allowed because the footer \
             grows with the row group count. Split it with py/gen_scale.py.",
            path.display()
        )));
    }
    Ok(reader.into_iter())
}
// [/LOCK]

/// Reads one row as F features followed by label and split.
///
/// The column order is fixed by py/descriptors.py; names are not consulted, so
/// a part written with a different layout is a hard error rather than silently
/// wrong numbers.
fn parse(row: &Row) -> Result<Sample, ParquetError> {
    let mut features = [0.0f32; F];
    let mut label = None;
    let mut split = None;
    for (i, (name, field)) in row.get_column_iter().enumerate() {
        match (i, field) {
            (i, Field::Float(v)) if i < F => features[i] = *v,
            (i, Field::Double(v)) if i < F => features[i] = *v as f32,
            (i, Field::Byte(v)) if i == F => label = Some(*v as u8),
            (i, Field::Byte(v)) if i == F + 1 => split = Some(*v),
            (i, Field::Int(v)) if i == F => label = Some(*v as u8),
            (i, Field::Int(v)) if i == F + 1 => split = Some(*v as i8),
            _ => {
                return Err(ParquetError::General(format!(
                    "column {i} ({name}) is {field:?}; expected {F} floats then label and split"
                )))
            }
        }
    }
    match (label, split) {
        (Some(label), Some(split)) => Ok(Sample { features, label, split }),
        _ => Err(ParquetError::General(format!(
            "row has {} columns; expected {}",
            row.get_column_iter().count(),
            F + 2
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Single test on purpose: LIVE_BULKS is process-wide and tests run in parallel.
    #[test]
    fn one_bulk_at_a_time() {
        let sample = Sample { features: [1.0; F], label: 1, split: TRAIN };
        let a = Bulk::new(vec![sample; 3]);
        assert_eq!(a.rows().copied().collect::<Vec<_>>(), vec![sample; 3]);

        let second = std::panic::catch_unwind(|| Bulk::new(Vec::new()));
        assert!(second.is_err(), "a second live bulk must be refused");

        drop(a);
        let _c = Bulk::new(Vec::new());
    }
}
