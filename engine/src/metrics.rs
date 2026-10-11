//! Peak RSS reporting.
//!
//! Reads the kernel's high-water mark once, after the loop has finished, so
//! nothing here runs inside the processing path.

use std::fs;

use crate::bulk::ReadStats;
use crate::carry::{Carry, CARRY_BYTES};

/// Peak resident set size of this process in KiB (VmHWM), if available.
pub fn peak_rss_kb() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
}

/// One key=value line on stderr, parsed by tests/test_memory_invariant.py.
pub fn report(stats: &ReadStats, bulk_size: usize, carry: &Carry) {
    eprintln!(
        "bulk_size={} bulks={} rows={} payload_bytes={} carry_bytes={} candidates={} peak_rss_kb={}",
        bulk_size,
        stats.bulks,
        stats.rows,
        stats.payload_bytes,
        CARRY_BYTES,
        carry.candidates().count(),
        peak_rss_kb().map_or_else(|| "unknown".to_string(), |v| v.to_string()),
    );
}
