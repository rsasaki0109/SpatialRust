//! Emits a versioned receipt for the canonical synthetic 1M-point workload.

use std::time::Instant;

use spatialrust_records::{
    canonical_streaming_workloads, MemoryBudget, MemoryTracker, StreamingReceipt,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workload =
        canonical_streaming_workloads().first().expect("canonical workload manifest is non-empty");
    let tracker = MemoryTracker::new(MemoryBudget::new(workload.memory_budget_bytes)?);
    let mut receipt = StreamingReceipt::new(format!("synthetic://{}", workload.id))?;
    let started = Instant::now();
    let mut remaining = workload.point_count;
    let mut checksum = 0.0_f64;

    while remaining > 0 {
        let point_count = remaining.min(workload.chunk_points as u64) as usize;
        let byte_count = u64::try_from(point_count)?
            .checked_mul(3 * u64::try_from(std::mem::size_of::<f32>())?)
            .ok_or("chunk byte count overflow")?;
        let reservation = tracker.try_reserve(byte_count)?;
        let points = vec![[1.0_f32, 2.0, 3.0]; point_count];
        checksum +=
            points.iter().map(|point| f64::from(point[0] + point[1] + point[2])).sum::<f64>();
        receipt.record_input_chunk(u64::try_from(point_count)?, byte_count)?;
        drop(points);
        drop(reservation);
        remaining -= u64::try_from(point_count)?;
    }

    receipt.record_phase(
        "synthetic-scan",
        u64::try_from(started.elapsed().as_nanos())?,
        receipt.bytes_read(),
    )?;
    receipt.capture_memory(&tracker);
    receipt.validate()?;
    assert_eq!(checksum, workload.point_count as f64 * 6.0);
    println!("{}", receipt.to_json()?);
    Ok(())
}
