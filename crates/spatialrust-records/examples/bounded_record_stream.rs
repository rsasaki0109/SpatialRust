//! Runs a deterministic leased stream while recycling one chunk buffer set.

use spatialrust_core::{PointCloudBuilder, StandardSchemas};
use spatialrust_records::{
    BoundedSpatialRecordSource, CancellationToken, MemoryBudget, RecyclingMemoryChunkSource,
    SchemaDescriptor, SchemaVersion, StreamOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = PointCloudBuilder::xyz();
    for index in 0..10_000 {
        builder.push_point([index as f32, 1.0, -1.0])?;
    }
    let cloud = builder.build()?;
    let schema = SchemaDescriptor::try_new(
        "synthetic-point",
        SchemaVersion::new(1, 0),
        StandardSchemas::point_xyz(),
    )?;
    let options = StreamOptions::new(1024, MemoryBudget::new(12 * 1024)?)?;
    let mut source =
        RecyclingMemoryChunkSource::try_new(schema, cloud, options, CancellationToken::default())?;
    let mut points = 0_usize;
    let mut chunks = 0_u64;

    while let Some(chunk) = source.next_chunk() {
        let chunk = chunk?;
        assert_eq!(chunk.identity().sequence, chunks);
        points += chunk.record().cloud().len();
        chunks += 1;
        drop(chunk);
    }

    println!(
        "points={points} chunks={chunks} buffer_set_allocations={} peak_bytes={}",
        source.buffer_set_allocations(),
        source.memory_tracker().snapshot().peak_bytes
    );
    Ok(())
}
