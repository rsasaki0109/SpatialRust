use spatialrust_core::PointCloudBuilder;
use spatialrust_io::SpoolOptions;
use spatialrust_pipeline::{StreamingVoxelConfig, StreamingVoxelSource};
use spatialrust_records::{
    BoundedSpatialRecordSource, CancellationToken, MemoryBudget, RecyclingMemoryChunkSource,
    SchemaDescriptor, SchemaVersion, StreamOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = PointCloudBuilder::xyz();
    for index in 0..100_000 {
        let x = index as f32 * 0.001;
        builder.push_point([x, 0.0, 0.0])?;
    }
    let cloud = builder.build()?;
    let schema =
        SchemaDescriptor::try_new("example.xyz", SchemaVersion::new(1, 0), cloud.schema().clone())?;
    let options = StreamOptions::new(16_384, MemoryBudget::new(16 * 1024 * 1024)?)?;
    let source =
        RecyclingMemoryChunkSource::try_new(schema, cloud, options, CancellationToken::default())?;
    let config = StreamingVoxelConfig::new(
        0.1,
        16_384,
        16,
        SpoolOptions::new(std::env::temp_dir(), 64 * 1024 * 1024)?,
    )?;
    let mut voxels = StreamingVoxelSource::try_build(source, config)?;
    let spill_bytes = voxels.spool_bytes();
    let mut output_points = 0_u64;
    while let Some(chunk) = voxels.next_chunk() {
        output_points += chunk?.record().cloud().len() as u64;
    }
    println!("output_points={output_points} spill_bytes={spill_bytes}");
    Ok(())
}
