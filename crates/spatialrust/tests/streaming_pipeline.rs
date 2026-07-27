#![cfg(feature = "pipeline-streaming")]

use spatialrust::io::SpoolOptions;
use spatialrust::records::{
    CancellationToken, MemoryBudget, RecyclingMemoryChunkSource, SchemaDescriptor, SchemaVersion,
    StreamOptions,
};
use spatialrust::{
    reduce_positions, PointCloudBuilder, StreamingVoxelConfig, StreamingVoxelSource,
};

#[test]
fn streaming_pipeline_public_surface_composes() {
    let mut builder = PointCloudBuilder::xyz();
    builder.push_point([0.1, 0.0, 0.0]).unwrap();
    builder.push_point([0.2, 0.0, 0.0]).unwrap();
    let cloud = builder.build().unwrap();
    let schema =
        SchemaDescriptor::try_new("public.xyz", SchemaVersion::new(1, 0), cloud.schema().clone())
            .unwrap();
    let options = StreamOptions::new(1, MemoryBudget::new(4096).unwrap()).unwrap();
    let source = RecyclingMemoryChunkSource::try_new(
        schema.clone(),
        cloud.clone(),
        options.clone(),
        CancellationToken::default(),
    )
    .unwrap();
    let config = StreamingVoxelConfig::new(
        1.0,
        1,
        4,
        SpoolOptions::new(std::env::temp_dir(), 4096).unwrap(),
    )
    .unwrap();
    let mut voxels = StreamingVoxelSource::try_build(source, config).unwrap();
    let reduction = reduce_positions(&mut voxels).unwrap();
    assert_eq!(reduction.point_count, 1);
    let centroid = reduction.centroid.unwrap();
    assert!((centroid[0] - 0.15).abs() < 1e-6);
}
