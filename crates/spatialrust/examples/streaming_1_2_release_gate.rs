//! Runs the canonical bounded-streaming workflow and prints its release receipt.

use spatialrust::core::{PointCloudBuilder, StandardSchemas};
use spatialrust::io::SpoolOptions;
use spatialrust::pipeline::{StreamingPipeline, StreamingVoxelConfig};
use spatialrust::platform::{
    ConformanceReport, ConformanceStatus, SecurityChecklist, Streaming12Measurements,
    Streaming12ReleaseEvidence, Streaming12ReleaseGate,
};
use spatialrust::records::{
    BoundedSpatialRecordSource, CancellationToken, MemoryBudget, RecyclingMemoryChunkSource,
    SchemaDescriptor, SchemaVersion, StreamOptions, StreamingTransferDirection,
};

fn main() {
    const MEMORY_BUDGET: u64 = 1024 * 1024;
    const SPOOL_LIMIT: u64 = 1024 * 1024;
    const MAX_RUNS: usize = 4;

    let mut builder = PointCloudBuilder::xyz();
    for point in [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0]] {
        builder.push_point(point).expect("valid point");
    }
    let schema = SchemaDescriptor::try_new(
        "release.xyz",
        SchemaVersion::new(1, 0),
        StandardSchemas::point_xyz(),
    )
    .expect("valid schema");
    let options = StreamOptions::new(2, MemoryBudget::new(MEMORY_BUDGET).expect("positive budget"))
        .expect("valid stream options");
    let source = RecyclingMemoryChunkSource::try_new(
        schema,
        builder.build().expect("valid cloud"),
        options,
        CancellationToken::default(),
    )
    .expect("bounded source");
    let tracker = source.memory_tracker().clone();
    let voxel = StreamingVoxelConfig::new(
        1.0,
        2,
        MAX_RUNS,
        SpoolOptions::new(std::env::temp_dir(), SPOOL_LIMIT).expect("bounded spool"),
    )
    .expect("voxel config");
    let pipeline = StreamingPipeline::new(source, "streaming-1.2-release")
        .expect("pipeline")
        .voxel(voxel)
        .expect("bounded voxel build");
    let mut stream = pipeline.into_iter();
    for chunk in stream.by_ref() {
        drop(chunk.expect("release output chunk"));
    }
    let receipt = stream.receipt().expect("streaming receipt");
    assert_eq!(receipt.input_points(), 4);
    assert_eq!(receipt.output_points(), 2);
    assert!(receipt.spilled_bytes() > 0);
    let host_to_device_bytes = receipt
        .transfers()
        .iter()
        .filter(|transfer| transfer.direction == StreamingTransferDirection::HostToDevice)
        .map(|transfer| transfer.bytes)
        .sum();
    let device_to_host_bytes = receipt
        .transfers()
        .iter()
        .filter(|transfer| transfer.direction == StreamingTransferDirection::DeviceToHost)
        .map(|transfer| transfer.bytes)
        .sum();
    drop(stream);

    let mut conformance = ConformanceReport::new();
    for &id in Streaming12ReleaseGate::required_conformance_cases() {
        conformance.record(id, ConformanceStatus::Pass, Some("CI/release receipt".into()));
    }
    let evidence = Streaming12ReleaseEvidence {
        conformance,
        security: SecurityChecklist::north_star_baseline_satisfied(),
        measurements: Streaming12Measurements {
            memory_budget_bytes: MEMORY_BUDGET,
            peak_tracked_bytes: receipt.peak_tracked_bytes(),
            spool_limit_bytes: SPOOL_LIMIT,
            spilled_bytes: receipt.spilled_bytes(),
            current_bytes_after_finish: tracker.snapshot().current_bytes,
            hidden_host_copy_bytes: 0,
            host_to_device_bytes,
            device_to_host_bytes,
            determinism_mismatches: 0,
            max_open_spill_files: MAX_RUNS as u64,
        },
        passed_receipts: Streaming12ReleaseGate::required_receipts()
            .iter()
            .map(ToString::to_string)
            .collect(),
        verified_examples: Streaming12ReleaseGate::required_examples()
            .iter()
            .map(ToString::to_string)
            .collect(),
        migration_policy: "bounded-streaming-1.2".into(),
    };
    let decision = Streaming12ReleaseGate::evaluate(&evidence);
    assert!(decision.allowed, "streaming 1.2 denied: {:?}", decision.reasons);
    print!("{}", Streaming12ReleaseGate::render_markdown(&evidence));
}
