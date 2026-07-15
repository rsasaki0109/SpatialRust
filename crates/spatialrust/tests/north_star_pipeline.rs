//! North-star end-to-end smoke: image → AI → episode → MCAP/ROS2 → scene → interchange.
//!
//! Exercises portable deepenings (MCAP XYZ, ROS 2 CDR PointCloud2, USDA ASCII,
//! Gaussian CPU soft-splat) without install-time rclrs/libusd.

#![cfg(feature = "north-star-e2e")]

use spatialrust::ai::{
    CopyPolicy, InferenceBackend, MockInferenceBackend, MockProfile, ModelSource, NamedTensors,
    RunOptions, SessionOptions,
};
use spatialrust::distribute::{ExecutionPartition, PartitionGraph};
use spatialrust::episode::{Episode, ModelProvenance};
use spatialrust::interchange::{
    export_stage_usda, export_triangle_mesh_gltf_json, import_mesh_from_usda,
    MemoryUsdStageAdapter, UsdPrimPath, UsdStageAdapter,
};
use spatialrust::mapping::{PoseGraph, PoseGraphEdge, PoseNodeId, StampedPose, Trajectory};
use spatialrust::platform::{
    ApiStabilityClass, ConformanceReport, ConformanceStatus, LtsPolicy, ReleaseGate,
    SecurityChecklist, StabilityRegistry,
};
use spatialrust::records::{SchemaVersion, SpatialRecord};
use spatialrust::runtime::{
    decode_point_cloud2_xyz, encode_point_cloud2_xyz, BoundedPipeline, LoopbackRos2Node,
    PipelineConfig, PipelineStage, PointCloud2Xyz, POINT_CLOUD2_TYPE,
};
use spatialrust::scene::{
    render_gaussians_cpu, GaussianCamera, GaussianPrimitive, GaussianScene, TsdfVolume,
};
use spatialrust::semantic::{
    Embedding, EntityId, MultimodalFusion, OpenVocabLabel, SemanticEntity, SemanticSearchIndex,
};
use spatialrust::sync::{
    read_memory_episode_mcap, write_memory_episode_mcap, ClockDomain, DeterministicReplayer,
    MemoryEpisode, StampedRecord, StampedTime, TopicId,
};
use spatialrust::{
    depth_map_to_point_cloud, depth_tensor_to_depth_map, rgb_u8_to_nchw_f32, CameraIntrinsics,
    DepthConversionOptions, Image, Interpolation, Isometry3, PinholeCamera, PointCloud, Pose3, Quat,
    Timestamp, Vec3,
};

#[test]
fn north_star_image_to_gltf_pipeline() {
    let started = std::time::Instant::now();
    // 1) Image → mock depth → XYZ cloud
    let width = 24usize;
    let height = 24usize;
    let mut pixels = vec![30_u8; width * height * 3];
    for y in 8..16 {
        for x in 8..16 {
            let index = (y * width + x) * 3;
            pixels[index..index + 3].copy_from_slice(&[200, 200, 200]);
        }
    }
    let image = Image::<u8, 3>::try_new(width, height, pixels).unwrap();
    let (tensor, _) = rgb_u8_to_nchw_f32(
        image.view(),
        width,
        height,
        Interpolation::Nearest,
        [0, 0, 0],
        1.0 / 255.0,
        [0.0; 3],
        [1.0; 3],
    )
    .unwrap();

    let mut session = MockInferenceBackend
        .create_session(
            &ModelSource::Mock(MockProfile::SyntheticDepth),
            &SessionOptions::default(),
        )
        .unwrap();
    let mut inputs = NamedTensors::new();
    inputs.insert("images", tensor).unwrap();
    let outputs = session
        .run_with_options(
            inputs,
            RunOptions {
                input_copy: CopyPolicy::Forbid,
                output_copy: CopyPolicy::Allow,
            },
        )
        .unwrap();
    let depth = depth_tensor_to_depth_map(outputs.get("depth").unwrap()).unwrap();
    let camera = PinholeCamera::new(
        CameraIntrinsics::try_new(
            width as f64,
            height as f64,
            (width as f64 - 1.0) * 0.5,
            (height as f64 - 1.0) * 0.5,
            width,
            height,
        )
        .unwrap(),
    );
    let cloud =
        depth_map_to_point_cloud(&depth, &camera, DepthConversionOptions::default()).unwrap();
    assert!(!cloud.is_empty());
    let xyz = collect_xyz(&cloud);

    // 2) Versioned record → stamped multimodal episode
    let record = SpatialRecord::try_from_cloud("point", SchemaVersion::new(1, 0), cloud.clone())
        .unwrap();
    let stamp = StampedTime::exact("host", ClockDomain::HostSteady, Timestamp::from_nanos(1_000));
    let stamped = StampedRecord::new("camera/depth_cloud", stamp.clone(), record);
    let memory = MemoryEpisode::from_records(vec![stamped]);
    let mut episode = Episode::try_new("north-star-demo", memory).unwrap();
    episode.provenance.push(ModelProvenance {
        model: "mock-synthetic-depth".into(),
        revision: "epic90".into(),
        dataset: Some("synthetic-rgb".into()),
    });
    let mut replayer = DeterministicReplayer::new(&episode.memory);
    assert!(replayer.next_record().is_some());

    // 2b) MCAP file round-trip for the same episode
    let mcap_path = std::env::temp_dir().join(format!(
        "spatialrust-north-star-{}-{}.mcap",
        std::process::id(),
        stamp.as_nanos()
    ));
    write_memory_episode_mcap(&mcap_path, &episode.memory).unwrap();
    let from_mcap = read_memory_episode_mcap(&mcap_path).unwrap();
    let _ = std::fs::remove_file(&mcap_path);
    assert_eq!(from_mcap.records().len(), episode.memory.records().len());

    // 2c) ROS 2 CDR PointCloud2 loopback (no rclrs)
    let pc2 = PointCloud2Xyz::try_new("camera", 0, 1_000, xyz.clone()).unwrap();
    let cdr = encode_point_cloud2_xyz(&pc2).unwrap();
    let mut ros_node = LoopbackRos2Node::new();
    ros_node.publish("/camera/points", cdr);
    let taken = ros_node.take("/camera/points").unwrap();
    let decoded = decode_point_cloud2_xyz(&taken).unwrap();
    assert_eq!(decoded.xyz.len(), xyz.len());
    assert_eq!(POINT_CLOUD2_TYPE, "sensor_msgs/msg/PointCloud2");

    // 3) Localization substrate: trajectory + pose graph
    let mut traj = Trajectory::new();
    let pose = Pose3::new(Isometry3::new(Quat::new(0.0, 0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 0.0)));
    traj.push(StampedPose::new(stamp.clone(), pose)).unwrap();
    traj.push(StampedPose::new(
        StampedTime::exact("host", ClockDomain::HostSteady, Timestamp::from_nanos(2_000)),
        Pose3::new(Isometry3::new(Quat::new(0.0, 0.0, 0.0, 1.0), Vec3::new(0.1, 0.0, 0.0))),
    ))
    .unwrap();
    assert!((traj.interpolate(1_500).unwrap().isometry.translation().x - 0.05).abs() < 1e-4);

    let mut graph = PoseGraph::new();
    graph.upsert_node("n0", traj.samples()[0].clone());
    graph.upsert_node("n1", traj.samples()[1].clone());
    graph
        .add_edge(PoseGraphEdge {
            from: PoseNodeId::new("n0"),
            to: PoseNodeId::new("n1"),
            to_t_from: Isometry3::new(Quat::new(0.0, 0.0, 0.0, 1.0), Vec3::new(0.1, 0.0, 0.0)),
            loop_closure: false,
        })
        .unwrap();
    graph.localize_from_root(&PoseNodeId::new("n0")).unwrap();

    // 4) TSDF → mesh → glTF + USDA
    let mut volume =
        TsdfVolume::try_new(Vec3::new(-2.0, -2.0, 0.0), 0.2, [20, 20, 20], 0.5).unwrap();
    volume.integrate_xyz(&xyz, Vec3::new(0.0, 0.0, 0.0)).unwrap();
    let mesh = volume.extract_mesh(0.5);
    assert!(!mesh.is_empty());
    let gltf = export_triangle_mesh_gltf_json(&mesh).unwrap();
    assert!(gltf.contains("\"asset\""));
    assert!(gltf.contains("VEC3"));

    let mut usd = MemoryUsdStageAdapter::new("north_star.usda");
    usd.declare_mesh(UsdPrimPath::try_new("/World/TsdfMesh").unwrap(), &mesh)
        .unwrap();
    let usda = export_stage_usda(&usd).unwrap();
    assert!(usda.starts_with("#usda 1.0"));
    let (prim, imported) = import_mesh_from_usda(&usda).unwrap();
    assert_eq!(prim.0, "/World/TsdfMesh");
    assert_eq!(imported.triangle_count(), mesh.triangle_count());

    // 4b) Gaussian CPU soft-splat from a few cloud samples
    let mut gaussians = GaussianScene::new();
    for chunk in xyz.chunks_exact(3).take(8) {
        gaussians
            .push(GaussianPrimitive {
                mean: Vec3::new(chunk[0], chunk[1], chunk[2].max(0.5)),
                scale: Vec3::new(0.08, 0.08, 0.08),
                rotation: Quat::<f32>::identity(),
                opacity: 0.9,
                color: [0.2, 0.7, 1.0],
            })
            .unwrap();
    }
    let gauss_cam = GaussianCamera::look_along_z(48, 48, 60.0, 60.0);
    let fb = render_gaussians_cpu(&gaussians, &gauss_cam).unwrap();
    assert_eq!(fb.rgba.len(), 48 * 48 * 4);
    assert!(fb.rgba.iter().any(|c| *c > 0));

    // 5) Semantic search on reconstructed entity
    let mut index = SemanticSearchIndex::new();
    index.insert(SemanticEntity {
        id: EntityId::new("surface"),
        centroid: Some(Vec3::new(0.0, 0.0, 1.0)),
        labels: vec![OpenVocabLabel {
            text: "plane".into(),
            confidence: 0.8,
        }],
        embedding: Some(Embedding::try_new(vec![1.0, 0.0, 0.0]).unwrap()),
    });
    let hits = index
        .search(
            &Embedding::try_new(vec![1.0, 0.0, 0.0]).unwrap(),
            MultimodalFusion::default(),
            1,
        )
        .unwrap();
    assert_eq!(hits[0].0, EntityId::new("surface"));

    // 6) Bounded runtime + distribute graph
    let mut pipeline = BoundedPipeline::new(PipelineConfig { max_inflight: 4 });
    pipeline
        .push(PipelineStage::new("reconstruct"), mesh.triangle_count())
        .unwrap();
    assert_eq!(pipeline.pop().unwrap().1, mesh.triangle_count());

    let mut partitions = PartitionGraph::new();
    partitions
        .insert_partition(ExecutionPartition {
            id: "edge".into(),
            nodes: vec!["camera".into()],
        })
        .unwrap();
    partitions
        .insert_partition(ExecutionPartition {
            id: "host".into(),
            nodes: vec!["scene".into()],
        })
        .unwrap();
    partitions.connect("edge", "host").unwrap();

    // 7) Platform release gate: stability + conformance + security + LTS + budgets
    let mut gate = ReleaseGate::north_star_defaults();
    gate.stability = Some({
        let mut registry = StabilityRegistry::north_star_surface();
        registry.register("spatialrust-scene::TsdfVolume", ApiStabilityClass::Provisional);
        registry
    });
    assert_eq!(LtsPolicy::spatialrust_v1().window_for("1.x").unwrap().total_months(), 24);

    let mut report = ConformanceReport::new();
    report.record(
        "north-star-e2e",
        ConformanceStatus::Pass,
        Some(TopicId::new("camera/depth_cloud").0),
    );
    report.record("north-star-e2e-mcap", ConformanceStatus::Pass, None);
    report.record("north-star-e2e-ros2-cdr", ConformanceStatus::Pass, None);
    report.record("north-star-e2e-usda", ConformanceStatus::Pass, None);
    report.record("north-star-e2e-gaussian-cpu", ConformanceStatus::Pass, None);
    report.assert_no_failures().unwrap();
    assert!(report.pass_count() >= 5);
    gate.conformance = Some(report);
    gate.security = Some(SecurityChecklist::north_star_baseline_satisfied());
    if let Some(budgets) = gate.budgets.as_mut() {
        let elapsed_ms = started.elapsed().as_millis() as u64;
        budgets.sample("north-star-e2e-latency-ms", elapsed_ms);
        budgets.sample(
            "north-star-e2e-bytes-copied",
            (xyz.len() * 4 + mesh.positions.len() * 4 + fb.rgba.len()) as u64,
        );
    }
    gate.assert_allowed().unwrap();
}

fn collect_xyz(cloud: &PointCloud) -> Vec<f32> {
    let xs = cloud.field("x").unwrap().as_f32().unwrap();
    let ys = cloud.field("y").unwrap().as_f32().unwrap();
    let zs = cloud.field("z").unwrap().as_f32().unwrap();
    let mut out = Vec::with_capacity(xs.len() * 3);
    for i in 0..xs.len() {
        out.extend_from_slice(&[xs[i], ys[i], zs[i]]);
    }
    out
}
