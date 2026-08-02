//! North-star facade demo: RGB → depth → MCAP/ROS2 → TSDF/USDA/Gaussian.
//!
//! Run with:
//! `cargo run -p spatialrust --example north_star_demo --features north-star-e2e`

use spatialrust::ai::{
    CopyPolicy, InferenceBackend, MockInferenceBackend, MockProfile, ModelSource, NamedTensors,
    RunOptions, SessionOptions,
};
use spatialrust::episode::{Episode, ModelProvenance};
use spatialrust::interchange::{
    export_stage_usda, export_triangle_mesh_gltf_json, MemoryUsdStageAdapter, UsdPrimPath,
    UsdStageAdapter,
};
use spatialrust::records::{SchemaVersion, SpatialRecord};
use spatialrust::runtime::{
    decode_point_cloud2_xyz, encode_point_cloud2_xyz, LoopbackRos2Node, PointCloud2Xyz,
};
use spatialrust::scene::{
    render_gaussians_cpu, GaussianCamera, GaussianPrimitive, GaussianScene, TsdfVolume,
};
use spatialrust::sync::{
    read_memory_episode_mcap, write_memory_episode_mcap, ClockDomain, MemoryEpisode, StampedRecord,
    StampedTime,
};
use spatialrust::{
    depth_map_to_point_cloud, depth_tensor_to_depth_map, rgb_u8_to_nchw_f32, CameraIntrinsics,
    DepthConversionOptions, Image, Interpolation, PinholeCamera, PointCloud, Quat, Timestamp, Vec3,
};

fn main() {
    let width = 32usize;
    let height = 32usize;
    let pixels = vec![180_u8; width * height * 3];
    let image = Image::<u8, 3>::try_new(width, height, pixels).expect("image");
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
    .expect("nchw");

    let mut session = MockInferenceBackend
        .create_session(&ModelSource::Mock(MockProfile::SyntheticDepth), &SessionOptions::default())
        .expect("session");
    let mut inputs = NamedTensors::new();
    inputs.insert("images", tensor).expect("inputs");
    let outputs = session
        .run_with_options(
            inputs,
            RunOptions { input_copy: CopyPolicy::Forbid, output_copy: CopyPolicy::Allow },
        )
        .expect("infer");
    let depth = depth_tensor_to_depth_map(outputs.get("depth").expect("depth")).expect("decode");
    let camera = PinholeCamera::new(
        CameraIntrinsics::try_new(
            width as f64,
            height as f64,
            (width as f64 - 1.0) * 0.5,
            (height as f64 - 1.0) * 0.5,
            width,
            height,
        )
        .expect("intrinsics"),
    );
    let cloud =
        depth_map_to_point_cloud(&depth, &camera, DepthConversionOptions::default()).expect("xyz");
    let xyz = collect_xyz(&cloud);

    let record = SpatialRecord::try_from_cloud("point", SchemaVersion::new(1, 0), cloud.clone())
        .expect("record");
    let stamped = StampedRecord::new(
        "camera/depth_cloud",
        StampedTime::exact("host", ClockDomain::HostSteady, Timestamp::from_nanos(1)),
        record,
    );
    let mut episode =
        Episode::try_new("north-star-demo", MemoryEpisode::from_records(vec![stamped]))
            .expect("episode");
    episode.provenance.push(ModelProvenance {
        model: "mock-synthetic-depth".into(),
        revision: "demo".into(),
        dataset: None,
    });

    let mcap_path =
        std::env::temp_dir().join(format!("spatialrust-demo-{}-{}.mcap", std::process::id(), 1));
    write_memory_episode_mcap(&mcap_path, &episode.memory).expect("mcap write");
    let mcap_episode = read_memory_episode_mcap(&mcap_path).expect("mcap read");
    let _ = std::fs::remove_file(&mcap_path);

    let pc2 = PointCloud2Xyz::try_new("camera", 0, 1, xyz.clone()).expect("pc2");
    let cdr = encode_point_cloud2_xyz(&pc2).expect("cdr");
    let mut node = LoopbackRos2Node::new();
    node.publish("/camera/points", cdr);
    let decoded =
        decode_point_cloud2_xyz(&node.take("/camera/points").expect("take")).expect("decode");

    let mut volume =
        TsdfVolume::try_new(Vec3::new(-2.0, -2.0, 0.0), 0.25, [16, 16, 16], 0.5).expect("tsdf");
    volume.integrate_xyz(&xyz, Vec3::new(0.0, 0.0, 0.0)).expect("integrate");
    let mesh = volume.extract_mesh(0.5);
    let gltf = export_triangle_mesh_gltf_json(&mesh).expect("gltf");

    let mut usd = MemoryUsdStageAdapter::new("demo.usda");
    usd.declare_mesh(UsdPrimPath::try_new("/World/Mesh").unwrap(), &mesh).expect("usd mesh");
    let usda = export_stage_usda(&usd).expect("usda");

    let mut gaussians = GaussianScene::new();
    for chunk in xyz.chunks_exact(3).take(4) {
        gaussians
            .push(GaussianPrimitive {
                mean: Vec3::new(chunk[0], chunk[1], chunk[2].max(0.5)),
                scale: Vec3::new(0.1, 0.1, 0.1),
                rotation: Quat::<f32>::identity(),
                opacity: 0.85,
                color: [1.0, 0.4, 0.1],
            })
            .expect("gaussian");
    }
    let fb = render_gaussians_cpu(&gaussians, &GaussianCamera::look_along_z(64, 64, 80.0, 80.0))
        .expect("render");

    println!("episode={}", episode.id.0);
    println!("mcap_records={}", mcap_episode.records().len());
    println!("ros2_points={}", decoded.point_count());
    println!("points={}", cloud.len());
    println!("triangles={}", mesh.triangle_count());
    println!("gltf_bytes={}", gltf.len());
    println!("usda_bytes={}", usda.len());
    println!("gaussian_rgba={}", fb.rgba.len());
    println!("gltf_head={}", &gltf[..gltf.len().min(120)]);
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
