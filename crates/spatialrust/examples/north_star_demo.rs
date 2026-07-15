//! North-star facade demo: RGB → mock depth → episode → TSDF mesh → glTF JSON.
//!
//! Run with:
//! `cargo run -p spatialrust --example north_star_demo --features north-star-e2e`

use spatialrust::ai::{
    CopyPolicy, InferenceBackend, MockInferenceBackend, MockProfile, ModelSource, NamedTensors,
    RunOptions, SessionOptions,
};
use spatialrust::episode::{Episode, ModelProvenance};
use spatialrust::interchange::export_triangle_mesh_gltf_json;
use spatialrust::records::{SchemaVersion, SpatialRecord};
use spatialrust::scene::TsdfVolume;
use spatialrust::sync::{ClockDomain, MemoryEpisode, StampedRecord, StampedTime};
use spatialrust::{
    depth_map_to_point_cloud, depth_tensor_to_depth_map, rgb_u8_to_nchw_f32, CameraIntrinsics,
    DepthConversionOptions, Image, Interpolation, PinholeCamera, PointCloud, Timestamp, Vec3,
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
        .create_session(
            &ModelSource::Mock(MockProfile::SyntheticDepth),
            &SessionOptions::default(),
        )
        .expect("session");
    let mut inputs = NamedTensors::new();
    inputs.insert("images", tensor).expect("inputs");
    let outputs = session
        .run_with_options(
            inputs,
            RunOptions {
                input_copy: CopyPolicy::Forbid,
                output_copy: CopyPolicy::Allow,
            },
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

    let record =
        SpatialRecord::try_from_cloud("point", SchemaVersion::new(1, 0), cloud.clone()).expect("record");
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

    let mut volume =
        TsdfVolume::try_new(Vec3::new(-2.0, -2.0, 0.0), 0.25, [16, 16, 16], 0.5).expect("tsdf");
    volume
        .integrate_xyz(&collect_xyz(&cloud), Vec3::new(0.0, 0.0, 0.0))
        .expect("integrate");
    let mesh = volume.extract_mesh(0.5);
    let gltf = export_triangle_mesh_gltf_json(&mesh).expect("gltf");

    println!("episode={}", episode.id.0);
    println!("points={}", cloud.len());
    println!("triangles={}", mesh.triangle_count());
    println!("gltf_bytes={}", gltf.len());
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
