//! Generates the README preview image from a real MVP pipeline run.
//!
//! ```bash
//! cargo run -p spatialrust --features mvp --example readme_mvp_preview
//! ```

use std::{fmt::Write as _, fs, path::PathBuf};

use spatialrust::{
    EuclideanClusterConfig, HasPositions3, MvpPipeline, MvpPipelineConfig, NormalEstimationConfig,
    PointCloud, PointCloudBuilder, RansacPlaneConfig, Vec3,
};

fn sample_scene() -> PointCloud {
    let mut builder = PointCloudBuilder::xyz();
    for x in 0..10 {
        for y in 0..10 {
            builder
                .push_point([x as f32 * 0.1, y as f32 * 0.1, 0.0])
                .unwrap();
        }
    }
    for point in [(0.0, 0.0, 0.5), (0.1, 0.0, 0.5), (0.0, 0.1, 0.5)] {
        builder.push_point([point.0, point.1, point.2]).unwrap();
    }
    builder.build().unwrap()
}

fn label_color(label: i32) -> &'static str {
    match label {
        0 => "#f97316",
        1 => "#06b6d4",
        2 => "#a855f7",
        3 => "#22c55e",
        _ => "#64748b",
    }
}

fn bounds_xy(cloud: &PointCloud) -> ([f32; 2], [f32; 2]) {
    let (x, y, _) = cloud.positions3().expect("positions");
    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    for index in 0..cloud.len() {
        min[0] = min[0].min(x[index]);
        min[1] = min[1].min(y[index]);
        max[0] = max[0].max(x[index]);
        max[1] = max[1].max(y[index]);
    }
    (min, max)
}

fn merge_bounds(
    (mut min, mut max): ([f32; 2], [f32; 2]),
    other: ([f32; 2], [f32; 2]),
) -> ([f32; 2], [f32; 2]) {
    for axis in 0..2 {
        min[axis] = min[axis].min(other.0[axis]);
        max[axis] = max[axis].max(other.1[axis]);
    }
    (min, max)
}

fn project(x: f32, y: f32, min: [f32; 2], max: [f32; 2], width: f32, height: f32, pad: f32) -> (f32, f32) {
    let span_x = (max[0] - min[0]).max(1e-3);
    let span_y = (max[1] - min[1]).max(1e-3);
    let scale = ((width - 2.0 * pad) / span_x).min((height - 2.0 * pad) / span_y);
    let px = pad + (x - min[0]) * scale;
    let py = height - pad - (y - min[1]) * scale;
    (px, py)
}

fn write_plane_points(svg: &mut String, cloud: &PointCloud, min: [f32; 2], max: [f32; 2]) {
    let (x, y, _) = cloud.positions3().expect("plane positions");
    for index in 0..cloud.len() {
        let (px, py) = project(x[index], y[index], min, max, 320.0, 240.0, 24.0);
        let _ = write!(
            svg,
            r##"<circle cx="{px:.2}" cy="{py:.2}" r="3.2" fill="#cbd5e1" fill-opacity="0.85"/>"##,
        );
    }
}

fn write_cluster_points(svg: &mut String, cloud: &PointCloud, min: [f32; 2], max: [f32; 2]) {
    let (x, y, _) = cloud.positions3().expect("cluster positions");
    let labels = match cloud.field("label").expect("labels") {
        spatialrust::PointBuffer::I32(values) => values.as_slice(),
        other => panic!("expected i32 labels, got {:?}", other.dtype()),
    };
    for index in 0..cloud.len() {
        let (px, py) = project(x[index], y[index], min, max, 320.0, 240.0, 24.0);
        let color = label_color(labels[index]);
        let _ = write!(
            svg,
            r#"<circle cx="{px:.2}" cy="{py:.2}" r="4.6" fill="{color}"/>"#,
        );
    }
}

fn render_svg(plane: &PointCloud, clusters: &PointCloud, cluster_count: usize) -> String {
    let bounds = merge_bounds(bounds_xy(plane), bounds_xy(clusters));
    let (min, max) = bounds;

    let mut svg = String::new();
    svg.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    svg.push('\n');
    svg.push_str(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="960" height="320" viewBox="0 0 960 320" role="img" aria-labelledby="title desc">"#,
    );
    svg.push('\n');
    svg.push_str(r#"<title id="title">SpatialRust MVP pipeline preview</title>"#);
    svg.push('\n');
    svg.push_str(r#"<desc id="desc">Top-down XY view of plane inliers and Euclidean cluster labels produced by the MVP pipeline.</desc>"#);
    svg.push('\n');
    svg.push_str(r##"<rect width="960" height="320" fill="#0f172a"/>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="24" y="34" fill="#e2e8f0" font-family="ui-sans-serif, system-ui, sans-serif" font-size="20" font-weight="700">SpatialRust MVP pipeline</text>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="24" y="58" fill="#94a3b8" font-family="ui-sans-serif, system-ui, sans-serif" font-size="13">Real output from cargo run --example readme_mvp_preview</text>"##);
    svg.push('\n');

    svg.push_str(r#"<g transform="translate(24 72)">"#);
    svg.push('\n');
    svg.push_str(r##"<rect x="0" y="0" width="320" height="240" rx="12" fill="#111827" stroke="#334155"/>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="16" y="24" fill="#cbd5e1" font-family="ui-sans-serif, system-ui, sans-serif" font-size="14" font-weight="600">Plane inliers (RANSAC)</text>"##);
    svg.push('\n');
    write_plane_points(&mut svg, plane, min, max);
    svg.push('\n');
    svg.push_str("</g>\n");

    svg.push_str(r#"<g transform="translate(360 72)">"#);
    svg.push('\n');
    svg.push_str(r##"<rect x="0" y="0" width="320" height="240" rx="12" fill="#111827" stroke="#334155"/>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="16" y="24" fill="#cbd5e1" font-family="ui-sans-serif, system-ui, sans-serif" font-size="14" font-weight="600">Cluster labels (Euclidean)</text>"##);
    svg.push('\n');
    write_cluster_points(&mut svg, clusters, min, max);
    svg.push('\n');
    svg.push_str("</g>\n");

    svg.push_str(r#"<g transform="translate(696 72)">"#);
    svg.push('\n');
    svg.push_str(r##"<rect x="0" y="0" width="240" height="240" rx="12" fill="#111827" stroke="#334155"/>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="16" y="24" fill="#cbd5e1" font-family="ui-sans-serif, system-ui, sans-serif" font-size="14" font-weight="600">Pipeline</text>"##);
    svg.push('\n');
    let stages = [
        "PCD / LAS / COPC",
        "Voxel downsample",
        "Normal estimation",
        "Plane RANSAC",
        "Euclidean cluster",
        "Optional ICP",
    ];
    for (index, stage) in stages.iter().enumerate() {
        let y = 52.0 + index as f32 * 28.0;
        let _ = write!(
            svg,
            r##"<text x="16" y="{y:.1}" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="12">{index}. {stage}</text>"##,
        );
    }
    let _ = write!(
        svg,
        r##"<text x="16" y="228" fill="#64748b" font-family="ui-sans-serif, system-ui, sans-serif" font-size="11">{cluster_count} cluster(s) in preview scene</text>"##,
    );
    svg.push('\n');
    svg.push_str("</g>\n");
    svg.push_str("</svg>\n");
    svg
}

fn output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/assets/readme_mvp_preview.svg")
}

fn main() {
    let result = MvpPipeline::new(MvpPipelineConfig {
        voxel: spatialrust::VoxelGridDownsampleConfig::centroid(0.2),
        normals: NormalEstimationConfig {
            k_neighbors: 8,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.0, 0.0, 10.0)),
            ..NormalEstimationConfig::default()
        },
        plane: RansacPlaneConfig {
            distance_threshold: 0.05,
            max_iterations: 500,
            min_inliers: 10,
            seed: 17,
        },
        cluster: EuclideanClusterConfig {
            cluster_tolerance: 0.3,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
        },
        icp: None,
        ..MvpPipelineConfig::default()
    })
    .run(&sample_scene())
    .expect("mvp pipeline preview run");

    let svg = render_svg(
        &result.plane.inliers,
        &result.output,
        result.clusters.cluster_count,
    );
    let path = output_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create docs/assets directory");
    }
    fs::write(&path, svg).expect("write readme preview svg");
    println!("wrote {}", path.display());
}
