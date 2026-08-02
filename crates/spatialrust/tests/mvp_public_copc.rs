//! Public-dataset COPC validation (Epic 60).
//!
//! Requires the public PCL `table_scene_lms400.pcd` sample under
//! `target/bench-data/` (or `SPATIALRUST_PUBLIC_PCD`).

#![cfg(feature = "mvp")]

use std::path::PathBuf;

fn public_pcd_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("SPATIALRUST_PUBLIC_PCD") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for relative in [
        "../../target/bench-data/table_scene_lms400.pcd",
        "../../target/readme-data/table_scene_lms400.pcd",
    ] {
        let candidate = manifest.join(relative);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn skip_without_public_pcd() -> Option<PathBuf> {
    public_pcd_path().or_else(|| {
        eprintln!(
            "skipping public COPC validation (PCD not found); fetch with:\n  \
             python bench/public_copc/run.py --fetch-only"
        );
        None
    })
}

fn inner_roi_bounds(cloud: &spatialrust::PointCloud, fraction: f32) -> spatialrust::CopcBounds {
    use spatialrust::HasPositions3;

    let (x, y, z) = cloud.positions3().expect("positions");
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for i in 0..cloud.len() {
        min[0] = min[0].min(x[i]);
        min[1] = min[1].min(y[i]);
        min[2] = min[2].min(z[i]);
        max[0] = max[0].max(x[i]);
        max[1] = max[1].max(y[i]);
        max[2] = max[2].max(z[i]);
    }
    let center = [0.5 * (min[0] + max[0]), 0.5 * (min[1] + max[1]), 0.5 * (min[2] + max[2])];
    let half = [
        0.5 * fraction * (max[0] - min[0]),
        0.5 * fraction * (max[1] - min[1]),
        0.5 * fraction * (max[2] - min[2]),
    ];
    spatialrust::CopcBounds::new(
        [
            f64::from(center[0] - half[0]),
            f64::from(center[1] - half[1]),
            f64::from(center[2] - half[2]),
        ],
        [
            f64::from(center[0] + half[0]),
            f64::from(center[1] + half[1]),
            f64::from(center[2] + half[2]),
        ],
    )
}

#[cfg(all(feature = "mvp", feature = "mvp-http"))]
fn inner_xy_bounds(bounds: spatialrust::CopcBounds, fraction: f64) -> spatialrust::CopcBounds {
    let center_x = 0.5 * (bounds.min[0] + bounds.max[0]);
    let center_y = 0.5 * (bounds.min[1] + bounds.max[1]);
    let half_x = 0.5 * fraction * (bounds.max[0] - bounds.min[0]);
    let half_y = 0.5 * fraction * (bounds.max[1] - bounds.min[1]);
    spatialrust::CopcBounds::from_ranges(
        (center_x - half_x, center_x + half_x),
        (center_y - half_y, center_y + half_y),
        (bounds.min[2], bounds.max[2]),
    )
}

#[cfg(feature = "mvp")]
#[test]
fn mvp_public_pcd_copc_bounds_resolution_and_pipeline() {
    use spatialrust::{
        read_copc_file, read_copc_file_info, read_copc_file_with_query, read_point_cloud_file,
        write_copc_file_with_params, CopcQuery, CopcWriterParams, MvpPipeline, MvpPipelineConfig,
    };

    let Some(pcd_path) = skip_without_public_pcd() else {
        return;
    };
    let cloud = read_point_cloud_file(&pcd_path).expect("read public PCD");
    assert!(cloud.len() > 10_000, "unexpected point count: {}", cloud.len());

    let copc_path = std::env::temp_dir()
        .join(format!("spatialrust_public_copc_{}.copc.laz", std::process::id()));
    write_copc_file_with_params(
        &copc_path,
        &cloud,
        &CopcWriterParams { max_points_per_node: 512, max_depth: 10 },
    )
    .expect("write public COPC");

    let info = read_copc_file_info(&copc_path).expect("copc info");
    let full_count = read_copc_file(&copc_path).expect("full read").len();
    let roi_bounds = inner_roi_bounds(&cloud, 0.6);
    let coarse_resolution = info.spacing * 4.0;

    let bounds_only_count = read_copc_file_with_query(&copc_path, &CopcQuery::bounds(roi_bounds))
        .expect("bounds query")
        .len();
    let combined_count = read_copc_file_with_query(
        &copc_path,
        &CopcQuery::with_resolution(roi_bounds, coarse_resolution),
    )
    .expect("bounds+resolution query")
    .len();

    assert_eq!(full_count, cloud.len());
    assert!(bounds_only_count < full_count, "{bounds_only_count} vs {full_count}");
    assert!(combined_count <= bounds_only_count, "{combined_count} vs {bounds_only_count}");
    assert!(combined_count < full_count, "{combined_count} vs {full_count}");

    let queried = read_copc_file_with_query(
        &copc_path,
        &CopcQuery::with_resolution(info.root_bounds, coarse_resolution),
    )
    .expect("root resolution query");
    let pipeline = MvpPipeline::new(MvpPipelineConfig::default());
    let result = pipeline.run(&queried).expect("mvp pipeline on public COPC query");
    assert!(result.plane.inlier_count > 0, "expected plane inliers on public scene");
    assert!(result.clusters.cluster_count >= 1, "expected at least one cluster");

    eprintln!("public PCD COPC validation ({})", pcd_path.display());
    eprintln!("  source points     : {full_count}");
    eprintln!("  roi bounds points : {bounds_only_count}");
    eprintln!("  roi+res points    : {combined_count}");
    eprintln!("  coarse spacing    : {coarse_resolution:.6}");
    eprintln!("  plane inliers     : {}", result.plane.inlier_count);
    eprintln!("  clusters          : {}", result.clusters.cluster_count);

    let _ = std::fs::remove_file(&copc_path);
}

#[cfg(all(feature = "mvp", feature = "mvp-http"))]
#[test]
#[ignore = "requires network; run bench/public_copc/run.py --http"]
fn mvp_http_autzen_copc_bounds_smoke() {
    use spatialrust::{read_copc_url_info, read_copc_url_with_query, CopcQuery};

    const AUTZEN_URL: &str = "https://s3.amazonaws.com/hobu-lidar/autzen-classified.copc.laz";

    let info = read_copc_url_info(AUTZEN_URL).expect("autzen copc info");
    assert!(info.point_count > 0, "Autzen COPC header reported no points");
    let roi = inner_xy_bounds(info.root_bounds, 0.25);
    let bounds_count = read_copc_url_with_query(AUTZEN_URL, Some(&CopcQuery::bounds(roi)))
        .expect("autzen bounds")
        .len();

    assert!(bounds_count > 0, "central Autzen ROI returned no points");
    assert!(
        u64::try_from(bounds_count).unwrap() < info.point_count,
        "{bounds_count} vs {}",
        info.point_count
    );
    eprintln!("HTTP Autzen COPC smoke");
    eprintln!("  header points : {}", info.point_count);
    eprintln!("  roi points    : {bounds_count}");
}
