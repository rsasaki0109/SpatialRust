//! Generates README marketing assets from a real MVP pipeline run.
//!
//! ```bash
//! cargo run -p spatialrust --features mvp --example readme_mvp_preview
//! ```

use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use spatialrust::{
    EuclideanClusterConfig, HasPositions3, MvpPipeline, MvpPipelineConfig, MvpPipelineResult,
    NormalEstimationConfig, PointCloud, PointCloudBuilder, RansacPlaneConfig, Vec3,
};

const GIF_WIDTH: u32 = 720;
const GIF_HEIGHT: u32 = 480;

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

fn pipeline_config() -> MvpPipelineConfig {
    MvpPipelineConfig {
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

fn project_f32(
    x: f32,
    y: f32,
    min: [f32; 2],
    max: [f32; 2],
    width: f32,
    height: f32,
    pad: f32,
) -> (f32, f32) {
    let span_x = (max[0] - min[0]).max(1e-3);
    let span_y = (max[1] - min[1]).max(1e-3);
    let scale = ((width - 2.0 * pad) / span_x).min((height - 2.0 * pad) / span_y);
    let px = pad + (x - min[0]) * scale;
    let py = height - pad - (y - min[1]) * scale;
    (px, py)
}

fn project(
    x: f32,
    y: f32,
    min: [f32; 2],
    max: [f32; 2],
    width: f32,
    height: f32,
    pad: f32,
) -> (i32, i32) {
    let (px, py) = project_f32(x, y, min, max, width, height, pad);
    (px.round() as i32, py.round() as i32)
}

#[derive(Clone, Copy)]
struct Rgb(u8, u8, u8);

struct Canvas {
    pixels: Vec<u8>,
}

impl Canvas {
    fn new(width: u32, height: u32, background: Rgb) -> Self {
        let mut pixels = vec![0_u8; (width * height * 3) as usize];
        for chunk in pixels.chunks_mut(3) {
            chunk[0] = background.0;
            chunk[1] = background.1;
            chunk[2] = background.2;
        }
        Self { pixels }
    }

    fn put(&mut self, width: u32, height: u32, x: i32, y: i32, color: Rgb) {
        if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
            return;
        }
        let index = ((y as u32 * width + x as u32) * 3) as usize;
        self.pixels[index] = color.0;
        self.pixels[index + 1] = color.1;
        self.pixels[index + 2] = color.2;
    }

    fn fill_circle(&mut self, width: u32, height: u32, cx: i32, cy: i32, radius: i32, color: Rgb) {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    self.put(width, height, cx + dx, cy + dy, color);
                }
            }
        }
    }

    fn draw_stage_bar(&mut self, width: u32, stage: usize) {
        let bar_y = 18;
        for index in 0..4 {
            let x0 = 24 + index as u32 * 168;
            let active = index == stage;
            let color = if active {
                Rgb(56, 189, 248)
            } else {
                Rgb(51, 65, 85)
            };
            for x in x0..x0 + 150 {
                for y in bar_y..bar_y + 6 {
                    self.put(width, GIF_HEIGHT, x as i32, y as i32, color);
                }
            }
        }
    }

    fn write_ppm(&self, width: u32, height: u32, path: &Path) {
        let mut file = Vec::new();
        file.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
        file.extend_from_slice(&self.pixels);
        fs::write(path, file).expect("write ppm frame");
    }
}

fn label_rgb(label: i32) -> Rgb {
    match label {
        0 => Rgb(249, 115, 22),
        1 => Rgb(6, 182, 212),
        2 => Rgb(168, 85, 247),
        3 => Rgb(34, 197, 94),
        _ => Rgb(100, 116, 139),
    }
}

fn draw_points(
    canvas: &mut Canvas,
    cloud: &PointCloud,
    min: [f32; 2],
    max: [f32; 2],
    radius: i32,
    color: Rgb,
) {
    let (x, y, _) = cloud.positions3().expect("positions");
    for index in 0..cloud.len() {
        let (px, py) = project(x[index], y[index], min, max, GIF_WIDTH as f32, GIF_HEIGHT as f32, 48.0);
        canvas.fill_circle(GIF_WIDTH, GIF_HEIGHT, px, py, radius, color);
    }
}

fn draw_plane_stage(
    canvas: &mut Canvas,
    plane: &spatialrust::RansacPlaneSegmentation,
    min: [f32; 2],
    max: [f32; 2],
) {
    draw_points(
        canvas,
        &plane.inliers,
        min,
        max,
        4,
        Rgb(203, 213, 225),
    );
    draw_points(
        canvas,
        &plane.outliers,
        min,
        max,
        5,
        Rgb(251, 146, 60),
    );
}

fn draw_cluster_stage(
    canvas: &mut Canvas,
    output: &PointCloud,
    min: [f32; 2],
    max: [f32; 2],
) {
    let (x, y, _) = output.positions3().expect("positions");
    let labels = match output.field("label").expect("labels") {
        spatialrust::PointBuffer::I32(values) => values.as_slice(),
        other => panic!("expected i32 labels, got {:?}", other.dtype()),
    };
    for index in 0..output.len() {
        let (px, py) = project(x[index], y[index], min, max, GIF_WIDTH as f32, GIF_HEIGHT as f32, 48.0);
        canvas.fill_circle(GIF_WIDTH, GIF_HEIGHT, px, py, 6, label_rgb(labels[index]));
    }
}

fn render_gif_frames(input: &PointCloud, result: &MvpPipelineResult, temp_dir: &Path) {
    let bounds = merge_bounds(bounds_xy(input), bounds_xy(&result.output));
    let (min, max) = bounds;
    let mut frame_index = 0_u32;

    for (stage, repeats) in [(0, 8_usize), (1, 8), (2, 8), (3, 10)] {
        for _ in 0..repeats {
            let mut canvas = Canvas::new(GIF_WIDTH, GIF_HEIGHT, Rgb(15, 23, 42));
            canvas.draw_stage_bar(GIF_WIDTH, stage);
            match stage {
                0 => draw_points(
                    &mut canvas,
                    input,
                    min,
                    max,
                    3,
                    Rgb(148, 163, 184),
                ),
                1 => draw_points(
                    &mut canvas,
                    &result.downsampled,
                    min,
                    max,
                    4,
                    Rgb(226, 232, 240),
                ),
                2 => draw_plane_stage(&mut canvas, &result.plane, min, max),
                3 => draw_cluster_stage(&mut canvas, &result.output, min, max),
                _ => unreachable!(),
            }
            let path = temp_dir.join(format!("frame_{frame_index:03}.ppm"));
            canvas.write_ppm(GIF_WIDTH, GIF_HEIGHT, &path);
            frame_index += 1;
        }
    }
}

fn encode_gif(temp_dir: &Path, output: &Path) {
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate",
            "8",
            "-i",
        ])
        .arg(temp_dir.join("frame_%03d.ppm"))
        .args([
            "-vf",
            "split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer",
            "-loop",
            "0",
        ])
        .arg(output)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg gif encode failed");
}

fn label_color_hex(label: i32) -> &'static str {
    match label {
        0 => "#f97316",
        1 => "#06b6d4",
        2 => "#a855f7",
        3 => "#22c55e",
        _ => "#64748b",
    }
}

fn write_plane_points_svg(svg: &mut String, cloud: &PointCloud, min: [f32; 2], max: [f32; 2]) {
    let (x, y, _) = cloud.positions3().expect("plane positions");
    for index in 0..cloud.len() {
        let (px, py) = project_f32(x[index], y[index], min, max, 320.0, 240.0, 24.0);
        let _ = write!(
            svg,
            r##"<circle cx="{px:.2}" cy="{py:.2}" r="3.2" fill="#cbd5e1" fill-opacity="0.85"/>"##,
        );
    }
}

fn write_cluster_points_svg(svg: &mut String, cloud: &PointCloud, min: [f32; 2], max: [f32; 2]) {
    let (x, y, _) = cloud.positions3().expect("cluster positions");
    let labels = match cloud.field("label").expect("labels") {
        spatialrust::PointBuffer::I32(values) => values.as_slice(),
        other => panic!("expected i32 labels, got {:?}", other.dtype()),
    };
    for index in 0..cloud.len() {
        let (px, py) = project_f32(x[index], y[index], min, max, 320.0, 240.0, 24.0);
        let color = label_color_hex(labels[index]);
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
    write_plane_points_svg(&mut svg, plane, min, max);
    svg.push('\n');
    svg.push_str("</g>\n");

    svg.push_str(r#"<g transform="translate(360 72)">"#);
    svg.push('\n');
    svg.push_str(r##"<rect x="0" y="0" width="320" height="240" rx="12" fill="#111827" stroke="#334155"/>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="16" y="24" fill="#cbd5e1" font-family="ui-sans-serif, system-ui, sans-serif" font-size="14" font-weight="600">Cluster labels (Euclidean)</text>"##);
    svg.push('\n');
    write_cluster_points_svg(&mut svg, clusters, min, max);
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

fn render_social_card() -> String {
    r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="640" viewBox="0 0 1280 640" role="img" aria-labelledby="title">
  <title id="title">SpatialRust social preview</title>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="#111827"/>
    </linearGradient>
  </defs>
  <rect width="1280" height="640" fill="url(#bg)"/>
  <text x="80" y="150" fill="#f8fafc" font-family="ui-sans-serif, system-ui, sans-serif" font-size="72" font-weight="800">SpatialRust</text>
  <text x="80" y="220" fill="#38bdf8" font-family="ui-sans-serif, system-ui, sans-serif" font-size="34" font-weight="600">PyTorch for Spatial Computing</text>
  <text x="80" y="290" fill="#cbd5e1" font-family="ui-sans-serif, system-ui, sans-serif" font-size="28">Point clouds · wgpu · COPC · RANSAC · ICP · native Rust</text>
  <text x="80" y="380" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="24">cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- scan.las out.las</text>
  <rect x="860" y="110" width="340" height="420" rx="24" fill="#111827" stroke="#334155"/>
  <text x="890" y="160" fill="#e2e8f0" font-family="ui-sans-serif, system-ui, sans-serif" font-size="22" font-weight="700">MVP pipeline</text>
  <text x="890" y="205" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="18">1. IO (PCD/LAS/COPC)</text>
  <text x="890" y="245" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="18">2. Voxel downsample</text>
  <text x="890" y="285" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="18">3. Normal estimation</text>
  <text x="890" y="325" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="18">4. Plane RANSAC</text>
  <text x="890" y="365" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="18">5. Euclidean cluster</text>
  <text x="890" y="405" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="18">6. Optional ICP</text>
  <circle cx="980" cy="470" r="16" fill="#cbd5e1"/>
  <circle cx="1040" cy="470" r="16" fill="#cbd5e1"/>
  <circle cx="1100" cy="470" r="16" fill="#cbd5e1"/>
  <circle cx="980" cy="520" r="18" fill="#f97316"/>
</svg>
"##
    .to_string()
}

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/assets")
}

fn main() {
    let input = sample_scene();
    let result = MvpPipeline::new(pipeline_config())
        .run(&input)
        .expect("mvp pipeline preview run");

    let assets = assets_dir();
    fs::create_dir_all(&assets).expect("create docs/assets directory");

    let svg_path = assets.join("readme_mvp_preview.svg");
    fs::write(
        &svg_path,
        render_svg(
            &result.plane.inliers,
            &result.output,
            result.clusters.cluster_count,
        ),
    )
    .expect("write svg");

    let temp_dir = std::env::temp_dir().join(format!("spatialrust_readme_gif_{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).expect("create temp gif frames");
    render_gif_frames(&input, &result, &temp_dir);
    let gif_path = assets.join("readme_mvp_pipeline.gif");
    encode_gif(&temp_dir, &gif_path);
    let _ = fs::remove_dir_all(&temp_dir);

    let social_path = assets.join("social_preview.svg");
    fs::write(&social_path, render_social_card()).expect("write social preview");

    println!("wrote {}", svg_path.display());
    println!("wrote {}", gif_path.display());
    println!("wrote {}", social_path.display());
}
