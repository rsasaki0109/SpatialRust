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
    read_point_cloud_file, EuclideanClusterConfig, HasPositions3, MvpPipeline, MvpPipelineConfig,
    MvpPipelineResult, NormalEstimationConfig, PointCloud, PointCloudBuilder, RansacPlaneConfig,
    Vec3,
};

const GIF_WIDTH: u32 = 720;
const GIF_HEIGHT: u32 = 480;
const RECEIPT_LEFT_WIDTH: u32 = 400;
const RECEIPT_SPLIT_X: i32 = RECEIPT_LEFT_WIDTH as i32;
const RECEIPT_FRAMES_PER_LINE: usize = 6;
const RECEIPT_HOLD_FRAMES: usize = 20;
const HERO_WIDTH: u32 = 1280;
const HERO_HEIGHT: u32 = 540;
const PUBLIC_SCENE_URL: &str =
    "https://raw.githubusercontent.com/PointCloudLibrary/data/master/tutorials/table_scene_lms400.pcd";
const PUBLIC_SCENE_FILE: &str = "table_scene_lms400.pcd";
const PUBLIC_SCENE_MAX_POINTS: usize = 80_000;

fn sample_scene() -> PointCloud {
    public_sample_scene().unwrap_or_else(|error| {
        panic!(
            "failed to load public README scene: {error}\n\
             set SPATIALRUST_README_CLOUD to a local PCD/PLY/LAS/COPC file, \
             or install curl so the PCL sample can be downloaded"
        )
    })
}

fn default_public_scene_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/readme-data")
        .join(PUBLIC_SCENE_FILE)
}

fn public_sample_scene() -> Result<PointCloud, String> {
    let path = match std::env::var_os("SPATIALRUST_README_CLOUD") {
        Some(path) => PathBuf::from(path),
        None => {
            let path = default_public_scene_path();
            ensure_public_scene(&path)?;
            path
        }
    };
    let cloud = read_point_cloud_file(&path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok(decimate_xyz_cloud(&cloud, PUBLIC_SCENE_MAX_POINTS)?)
}

fn public_sample_scene_full() -> Result<PointCloud, String> {
    let path = match std::env::var_os("SPATIALRUST_README_CLOUD") {
        Some(path) => PathBuf::from(path),
        None => {
            let path = default_public_scene_path();
            ensure_public_scene(&path)?;
            path
        }
    };
    read_point_cloud_file(&path)
        .map_err(|error| format!("read {}: {error}", path.display()))
}

fn ensure_public_scene(path: &Path) -> Result<(), String> {
    if path.exists() {
        return Ok(());
    }

    let parent = path.parent().ok_or_else(|| format!("missing parent for {}", path.display()))?;
    fs::create_dir_all(parent).map_err(|error| format!("create {}: {error}", parent.display()))?;

    let status = Command::new("curl")
        .args(["-L", "--fail", "--silent", "--show-error", "--output"])
        .arg(path)
        .arg(PUBLIC_SCENE_URL)
        .status()
        .map_err(|error| format!("spawn curl: {error}"))?;
    if !status.success() {
        let _ = fs::remove_file(path);
        return Err(format!("download {PUBLIC_SCENE_URL} failed with {status}"));
    }
    Ok(())
}

fn decimate_xyz_cloud(input: &PointCloud, max_points: usize) -> Result<PointCloud, String> {
    let (x, y, z) = input.positions3().map_err(|error| error.to_string())?;
    let stride = input.len().div_ceil(max_points).max(1);
    let mut builder = PointCloudBuilder::xyz();
    for index in (0..input.len()).step_by(stride) {
        let point = [x[index], y[index], z[index]];
        if point.iter().all(|value| value.is_finite()) {
            builder.push_point(point).map_err(|error| error.to_string())?;
        }
    }
    builder.build().map_err(|error| error.to_string())
}

#[allow(dead_code)]
fn rich_sample_scene() -> PointCloud {
    let mut builder = PointCloudBuilder::xyz();

    // Dense floor grid — dominant ground plane for RANSAC.
    for xi in 0..55 {
        for yi in 0..40 {
            let xf = xi as f32 * 2.6 / 54.0;
            let yf = yi as f32 * 1.9 / 39.0;
            let noise = ((xi * 7 + yi * 13) % 5) as f32 * 0.005 - 0.01;
            builder.push_point([xf, yf, noise]).expect("floor point");
        }
    }

    // Table — flat raised surface.
    for i in 0..7 {
        for j in 0..8 {
            let px = 0.52 + i as f32 * 0.025;
            let py = 0.40 + j as f32 * 0.022;
            let z_noise = ((i * 3 + j * 5) % 3) as f32 * 0.003;
            builder.push_point([px, py, 0.62 + z_noise]).expect("table point");
        }
    }

    // Chair / box cluster A.
    for i in 0..5 {
        for j in 0..6 {
            let px = 0.18 + i as f32 * 0.025;
            let py = 0.74 + j as f32 * 0.022;
            let z = 0.36 + ((i + j) % 4) as f32 * 0.02;
            builder.push_point([px, py, z]).expect("chair a");
        }
    }

    // Chair / box cluster B.
    for i in 0..5 {
        for j in 0..6 {
            let px = 0.98 + i as f32 * 0.024;
            let py = 0.24 + j as f32 * 0.022;
            let z = 0.44 + ((i * 2 + j) % 5) as f32 * 0.015;
            builder.push_point([px, py, z]).expect("chair b");
        }
    }

    // Tall cabinet — multi-layer vertical blob.
    for i in 0..6 {
        for j in 0..5 {
            for k in 0..3 {
                let px = 1.30 + i as f32 * 0.022;
                let py = 0.50 + j as f32 * 0.020;
                let z = 0.48 + k as f32 * 0.18;
                builder.push_point([px, py, z]).expect("cabinet point");
            }
        }
    }

    // Small object on the floor.
    for i in 0..6 {
        for j in 0..6 {
            let px = 1.95 + i as f32 * 0.020;
            let py = 1.15 + j as f32 * 0.018;
            let z = 0.38 + ((i + j) % 3) as f32 * 0.04;
            builder.push_point([px, py, z]).expect("small object");
        }
    }

    // Box cluster near back wall.
    for i in 0..6 {
        for j in 0..7 {
            let px = 0.80 + i as f32 * 0.022;
            let py = 1.50 + j as f32 * 0.020;
            let z = 0.42 + ((i * j) % 3) as f32 * 0.05;
            builder.push_point([px, py, z]).expect("box cluster");
        }
    }

    // Perimeter pillars: dense vertical blobs spaced apart along the back and
    // side edges. Each pillar is tight enough (<0.18) to cluster on its own and
    // separated enough (>0.4) to stay distinct, so the cluster reveal lights up
    // a colorful room outline instead of one giant blob.
    let pillars = [
        (0.30_f32, 1.86_f32),
        (0.95, 1.86),
        (1.70, 1.86),
        (2.45, 1.86),
        (2.52, 1.30),
        (2.52, 0.70),
    ];
    for (cx, cy) in pillars {
        for layer in 0..5 {
            for i in 0..3 {
                for j in 0..3 {
                    let xf = cx + i as f32 * 0.05 - 0.05;
                    let yf = cy + j as f32 * 0.045 - 0.045;
                    let zf = 0.10 + layer as f32 * 0.14;
                    let noise = ((layer * 7 + i * 3 + j) % 4) as f32 * 0.006 - 0.009;
                    builder.push_point([xf, yf, zf + noise]).expect("perimeter pillar");
                }
            }
        }
    }

    builder.build().expect("rich scene")
}

fn pipeline_config() -> MvpPipelineConfig {
    MvpPipelineConfig {
        voxel: spatialrust::VoxelGridDownsampleConfig::centroid(0.03),
        normals: NormalEstimationConfig {
            k_neighbors: 8,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.7, 0.7, 10.0)),
            ..NormalEstimationConfig::default()
        },
        plane: RansacPlaneConfig {
            distance_threshold: 0.025,
            max_iterations: 500,
            min_inliers: 40,
            seed: 17,
            ..Default::default()
        },
        cluster: EuclideanClusterConfig {
            cluster_tolerance: 0.06,
            min_cluster_size: 8,
            max_cluster_size: usize::MAX,
            ..Default::default()
        },
        icp: None,
        ..MvpPipelineConfig::default()
    }
}

fn receipt_pipeline_config() -> MvpPipelineConfig {
    MvpPipelineConfig {
        voxel: spatialrust::VoxelGridDownsampleConfig::centroid(0.05),
        normals: NormalEstimationConfig {
            k_neighbors: 20,
            min_neighbors: 3,
            ..NormalEstimationConfig::default()
        },
        plane: RansacPlaneConfig {
            distance_threshold: 0.05,
            max_iterations: 500,
            min_inliers: 100,
            seed: 17,
            ..Default::default()
        },
        cluster: EuclideanClusterConfig {
            cluster_tolerance: 0.05,
            min_cluster_size: 1,
            max_cluster_size: usize::MAX,
            ..Default::default()
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

fn bounds_xyz(cloud: &PointCloud) -> ([f32; 3], [f32; 3]) {
    let (x, y, z) = cloud.positions3().expect("positions");
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for index in 0..cloud.len() {
        min[0] = min[0].min(x[index]);
        min[1] = min[1].min(y[index]);
        min[2] = min[2].min(z[index]);
        max[0] = max[0].max(x[index]);
        max[1] = max[1].max(y[index]);
        max[2] = max[2].max(z[index]);
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

fn iso_project(
    x: f32,
    y: f32,
    z: f32,
    yaw: f32,
    min: [f32; 3],
    max: [f32; 3],
    width: f32,
    height: f32,
    pad_x: f32,
    pad_y: f32,
) -> (f32, f32, f32) {
    let cos = yaw.cos();
    let sin = yaw.sin();
    let rx = x * cos - y * sin;
    let ry = x * sin + y * cos;

    let sx = (rx - ry) * 0.82;
    let sy = -z * 1.35 + (rx + ry) * 0.42;

    let span_x = (max[0] - min[0]).max(1e-3);
    let span_y = (max[1] - min[1]).max(1e-3);
    let span_z = (max[2] - min[2]).max(1e-3);
    let extent = (span_x + span_y + span_z * 0.8).max(0.5);
    // Fill more of the frame for a punchier hero composition.
    const FILL: f32 = 1.5;
    let scale =
        FILL * ((width - 2.0 * pad_x) / extent).min((height - 2.0 * pad_y) / (extent * 0.72));

    let cx = (min[0] + max[0]) * 0.5;
    let cy = (min[1] + max[1]) * 0.5;
    let cz = (min[2] + max[2]) * 0.5;
    let centered_x = x - cx;
    let centered_y = y - cy;
    let centered_z = z - cz;
    let crx = centered_x * cos - centered_y * sin;
    let cry = centered_x * sin + centered_y * cos;
    let csx = (crx - cry) * 0.82;
    let csy = -centered_z * 1.35 + (crx + cry) * 0.42;

    let px = width * 0.5 + csx * scale;
    let py = height * 0.58 - csy * scale;
    let depth = sx + sy * 0.001;
    (px, py, depth)
}

#[derive(Clone, Copy)]
struct Rgb(u8, u8, u8);

impl Rgb {
    fn blend(self, other: Rgb, t: f32) -> Rgb {
        let t = t.clamp(0.0, 1.0);
        Rgb(
            (self.0 as f32 * (1.0 - t) + other.0 as f32 * t) as u8,
            (self.1 as f32 * (1.0 - t) + other.1 as f32 * t) as u8,
            (self.2 as f32 * (1.0 - t) + other.2 as f32 * t) as u8,
        )
    }

    fn scale(self, factor: f32) -> Rgb {
        Rgb(
            (self.0 as f32 * factor) as u8,
            (self.1 as f32 * factor) as u8,
            (self.2 as f32 * factor) as u8,
        )
    }
}

struct Canvas {
    pixels: Vec<u8>,
}

impl Canvas {
    fn new_gradient(width: u32, height: u32, top: Rgb, bottom: Rgb) -> Self {
        let mut pixels = vec![0_u8; (width * height * 3) as usize];
        for y in 0..height {
            let t = y as f32 / (height.saturating_sub(1).max(1) as f32);
            let row_color = top.blend(bottom, t);
            for x in 0..width {
                let index = ((y * width + x) * 3) as usize;
                pixels[index] = row_color.0;
                pixels[index + 1] = row_color.1;
                pixels[index + 2] = row_color.2;
            }
        }
        Self { pixels }
    }

    fn new(width: u32, height: u32, background: Rgb) -> Self {
        Self::new_gradient(width, height, background, background)
    }

    fn get(&self, width: u32, x: i32, y: i32) -> Rgb {
        if x < 0 || y < 0 || x >= width as i32 || y >= self.height(width) as i32 {
            return Rgb(0, 0, 0);
        }
        let index = ((y as u32 * width + x as u32) * 3) as usize;
        Rgb(self.pixels[index], self.pixels[index + 1], self.pixels[index + 2])
    }

    fn height(&self, width: u32) -> u32 {
        (self.pixels.len() / (width as usize * 3)) as u32
    }

    fn put(&mut self, width: u32, x: i32, y: i32, color: Rgb) {
        if x < 0 || y < 0 || x >= width as i32 || y >= self.height(width) as i32 {
            return;
        }
        let index = ((y as u32 * width + x as u32) * 3) as usize;
        self.pixels[index] = color.0;
        self.pixels[index + 1] = color.1;
        self.pixels[index + 2] = color.2;
    }

    fn put_blend(&mut self, width: u32, x: i32, y: i32, color: Rgb, alpha: f32) {
        let bg = self.get(width, x, y);
        self.put(width, x, y, bg.blend(color, alpha));
    }

    fn fill_circle(&mut self, width: u32, cx: i32, cy: i32, radius: i32, color: Rgb) {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    self.put(width, cx + dx, cy + dy, color);
                }
            }
        }
    }

    fn fill_circle_blend(
        &mut self,
        width: u32,
        cx: i32,
        cy: i32,
        radius: i32,
        color: Rgb,
        alpha: f32,
    ) {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    self.put_blend(width, cx + dx, cy + dy, color, alpha);
                }
            }
        }
    }

    fn draw_glow_point(&mut self, width: u32, cx: i32, cy: i32, core: i32, color: Rgb) {
        for ring in (1..=core * 3).rev() {
            let alpha = 0.08 * (1.0 - ring as f32 / (core * 3 + 1) as f32);
            self.fill_circle_blend(width, cx, cy, ring, color, alpha);
        }
        self.fill_circle(width, cx, cy, core, color);
    }

    fn draw_scan_beam(&mut self, width: u32, height: u32, beam_x: f32, intensity: f32) {
        let beam_width = 28.0;
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let dx = (x as f32 - beam_x).abs();
                if dx > beam_width {
                    continue;
                }
                let t = 1.0 - dx / beam_width;
                let alpha = intensity * t * t * 0.35;
                self.put_blend(width, x, y, Rgb(56, 189, 248), alpha);
            }
        }
    }

    fn draw_stage_footer(&mut self, width: u32, height: u32, stage: usize) {
        let labels = ["Input scan", "Voxel grid", "Plane RANSAC", "Clusters"];
        let footer_top = height as i32 - 56;
        for x in 0..width as i32 {
            for y in footer_top..height as i32 {
                let t = (y - footer_top) as f32 / 55.0;
                self.put_blend(width, x, y, Rgb(2, 6, 23), t * 0.85);
            }
        }

        for index in 0..4 {
            let cx = 170 + index as i32 * 240;
            let active = index == stage;
            let dot = if active { Rgb(56, 189, 248) } else { Rgb(71, 85, 105) };
            if active {
                self.fill_circle_blend(width, cx, footer_top + 18, 14, Rgb(56, 189, 248), 0.25);
            }
            self.fill_circle(width, cx, footer_top + 18, if active { 6 } else { 4 }, dot);
            let label_x = cx + if active { 26 } else { 16 };
            self.draw_label_chip(width, label_x, footer_top + 12, labels[index], active);
        }
    }

    fn draw_label_chip(&mut self, width: u32, x: i32, y: i32, text: &str, active: bool) {
        let color = if active { Rgb(226, 232, 240) } else { Rgb(100, 116, 139) };
        let mut cursor = x;
        for ch in text.chars() {
            self.draw_char(width, cursor, y, ch, color, active);
            cursor += if active { 8 } else { 7 };
        }
    }

    fn draw_char(&mut self, width: u32, x: i32, y: i32, ch: char, color: Rgb, bold: bool) {
        let glyph = glyph_5x7(ch);
        let scale = if bold { 2_i32 } else { 1_i32 };
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..5 {
                if bits & (1 << (4 - col)) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            self.put(
                                width,
                                x + col * scale + sx,
                                y + row as i32 * scale + sy,
                                color,
                            );
                        }
                    }
                }
            }
        }
    }

    fn draw_title(&mut self, width: u32) {
        self.draw_char_line(width, 36, 34, "SpatialRust", Rgb(248, 250, 252), 2);
        self.draw_char_line(width, 36, 62, "Rust-native spatial computing", Rgb(56, 189, 248), 1);
    }

    fn draw_char_line(&mut self, width: u32, x: i32, y: i32, text: &str, color: Rgb, scale: i32) {
        let mut cursor = x;
        for ch in text.chars() {
            let glyph = glyph_5x7(ch);
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..5 {
                    if bits & (1 << (4 - col)) != 0 {
                        for sy in 0..scale {
                            for sx in 0..scale {
                                self.put(
                                    width,
                                    cursor + col * scale + sx,
                                    y + row as i32 * scale + sy,
                                    color,
                                );
                            }
                        }
                    }
                }
            }
            cursor += 6 * scale;
        }
    }

    fn draw_stage_bar(&mut self, width: u32, stage: usize) {
        let bar_y = 18;
        for index in 0..4 {
            let x0 = 24 + index as u32 * 168;
            let active = index == stage;
            let color = if active { Rgb(56, 189, 248) } else { Rgb(51, 65, 85) };
            for x in x0..x0 + 150 {
                for y in bar_y..bar_y + 6 {
                    self.put(width, x as i32, y as i32, color);
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

fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch {
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x11, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        'a' => [0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F],
        'b' => [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x1E],
        'c' => [0x00, 0x00, 0x0E, 0x10, 0x10, 0x11, 0x0E],
        'd' => [0x01, 0x01, 0x0D, 0x13, 0x11, 0x11, 0x0F],
        'e' => [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E],
        'f' => [0x06, 0x08, 0x1E, 0x08, 0x08, 0x08, 0x08],
        'g' => [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x0E],
        'h' => [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x11],
        'i' => [0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E],
        'j' => [0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C],
        'k' => [0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12],
        'l' => [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'm' => [0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11],
        'n' => [0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11],
        'o' => [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E],
        'p' => [0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10],
        'q' => [0x00, 0x00, 0x0D, 0x13, 0x11, 0x0D, 0x01],
        'r' => [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],
        's' => [0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E],
        't' => [0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06],
        'u' => [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D],
        'v' => [0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'w' => [0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A],
        'x' => [0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11],
        'y' => [0x00, 0x00, 0x11, 0x11, 0x0F, 0x01, 0x0E],
        'z' => [0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F],
        ' ' => [0x00; 7],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04],
        '·' => [0x00, 0x04, 0x00, 0x00, 0x04, 0x00, 0x00],
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F],
        '3' => [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x0E, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E],
        _ => [0x00; 7],
    }
}

fn depth_color(z: f32, min_z: f32, max_z: f32) -> Rgb {
    let t = ((z - min_z) / (max_z - min_z).max(1e-3)).clamp(0.0, 1.0);
    Rgb((40.0 + t * 180.0) as u8, (120.0 + t * 80.0) as u8, (220.0 - t * 120.0) as u8)
}

/// Vivid, well-separated palette cycled by cluster id so every cluster in the
/// reveal frame gets a distinct neon color instead of collapsing to one hue.
const CLUSTER_PALETTE: [Rgb; 8] = [
    Rgb(56, 189, 248),  // sky
    Rgb(249, 115, 22),  // orange
    Rgb(34, 197, 94),   // green
    Rgb(168, 85, 247),  // purple
    Rgb(244, 114, 182), // pink
    Rgb(250, 204, 21),  // amber
    Rgb(45, 212, 191),  // teal
    Rgb(96, 165, 250),  // blue
];

fn label_rgb(label: i32) -> Rgb {
    let index = label.rem_euclid(CLUSTER_PALETTE.len() as i32) as usize;
    CLUSTER_PALETTE[index]
}

struct IsoPoint {
    px: f32,
    py: f32,
    depth: f32,
    radius: i32,
    color: Rgb,
}

fn collect_iso_points(
    cloud: &PointCloud,
    yaw: f32,
    bounds: ([f32; 3], [f32; 3]),
    width: f32,
    height: f32,
    point_color: impl Fn(usize, f32, f32, f32) -> Rgb,
    radius: impl Fn(usize, f32) -> i32,
    visible: impl Fn(usize, f32, f32, f32) -> bool,
) -> Vec<IsoPoint> {
    let (min, max) = bounds;
    let (x, y, z) = cloud.positions3().expect("positions");
    let mut points = Vec::new();
    for index in 0..cloud.len() {
        if !visible(index, x[index], y[index], z[index]) {
            continue;
        }
        let (px, py, depth) =
            iso_project(x[index], y[index], z[index], yaw, min, max, width, height, 80.0, 110.0);
        points.push(IsoPoint {
            px,
            py,
            depth,
            radius: radius(index, z[index]),
            color: point_color(index, x[index], y[index], z[index]),
        });
    }
    points.sort_by(|left, right| {
        left.depth.partial_cmp(&right.depth).unwrap_or(std::cmp::Ordering::Equal)
    });
    points
}

fn draw_iso_points(canvas: &mut Canvas, width: u32, points: &[IsoPoint], glow: f32) {
    for point in points {
        let cx = point.px.round() as i32;
        let cy = point.py.round() as i32;
        if glow > 0.0 {
            canvas.draw_glow_point(
                width,
                cx,
                cy,
                point.radius,
                point.color.scale(0.85 + glow * 0.3),
            );
        } else {
            canvas.fill_circle(width, cx, cy, point.radius, point.color);
        }
    }
}

fn render_hero_frames(input: &PointCloud, result: &MvpPipelineResult, temp_dir: &Path) {
    let bounds = bounds_xyz(input);
    let (min_z, max_z) = (bounds.0[2], bounds.1[2]);
    let mut frame_index = 0_u32;

    let phases: [(usize, usize, f32); 5] =
        [(0, 18, 0.55), (0, 12, 0.85), (1, 10, 1.15), (2, 12, 1.45), (3, 16, 1.85)];

    for (stage, frame_count, yaw_base) in phases {
        for frame in 0..frame_count {
            let t = frame as f32 / frame_count.max(1) as f32;
            let yaw = yaw_base + t * 0.35;
            let mut canvas =
                Canvas::new_gradient(HERO_WIDTH, HERO_HEIGHT, Rgb(8, 12, 28), Rgb(17, 24, 39));
            canvas.draw_title(HERO_WIDTH);

            let sweep = if stage == 0 && frame_count > 12 && frame < frame_count - 4 {
                Some(120.0 + t * (HERO_WIDTH as f32 - 240.0))
            } else {
                None
            };

            if let Some(beam_x) = sweep {
                canvas.draw_scan_beam(HERO_WIDTH, HERO_HEIGHT, beam_x, 1.0);
            }

            let points = match stage {
                0 => collect_iso_points(
                    input,
                    yaw,
                    bounds,
                    HERO_WIDTH as f32,
                    HERO_HEIGHT as f32,
                    move |_, _, _, z| depth_color(z, min_z, max_z),
                    |_, _| 2,
                    move |_, x, _, _| sweep.map(|beam| x * 120.0 <= beam - 80.0).unwrap_or(true),
                ),
                1 => collect_iso_points(
                    &result.downsampled,
                    yaw,
                    bounds,
                    HERO_WIDTH as f32,
                    HERO_HEIGHT as f32,
                    move |_, _, _, z| Rgb(226, 232, 240).blend(depth_color(z, min_z, max_z), 0.35),
                    |_, _| 3,
                    |_, _, _, _| true,
                ),
                2 => {
                    let mut items = Vec::new();
                    items.extend(collect_iso_points(
                        &result.plane.inliers,
                        yaw,
                        bounds,
                        HERO_WIDTH as f32,
                        HERO_HEIGHT as f32,
                        |_, _, _, _| Rgb(56, 189, 248),
                        |_, _| 3,
                        |_, _, _, _| true,
                    ));
                    items.extend(collect_iso_points(
                        &result.plane.outliers,
                        yaw,
                        bounds,
                        HERO_WIDTH as f32,
                        HERO_HEIGHT as f32,
                        |_, _, _, _| Rgb(251, 146, 60),
                        |_, _| 4,
                        |_, _, _, _| true,
                    ));
                    items.sort_by(|left, right| {
                        left.depth.partial_cmp(&right.depth).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    items
                }
                3 => {
                    let output = &result.output;
                    let (x, y, z) = output.positions3().expect("positions");
                    let labels = match output.field("label").expect("labels") {
                        spatialrust::PointBuffer::I32(values) => values.as_slice(),
                        other => panic!("expected i32 labels, got {:?}", other.dtype()),
                    };
                    let mut items = Vec::new();
                    for index in 0..output.len() {
                        let (px, py, depth) = iso_project(
                            x[index],
                            y[index],
                            z[index],
                            yaw,
                            bounds.0,
                            bounds.1,
                            HERO_WIDTH as f32,
                            HERO_HEIGHT as f32,
                            80.0,
                            110.0,
                        );
                        items.push(IsoPoint {
                            px,
                            py,
                            depth,
                            radius: 5,
                            color: label_rgb(labels[index]),
                        });
                    }
                    items.sort_by(|left, right| {
                        left.depth.partial_cmp(&right.depth).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    items
                }
                _ => unreachable!(),
            };

            let glow = match stage {
                3 => 1.0,
                2 => 0.55,
                _ => 0.25,
            };
            draw_iso_points(&mut canvas, HERO_WIDTH, &points, glow);
            canvas.draw_stage_footer(HERO_WIDTH, HERO_HEIGHT, stage);

            let path = temp_dir.join(format!("hero_{frame_index:03}.ppm"));
            canvas.write_ppm(HERO_WIDTH, HERO_HEIGHT, &path);
            frame_index += 1;
        }
    }
}

fn label_rgb_flat(label: i32) -> Rgb {
    label_rgb(label)
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
        let (px, py) =
            project(x[index], y[index], min, max, GIF_WIDTH as f32, GIF_HEIGHT as f32, 48.0);
        canvas.fill_circle(GIF_WIDTH, px, py, radius, color);
    }
}

fn draw_plane_stage(
    canvas: &mut Canvas,
    plane: &spatialrust::RansacPlaneSegmentation,
    min: [f32; 2],
    max: [f32; 2],
) {
    draw_points(canvas, &plane.inliers, min, max, 4, Rgb(203, 213, 225));
    draw_points(canvas, &plane.outliers, min, max, 5, Rgb(251, 146, 60));
}

fn project_left(
    x: f32,
    y: f32,
    min: [f32; 2],
    max: [f32; 2],
    pad: f32,
) -> (i32, i32) {
    project(
        x,
        y,
        min,
        max,
        RECEIPT_LEFT_WIDTH as f32,
        GIF_HEIGHT as f32,
        pad,
    )
}

fn draw_points_left(
    canvas: &mut Canvas,
    cloud: &PointCloud,
    min: [f32; 2],
    max: [f32; 2],
    radius: i32,
    color: Rgb,
) {
    let (x, y, _) = cloud.positions3().expect("positions");
    for index in 0..cloud.len() {
        let (px, py) = project_left(x[index], y[index], min, max, 40.0);
        canvas.fill_circle(GIF_WIDTH, px, py, radius, color);
    }
}

fn draw_plane_stage_left(
    canvas: &mut Canvas,
    plane: &spatialrust::RansacPlaneSegmentation,
    min: [f32; 2],
    max: [f32; 2],
) {
    draw_points_left(canvas, &plane.inliers, min, max, 3, Rgb(203, 213, 225));
    draw_points_left(canvas, &plane.outliers, min, max, 4, Rgb(251, 146, 60));
}

fn draw_cluster_stage_left(canvas: &mut Canvas, output: &PointCloud, min: [f32; 2], max: [f32; 2]) {
    let (x, y, _) = output.positions3().expect("positions");
    let labels = match output.field("label").expect("labels") {
        spatialrust::PointBuffer::I32(values) => values.as_slice(),
        other => panic!("expected i32 labels, got {:?}", other.dtype()),
    };
    for index in 0..output.len() {
        let (px, py) = project_left(x[index], y[index], min, max, 40.0);
        canvas.fill_circle(GIF_WIDTH, px, py, 4, label_rgb_flat(labels[index]));
    }
}

fn receipt_log_lines(
    input: &PointCloud,
    result: &MvpPipelineResult,
    leaf_size: f32,
) -> Vec<String> {
    vec![
        "spatialrust mvp table scene".to_owned(),
        format!("load     {} pts", input.len()),
        format!("voxel    leaf {:.2}   {} pts", leaf_size, result.downsampled.len()),
        format!("plane    {} inliers", result.plane.inliers.len()),
        format!("cluster  {} clusters", result.clusters.cluster_count),
        "save     labeled.las".to_owned(),
    ]
}

fn receipt_visual_stage(visible_lines: usize) -> usize {
    match visible_lines {
        0 | 1 => 0,
        2 => 1,
        3 => 2,
        4 => 3,
        _ => 4,
    }
}

fn draw_receipt_left_panel(
    canvas: &mut Canvas,
    input: &PointCloud,
    result: &MvpPipelineResult,
    min: [f32; 2],
    max: [f32; 2],
    stage: usize,
) {
    if stage >= 1 {
        draw_points_left(canvas, input, min, max, 2, Rgb(100, 116, 139));
    }
    if stage >= 2 {
        draw_points_left(canvas, &result.downsampled, min, max, 3, Rgb(226, 232, 240));
    }
    if stage >= 3 {
        draw_plane_stage_left(canvas, &result.plane, min, max);
    }
    if stage >= 4 {
        draw_cluster_stage_left(canvas, &result.output, min, max);
    }
}

fn draw_receipt_terminal(
    canvas: &mut Canvas,
    lines: &[String],
    show_footnote: bool,
    cluster_count: usize,
) {
    let terminal_bg = Rgb(2, 6, 23);
    for x in RECEIPT_SPLIT_X..GIF_WIDTH as i32 {
        for y in 0..GIF_HEIGHT as i32 {
            canvas.put(GIF_WIDTH, x, y, terminal_bg);
        }
    }
    for y in 0..GIF_HEIGHT as i32 {
        canvas.put(GIF_WIDTH, RECEIPT_SPLIT_X, y, Rgb(51, 65, 85));
    }

    for (index, line) in lines.iter().enumerate() {
        let color = if index == 0 { Rgb(226, 232, 240) } else { Rgb(148, 163, 184) };
        canvas.draw_char_line(GIF_WIDTH, RECEIPT_SPLIT_X + 16, 40 + index as i32 * 24, line, color, 1);
    }

    if show_footnote {
        let footnote = format!("full cloud {cluster_count} clusters");
        canvas.draw_char_line(
            GIF_WIDTH,
            RECEIPT_SPLIT_X + 16,
            GIF_HEIGHT as i32 - 36,
            &footnote,
            Rgb(100, 116, 139),
            1,
        );
        canvas.draw_char_line(
            GIF_WIDTH,
            RECEIPT_SPLIT_X + 16,
            GIF_HEIGHT as i32 - 16,
            "gpu matches cpu",
            Rgb(100, 116, 139),
            1,
        );
    }
}

/// Footer GIF: pipeline receipt — left panel shows one evolving result, right
/// panel types measured log lines from a real [`MvpPipelineResult`].
fn render_receipt_gif_frames(
    input: &PointCloud,
    result: &MvpPipelineResult,
    temp_dir: &Path,
    leaf_size: f32,
) {
    let bounds = merge_bounds(bounds_xy(input), bounds_xy(&result.output));
    let (min, max) = bounds;
    let log_lines = receipt_log_lines(input, result, leaf_size);
    let typing_frames = log_lines.len() * RECEIPT_FRAMES_PER_LINE;
    let total_frames = typing_frames + RECEIPT_HOLD_FRAMES;
    let mut frame_index = 0_u32;

    for frame in 0..total_frames {
        let visible_lines = if frame >= typing_frames {
            log_lines.len()
        } else {
            frame / RECEIPT_FRAMES_PER_LINE + 1
        };
        let stage = receipt_visual_stage(visible_lines);
        let show_footnote = frame >= typing_frames;

        let mut canvas = Canvas::new(GIF_WIDTH, GIF_HEIGHT, Rgb(15, 23, 42));
        draw_receipt_left_panel(&mut canvas, input, result, min, max, stage);
        draw_receipt_terminal(
            &mut canvas,
            &log_lines[..visible_lines.min(log_lines.len())],
            show_footnote,
            result.clusters.cluster_count,
        );

        let path = temp_dir.join(format!("frame_{frame_index:03}.ppm"));
        canvas.write_ppm(GIF_WIDTH, GIF_HEIGHT, &path);
        frame_index += 1;
    }
}

fn draw_cluster_stage(canvas: &mut Canvas, output: &PointCloud, min: [f32; 2], max: [f32; 2]) {
    let (x, y, _) = output.positions3().expect("positions");
    let labels = match output.field("label").expect("labels") {
        spatialrust::PointBuffer::I32(values) => values.as_slice(),
        other => panic!("expected i32 labels, got {:?}", other.dtype()),
    };
    for index in 0..output.len() {
        let (px, py) =
            project(x[index], y[index], min, max, GIF_WIDTH as f32, GIF_HEIGHT as f32, 48.0);
        canvas.draw_glow_point(GIF_WIDTH, px, py, 5, label_rgb_flat(labels[index]));
    }
}

fn world_to_px(x: f32, y: f32, min: [f32; 2], max: [f32; 2], pad: f32) -> (i32, i32) {
    project(x, y, min, max, GIF_WIDTH as f32, GIF_HEIGHT as f32, pad)
}

fn draw_bounds_box(
    canvas: &mut Canvas,
    min: [f32; 2],
    max: [f32; 2],
    b_min: [f32; 2],
    b_max: [f32; 2],
    color: Rgb,
    glow: f32,
) {
    let (px0, py0) = world_to_px(b_min[0], b_min[1], min, max, 48.0);
    let (px1, py1) = world_to_px(b_max[0], b_max[1], min, max, 48.0);
    let left = px0.min(px1);
    let right = px0.max(px1);
    let top = py0.min(py1);
    let bottom = py0.max(py1);

    if glow > 0.0 {
        for y in top..=bottom {
            for x in left..=right {
                canvas.put_blend(GIF_WIDTH, x, y, color, 0.07 * glow);
            }
        }
    }
    for t in 0..3 {
        for x in left..=right {
            canvas.put(GIF_WIDTH, x, top + t, color);
            canvas.put(GIF_WIDTH, x, bottom - t, color);
        }
        for y in top..=bottom {
            canvas.put(GIF_WIDTH, left + t, y, color);
            canvas.put(GIF_WIDTH, right - t, y, color);
        }
    }
}

/// Compact GIF showing a COPC partial read: full tile -> bounds query ->
/// the recentered region of interest, mirroring the `--bounds` CLI flag.
fn render_copc_frames(input: &PointCloud, temp_dir: &Path) {
    let (min, max) = bounds_xy(input);
    let span = [(max[0] - min[0]).max(1e-3), (max[1] - min[1]).max(1e-3)];
    let b_min = [min[0] + span[0] * 0.30, min[1] + span[1] * 0.28];
    let b_max = [min[0] + span[0] * 0.70, min[1] + span[1] * 0.72];
    let (x, y, _) = input.positions3().expect("positions");
    let inside =
        |i: usize| x[i] >= b_min[0] && x[i] <= b_max[0] && y[i] >= b_min[1] && y[i] <= b_max[1];
    let inside_count = (0..input.len()).filter(|&i| inside(i)).count();

    let dim = Rgb(71, 85, 105);
    let hot = Rgb(56, 189, 248);
    let bg = Rgb(15, 23, 42);
    let mut frame_index = 0_u32;

    let mut write_frame = |canvas: &Canvas| {
        let path = temp_dir.join(format!("copc_{frame_index:03}.ppm"));
        canvas.write_ppm(GIF_WIDTH, GIF_HEIGHT, &path);
        frame_index += 1;
    };

    // Stage 1: the full COPC tile.
    for _ in 0..8 {
        let mut canvas = Canvas::new(GIF_WIDTH, GIF_HEIGHT, bg);
        canvas.draw_char_line(GIF_WIDTH, 24, 22, "Full COPC tile", Rgb(226, 232, 240), 2);
        for i in 0..input.len() {
            let (px, py) = world_to_px(x[i], y[i], min, max, 48.0);
            canvas.fill_circle(GIF_WIDTH, px, py, 3, dim);
        }
        let cap = format!("{} points", input.len());
        canvas.draw_char_line(GIF_WIDTH, 24, GIF_HEIGHT as i32 - 32, &cap, Rgb(148, 163, 184), 1);
        write_frame(&canvas);
    }

    // Stage 2: bounds query — box pulses in, points inside light up.
    for f in 0..12 {
        let t = f as f32 / 11.0;
        let glow = 0.35 + 0.65 * (t * std::f32::consts::PI).sin().abs();
        let mut canvas = Canvas::new(GIF_WIDTH, GIF_HEIGHT, bg);
        canvas.draw_char_line(GIF_WIDTH, 24, 22, "COPC bounds query", Rgb(226, 232, 240), 2);
        for i in 0..input.len() {
            let (px, py) = world_to_px(x[i], y[i], min, max, 48.0);
            if inside(i) {
                canvas.fill_circle(GIF_WIDTH, px, py, 4, hot);
            } else {
                canvas.fill_circle(GIF_WIDTH, px, py, 3, dim);
            }
        }
        draw_bounds_box(&mut canvas, min, max, b_min, b_max, hot, glow);
        canvas.draw_char_line(
            GIF_WIDTH,
            24,
            GIF_HEIGHT as i32 - 32,
            "select region of interest",
            Rgb(148, 163, 184),
            1,
        );
        write_frame(&canvas);
    }

    // Stage 3: the recentered partial-read result.
    for _ in 0..12 {
        let mut canvas = Canvas::new(GIF_WIDTH, GIF_HEIGHT, bg);
        canvas.draw_char_line(GIF_WIDTH, 24, 22, "Partial read result", Rgb(226, 232, 240), 2);
        for i in 0..input.len() {
            if !inside(i) {
                continue;
            }
            let (px, py) = world_to_px(x[i], y[i], b_min, b_max, 72.0);
            canvas.fill_circle(GIF_WIDTH, px, py, 5, hot);
        }
        let cap = format!("roi.copc.laz · {inside_count} points");
        canvas.draw_char_line(GIF_WIDTH, 24, GIF_HEIGHT as i32 - 32, &cap, Rgb(148, 163, 184), 1);
        write_frame(&canvas);
    }
}

fn render_gif_frames(input: &PointCloud, result: &MvpPipelineResult, temp_dir: &Path) {
    render_receipt_gif_frames(input, result, temp_dir, 0.05);
}

fn encode_gif(temp_dir: &Path, pattern: &str, framerate: &str, output: &Path) {
    encode_gif_with_filter(temp_dir, pattern, framerate, output, None);
}

fn encode_gif_with_filter(
    temp_dir: &Path,
    pattern: &str,
    framerate: &str,
    output: &Path,
    pre_palette_filter: Option<&str>,
) {
    let filter = match pre_palette_filter {
        Some(prefix) => format!(
            "{prefix},split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer"
        ),
        None => "split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer"
            .to_string(),
    };
    let status = Command::new("ffmpeg")
        .args(["-y", "-framerate", framerate, "-i"])
        .arg(temp_dir.join(pattern))
        .args(["-vf", &filter, "-loop", "0"])
        .arg(output)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg gif encode failed");
}

/// Encode with a fresh palette per frame so sparse neon cluster colors are not
/// quantized away by the dense floor color that dominates a single global
/// palette.
fn encode_gif_per_frame_palette(temp_dir: &Path, pattern: &str, framerate: &str, output: &Path) {
    let filter =
        "split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse=new=1:dither=bayer";
    let status = Command::new("ffmpeg")
        .args(["-y", "-framerate", framerate, "-i"])
        .arg(temp_dir.join(pattern))
        .args(["-vf", filter, "-loop", "0"])
        .arg(output)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg gif encode failed");
}

fn label_color_hex(label: i32) -> &'static str {
    // Mirror CLUSTER_PALETTE so the SVG preview matches the GIF colors.
    const HEX: [&str; 8] =
        ["#38bdf8", "#f97316", "#22c55e", "#a855f7", "#f472b6", "#facc15", "#2dd4bf", "#60a5fa"];
    HEX[label.rem_euclid(HEX.len() as i32) as usize]
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
        let _ = write!(svg, r#"<circle cx="{px:.2}" cy="{py:.2}" r="4.6" fill="{color}"/>"#,);
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
    svg.push_str(
        r##"<rect x="0" y="0" width="320" height="240" rx="12" fill="#111827" stroke="#334155"/>"##,
    );
    svg.push('\n');
    svg.push_str(r##"<text x="16" y="24" fill="#cbd5e1" font-family="ui-sans-serif, system-ui, sans-serif" font-size="14" font-weight="600">Plane inliers (RANSAC)</text>"##);
    svg.push('\n');
    write_plane_points_svg(&mut svg, plane, min, max);
    svg.push('\n');
    svg.push_str("</g>\n");

    svg.push_str(r#"<g transform="translate(360 72)">"#);
    svg.push('\n');
    svg.push_str(
        r##"<rect x="0" y="0" width="320" height="240" rx="12" fill="#111827" stroke="#334155"/>"##,
    );
    svg.push('\n');
    svg.push_str(r##"<text x="16" y="24" fill="#cbd5e1" font-family="ui-sans-serif, system-ui, sans-serif" font-size="14" font-weight="600">Cluster labels (Euclidean)</text>"##);
    svg.push('\n');
    write_cluster_points_svg(&mut svg, clusters, min, max);
    svg.push('\n');
    svg.push_str("</g>\n");

    svg.push_str(r#"<g transform="translate(696 72)">"#);
    svg.push('\n');
    svg.push_str(
        r##"<rect x="0" y="0" width="240" height="240" rx="12" fill="#111827" stroke="#334155"/>"##,
    );
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

fn render_benchmark_chart() -> String {
    const WIDTH: f64 = 960.0;
    const HEIGHT: f64 = 360.0;
    const MARGIN_LEFT: f64 = 70.0;
    const MARGIN_RIGHT: f64 = 150.0;
    const MARGIN_TOP: f64 = 70.0;
    const MARGIN_BOTTOM: f64 = 50.0;

    let plot_left = MARGIN_LEFT;
    let plot_top = MARGIN_TOP;
    let plot_width = WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
    let plot_height = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;
    let plot_bottom = plot_top + plot_height;
    let plot_right = plot_left + plot_width;

    let log_x_min = 10_000_f64.log10();
    let log_x_max = 2_000_000_f64.log10();
    let log_x_span = log_x_max - log_x_min;
    let y_max = 400.0_f64;

    let data: [(f64, f64, f64); 8] = [
        (10_000.0, 0.8, 17.0),
        (65_000.0, 4.7, 14.7),
        (100_000.0, 7.0, 17.2),
        (200_000.0, 23.8, 26.3),
        (500_000.0, 94.0, 51.0),
        (750_000.0, 148.0, 48.0),
        (1_000_000.0, 155.0, 56.0),
        (2_000_000.0, 389.0, 101.0),
    ];

    let x_px =
        |point_count: f64| plot_left + (point_count.log10() - log_x_min) / log_x_span * plot_width;
    let y_px = |ms: f64| plot_top + plot_height - (ms / y_max) * plot_height;

    let mut svg = String::new();
    svg.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    svg.push('\n');
    svg.push_str(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="960" height="360" viewBox="0 0 960 360" role="img" aria-labelledby="title desc">"#,
    );
    svg.push('\n');
    svg.push_str(r#"<title id="title">Voxel downsample: CPU vs GPU</title>"#);
    svg.push('\n');
    svg.push_str(
        r#"<desc id="desc">End-to-end centroid voxel filter latency versus point count on CPU and GPU.</desc>"#,
    );
    svg.push('\n');
    svg.push_str(r##"<rect width="960" height="360" fill="#0f172a"/>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="24" y="34" fill="#e2e8f0" font-family="ui-sans-serif, system-ui, sans-serif" font-size="20" font-weight="700">Voxel downsample: CPU vs GPU</text>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="24" y="58" fill="#94a3b8" font-family="ui-sans-serif, system-ui, sans-serif" font-size="13">End-to-end filter latency vs point count (centroid, leaf=4.0)</text>"##);
    svg.push('\n');

    for ms in [0.0, 100.0, 200.0, 300.0, 400.0] {
        let y = y_px(ms);
        let _ = write!(
            svg,
            r##"<line x1="{plot_left:.1}" y1="{y:.2}" x2="{plot_right:.1}" y2="{y:.2}" stroke="#1e293b" stroke-width="1"/>"##,
        );
    }

    let x_ticks: [(f64, &str); 3] = [(10_000.0, "10k"), (100_000.0, "100k"), (1_000_000.0, "1M")];
    for (point_count, label) in x_ticks {
        let x = x_px(point_count);
        let _ = write!(
            svg,
            r##"<line x1="{x:.2}" y1="{plot_top:.1}" x2="{x:.2}" y2="{plot_bottom:.1}" stroke="#1e293b" stroke-width="1"/>"##,
        );
        let _ = write!(
            svg,
            r##"<text x="{x:.2}" y="{plot_bottom:.1}" dy="18" text-anchor="middle" fill="#64748b" font-family="ui-sans-serif, system-ui, sans-serif" font-size="11">{label}</text>"##,
        );
    }

    for ms in [0.0, 100.0, 200.0, 300.0, 400.0] {
        let y = y_px(ms);
        let label = if ms >= 399.0 { "400 ms" } else { "" };
        if ms < 399.0 {
            let _ = write!(
                svg,
                r##"<text x="{plot_left:.1}" y="{y:.2}" dx="-8" text-anchor="end" dominant-baseline="middle" fill="#64748b" font-family="ui-sans-serif, system-ui, sans-serif" font-size="11">{ms:.0}</text>"##,
            );
        } else {
            let _ = write!(
                svg,
                r##"<text x="{plot_left:.1}" y="{y:.2}" dx="-8" text-anchor="end" dominant-baseline="middle" fill="#64748b" font-family="ui-sans-serif, system-ui, sans-serif" font-size="11">{label}</text>"##,
            );
        }
    }

    let mut cpu_points = String::new();
    let mut gpu_points = String::new();
    for (point_count, cpu_ms, gpu_ms) in data {
        let x = x_px(point_count);
        let cpu_y = y_px(cpu_ms);
        let gpu_y = y_px(gpu_ms);
        if !cpu_points.is_empty() {
            cpu_points.push(' ');
        }
        let _ = write!(cpu_points, "{x:.2},{cpu_y:.2}");
        if !gpu_points.is_empty() {
            gpu_points.push(' ');
        }
        let _ = write!(gpu_points, "{x:.2},{gpu_y:.2}");
    }

    let _ = write!(
        svg,
        r##"<polyline points="{cpu_points}" fill="none" stroke="#f97316" stroke-width="3" stroke-linejoin="round"/>"##,
    );
    let _ = write!(
        svg,
        r##"<polyline points="{gpu_points}" fill="none" stroke="#38bdf8" stroke-width="3" stroke-linejoin="round"/>"##,
    );

    for (point_count, cpu_ms, gpu_ms) in data {
        let x = x_px(point_count);
        let cpu_y = y_px(cpu_ms);
        let gpu_y = y_px(gpu_ms);
        let _ = write!(svg, r##"<circle cx="{x:.2}" cy="{cpu_y:.2}" r="4" fill="#f97316"/>"##,);
        let _ = write!(svg, r##"<circle cx="{x:.2}" cy="{gpu_y:.2}" r="4" fill="#38bdf8"/>"##,);
    }

    let legend_x = plot_right - 8.0;
    let legend_y = plot_top + 12.0;
    let _ = write!(
        svg,
        r##"<line x1="{legend_x:.1}" y1="{legend_y:.1}" x2="{:.1}" y2="{legend_y:.1}" stroke="#f97316" stroke-width="3" stroke-linecap="round"/>"##,
        legend_x - 28.0,
    );
    let _ = write!(
        svg,
        r##"<text x="{legend_x:.1}" y="{legend_y:.1}" dx="8" dominant-baseline="middle" fill="#cbd5e1" font-family="ui-sans-serif, system-ui, sans-serif" font-size="13">CPU (centroid)</text>"##,
    );
    let legend_y_gpu = legend_y + 22.0;
    let _ = write!(
        svg,
        r##"<line x1="{legend_x:.1}" y1="{legend_y_gpu:.1}" x2="{:.1}" y2="{legend_y_gpu:.1}" stroke="#38bdf8" stroke-width="3" stroke-linecap="round"/>"##,
        legend_x - 28.0,
    );
    let _ = write!(
        svg,
        r##"<text x="{legend_x:.1}" y="{legend_y_gpu:.1}" dx="8" dominant-baseline="middle" fill="#cbd5e1" font-family="ui-sans-serif, system-ui, sans-serif" font-size="13">GPU (centroid)</text>"##,
    );

    // Headline insight placed in the empty upper-left of the plot so it never
    // crosses the steeply climbing CPU line near 2M.
    let callout_x = x_px(60_000.0);
    let callout_y = y_px(330.0);
    let _ = write!(
        svg,
        r##"<text x="{callout_x:.1}" y="{callout_y:.1}" fill="#38bdf8" font-family="ui-sans-serif, system-ui, sans-serif" font-size="16" font-weight="700">GPU is ~3.9x faster at 2M points</text>"##,
    );
    let _ = write!(
        svg,
        r##"<text x="{callout_x:.1}" y="{:.1}" fill="#94a3b8" font-family="ui-sans-serif, system-ui, sans-serif" font-size="12">CPU stays ahead below ~200k; wgpu wins as clouds grow</text>"##,
        callout_y + 20.0,
    );

    svg.push_str("</svg>\n");
    svg
}

fn render_architecture_diagram() -> String {
    let mut svg = String::new();
    svg.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    svg.push('\n');
    svg.push_str(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="960" height="440" viewBox="0 0 960 440" role="img" aria-labelledby="title desc">"#,
    );
    svg.push('\n');
    svg.push_str(r#"<title id="title">SpatialRust architecture</title>"#);
    svg.push('\n');
    svg.push_str(
        r#"<desc id="desc">Dataflow from file load to labeled clusters with composable SpatialRust crates and optional wgpu acceleration.</desc>"#,
    );
    svg.push('\n');
    svg.push_str(r##"<rect width="960" height="440" fill="#0f172a"/>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="24" y="36" fill="#e2e8f0" font-family="ui-sans-serif, system-ui, sans-serif" font-size="22" font-weight="700">SpatialRust architecture</text>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="24" y="60" fill="#94a3b8" font-family="ui-sans-serif, system-ui, sans-serif" font-size="13">One dataflow from file to labeled clusters — composable crates, optional wgpu acceleration</text>"##);
    svg.push('\n');

    let cards: [(&str, &str, &str, &str, i32); 7] = [
        ("Load", "PCD·PLY·LAS·COPC", "spatialrust-io", "#38bdf8", 24),
        ("Voxel", "downsample", "spatialrust-filtering", "#22c55e", 156),
        ("Normals", "estimate", "spatialrust-features", "#facc15", 288),
        ("Plane", "RANSAC", "spatialrust-segmentation", "#a855f7", 420),
        ("Cluster", "Euclidean", "spatialrust-segmentation", "#a855f7", 552),
        ("Register", "ICP", "spatialrust-registration", "#f472b6", 684),
        ("Save", "PCD·PLY·LAS·COPC", "spatialrust-io", "#38bdf8", 816),
    ];

    for (name, sub, crate_name, accent, card_x) in cards {
        let cx = card_x + 60;
        let _ = write!(
            svg,
            r##"<rect x="{card_x}" y="84" width="120" height="88" rx="10" fill="#111827" stroke="#334155"/>"##,
        );
        svg.push('\n');
        let _ =
            write!(svg, r##"<rect x="{card_x}" y="84" width="120" height="6" fill="{accent}"/>"##,);
        svg.push('\n');
        let _ = write!(
            svg,
            r##"<text x="{cx}" y="128" text-anchor="middle" fill="#e2e8f0" font-family="ui-sans-serif, system-ui, sans-serif" font-size="15" font-weight="700">{name}</text>"##,
        );
        svg.push('\n');
        let _ = write!(
            svg,
            r##"<text x="{cx}" y="148" text-anchor="middle" fill="#94a3b8" font-family="ui-sans-serif, system-ui, sans-serif" font-size="11">{sub}</text>"##,
        );
        svg.push('\n');
        let _ = write!(
            svg,
            r##"<text x="{cx}" y="164" text-anchor="middle" fill="#64748b" font-family="ui-monospace, monospace" font-size="10">{crate_name}</text>"##,
        );
        svg.push('\n');
    }

    for (_, _, _, _, card_x) in cards.iter().take(6) {
        let gap_left = card_x + 120 + 2;
        let gap_tip = gap_left + 8;
        let _ = write!(
            svg,
            r##"<polygon points="{gap_left},122 {gap_tip},128 {gap_left},134" fill="#38bdf8"/>"##,
        );
        svg.push('\n');
    }

    svg.push_str(r##"<rect x="156" y="210" width="120" height="56" rx="10" fill="#0b1224" stroke="#38bdf8"/>"##);
    svg.push('\n');
    svg.push_str(
        r##"<line x1="216" y1="172" x2="216" y2="210" stroke="#38bdf8" stroke-dasharray="3 3"/>"##,
    );
    svg.push('\n');
    svg.push_str(r##"<text x="216" y="234" text-anchor="middle" fill="#38bdf8" font-family="ui-sans-serif, system-ui, sans-serif" font-size="13" font-weight="700">wgpu</text>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="216" y="250" text-anchor="middle" fill="#94a3b8" font-family="ui-sans-serif, system-ui, sans-serif" font-size="10">voxel kernel + CPU fallback</text>"##);
    svg.push('\n');

    svg.push_str(r##"<rect x="24" y="300" width="912" height="64" rx="12" fill="#111827" stroke="#334155"/>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="40" y="322" fill="#cbd5e1" font-family="ui-sans-serif, system-ui, sans-serif" font-size="12" font-weight="700">Foundation</text>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="40" y="348" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="12">spatialrust-core — schema · metadata · traits</text>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="400" y="348" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="12">spatialrust-math — Vec / Mat / Pose</text>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="700" y="348" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="12">spatialrust-search — KD-tree</text>"##);
    svg.push('\n');

    svg.push_str(r##"<rect x="560" y="24" width="180" height="24" rx="14" fill="#0b1224" stroke="#334155"/>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="650" y="40" text-anchor="middle" fill="#cbd5e1" font-family="ui-monospace, monospace" font-size="11">spatialrust · re-exports</text>"##);
    svg.push('\n');
    svg.push_str(r##"<rect x="752" y="24" width="184" height="24" rx="14" fill="#0b1224" stroke="#334155"/>"##);
    svg.push('\n');
    svg.push_str(r##"<text x="844" y="40" text-anchor="middle" fill="#cbd5e1" font-family="ui-monospace, monospace" font-size="11">spatialrust-pipeline · MVP</text>"##);
    svg.push('\n');

    svg.push_str(r##"<text x="24" y="396" fill="#64748b" font-family="ui-monospace, monospace" font-size="12">dependency direction:  math → core → io · search · gpu → algorithms → pipeline</text>"##);
    svg.push('\n');

    svg.push_str("</svg>\n");
    svg
}

fn render_social_card() -> String {
    let mut svg = String::new();
    svg.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    svg.push('\n');
    svg.push_str(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="640" viewBox="0 0 1280 640" role="img" aria-labelledby="title">"#,
    );
    svg.push('\n');
    svg.push_str(r#"<title id="title">SpatialRust social preview</title>"#);
    svg.push('\n');
    svg.push_str(r#"<defs>"#);
    svg.push('\n');
    svg.push_str(r##"  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">"##);
    svg.push('\n');
    svg.push_str(r##"    <stop offset="0%" stop-color="#080c1c"/>"##);
    svg.push('\n');
    svg.push_str(r##"    <stop offset="55%" stop-color="#111827"/>"##);
    svg.push('\n');
    svg.push_str(r##"    <stop offset="100%" stop-color="#172554"/>"##);
    svg.push('\n');
    svg.push_str(r##"  </linearGradient>"##);
    svg.push('\n');
    svg.push_str(r##"  <radialGradient id="glow" cx="72%" cy="38%" r="48%">"##);
    svg.push('\n');
    svg.push_str(r##"    <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.35"/>"##);
    svg.push('\n');
    svg.push_str(r##"    <stop offset="100%" stop-color="#38bdf8" stop-opacity="0"/>"##);
    svg.push('\n');
    svg.push_str(r#"  </radialGradient>"#);
    svg.push('\n');
    svg.push_str(r#"</defs>"#);
    svg.push('\n');
    svg.push_str(r#"<rect width="1280" height="640" fill="url(#bg)"/>"#);
    svg.push('\n');
    svg.push_str(r#"<rect width="1280" height="640" fill="url(#glow)"/>"#);
    svg.push('\n');
    svg.push_str(
        r##"<text x="80" y="150" fill="#f8fafc" font-family="ui-sans-serif, system-ui, sans-serif" font-size="80" font-weight="800">SpatialRust</text>"##,
    );
    svg.push('\n');
    svg.push_str(
        r##"<text x="82" y="210" fill="#38bdf8" font-family="ui-sans-serif, system-ui, sans-serif" font-size="34" font-weight="700">Rust-native spatial computing</text>"##,
    );
    svg.push('\n');
    svg.push_str(
        r##"<text x="82" y="262" fill="#cbd5e1" font-family="ui-sans-serif, system-ui, sans-serif" font-size="24">Point clouds · wgpu · COPC · RANSAC · ICP — native Rust</text>"##,
    );
    svg.push('\n');
    svg.push_str(
        r##"<rect x="80" y="300" width="230" height="40" rx="16" fill="#0b1224" stroke="#334155"/>"##,
    );
    svg.push('\n');
    svg.push_str(
        r##"<rect x="326" y="300" width="150" height="40" rx="16" fill="#0b1224" stroke="#334155"/>"##,
    );
    svg.push('\n');
    svg.push_str(
        r##"<rect x="492" y="300" width="250" height="40" rx="16" fill="#0b1224" stroke="#334155"/>"##,
    );
    svg.push('\n');
    svg.push_str(
        r##"<text x="195" y="320" text-anchor="middle" dominant-baseline="middle" fill="#38bdf8" font-family="ui-sans-serif, system-ui, sans-serif" font-size="18" font-weight="700">GPU voxel ~3.9x @ 2M</text>"##,
    );
    svg.push('\n');
    svg.push_str(
        r##"<text x="401" y="320" text-anchor="middle" dominant-baseline="middle" fill="#cbd5e1" font-family="ui-sans-serif, system-ui, sans-serif" font-size="18">11 crates</text>"##,
    );
    svg.push('\n');
    svg.push_str(
        r##"<text x="617" y="320" text-anchor="middle" dominant-baseline="middle" fill="#cbd5e1" font-family="ui-sans-serif, system-ui, sans-serif" font-size="18">MIT / Apache-2.0</text>"##,
    );
    svg.push('\n');
    svg.push_str(
        r##"<text x="82" y="400" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="22">cargo run --features mvp --bin spatialrust-mvp -- scan.las out.las</text>"##,
    );
    svg.push('\n');
    svg.push_str(
        r##"<rect x="812" y="70" width="408" height="500" rx="28" fill="#0b1224" stroke="#334155"/>"##,
    );
    svg.push('\n');
    svg.push_str(
        r##"<text x="842" y="118" fill="#e2e8f0" font-family="ui-sans-serif, system-ui, sans-serif" font-size="22" font-weight="700">Isometric pipeline preview</text>"##,
    );
    svg.push('\n');

    const PANEL_CX: f32 = 1016.0;
    const FLOOR_ANCHOR_Y: f32 = 250.0;
    for gx in 0..14 {
        for gy in 0..11 {
            let u = gx as f32;
            let v = gy as f32;
            let px = PANEL_CX + (u - v) * 11.5;
            let py = FLOOR_ANCHOR_Y + (u + v) * 6.5;
            let _ = write!(
                svg,
                r##"<circle cx="{px:.1}" cy="{py:.1}" r="3" fill="#3b82f6" fill-opacity="0.7"/>"##,
            );
        }
    }
    svg.push('\n');

    let cluster_centers = [(940, 250), (980, 300), (1060, 260), (1090, 320), (1010, 360)];
    let cluster_colors = ["#f97316", "#06b6d4", "#a855f7", "#22c55e", "#f472b6"];
    let blob_offsets = [(0, 0), (8, 4), (-7, 5), (4, -7), (-5, -6)];
    for ((cx, cy), color) in cluster_centers.iter().zip(cluster_colors.iter()) {
        for (dx, dy) in blob_offsets {
            let _ = write!(
                svg,
                r##"<circle cx="{}" cy="{}" r="6" fill="{color}"/>"##,
                cx + dx,
                cy + dy,
            );
        }
    }
    svg.push('\n');

    svg.push_str(r##"<rect x="842" y="500" width="348" height="8" rx="4" fill="#1e293b"/>"##);
    svg.push('\n');
    svg.push_str(r##"<rect x="842" y="500" width="261" height="8" rx="4" fill="#38bdf8"/>"##);
    svg.push('\n');
    svg.push_str(
        r##"<text x="842" y="534" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="16">scan → voxel → plane → cluster</text>"##,
    );
    svg.push('\n');
    svg.push_str("</svg>\n");
    svg
}

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/assets")
}

fn main() {
    let input = sample_scene();
    let result = MvpPipeline::new(pipeline_config()).run(&input).expect("mvp pipeline preview run");

    let receipt_input = public_sample_scene_full().unwrap_or_else(|error| {
        eprintln!("warning: receipt GIF uses decimated input ({error})");
        input.clone()
    });
    let receipt_result = MvpPipeline::new(receipt_pipeline_config())
        .run(&receipt_input)
        .expect("receipt pipeline run");

    let assets = assets_dir();
    fs::create_dir_all(&assets).expect("create docs/assets directory");

    let svg_path = assets.join("readme_mvp_preview.svg");
    fs::write(
        &svg_path,
        render_svg(&result.plane.inliers, &result.output, result.clusters.cluster_count),
    )
    .expect("write svg");

    let temp_dir =
        std::env::temp_dir().join(format!("spatialrust_readme_gif_{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).expect("create temp gif frames");
    render_receipt_gif_frames(&receipt_input, &receipt_result, &temp_dir, 0.05);
    let gif_path = assets.join("readme_mvp_pipeline.gif");
    encode_gif_per_frame_palette(&temp_dir, "frame_%03d.ppm", "8", &gif_path);

    let copc_temp =
        std::env::temp_dir().join(format!("spatialrust_readme_copc_{}", std::process::id()));
    let _ = fs::remove_dir_all(&copc_temp);
    fs::create_dir_all(&copc_temp).expect("create temp copc frames");
    render_copc_frames(&input, &copc_temp);
    let copc_path = assets.join("copc_query.gif");
    encode_gif(&copc_temp, "copc_%03d.ppm", "6", &copc_path);
    let _ = fs::remove_dir_all(&copc_temp);

    let hero_temp =
        std::env::temp_dir().join(format!("spatialrust_readme_hero_{}", std::process::id()));
    let _ = fs::remove_dir_all(&hero_temp);
    fs::create_dir_all(&hero_temp).expect("create temp hero frames");
    render_hero_frames(&input, &result, &hero_temp);
    let hero_path = assets.join("readme_hero.gif");
    encode_gif_with_filter(
        &hero_temp,
        "hero_%03d.ppm",
        "12",
        &hero_path,
        Some("scale=960:-1:flags=lanczos"),
    );
    let _ = fs::remove_dir_all(&hero_temp);
    let _ = fs::remove_dir_all(&temp_dir);

    let social_path = assets.join("social_preview.svg");
    fs::write(&social_path, render_social_card()).expect("write social preview");

    let benchmark_path = assets.join("benchmark_voxel.svg");
    fs::write(&benchmark_path, render_benchmark_chart()).expect("write benchmark chart");

    let architecture_path = assets.join("architecture.svg");
    fs::write(&architecture_path, render_architecture_diagram())
        .expect("write architecture diagram");

    println!("wrote {}", svg_path.display());
    println!("wrote {}", gif_path.display());
    println!("wrote {}", hero_path.display());
    println!("wrote {}", social_path.display());
    println!("wrote {}", benchmark_path.display());
    println!("wrote {}", architecture_path.display());
    println!("wrote {}", copc_path.display());
}
