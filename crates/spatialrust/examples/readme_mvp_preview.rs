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
const HERO_WIDTH: u32 = 1280;
const HERO_HEIGHT: u32 = 540;

fn sample_scene() -> PointCloud {
    rich_sample_scene()
}

fn rich_sample_scene() -> PointCloud {
    let mut builder = PointCloudBuilder::xyz();

    for x in 0..18 {
        for y in 0..18 {
            let xf = x as f32 * 0.08;
            let yf = y as f32 * 0.08;
            let noise = ((x * 7 + y * 13) % 5) as f32 * 0.004 - 0.008;
            builder
                .push_point([xf, yf, noise])
                .expect("floor point");
        }
    }

    for x in 0..6 {
        for y in 0..4 {
            builder
                .push_point([
                    0.55 + x as f32 * 0.06,
                    0.45 + y as f32 * 0.06,
                    0.62,
                ])
                .expect("table point");
        }
    }

    for (cx, cy, cz) in [
        (0.22, 0.78, 0.38),
        (0.28, 0.82, 0.41),
        (0.18, 0.74, 0.36),
        (0.25, 0.76, 0.39),
        (0.30, 0.80, 0.40),
    ] {
        builder.push_point([cx, cy, cz]).expect("cluster a");
    }

    for (cx, cy, cz) in [
        (1.02, 0.28, 0.48),
        (1.06, 0.32, 0.46),
        (1.00, 0.30, 0.50),
        (1.04, 0.26, 0.47),
    ] {
        builder.push_point([cx, cy, cz]).expect("cluster b");
    }

    for z in 0..8 {
        builder
            .push_point([1.28, 0.12 + z as f32 * 0.08, 0.05 + z as f32 * 0.07])
            .expect("wall point");
    }

    builder.build().expect("rich scene")
}

fn pipeline_config() -> MvpPipelineConfig {
    MvpPipelineConfig {
        voxel: spatialrust::VoxelGridDownsampleConfig::centroid(0.12),
        normals: NormalEstimationConfig {
            k_neighbors: 8,
            min_neighbors: 3,
            viewpoint: Some(Vec3::new(0.7, 0.7, 10.0)),
            ..NormalEstimationConfig::default()
        },
        plane: RansacPlaneConfig {
            distance_threshold: 0.05,
            max_iterations: 500,
            min_inliers: 40,
            seed: 17,
        },
        cluster: EuclideanClusterConfig {
            cluster_tolerance: 0.18,
            min_cluster_size: 2,
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
    let scale = ((width - 2.0 * pad_x) / extent).min((height - 2.0 * pad_y) / (extent * 0.72));

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
        Rgb(
            self.pixels[index],
            self.pixels[index + 1],
            self.pixels[index + 2],
        )
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
            let dot = if active {
                Rgb(56, 189, 248)
            } else {
                Rgb(71, 85, 105)
            };
            if active {
                self.fill_circle_blend(width, cx, footer_top + 18, 14, Rgb(56, 189, 248), 0.25);
            }
            self.fill_circle(width, cx, footer_top + 18, if active { 6 } else { 4 }, dot);
            self.draw_label_chip(width, cx + 16, footer_top + 12, labels[index], active);
        }
    }

    fn draw_label_chip(&mut self, width: u32, x: i32, y: i32, text: &str, active: bool) {
        let color = if active {
            Rgb(226, 232, 240)
        } else {
            Rgb(100, 116, 139)
        };
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
        self.draw_char_line(width, 36, 62, "PyTorch for Spatial Computing", Rgb(56, 189, 248), 1);
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
            let color = if active {
                Rgb(56, 189, 248)
            } else {
                Rgb(51, 65, 85)
            };
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
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'a' => [0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F],
        'c' => [0x00, 0x00, 0x0E, 0x10, 0x10, 0x11, 0x0E],
        'e' => [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E],
        'f' => [0x06, 0x08, 0x1E, 0x08, 0x08, 0x08, 0x08],
        'g' => [0x00, 0x00, 0x0F, 0x11, 0x0F, 0x01, 0x0E],
        'h' => [0x10, 0x10, 0x16, 0x19, 0x11, 0x11, 0x11],
        'i' => [0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E],
        'l' => [0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'm' => [0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11],
        'n' => [0x00, 0x00, 0x16, 0x19, 0x11, 0x11, 0x11],
        'o' => [0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E],
        'p' => [0x00, 0x00, 0x1E, 0x11, 0x1E, 0x10, 0x10],
        'r' => [0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10],
        's' => [0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E],
        't' => [0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06],
        'u' => [0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D],
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
    Rgb(
        (40.0 + t * 180.0) as u8,
        (120.0 + t * 80.0) as u8,
        (220.0 - t * 120.0) as u8,
    )
}

fn label_rgb(label: i32) -> Rgb {
    match label {
        0 => Rgb(249, 115, 22),
        1 => Rgb(6, 182, 212),
        2 => Rgb(168, 85, 247),
        3 => Rgb(34, 197, 94),
        _ => Rgb(244, 114, 182),
    }
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
        let (px, py, depth) = iso_project(
            x[index],
            y[index],
            z[index],
            yaw,
            min,
            max,
            width,
            height,
            80.0,
            110.0,
        );
        points.push(IsoPoint {
            px,
            py,
            depth,
            radius: radius(index, z[index]),
            color: point_color(index, x[index], y[index], z[index]),
        });
    }
    points.sort_by(|left, right| {
        left.depth
            .partial_cmp(&right.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
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

    let phases: [(usize, usize, f32); 5] = [
        (0, 18, 0.55),
        (0, 12, 0.85),
        (1, 10, 1.15),
        (2, 12, 1.45),
        (3, 16, 1.85),
    ];

    for (stage, frame_count, yaw_base) in phases {
        for frame in 0..frame_count {
            let t = frame as f32 / frame_count.max(1) as f32;
            let yaw = yaw_base + t * 0.35;
            let mut canvas = Canvas::new_gradient(
                HERO_WIDTH,
                HERO_HEIGHT,
                Rgb(8, 12, 28),
                Rgb(17, 24, 39),
            );
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
                        left.depth
                            .partial_cmp(&right.depth)
                            .unwrap_or(std::cmp::Ordering::Equal)
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
                        left.depth
                            .partial_cmp(&right.depth)
                            .unwrap_or(std::cmp::Ordering::Equal)
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
        let (px, py) = project(
            x[index],
            y[index],
            min,
            max,
            GIF_WIDTH as f32,
            GIF_HEIGHT as f32,
            48.0,
        );
        canvas.fill_circle(GIF_WIDTH, px, py, radius, color);
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
        let (px, py) = project(
            x[index],
            y[index],
            min,
            max,
            GIF_WIDTH as f32,
            GIF_HEIGHT as f32,
            48.0,
        );
        canvas.fill_circle(GIF_WIDTH, px, py, 6, label_rgb_flat(labels[index]));
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
      <stop offset="0%" stop-color="#080c1c"/>
      <stop offset="55%" stop-color="#111827"/>
      <stop offset="100%" stop-color="#172554"/>
    </linearGradient>
    <radialGradient id="glow" cx="72%" cy="42%" r="45%">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="#38bdf8" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="1280" height="640" fill="url(#bg)"/>
  <rect width="1280" height="640" fill="url(#glow)"/>
  <text x="80" y="150" fill="#f8fafc" font-family="ui-sans-serif, system-ui, sans-serif" font-size="72" font-weight="800">SpatialRust</text>
  <text x="80" y="220" fill="#38bdf8" font-family="ui-sans-serif, system-ui, sans-serif" font-size="34" font-weight="600">PyTorch for Spatial Computing</text>
  <text x="80" y="290" fill="#cbd5e1" font-family="ui-sans-serif, system-ui, sans-serif" font-size="28">Point clouds · wgpu · COPC · RANSAC · ICP · native Rust</text>
  <text x="80" y="380" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="24">cargo run -p spatialrust --features mvp --bin spatialrust-mvp -- scan.las out.las</text>
  <rect x="820" y="90" width="400" height="460" rx="28" fill="#0b1224" stroke="#334155"/>
  <text x="850" y="140" fill="#e2e8f0" font-family="ui-sans-serif, system-ui, sans-serif" font-size="22" font-weight="700">Isometric pipeline preview</text>
  <circle cx="920" cy="250" r="5" fill="#94a3b8"/>
  <circle cx="940" cy="230" r="5" fill="#64748b"/>
  <circle cx="960" cy="260" r="5" fill="#cbd5e1"/>
  <circle cx="980" cy="240" r="5" fill="#64748b"/>
  <circle cx="1000" cy="270" r="5" fill="#94a3b8"/>
  <circle cx="1040" cy="220" r="7" fill="#38bdf8"/>
  <circle cx="1060" cy="210" r="7" fill="#38bdf8"/>
  <circle cx="1080" cy="225" r="7" fill="#22d3ee"/>
  <circle cx="1110" cy="320" r="9" fill="#f97316"/>
  <circle cx="1135" cy="305" r="9" fill="#06b6d4"/>
  <circle cx="1160" cy="330" r="9" fill="#a855f7"/>
  <circle cx="1185" cy="315" r="9" fill="#22c55e"/>
  <rect x="850" y="390" width="340" height="8" rx="4" fill="#1e293b"/>
  <rect x="850" y="390" width="255" height="8" rx="4" fill="#38bdf8"/>
  <text x="850" y="430" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="16">scan → voxel → plane → cluster</text>
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
    encode_gif(&temp_dir, "frame_%03d.ppm", "8", &gif_path);

    let hero_temp = std::env::temp_dir().join(format!("spatialrust_readme_hero_{}", std::process::id()));
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

    println!("wrote {}", svg_path.display());
    println!("wrote {}", gif_path.display());
    println!("wrote {}", hero_path.display());
    println!("wrote {}", social_path.display());
}
