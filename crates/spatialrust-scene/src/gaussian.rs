//! Feature-gated Gaussian scene primitives and CPU soft-splat renderer.

use spatialrust_math::{Quat, Vec3};

use crate::{SceneError, SceneResult};

/// One anisotropic Gaussian primitive.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianPrimitive {
    /// Mean position.
    pub mean: Vec3<f32>,
    /// Per-axis scale.
    pub scale: Vec3<f32>,
    /// Orientation quaternion.
    pub rotation: Quat<f32>,
    /// Opacity in `[0, 1]`.
    pub opacity: f32,
    /// RGB color in `[0, 1]`.
    pub color: [f32; 3],
}

/// Host-side Gaussian scene container.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GaussianScene {
    primitives: Vec<GaussianPrimitive>,
}

impl GaussianScene {
    /// Creates an empty scene.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a validated Gaussian.
    pub fn push(&mut self, primitive: GaussianPrimitive) -> SceneResult<()> {
        validate_primitive(&primitive)?;
        self.primitives.push(primitive);
        Ok(())
    }

    /// Returns primitives.
    #[must_use]
    pub fn primitives(&self) -> &[GaussianPrimitive] {
        &self.primitives
    }

    /// Returns primitive count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.primitives.len()
    }

    /// Returns true when empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }
}

fn validate_primitive(primitive: &GaussianPrimitive) -> SceneResult<()> {
    if !(0.0..=1.0).contains(&primitive.opacity) {
        return Err(SceneError::InvalidConfiguration(
            "opacity must be in [0, 1]".into(),
        ));
    }
    if primitive.color.iter().any(|c| !(0.0..=1.0).contains(c)) {
        return Err(SceneError::InvalidConfiguration(
            "color channels must be in [0, 1]".into(),
        ));
    }
    if !(primitive.scale.x.is_finite()
        && primitive.scale.y.is_finite()
        && primitive.scale.z.is_finite())
        || primitive.scale.x <= 0.0
        || primitive.scale.y <= 0.0
        || primitive.scale.z <= 0.0
    {
        return Err(SceneError::InvalidConfiguration(
            "scale components must be finite and > 0".into(),
        ));
    }
    Ok(())
}

/// Pinhole camera used by the CPU Gaussian soft-splat renderer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianCamera {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Focal length x (pixels).
    pub fx: f32,
    /// Focal length y (pixels).
    pub fy: f32,
    /// Principal point x.
    pub cx: f32,
    /// Principal point y.
    pub cy: f32,
    /// Camera translation in world (world-from-camera origin).
    pub eye: Vec3<f32>,
    /// Camera orientation (camera-from-world rotation as quaternion).
    pub rotation_camera_from_world: Quat<f32>,
}

impl GaussianCamera {
    /// Creates a camera looking along +Z from the origin with identity rotation.
    #[must_use]
    pub fn look_along_z(width: u32, height: u32, fx: f32, fy: f32) -> Self {
        Self {
            width,
            height,
            fx,
            fy,
            cx: width as f32 * 0.5,
            cy: height as f32 * 0.5,
            eye: Vec3::new(0.0, 0.0, 0.0),
            rotation_camera_from_world: Quat::<f32>::identity(),
        }
    }
}

/// Packed RGBA8 framebuffer from [`render_gaussians_cpu`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GaussianFramebuffer {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Interleaved RGBA bytes (`width * height * 4`).
    pub rgba: Vec<u8>,
}

impl GaussianFramebuffer {
    /// Returns pixel count.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

/// Renders a Gaussian scene with a CPU alpha soft-splat (front-to-back).
///
/// Each primitive is projected with the pinhole camera, sorted by depth, and
/// composited with a screen-space isotropic Gaussian whose radius tracks the
/// projected max scale. This is a portable reference path; a GPU rasterizer can
/// later share the same [`GaussianScene`] container.
pub fn render_gaussians_cpu(
    scene: &GaussianScene,
    camera: &GaussianCamera,
) -> SceneResult<GaussianFramebuffer> {
    if camera.width == 0 || camera.height == 0 {
        return Err(SceneError::InvalidConfiguration(
            "camera dimensions must be non-zero".into(),
        ));
    }
    if !(camera.fx.is_finite() && camera.fy.is_finite() && camera.fx > 0.0 && camera.fy > 0.0) {
        return Err(SceneError::InvalidConfiguration(
            "fx/fy must be finite and > 0".into(),
        ));
    }

    let rot = camera.rotation_camera_from_world.normalize().to_mat3();
    let mut projected = Vec::with_capacity(scene.primitives.len());
    for prim in &scene.primitives {
        let world = prim.mean - camera.eye;
        let cam = rot.mul_vec3(world);
        if !(cam.z.is_finite() && cam.z > 1e-4) {
            continue;
        }
        let u = camera.fx * (cam.x / cam.z) + camera.cx;
        let v = camera.fy * (cam.y / cam.z) + camera.cy;
        let radius_world = prim.scale.x.max(prim.scale.y).max(prim.scale.z);
        let radius_px = (camera.fx * radius_world / cam.z).max(0.5);
        projected.push(ProjectedSplat {
            u,
            v,
            depth: cam.z,
            radius_px,
            opacity: prim.opacity,
            color: prim.color,
        });
    }
    projected.sort_by(|a, b| {
        a.depth
            .partial_cmp(&b.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let pixels = (camera.width as usize) * (camera.height as usize);
    let mut color = vec![0.0f32; pixels * 3];
    let mut alpha = vec![0.0f32; pixels];

    for splat in &projected {
        let r = splat.radius_px * 3.0;
        let min_x = ((splat.u - r).floor() as i32).max(0) as u32;
        let max_x = ((splat.u + r).ceil() as i32).min(camera.width as i32 - 1) as u32;
        let min_y = ((splat.v - r).floor() as i32).max(0) as u32;
        let max_y = ((splat.v + r).ceil() as i32).min(camera.height as i32 - 1) as u32;
        let sigma2 = (splat.radius_px * splat.radius_px).max(1e-4);
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 + 0.5 - splat.u;
                let dy = y as f32 + 0.5 - splat.v;
                let w = (-0.5 * (dx * dx + dy * dy) / sigma2).exp() * splat.opacity;
                if w <= 1e-5 {
                    continue;
                }
                let idx = (y as usize) * (camera.width as usize) + (x as usize);
                let one_m_a = 1.0 - alpha[idx];
                let contrib = w * one_m_a;
                color[idx * 3] += splat.color[0] * contrib;
                color[idx * 3 + 1] += splat.color[1] * contrib;
                color[idx * 3 + 2] += splat.color[2] * contrib;
                alpha[idx] += contrib;
            }
        }
    }

    let mut rgba = Vec::with_capacity(pixels * 4);
    for i in 0..pixels {
        rgba.push(to_u8(color[i * 3]));
        rgba.push(to_u8(color[i * 3 + 1]));
        rgba.push(to_u8(color[i * 3 + 2]));
        rgba.push(to_u8(alpha[i]));
    }
    Ok(GaussianFramebuffer {
        width: camera.width,
        height: camera.height,
        rgba,
    })
}

struct ProjectedSplat {
    u: f32,
    v: f32,
    depth: f32,
    radius_px: f32,
    opacity: f32,
    color: [f32; 3],
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::{
        render_gaussians_cpu, GaussianCamera, GaussianPrimitive, GaussianScene,
    };
    use spatialrust_math::{Quat, Vec3};

    #[test]
    fn renders_opaque_splat_near_center() {
        let mut scene = GaussianScene::new();
        scene
            .push(GaussianPrimitive {
                mean: Vec3::new(0.0, 0.0, 2.0),
                scale: Vec3::new(0.15, 0.15, 0.15),
                rotation: Quat::<f32>::identity(),
                opacity: 1.0,
                color: [1.0, 0.0, 0.0],
            })
            .unwrap();
        let camera = GaussianCamera::look_along_z(32, 32, 40.0, 40.0);
        let fb = render_gaussians_cpu(&scene, &camera).unwrap();
        assert_eq!(fb.rgba.len(), 32 * 32 * 4);
        let center = ((16 * 32 + 16) * 4) as usize;
        assert!(fb.rgba[center] > 20, "expected red contribution at center");
        assert!(fb.rgba[center + 3] > 20, "expected non-zero alpha");
    }

    #[test]
    fn rejects_non_positive_scale() {
        let mut scene = GaussianScene::new();
        let err = scene
            .push(GaussianPrimitive {
                mean: Vec3::new(0.0, 0.0, 1.0),
                scale: Vec3::new(0.0, 1.0, 1.0),
                rotation: Quat::<f32>::identity(),
                opacity: 1.0,
                color: [1.0, 1.0, 1.0],
            })
            .unwrap_err();
        assert!(err.to_string().contains("scale"));
    }
}

