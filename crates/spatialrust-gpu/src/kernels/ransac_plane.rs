use bytemuck::{Pod, Zeroable};
use spatialrust_core::{SpatialError, SpatialResult};
use wgpu::util::DeviceExt;

use crate::runtime::WgpuRuntime;

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RansacPlaneUniform {
    point_count: u32,
    hypothesis_count: u32,
    distance_threshold: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RansacHypothesisPod {
    i0: u32,
    i1: u32,
    i2: u32,
    _pad: u32,
}

/// GPU score for one RANSAC plane hypothesis.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuPlaneScore {
    /// Number of inlier points for this hypothesis.
    pub inlier_count: u32,
    /// Unit plane normal `(nx, ny, nz)`.
    pub normal: [f32; 3],
    /// Plane offset term in Hessian form.
    pub d: f32,
}

const RANSAC_PLANE_WGSL: &str = r#"
struct Params {
    point_count: u32,
    hypothesis_count: u32,
    distance_threshold: f32,
    pad: f32,
};

struct Hypothesis {
    i0: u32,
    i1: u32,
    i2: u32,
    pad: u32,
};

struct Score {
    inlier_count: u32,
    nx: f32,
    ny: f32,
    nz: f32,
    d: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> xs: array<f32>;
@group(0) @binding(2) var<storage, read> ys: array<f32>;
@group(0) @binding(3) var<storage, read> zs: array<f32>;
@group(0) @binding(4) var<storage, read> hypotheses: array<Hypothesis>;
@group(0) @binding(5) var<storage, read_write> scores: array<Score>;

fn plane_from_indices(i0: u32, i1: u32, i2: u32) -> Score {
    let p0 = vec3<f32>(xs[i0], ys[i0], zs[i0]);
    let p1 = vec3<f32>(xs[i1], ys[i1], zs[i1]);
    let p2 = vec3<f32>(xs[i2], ys[i2], zs[i2]);
    let v1 = p1 - p0;
    let v2 = p2 - p0;
    var normal = cross(v1, v2);
    if (dot(normal, normal) < 1e-12) {
        return Score(0u, 0.0, 0.0, 0.0, 0.0);
    }
    normal = normalize(normal);
    let d = -dot(normal, p0);
    var count = 0u;
    for (var i: u32 = 0u; i < params.point_count; i = i + 1u) {
        let dist = abs(normal.x * xs[i] + normal.y * ys[i] + normal.z * zs[i] + d);
        if (dist <= params.distance_threshold) {
            count = count + 1u;
        }
    }
    return Score(count, normal.x, normal.y, normal.z, d);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= params.hypothesis_count) {
        return;
    }
    let hyp = hypotheses[h];
    scores[h] = plane_from_indices(hyp.i0, hyp.i1, hyp.i2);
}
"#;

/// Scores RANSAC plane hypotheses in parallel on the GPU.
pub fn score_ransac_plane_hypotheses_gpu(
    runtime: &WgpuRuntime,
    x: &[f32],
    y: &[f32],
    z: &[f32],
    hypotheses: &[[u32; 3]],
    distance_threshold: f32,
) -> SpatialResult<Vec<GpuPlaneScore>> {
    if x.len() != y.len() || x.len() != z.len() {
        return Err(SpatialError::InvalidArgument(
            "xyz arrays must have equal length".to_owned(),
        ));
    }
    if hypotheses.is_empty() {
        return Ok(Vec::new());
    }

    let device = runtime.device();
    let queue = runtime.queue();
    let point_count = x.len();

    let x_buffer = runtime.upload_f32_storage("ransac-plane-x", x)?;
    let y_buffer = runtime.upload_f32_storage("ransac-plane-y", y)?;
    let z_buffer = runtime.upload_f32_storage("ransac-plane-z", z)?;

    let hypothesis_pods: Vec<RansacHypothesisPod> = hypotheses
        .iter()
        .map(|indices| RansacHypothesisPod {
            i0: indices[0],
            i1: indices[1],
            i2: indices[2],
            _pad: 0,
        })
        .collect();
    let hypothesis_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ransac-plane-hypotheses"),
        contents: bytemuck::cast_slice(&hypothesis_pods),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let hypothesis_count = hypotheses.len();
    let output_len = (hypothesis_count * std::mem::size_of::<GpuPlaneScore>()) as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ransac-plane-scores"),
        size: output_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let uniform = RansacPlaneUniform {
        point_count: point_count as u32,
        hypothesis_count: hypothesis_count as u32,
        distance_threshold,
        _pad: 0,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ransac-plane-uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ransac-plane-shader"),
        source: wgpu::ShaderSource::Wgsl(RANSAC_PLANE_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ransac-plane-pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ransac-plane-bind-group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: x_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: z_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: hypothesis_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output_buffer.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ransac-plane"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ransac-plane-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            hypothesis_count.div_ceil(WORKGROUP_SIZE as usize) as u32,
            1,
            1,
        );
    }
    queue.submit(Some(encoder.finish()));

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ransac-plane-staging"),
        size: output_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ransac-plane-readback"),
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_len);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| SpatialError::InvalidArgument("failed to receive wgpu map result".to_owned()))?
        .map_err(|error| {
            SpatialError::InvalidArgument(format!("failed to map wgpu buffer: {error}"))
        })?;

    let data = slice.get_mapped_range();
    let scores: Vec<GpuPlaneScore> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    runtime.recycle_storage(
        (point_count * std::mem::size_of::<f32>()) as u64,
        x_buffer,
    );
    runtime.recycle_storage(
        (point_count * std::mem::size_of::<f32>()) as u64,
        y_buffer,
    );
    runtime.recycle_storage(
        (point_count * std::mem::size_of::<f32>()) as u64,
        z_buffer,
    );

    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::{score_ransac_plane_hypotheses_gpu, GpuPlaneScore};
    use crate::runtime::WgpuRuntime;

    #[test]
    fn scores_planar_patch_hypothesis() {
        let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut z = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                x.push(i as f32);
                y.push(j as f32);
                z.push(0.0);
            }
        }
        let hypotheses = [[0u32, 1, 10], [5, 15, 50]];
        let scores = score_ransac_plane_hypotheses_gpu(
            &runtime,
            &x,
            &y,
            &z,
            &hypotheses,
            0.05,
        )
        .expect("gpu scores");
        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|score: &GpuPlaneScore| score.inlier_count >= 90));
        assert!(scores[0].normal[2].abs() > 0.9);
    }
}
