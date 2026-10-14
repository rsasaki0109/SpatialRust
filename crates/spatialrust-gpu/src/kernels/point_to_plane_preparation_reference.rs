// Frozen preparation and constructor from merged PR 203; native tests only.
use super::*;

pub(crate) fn previous_normal_preparation(
    target_normals: &[[f32; 3]],
) -> SpatialResult<(Vec<[f32; 4]>, f32)> {
    let mut normal_scale = 0.0_f32;
    for &v in target_normals.iter().flatten() {
        if !v.is_finite() {
            return Err(invalid());
        }
        normal_scale = normal_scale.max(v.abs());
    }
    if normal_scale == 0.0 {
        normal_scale = 1.0;
    }
    if normal_scale < f32::MIN_POSITIVE {
        return Err(invalid());
    }
    let mut packed = Vec::with_capacity(target_normals.len());
    for n in target_normals {
        let row = n.map(|v| v / normal_scale);
        if (0..3).any(|i| n[i] != 0.0 && row[i].abs() < f32::MIN_POSITIVE) {
            return Err(invalid());
        }
        packed.push([row[0], row[1], row[2], 0.0]);
    }
    Ok((packed, normal_scale))
}

pub(crate) fn previous_upload(
    runtime: &WgpuRuntime,
    source: &[[f32; 3]],
    target: &KdTree,
    target_normals: &[[f32; 3]],
    gpu_gather: bool,
) -> SpatialResult<GpuPointToPlaneInput> {
    if target.layout().positions().0.len() != target_normals.len() {
        return Err(invalid());
    }
    plan(source.len() as u64, target_normals.len() as u64, &runtime.device().limits())?;
    let mut normal_scale = 0.0_f32;
    for &v in target_normals.iter().flatten() {
        if !v.is_finite() {
            return Err(invalid());
        }
        normal_scale = normal_scale.max(v.abs());
    }
    if normal_scale == 0.0 {
        normal_scale = 1.0;
    }
    if normal_scale < f32::MIN_POSITIVE {
        return Err(invalid());
    }
    let mut packed = Vec::with_capacity(target_normals.len());
    for n in target_normals {
        let row = n.map(|v| v / normal_scale);
        if (0..3).any(|i| n[i] != 0.0 && row[i].abs() < f32::MIN_POSITIVE) {
            return Err(invalid());
        }
        packed.push([row[0], row[1], row[2], 0.0]);
    }
    let positions = if gpu_gather {
        GpuIcpInput::upload_with_gpu_gather(runtime, source, target)?
    } else {
        GpuIcpInput::upload(runtime, source, target)?
    };
    let normals = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("point-to-plane-normals"),
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut transfers = positions.upload_transfers();
    transfers.record(TransferDirection::HostToDevice, target_normals.len() as u64 * 16);
    Ok(GpuPointToPlaneInput { positions, normals, normal_scale, transfers })
}
