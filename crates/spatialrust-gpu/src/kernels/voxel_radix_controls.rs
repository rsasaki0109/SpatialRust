//! Immutable, aligned per-digit uniforms for one complete radix invocation.

use super::RadixParams;
use wgpu::util::DeviceExt;

const PASSES: u64 = 24;
const PARAM_BYTES: u64 = std::mem::size_of::<RadixParams>() as u64;
// Bound both host padding and the larger uniform allocation independently of N.
const MAX_BATCH_BYTES: u64 = 65_536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Layout {
    stride: u64,
    bytes: usize,
}

fn layout(limits: &wgpu::Limits) -> Option<Layout> {
    let alignment = u64::from(limits.min_uniform_buffer_offset_alignment);
    if !alignment.is_power_of_two()
        || u64::from(limits.max_uniform_buffer_binding_size) < PARAM_BYTES
    {
        return None;
    }
    let stride = PARAM_BYTES.div_ceil(alignment).checked_mul(alignment)?;
    let bytes = stride.checked_mul(PASSES - 1)?.checked_add(PARAM_BYTES)?;
    if bytes > MAX_BATCH_BYTES || bytes > limits.max_buffer_size {
        return None;
    }
    Some(Layout { stride, bytes: usize::try_from(bytes).ok()? })
}

fn packed(layout: Layout, point_count: u32, group_count: u32) -> Vec<u8> {
    let mut bytes = vec![0u8; layout.bytes];
    for digit in 0..PASSES {
        let params = RadixParams {
            point_count,
            group_count,
            axis: 2 - digit as u32 / 8,
            shift: digit as u32 % 8 * 4,
        };
        let start = (digit * layout.stride) as usize;
        bytes[start..start + PARAM_BYTES as usize].copy_from_slice(bytemuck::bytes_of(&params));
    }
    bytes
}

pub(super) struct RadixControls {
    buffer: wgpu::Buffer,
    stride: u64,
}

impl RadixControls {
    pub(super) fn new(device: &wgpu::Device, point_count: u32, group_count: u32) -> Option<Self> {
        let layout = layout(&device.limits())?;
        let bytes = packed(layout, point_count, group_count);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-radix-batched-params"),
            contents: &bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        Some(Self { buffer, stride: layout.stride })
    }

    pub(super) fn binding(&self, axis: u32, shift: u32) -> wgpu::BindingResource<'_> {
        debug_assert!(axis < 3 && shift < 32 && shift % 4 == 0);
        let digit = u64::from((2 - axis) * 8 + shift / 4);
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: digit * self.stride,
            size: std::num::NonZeroU64::new(PARAM_BYTES),
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "voxel_radix_controls_tests.rs"]
mod tests;
