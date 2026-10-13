//! Private resident stable distance/index selection for ICP updates.
use super::*;

pub(super) const SHADER: &str = include_str!("../shaders/icp_trim.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct TrimParams {
    count: u32,
    groups: u32,
    step: u32,
    mantissa_lo: u32,
    mantissa_hi: u32,
    shift: u32,
    bound: f32,
    targets: u32,
}

pub(super) fn plan_trim(count: u64, limits: &wgpu::Limits) -> SpatialResult<u32> {
    let groups = plan(count, limits)?;
    let bound = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    if u64::from(groups) * 1024 > bound
        || limits.max_storage_buffers_per_shader_stage < 7
        || limits.max_bindings_per_bind_group < 8
        || limits.max_compute_workgroup_storage_size < 2048
    {
        return Err(invalid());
    }
    Ok(groups)
}

impl TrimParams {
    fn new(count: u32, groups: u32, fraction: f64, bound: f32, targets: u32) -> Self {
        let bits = fraction.to_bits();
        let exponent = ((bits >> 52) & 2047) as u32;
        let mantissa = (bits & ((1u64 << 52) - 1)) | if exponent == 0 { 0 } else { 1u64 << 52 };
        Self {
            count,
            groups,
            step: 0,
            mantissa_lo: mantissa as u32,
            mantissa_hi: (mantissa >> 32) as u32,
            shift: if exponent == 0 { 1074 } else { 1075 - exponent },
            bound,
            targets,
        }
    }
}

pub(crate) struct TrimPipelines {
    pub(super) reduction: IcpReductionPipelines,
    layout: wgpu::BindGroupLayout,
    histogram: wgpu::ComputePipeline,
    select: wgpu::ComputePipeline,
}

impl TrimPipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..4)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: binding == 1 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("icp-trim-layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("icp-trim-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("icp-trim-shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let [histogram, select] = ["histogram_prefix", "select_digit"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("icp-trim-pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Self {
            reduction: IcpReductionPipelines::with_shader(device, &trimmed_shader(), true),
            layout,
            histogram,
            select,
        }
    }

    pub(super) fn encode(
        &self,
        runtime: &WgpuRuntime,
        encoder: &mut wgpu::CommandEncoder,
        scratch: &mut ScratchBuffers<'_>,
        params: TrimParams,
        matches: &wgpu::Buffer,
        selection: &wgpu::Buffer,
    ) {
        let histograms = scratch.acquire("icp-trim-histograms", u64::from(params.groups) * 1024);
        encoder.clear_buffer(selection, 0, None);
        // Source indices have known leading zero bytes; do not dispatch those
        // digits. Distance still uses all four bytes, without a quantized cutoff.
        let first_index_step = 4 + params.count.saturating_sub(1).leading_zeros() / 8;
        for step in (0..4).chain(first_index_step..8) {
            let uniform = runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("icp-trim-params"),
                contents: bytemuck::bytes_of(&TrimParams { step, ..params }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let entries: Vec<_> = [&uniform, matches, &histograms, selection]
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            let group = runtime.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("icp-trim-bindings"),
                layout: &self.layout,
                entries: &entries,
            });
            dispatch(encoder, &self.histogram, &group, params.groups, 1);
            dispatch(encoder, &self.select, &group, 1, 1);
        }
    }
}

pub(super) fn encode_selection(
    runtime: &WgpuRuntime,
    encoder: &mut wgpu::CommandEncoder,
    scratch: &mut ScratchBuffers<'_>,
    params: &Params,
    fraction: f64,
    matches: &wgpu::Buffer,
    selection: &wgpu::Buffer,
) {
    runtime.icp_trim_pipelines.get().unwrap().encode(
        runtime,
        encoder,
        scratch,
        TrimParams::new(params.count, params.groups, fraction, params.bound, params.targets),
        matches,
        selection,
    );
}

fn replace_once(shader: &mut String, from: &str, to: &str) {
    assert_eq!(shader.matches(from).count(), 1, "ICP trim shader anchor changed: {from}");
    *shader = shader.replacen(from, to, 1);
}

// Keep the untrimmed numerical kernels canonical; change only selection/count
// semantics. Assertions make later edits to the shared shader fail explicitly.
fn trimmed_shader() -> String {
    let mut shader = include_str!("../shaders/icp_reduction.wgsl").replace("\r\n", "\n");
    shader.push_str("\nstruct Selection { gated:u32, retained:u32, rank:u32, distance:u32, source_index:u32, unused0:u32, unused1:u32, unused2:u32 }\n@group(0) @binding(7) var<storage,read> selection: Selection;\nfn retained_match(i:u32)->bool { let distance=bitcast<u32>(matches[i].distance)&0x7fffffffu; return selection.retained>0u && (distance<selection.distance || (distance==selection.distance && i<=selection.source_index)); }\n");
    replace_once(&mut shader, "if !valid_match(i) { continue; }\n        let p=", "if !valid_match(i) { continue; }\n        if term!=15u && !retained_match(i) { continue; }\n        let p=");
    replace_once(
        &mut shader,
        "if !valid_match(i) { continue; }\n            let s=",
        "if !valid_match(i) || !retained_match(i) { continue; }\n            let s=",
    );
    replace_once(&mut shader, "result.count=count;", "result.count=count; result.reserved0=selection.gated;\n    if count!=selection.retained { atomicOr(&result.status,2u); }");
    replace_once(
        &mut shader,
        "if count>0u {\n                if term<6u",
        "if count>0u || term==15u {\n                if term<6u",
    );
    replace_once(&mut shader, "if atomicLoad(&result.status)!=0u || result.count==0u {", "if result.count==0u { result.source_anchor=vec4<f32>(0.0); result.target_anchor=vec4<f32>(0.0); }\n    if atomicLoad(&result.status)!=0u || result.reserved0==0u {");
    replace_once(
        &mut shader,
        "result.count=0u; result.source_anchor=",
        "result.count=0u; result.reserved0=0u; result.source_anchor=",
    );
    shader
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "icp_trim_tests.rs"]
mod tests;
