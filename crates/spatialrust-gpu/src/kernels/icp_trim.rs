//! Private resident stable distance/index selection for ICP updates.
use super::*;

pub(super) const SHADER: &str = include_str!("../shaders/icp_trim.wgsl");

/// Arithmetic used for retained ICP moments; transform/search remain f32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuIcpTrimPrecision {
    /// Canonical f32 normalized coordinates/products with compensated sums.
    /// Lower arithmetic cost, but coordinate/product rounding can bias moments.
    CompensatedF32,
    /// Two f32 words through offsets, normalization, mean division and products.
    /// Carries quotient/product residuals to reduce moment bias; hardware f32
    /// transforms/search and their gate/tie precision remain unchanged.
    #[default]
    DoubleSingle,
}

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
    reduction: std::sync::OnceLock<IcpReductionPipelines>,
    precise: std::sync::OnceLock<IcpReductionPipelines>,
    layout: wgpu::BindGroupLayout,
    histogram: wgpu::ComputePipeline,
    select: wgpu::ComputePipeline,
    wide_select: std::sync::OnceLock<wgpu::ComputePipeline>,
}

fn use_wide_selector(groups: u32, limits: &wgpu::Limits) -> bool {
    groups >= 64
        && limits.max_compute_invocations_per_workgroup >= 256
        && limits.max_compute_workgroup_size_x >= 256
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
            reduction: std::sync::OnceLock::new(),
            precise: std::sync::OnceLock::new(),
            layout,
            histogram,
            select,
            wide_select: std::sync::OnceLock::new(),
        }
    }

    fn selector(&self, device: &wgpu::Device, groups: u32) -> &wgpu::ComputePipeline {
        if !use_wide_selector(groups, &device.limits()) {
            return &self.select;
        }
        self.wide_select.get_or_init(|| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("icp-trim-wide-selector"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("icp-trim-wide-selector-layout"),
                bind_group_layouts: &[&self.layout],
                push_constant_ranges: &[],
            });
            let constants = std::collections::HashMap::from([("select_lanes".to_owned(), 256.0)]);
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("icp-trim-wide-selector"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("select_digit"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        })
    }

    pub(super) fn reduction(
        &self,
        device: &wgpu::Device,
        precision: GpuIcpTrimPrecision,
    ) -> &IcpReductionPipelines {
        match precision {
            GpuIcpTrimPrecision::CompensatedF32 => self.reduction.get_or_init(|| {
                IcpReductionPipelines::with_shader(device, &trimmed_shader(), true)
            }),
            GpuIcpTrimPrecision::DoubleSingle => self.precise.get_or_init(|| {
                IcpReductionPipelines::with_shader(device, &precise_shader(), true)
            }),
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
        let select = self.selector(runtime.device(), params.groups);
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
            dispatch(encoder, select, &group, 1, 1);
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

fn precise_shader() -> String {
    let mut shader = trimmed_shader();
    shader.push_str(include_str!("../shaders/icp_trim_moments.wgsl"));
    replace_once(&mut shader, "var value=0.0;", "var value=vec2<f32>(0.0);");
    replace_once(
        &mut shader,
        "value=((p-anchor)/params.scale)[term];",
        "value=ds_offset(p[term],anchor[term],params.scale);",
    );
    replace_once(
        &mut shader,
        "value=((q-params.target_anchor.xyz)/params.scale)[term-3u];",
        "value=ds_offset(q[term-3u],params.target_anchor[term-3u],params.scale);",
    );
    replace_once(&mut shader, "value=dot3(d,d);", "value=vec2<f32>(dot3(d,d),0.0);");
    replace_once(&mut shader, "if !finite(value) {", "if !finite(value.x) || !finite(value.y) {");
    replace_once(
        &mut shader,
        "total=add(total,value);",
        "if term==15u { total=add(total,value.x); } else { total=precise_merge(total,value); }",
    );
    replace_once(&mut shader, "result.values[term]=merge(vec2<f32>(total.x/f32(count),0.0),vec2<f32>(total.y/f32(count),0.0));", "result.values[term]=ds_divide(total,f32(count));");
    replace_once(
        &mut shader,
        "let mean_s=result.values[row].x+result.values[row].y;",
        "let mean_s=result.values[row];",
    );
    replace_once(
        &mut shader,
        "let mean_t=result.values[col+3u].x+result.values[col+3u].y;",
        "let mean_t=result.values[col+3u];",
    );
    replace_once(
        &mut shader,
        "let s=(queries[i].xyz-result.source_anchor.xyz)/params.scale;",
        "let s=ds_offset(queries[i][row],result.source_anchor[row],params.scale);",
    );
    replace_once(&mut shader, "let t=(target_points[matches[i].index].xyz-result.target_anchor.xyz)/params.scale;", "let t=ds_offset(target_points[matches[i].index][col],result.target_anchor[col],params.scale);");
    replace_once(
        &mut shader,
        "total=add(total,(s[row]-mean_s)*(t[col]-mean_t));",
        "total=precise_merge(total,ds_product(precise_merge(s,-mean_s),precise_merge(t,-mean_t)));",
    );
    replace_once(&mut shader, "if lane<stride { sums[lane]=merge(sums[lane],sums[lane+stride]); counts[lane]+=counts[lane+stride]; }", "if lane<stride { if term==15u { sums[lane]=merge(sums[lane],sums[lane+stride]); } else { sums[lane]=precise_merge(sums[lane],sums[lane+stride]); } counts[lane]+=counts[lane+stride]; }");
    let group_merge =
        "for (var g=0u;g<params.groups;g+=1u) { total=merge(total,partials[g].values[term]); }";
    assert_eq!(shader.matches(group_merge).count(), 2);
    shader=shader.replacen(group_merge,"for (var g=0u;g<params.groups;g+=1u) { if term==15u { total=merge(total,partials[g].values[term]); } else { total=precise_merge(total,partials[g].values[term]); } }",1);
    replace_once(
        &mut shader,
        "if lane<stride { sums[lane]=merge(sums[lane],sums[lane+stride]); }",
        "if lane<stride { sums[lane]=precise_merge(sums[lane],sums[lane+stride]); }",
    );
    // The remaining group merge belongs to covariance finalization.
    replace_once(&mut shader, "for (var g=0u;g<params.groups;g+=1u) { total=merge(total,partials[g].values[term]); }", "for (var g=0u;g<params.groups;g+=1u) { total=precise_merge(total,partials[g].values[term]); }");
    shader
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "icp_trim_tests.rs"]
mod tests;
