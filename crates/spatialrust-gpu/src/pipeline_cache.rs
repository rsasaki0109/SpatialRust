//! Cached wgpu compute pipelines and bind group layouts for voxel kernels.

/// Cached compute pipelines shared across voxel kernel dispatches.
pub struct ComputePipelineCache {
    /// Voxel key generation pipelines.
    pub voxel_keys: VoxelKeysPipelines,
    /// Interleaved AoSoA voxel key generation pipelines.
    pub voxel_keys_aoso: VoxelKeysAoSoPipelines,
    /// Interleaved AoSoA centroid reduction pipelines.
    pub voxel_reduce_aoso: VoxelReduceAoSoPipelines,
    /// Interleaved AoSoA attribute reduction pipelines.
    pub voxel_reduce_attributes_aoso: VoxelReduceAttributesAoSoPipelines,
    /// Bitonic sort pipelines.
    pub voxel_sort: VoxelSortPipelines,
    /// Sort-entry construction pipelines.
    pub voxel_sort_build: VoxelSortBuildPipelines,
    /// Post-sort valid-entry gather pipelines.
    pub voxel_sort_filter: VoxelSortFilterPipelines,
    /// Segment compact pipelines.
    pub voxel_compact: VoxelCompactPipelines,
    /// Voxel reduction pipelines.
    pub voxel_reduce: VoxelReducePipelines,
    /// Voxel first-point gather pipelines.
    pub voxel_gather: VoxelGatherPipelines,
}

/// Cached pipelines for voxel key generation.
pub struct VoxelKeysPipelines {
    /// Bind group layout for voxel key dispatch.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Main voxel key compute pipeline.
    pub pipeline: wgpu::ComputePipeline,
    _pipeline_layout: wgpu::PipelineLayout,
    _shader: wgpu::ShaderModule,
}

/// Cached pipelines for voxel keys from interleaved XYZ positions.
pub struct VoxelKeysAoSoPipelines {
    /// Bind group layout for interleaved voxel key dispatch.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Main interleaved voxel key compute pipeline.
    pub pipeline: wgpu::ComputePipeline,
    _pipeline_layout: wgpu::PipelineLayout,
    _shader: wgpu::ShaderModule,
}

/// Cached pipelines for centroid reduction from interleaved XYZ positions.
pub struct VoxelReduceAoSoPipelines {
    /// Bind group layout for interleaved centroid reduction.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Main interleaved centroid reduction pipeline.
    pub pipeline: wgpu::ComputePipeline,
    _pipeline_layout: wgpu::PipelineLayout,
    _shader: wgpu::ShaderModule,
}

/// Cached pipelines for interleaved attribute record reduction.
pub struct VoxelReduceAttributesAoSoPipelines {
    /// Bind group layout for attribute reduction.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Main attribute reduction pipeline.
    pub pipeline: wgpu::ComputePipeline,
    _pipeline_layout: wgpu::PipelineLayout,
    _shader: wgpu::ShaderModule,
}

/// Cached pipelines for bitonic voxel sorting.
pub struct VoxelSortPipelines {
    /// Bind group layout for voxel sort dispatch.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Main bitonic sort pipeline.
    pub pipeline: wgpu::ComputePipeline,
    _pipeline_layout: wgpu::PipelineLayout,
    _shader: wgpu::ShaderModule,
}

/// Cached pipelines that build sort entries from voxel keys.
pub struct VoxelSortBuildPipelines {
    /// Bind group layout for sort-entry construction.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Sort-entry construction pipeline.
    pub pipeline: wgpu::ComputePipeline,
    _pipeline_layout: wgpu::PipelineLayout,
    _shader: wgpu::ShaderModule,
}

/// Cached pipelines that gather valid entries after sorting.
pub struct VoxelSortFilterPipelines {
    /// Bind group layout for post-sort filtering.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Marks valid entries.
    pub mark: wgpu::ComputePipeline,
    /// Initializes inclusive scan input.
    pub init: wgpu::ComputePipeline,
    /// Runs one inclusive-scan step.
    pub scan: wgpu::ComputePipeline,
    /// Scatters valid entries into dense output.
    pub scatter: wgpu::ComputePipeline,
    _pipeline_layout: wgpu::PipelineLayout,
    _shader: wgpu::ShaderModule,
}

/// Cached pipelines that compact sorted voxel entries into segments.
pub struct VoxelCompactPipelines {
    /// Bind group layout for segment compact dispatch.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Marks segment boundaries.
    pub mark: wgpu::ComputePipeline,
    /// Initializes inclusive scan input.
    pub init: wgpu::ComputePipeline,
    /// Runs one inclusive-scan step.
    pub scan: wgpu::ComputePipeline,
    /// Writes compact segment metadata.
    pub write: wgpu::ComputePipeline,
    _pipeline_layout: wgpu::PipelineLayout,
    _shader: wgpu::ShaderModule,
}

/// Cached pipelines for voxel reduction.
pub struct VoxelReducePipelines {
    /// Bind group layout for single-channel voxel reduce dispatch.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Single-channel voxel reduce pipeline.
    pub pipeline: wgpu::ComputePipeline,
    /// Bind group layout for xyz centroid reduce dispatch.
    pub xyz_bind_group_layout: wgpu::BindGroupLayout,
    /// XYZ centroid reduce pipeline.
    pub xyz_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for single-channel u8 voxel reduce dispatch.
    pub u8_bind_group_layout: wgpu::BindGroupLayout,
    /// Single-channel u8 voxel reduce pipeline.
    pub u8_pipeline: wgpu::ComputePipeline,
    _pipeline_layout: wgpu::PipelineLayout,
    _xyz_pipeline_layout: wgpu::PipelineLayout,
    _u8_pipeline_layout: wgpu::PipelineLayout,
    _shader: wgpu::ShaderModule,
    _xyz_shader: wgpu::ShaderModule,
    _u8_shader: wgpu::ShaderModule,
}

/// Cached pipelines for voxel first-point gather.
pub struct VoxelGatherPipelines {
    /// Bind group layout for single-channel gather.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Single-channel gather pipeline.
    pub pipeline: wgpu::ComputePipeline,
    /// Bind group layout for xyz gather.
    pub xyz_bind_group_layout: wgpu::BindGroupLayout,
    /// XYZ gather pipeline.
    pub xyz_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for single-channel u8 gather.
    pub u8_bind_group_layout: wgpu::BindGroupLayout,
    /// Single-channel u8 gather pipeline.
    pub u8_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for 2-channel gather.
    pub multi_bind_group_layout: wgpu::BindGroupLayout,
    /// 2-channel gather pipeline.
    pub multi_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for up-to-4-channel gather.
    pub multi4_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Up-to-4-channel gather pipeline.
    pub multi4_pipeline: Option<wgpu::ComputePipeline>,
    /// Bind group layout for fused xyz + 4 f32 attribute gather.
    pub xyz_attrs4_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Fused xyz + 4 f32 attribute gather pipeline.
    pub xyz_attrs4_pipeline: Option<wgpu::ComputePipeline>,
    /// Maximum channels supported by the cached multi gather pipelines.
    pub multi_max_channels: u32,
    _pipeline_layout: wgpu::PipelineLayout,
    _xyz_pipeline_layout: wgpu::PipelineLayout,
    _multi_pipeline_layout: wgpu::PipelineLayout,
    _multi4_pipeline_layout: Option<wgpu::PipelineLayout>,
    _xyz_attrs4_pipeline_layout: Option<wgpu::PipelineLayout>,
    _shader: wgpu::ShaderModule,
    _xyz_shader: wgpu::ShaderModule,
    _multi_shader: wgpu::ShaderModule,
    _multi4_shader: Option<wgpu::ShaderModule>,
    _xyz_attrs4_shader: Option<wgpu::ShaderModule>,
    _u8_pipeline_layout: wgpu::PipelineLayout,
    _u8_shader: wgpu::ShaderModule,
}

impl ComputePipelineCache {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            voxel_keys: VoxelKeysPipelines::new(device),
            voxel_keys_aoso: VoxelKeysAoSoPipelines::new(device),
            voxel_reduce_aoso: VoxelReduceAoSoPipelines::new(device),
            voxel_reduce_attributes_aoso: VoxelReduceAttributesAoSoPipelines::new(device),
            voxel_sort: VoxelSortPipelines::new(device),
            voxel_sort_build: VoxelSortBuildPipelines::new(device),
            voxel_sort_filter: VoxelSortFilterPipelines::new(device),
            voxel_compact: VoxelCompactPipelines::new(device),
            voxel_reduce: VoxelReducePipelines::new(device),
            voxel_gather: VoxelGatherPipelines::new(device),
        }
    }
}

impl VoxelReduceAttributesAoSoPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-reduce-attributes-aoso-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
            ],
        });
        let shader = load_shader(
            device,
            "voxel-reduce-attributes-aoso-shader",
            include_str!("shaders/voxel_reduce_attributes_aoso.wgsl"),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-reduce-attributes-aoso-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = build_compute_pipeline(
            device,
            &pipeline_layout,
            &shader,
            "voxel-reduce-attributes-aoso-pipeline",
            "main",
        );
        Self { bind_group_layout, pipeline, _pipeline_layout: pipeline_layout, _shader: shader }
    }
}

impl VoxelReduceAoSoPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-reduce-aoso-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
            ],
        });
        let shader = load_shader(
            device,
            "voxel-reduce-aoso-shader",
            include_str!("shaders/voxel_reduce_aoso.wgsl"),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-reduce-aoso-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = build_compute_pipeline(
            device,
            &pipeline_layout,
            &shader,
            "voxel-reduce-aoso-pipeline",
            "main",
        );
        Self { bind_group_layout, pipeline, _pipeline_layout: pipeline_layout, _shader: shader }
    }
}

impl VoxelKeysAoSoPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-key-aoso-layout"),
            entries: &[uniform_entry(0), storage_entry(1, true), storage_entry(2, false)],
        });
        let shader = load_shader(
            device,
            "voxel-key-aoso-shader",
            include_str!("shaders/voxel_keys_aoso.wgsl"),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-key-aoso-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = build_compute_pipeline(
            device,
            &pipeline_layout,
            &shader,
            "voxel-key-aoso-pipeline",
            "main",
        );
        Self { bind_group_layout, pipeline, _pipeline_layout: pipeline_layout, _shader: shader }
    }
}

impl VoxelKeysPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-key-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
            ],
        });
        let shader =
            load_shader(device, "voxel-key-shader", include_str!("shaders/voxel_keys.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-key-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline =
            build_compute_pipeline(device, &pipeline_layout, &shader, "voxel-key-pipeline", "main");
        Self { bind_group_layout, pipeline, _pipeline_layout: pipeline_layout, _shader: shader }
    }
}

impl VoxelSortPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-sort-layout"),
            entries: &[uniform_entry(0), storage_entry(1, false)],
        });
        let shader =
            load_shader(device, "voxel-sort-shader", include_str!("shaders/voxel_sort.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-sort-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = build_compute_pipeline(
            device,
            &pipeline_layout,
            &shader,
            "voxel-sort-pipeline",
            "main",
        );
        Self { bind_group_layout, pipeline, _pipeline_layout: pipeline_layout, _shader: shader }
    }
}

impl VoxelSortBuildPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-sort-build-layout"),
            entries: &[uniform_entry(0), storage_entry(1, true), storage_entry(2, false)],
        });
        let shader = load_shader(
            device,
            "voxel-sort-build-shader",
            include_str!("shaders/voxel_sort_entries.wgsl"),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-sort-build-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = build_compute_pipeline(
            device,
            &pipeline_layout,
            &shader,
            "voxel-sort-build-pipeline",
            "build_sort_entries",
        );
        Self { bind_group_layout, pipeline, _pipeline_layout: pipeline_layout, _shader: shader }
    }
}

impl VoxelSortFilterPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-sort-filter-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
        });
        let shader = load_shader(
            device,
            "voxel-sort-filter-shader",
            include_str!("shaders/voxel_sort_filter.wgsl"),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-sort-filter-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        Self {
            mark: build_compute_pipeline(
                device,
                &pipeline_layout,
                &shader,
                "voxel-sort-filter-mark",
                "mark_valid",
            ),
            init: build_compute_pipeline(
                device,
                &pipeline_layout,
                &shader,
                "voxel-sort-filter-init",
                "init_inclusive",
            ),
            scan: build_compute_pipeline(
                device,
                &pipeline_layout,
                &shader,
                "voxel-sort-filter-scan",
                "scan_step",
            ),
            scatter: build_compute_pipeline(
                device,
                &pipeline_layout,
                &shader,
                "voxel-sort-filter-scatter",
                "scatter_valid",
            ),
            bind_group_layout,
            _pipeline_layout: pipeline_layout,
            _shader: shader,
        }
    }
}

impl VoxelCompactPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-compact-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
            ],
        });
        let shader =
            load_shader(device, "voxel-compact-shader", include_str!("shaders/voxel_compact.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-compact-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        Self {
            mark: build_compute_pipeline(
                device,
                &pipeline_layout,
                &shader,
                "voxel-compact-mark",
                "mark_boundaries",
            ),
            init: build_compute_pipeline(
                device,
                &pipeline_layout,
                &shader,
                "voxel-compact-init",
                "init_inclusive",
            ),
            scan: build_compute_pipeline(
                device,
                &pipeline_layout,
                &shader,
                "voxel-compact-scan",
                "scan_step",
            ),
            write: build_compute_pipeline(
                device,
                &pipeline_layout,
                &shader,
                "voxel-compact-write",
                "write_segments",
            ),
            bind_group_layout,
            _pipeline_layout: pipeline_layout,
            _shader: shader,
        }
    }
}

impl VoxelReducePipelines {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-reduce-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
            ],
        });
        let shader =
            load_shader(device, "voxel-reduce-shader", include_str!("shaders/voxel_reduce.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-reduce-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = build_compute_pipeline(
            device,
            &pipeline_layout,
            &shader,
            "voxel-reduce-pipeline",
            "main",
        );

        let xyz_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel-reduce-xyz-layout"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, true),
                    storage_entry(6, false),
                    storage_entry(7, false),
                    storage_entry(8, false),
                ],
            });
        let xyz_shader = load_shader(
            device,
            "voxel-reduce-xyz-shader",
            include_str!("shaders/voxel_reduce_xyz.wgsl"),
        );
        let xyz_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-reduce-xyz-pipeline-layout"),
            bind_group_layouts: &[&xyz_bind_group_layout],
            push_constant_ranges: &[],
        });
        let xyz_pipeline = build_compute_pipeline(
            device,
            &xyz_pipeline_layout,
            &xyz_shader,
            "voxel-reduce-xyz-pipeline",
            "main",
        );

        let u8_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel-reduce-u8-layout"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, false),
                ],
            });
        let u8_shader = load_shader(
            device,
            "voxel-reduce-u8-shader",
            include_str!("shaders/voxel_reduce_u8.wgsl"),
        );
        let u8_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-reduce-u8-pipeline-layout"),
            bind_group_layouts: &[&u8_bind_group_layout],
            push_constant_ranges: &[],
        });
        let u8_pipeline = build_compute_pipeline(
            device,
            &u8_pipeline_layout,
            &u8_shader,
            "voxel-reduce-u8-pipeline",
            "main",
        );

        Self {
            bind_group_layout,
            pipeline,
            xyz_bind_group_layout,
            xyz_pipeline,
            u8_bind_group_layout,
            u8_pipeline,
            _pipeline_layout: pipeline_layout,
            _xyz_pipeline_layout: xyz_pipeline_layout,
            _u8_pipeline_layout: u8_pipeline_layout,
            _shader: shader,
            _xyz_shader: xyz_shader,
            _u8_shader: u8_shader,
        }
    }
}

impl VoxelGatherPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-gather-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
            ],
        });
        let shader =
            load_shader(device, "voxel-gather-shader", include_str!("shaders/voxel_gather.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-gather-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = build_compute_pipeline(
            device,
            &pipeline_layout,
            &shader,
            "voxel-gather-pipeline",
            "main",
        );

        let xyz_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel-gather-xyz-layout"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, true),
                    storage_entry(6, false),
                    storage_entry(7, false),
                    storage_entry(8, false),
                ],
            });
        let xyz_shader = load_shader(
            device,
            "voxel-gather-xyz-shader",
            include_str!("shaders/voxel_gather_xyz.wgsl"),
        );
        let xyz_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-gather-xyz-pipeline-layout"),
            bind_group_layouts: &[&xyz_bind_group_layout],
            push_constant_ranges: &[],
        });
        let xyz_pipeline = build_compute_pipeline(
            device,
            &xyz_pipeline_layout,
            &xyz_shader,
            "voxel-gather-xyz-pipeline",
            "main",
        );

        let u8_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel-gather-u8-layout"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, false),
                ],
            });
        let u8_shader = load_shader(
            device,
            "voxel-gather-u8-shader",
            include_str!("shaders/voxel_gather_u8.wgsl"),
        );
        let u8_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-gather-u8-pipeline-layout"),
            bind_group_layouts: &[&u8_bind_group_layout],
            push_constant_ranges: &[],
        });
        let u8_pipeline = build_compute_pipeline(
            device,
            &u8_pipeline_layout,
            &u8_shader,
            "voxel-gather-u8-pipeline",
            "main",
        );

        let multi_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel-gather-multi-layout"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, false),
                    storage_entry(6, false),
                ],
            });
        let multi_shader = load_shader(
            device,
            "voxel-gather-multi-shader",
            include_str!("shaders/voxel_gather_multi.wgsl"),
        );
        let multi_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("voxel-gather-multi-pipeline-layout"),
                bind_group_layouts: &[&multi_bind_group_layout],
                push_constant_ranges: &[],
            });
        let multi_pipeline = build_compute_pipeline(
            device,
            &multi_pipeline_layout,
            &multi_shader,
            "voxel-gather-multi-pipeline",
            "main",
        );

        let storage_limit = device.limits().max_storage_buffers_per_shader_stage;
        let multi_max_channels = if storage_limit >= crate::runtime::MULTI_GATHER4_STORAGE_BUFFERS {
            4
        } else if storage_limit >= crate::runtime::MULTI_GATHER2_STORAGE_BUFFERS {
            2
        } else {
            1
        };

        let (multi4_bind_group_layout, multi4_pipeline_layout, multi4_pipeline, multi4_shader) =
            if storage_limit >= crate::runtime::MULTI_GATHER4_STORAGE_BUFFERS {
                let multi4_bind_group_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("voxel-gather-multi4-layout"),
                        entries: &[
                            uniform_entry(0),
                            storage_entry(1, true),
                            storage_entry(2, true),
                            storage_entry(3, true),
                            storage_entry(4, true),
                            storage_entry(5, true),
                            storage_entry(6, true),
                            storage_entry(7, false),
                            storage_entry(8, false),
                            storage_entry(9, false),
                            storage_entry(10, false),
                        ],
                    });
                let multi4_shader = load_shader(
                    device,
                    "voxel-gather-multi4-shader",
                    include_str!("shaders/voxel_gather_multi4.wgsl"),
                );
                let multi4_pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("voxel-gather-multi4-pipeline-layout"),
                        bind_group_layouts: &[&multi4_bind_group_layout],
                        push_constant_ranges: &[],
                    });
                let multi4_pipeline = build_compute_pipeline(
                    device,
                    &multi4_pipeline_layout,
                    &multi4_shader,
                    "voxel-gather-multi4-pipeline",
                    "main",
                );
                (
                    Some(multi4_bind_group_layout),
                    Some(multi4_pipeline_layout),
                    Some(multi4_pipeline),
                    Some(multi4_shader),
                )
            } else {
                (None, None, None, None)
            };

        let (
            xyz_attrs4_bind_group_layout,
            xyz_attrs4_pipeline_layout,
            xyz_attrs4_pipeline,
            xyz_attrs4_shader,
        ) = if storage_limit >= crate::runtime::MULTI_GATHER4_STORAGE_BUFFERS {
            let xyz_attrs4_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("voxel-gather-xyz-attrs4-layout"),
                    entries: &[
                        uniform_entry(0),
                        storage_entry(1, true),
                        storage_entry(2, true),
                        storage_entry(3, true),
                        storage_entry(4, true),
                        storage_entry(5, true),
                        storage_entry(6, true),
                        storage_entry(7, true),
                        storage_entry(8, true),
                        storage_entry(9, true),
                        storage_entry(10, false),
                    ],
                });
            let xyz_attrs4_shader = load_shader(
                device,
                "voxel-gather-xyz-attrs4-shader",
                include_str!("shaders/voxel_gather_xyz_attrs4.wgsl"),
            );
            let xyz_attrs4_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("voxel-gather-xyz-attrs4-pipeline-layout"),
                    bind_group_layouts: &[&xyz_attrs4_bind_group_layout],
                    push_constant_ranges: &[],
                });
            let xyz_attrs4_pipeline = build_compute_pipeline(
                device,
                &xyz_attrs4_pipeline_layout,
                &xyz_attrs4_shader,
                "voxel-gather-xyz-attrs4-pipeline",
                "main",
            );
            (
                Some(xyz_attrs4_bind_group_layout),
                Some(xyz_attrs4_pipeline_layout),
                Some(xyz_attrs4_pipeline),
                Some(xyz_attrs4_shader),
            )
        } else {
            (None, None, None, None)
        };

        Self {
            bind_group_layout,
            pipeline,
            xyz_bind_group_layout,
            xyz_pipeline,
            u8_bind_group_layout,
            u8_pipeline,
            multi_bind_group_layout,
            multi_pipeline,
            multi4_bind_group_layout,
            multi4_pipeline,
            xyz_attrs4_bind_group_layout,
            xyz_attrs4_pipeline,
            multi_max_channels,
            _pipeline_layout: pipeline_layout,
            _xyz_pipeline_layout: xyz_pipeline_layout,
            _u8_pipeline_layout: u8_pipeline_layout,
            _multi_pipeline_layout: multi_pipeline_layout,
            _multi4_pipeline_layout: multi4_pipeline_layout,
            _xyz_attrs4_pipeline_layout: xyz_attrs4_pipeline_layout,
            _shader: shader,
            _xyz_shader: xyz_shader,
            _u8_shader: u8_shader,
            _multi_shader: multi_shader,
            _multi4_shader: multi4_shader,
            _xyz_attrs4_shader: xyz_attrs4_shader,
        }
    }
}

fn load_shader(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

fn build_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
