//! Explicit upload-once GPU-resident Vision 2 chain.

use spatialrust_core::SpatialResult;

use super::{GpuAiTensor, GpuImage, GpuImageBorder, GpuMorphology};
use crate::WgpuRuntime;

/// Configuration for the resident resize-to-AI chain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuVisionChainOptions {
    /// Resize output width.
    pub width: u32,
    /// Resize output height.
    pub height: u32,
    /// Odd box-blur kernel size.
    pub blur_kernel: u32,
    /// Odd morphology kernel size.
    pub morphology_kernel: u32,
    /// Morphology operation after Sobel.
    pub morphology: GpuMorphology,
    /// Input-value scale for AI packing.
    pub scale: f32,
    /// Per-channel means.
    pub mean: [f32; 4],
    /// Per-channel standard deviations.
    pub std: [f32; 4],
}

impl Default for GpuVisionChainOptions {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            blur_kernel: 3,
            morphology_kernel: 3,
            morphology: GpuMorphology::Dilate,
            scale: 1.0 / 255.0,
            mean: [0.0; 4],
            std: [1.0; 4],
        }
    }
}

/// Runs resize → gray → blur → Sobel → morphology → planar AI packing.
///
/// Matching source/output dimensions omit resize and its device transfer stage.
/// The caller owns the uploaded source. Every intermediate is recycled after
/// its consumer has been submitted, and no host readback occurs.
pub fn run_gpu_vision_chain(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    options: GpuVisionChainOptions,
) -> SpatialResult<GpuAiTensor> {
    run_gpu_vision_chain_with_profiler(runtime, source, options, None)
}

/// Runs the resident image chain with five to eight compute-pass timing samples.
///
/// Queries remain on the device until the caller explicitly collects the
/// profiler. Small blur/morphology workloads need five unused stages for matching
/// dimensions or six when resizing. Blur boxes with area at least 81 and at
/// least 65,536 pixel/sample visits add one
/// for the unrounded horizontal-sum pass. Morphology with area at least 81
/// and at least 1,048,576 visits adds a horizontal extrema sample.
pub fn run_gpu_vision_chain_profiled(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    options: GpuVisionChainOptions,
    profiler: &mut crate::GpuStageProfiler,
) -> SpatialResult<GpuAiTensor> {
    profiler.validate(
        runtime,
        3 + u32::from(source.width() != options.width || source.height() != options.height)
            + super::kernels::box_blur::blur_pass_count(
                options.width,
                options.height,
                options.blur_kernel,
                options.blur_kernel,
            )
            + super::kernels::spatial::morphology_pass_count(
                options.width,
                options.height,
                options.morphology_kernel,
                options.morphology_kernel,
            ),
    )?;
    let result = run_gpu_vision_chain_with_profiler(runtime, source, options, Some(profiler));
    profiler.finish_recording(&result);
    result
}

pub(crate) fn run_gpu_vision_chain_with_profiler(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    options: GpuVisionChainOptions,
    mut profiler: Option<&mut crate::GpuStageProfiler>,
) -> SpatialResult<GpuAiTensor> {
    let resized = if source.width() == options.width && source.height() == options.height {
        // Preserve the original resize admission even when its identity copy is omitted.
        source.validate_runtime(runtime)?;
        super::kernels::spatial::validate_resize_dispatch(
            [source.width(), source.height()],
            [options.width, options.height],
            &runtime.device().limits(),
        )?;
        None
    } else {
        Some(super::kernels::spatial::resize_nearest_gpu_with_profiler(
            runtime,
            source,
            options.width,
            options.height,
            profiler.as_deref_mut(),
        )?)
    };
    let gray = super::kernels::gray::rgb_to_gray_gpu_with_profiler(
        runtime,
        resized.as_ref().unwrap_or(source),
        profiler.as_deref_mut(),
    )?;
    if let Some(resized) = resized {
        resized.recycle(runtime);
    }
    let blurred = super::kernels::box_blur::box_blur_gpu_with_profiler(
        runtime,
        &gray,
        options.blur_kernel,
        options.blur_kernel,
        GpuImageBorder::Replicate,
        profiler.as_deref_mut(),
    )?;
    gray.recycle(runtime);
    let edges = super::kernels::spatial::sobel_gpu_with_profiler(
        runtime,
        &blurred,
        profiler.as_deref_mut(),
    )?;
    blurred.recycle(runtime);
    let morphology = super::kernels::spatial::morphology_gpu_with_profiler(
        runtime,
        &edges,
        options.morphology_kernel,
        options.morphology_kernel,
        options.morphology,
        profiler.as_deref_mut(),
    )?;
    edges.recycle(runtime);
    let tensor = super::ai_tensor::pack_ai_chw_gpu_with_profiler(
        runtime,
        &morphology,
        options.scale,
        options.mean,
        options.std,
        profiler,
    )?;
    morphology.recycle(runtime);
    Ok(tensor)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "vision_chain_identity_tests.rs"]
mod identity_tests;
