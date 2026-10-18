use crate::image::{GpuAiTensor, GpuImage, GpuImageBorder, GpuVisionChainOptions};
use crate::{GpuStageProfiler, WgpuRuntime};
use spatialrust_core::SpatialResult;

/// Runs resize → gray → blur → Sobel → morphology → planar AI packing.
///
/// The caller owns the uploaded source. Every intermediate is recycled after
/// its consumer has been submitted, and no host readback occurs.
pub fn previous(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    options: GpuVisionChainOptions,
) -> SpatialResult<GpuAiTensor> {
    previous_with_profiler(runtime, source, options, None)
}

/// Runs the resident image chain with six to eight compute-pass timing samples.
///
/// Queries remain on the device until the caller explicitly collects the
/// profiler. Small blur and morphology workloads need six unused stages. Blur
/// boxes with area at least 81 and at least 65,536 pixel/sample visits add one
/// for the unrounded horizontal-sum pass. Morphology with area at least 81
/// and at least 1,048,576 visits adds a horizontal extrema sample.
pub fn previous_profiled(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    options: GpuVisionChainOptions,
    profiler: &mut GpuStageProfiler,
) -> SpatialResult<GpuAiTensor> {
    profiler.validate(
        runtime,
        4 + crate::image::kernels::box_blur::blur_pass_count(
            options.width,
            options.height,
            options.blur_kernel,
            options.blur_kernel,
        ) + crate::image::kernels::spatial::morphology_pass_count(
            options.width,
            options.height,
            options.morphology_kernel,
            options.morphology_kernel,
        ),
    )?;
    let result = previous_with_profiler(runtime, source, options, Some(profiler));
    profiler.finish_recording(&result);
    result
}

pub(crate) fn previous_with_profiler(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    options: GpuVisionChainOptions,
    mut profiler: Option<&mut GpuStageProfiler>,
) -> SpatialResult<GpuAiTensor> {
    let resized = crate::image::kernels::spatial::resize_nearest_gpu_with_profiler(
        runtime,
        source,
        options.width,
        options.height,
        profiler.as_deref_mut(),
    )?;
    let gray = crate::image::kernels::gray::rgb_to_gray_gpu_with_profiler(
        runtime,
        &resized,
        profiler.as_deref_mut(),
    )?;
    resized.recycle(runtime);
    let blurred = crate::image::kernels::box_blur::box_blur_gpu_with_profiler(
        runtime,
        &gray,
        options.blur_kernel,
        options.blur_kernel,
        GpuImageBorder::Replicate,
        profiler.as_deref_mut(),
    )?;
    gray.recycle(runtime);
    let edges = crate::image::kernels::spatial::sobel_gpu_with_profiler(
        runtime,
        &blurred,
        profiler.as_deref_mut(),
    )?;
    blurred.recycle(runtime);
    let morphology = crate::image::kernels::spatial::morphology_gpu_with_profiler(
        runtime,
        &edges,
        options.morphology_kernel,
        options.morphology_kernel,
        options.morphology,
        profiler.as_deref_mut(),
    )?;
    edges.recycle(runtime);
    let tensor = crate::image::ai_tensor::pack_ai_chw_gpu_with_profiler(
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
