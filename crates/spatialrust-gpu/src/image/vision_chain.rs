//! Explicit upload-once GPU-resident Vision 2 chain.

use spatialrust_core::SpatialResult;

use super::{
    box_blur_gpu, morphology_gpu, pack_ai_chw_gpu, resize_nearest_gpu, rgb_to_gray_gpu, sobel_gpu,
    GpuAiTensor, GpuImage, GpuImageBorder, GpuMorphology,
};
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
/// The caller owns the uploaded source. Every intermediate is recycled after
/// its consumer has been submitted, and no host readback occurs.
pub fn run_gpu_vision_chain(
    runtime: &WgpuRuntime,
    source: &GpuImage,
    options: GpuVisionChainOptions,
) -> SpatialResult<GpuAiTensor> {
    let resized = resize_nearest_gpu(runtime, source, options.width, options.height)?;
    let gray = rgb_to_gray_gpu(runtime, &resized)?;
    resized.recycle(runtime);
    let blurred = box_blur_gpu(
        runtime,
        &gray,
        options.blur_kernel,
        options.blur_kernel,
        GpuImageBorder::Replicate,
    )?;
    gray.recycle(runtime);
    let edges = sobel_gpu(runtime, &blurred)?;
    blurred.recycle(runtime);
    let morphology = morphology_gpu(
        runtime,
        &edges,
        options.morphology_kernel,
        options.morphology_kernel,
        options.morphology,
    )?;
    edges.recycle(runtime);
    let tensor = pack_ai_chw_gpu(runtime, &morphology, options.scale, options.mean, options.std)?;
    morphology.recycle(runtime);
    Ok(tensor)
}
