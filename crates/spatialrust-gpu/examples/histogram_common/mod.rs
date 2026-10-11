use spatialrust_gpu::{histogram_u8_gpu, GpuImage, WgpuRuntime};
use spatialrust_image::Image;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Cpu,
    Resident,
    BinsReadback,
    UploadAndBins,
    ImageReadbackCpu,
}

pub const MODES: [Mode; 5] =
    [Mode::Cpu, Mode::Resident, Mode::BinsReadback, Mode::UploadAndBins, Mode::ImageReadbackCpu];

pub struct Fixture {
    pub image: Image<u8, 1>,
    pub uploaded: GpuImage,
    cpu_readback_source: GpuImage,
    pub expected: [u32; 256],
}

pub struct Sample {
    pub elapsed: Duration,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
}

impl Fixture {
    pub fn new(runtime: &WgpuRuntime, width: usize, height: usize, hot: bool) -> Self {
        let pixels: Vec<_> = (0..width * height)
            .map(|i| if hot { 43 } else { ((i * 37 + i / 997) % 256) as u8 })
            .collect();
        let mut expected = [0u32; 256];
        for &value in &pixels {
            expected[value as usize] += 1;
        }
        let image = Image::try_new(width, height, pixels).unwrap();
        let uploaded = GpuImage::upload_u8(runtime, image.view()).unwrap();
        // Keep full-image-readback accounting separate from the resident input.
        let cpu_readback_source = GpuImage::upload_u8(runtime, image.view()).unwrap();
        runtime.wait_idle();
        Self { image, uploaded, cpu_readback_source, expected }
    }

    pub fn sample(&mut self, runtime: &WgpuRuntime, mode: Mode) -> Sample {
        let physical_image_bytes = (self.image.width() * self.image.height() * 4) as u64;
        let (elapsed, host_to_device_bytes, device_to_host_bytes) = match mode {
            Mode::Cpu => {
                let start = Instant::now();
                let bins = spatialrust_vision::histogram_u8(self.image.view());
                let elapsed = start.elapsed();
                assert!(
                    bins.iter().copied().eq(self.expected.iter().map(|&n| u64::from(n))),
                    "CPU reference mismatch"
                );
                (elapsed, 0, 0)
            }
            Mode::ImageReadbackCpu => {
                let start = Instant::now();
                let host = self.cpu_readback_source.readback_u8::<1>(runtime).unwrap();
                let bins = spatialrust_vision::histogram_u8(host.view());
                let elapsed = start.elapsed();
                assert!(
                    bins.iter().copied().eq(self.expected.iter().map(|&n| u64::from(n))),
                    "image readback CPU mismatch"
                );
                (elapsed, 0, physical_image_bytes)
            }
            Mode::Resident | Mode::BinsReadback => {
                let start = Instant::now();
                let mut histogram = histogram_u8_gpu(runtime, &self.uploaded, 0).unwrap();
                let bins = if matches!(mode, Mode::BinsReadback) {
                    Some(histogram.readback_u32(runtime).unwrap())
                } else {
                    runtime.wait_idle();
                    None
                };
                let elapsed = start.elapsed();
                let readback = histogram.receipt().device_to_host_bytes();
                // Resident verification explicitly reads bins only after timing.
                let actual = bins.unwrap_or_else(|| histogram.readback_u32(runtime).unwrap());
                assert!(actual == self.expected, "GPU histogram mismatch");
                histogram.recycle(runtime);
                (elapsed, 0, readback)
            }
            Mode::UploadAndBins => {
                let start = Instant::now();
                let source = GpuImage::upload_u8(runtime, self.image.view()).unwrap();
                let mut histogram = histogram_u8_gpu(runtime, &source, 0).unwrap();
                let bins = histogram.readback_u32(runtime).unwrap();
                let elapsed = start.elapsed();
                assert!(bins == self.expected, "upload + GPU histogram mismatch");
                histogram.recycle(runtime);
                source.recycle(runtime);
                (elapsed, physical_image_bytes, 1024)
            }
        };
        Sample { elapsed, host_to_device_bytes, device_to_host_bytes }
    }

    pub fn recycle(self, runtime: &WgpuRuntime) {
        self.uploaded.recycle(runtime);
        self.cpu_readback_source.recycle(runtime);
    }
}
