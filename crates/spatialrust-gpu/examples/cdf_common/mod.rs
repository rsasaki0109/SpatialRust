use spatialrust_gpu::{
    histogram_cdf_gpu, histogram_u8_gpu, GpuImage, GpuImageHistogram, WgpuRuntime,
};
use spatialrust_image::Image;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub enum Mode {
    CpuScan,
    Resident,
    CdfReadback,
    BinsReadbackCpu,
    ImageChainResident,
    ImageChainReadback,
    ImageReadbackCpu,
}

pub const MODES: [Mode; 7] = [
    Mode::CpuScan,
    Mode::Resident,
    Mode::CdfReadback,
    Mode::BinsReadbackCpu,
    Mode::ImageChainResident,
    Mode::ImageChainReadback,
    Mode::ImageReadbackCpu,
];

pub struct Fixture {
    pub uploaded: GpuImage,
    pub histogram: GpuImageHistogram,
    cpu_histogram: GpuImageHistogram,
    cpu_image: GpuImage,
    bins: [u32; 256],
    expected: [u32; 256],
    image_bytes: u64,
}

pub struct Sample {
    pub elapsed: Duration,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
}

fn inclusive(bins: &[u32; 256]) -> [u32; 256] {
    let mut sum = 0;
    std::array::from_fn(|i| {
        sum += bins[i];
        sum
    })
}

impl Fixture {
    pub fn new(runtime: &WgpuRuntime, width: usize, height: usize, hot: bool) -> Self {
        let pixels: Vec<_> = (0..width * height)
            .map(|i| if hot { 43 } else { ((i * 37 + i / 997) % 256) as u8 })
            .collect();
        let mut bins = [0u32; 256];
        for &value in &pixels {
            bins[value as usize] += 1;
        }
        let expected = inclusive(&bins);
        let image = Image::<u8, 1>::try_new(width, height, pixels).unwrap();
        let uploaded = GpuImage::upload_u8(runtime, image.view()).unwrap();
        let cpu_image = GpuImage::upload_u8(runtime, image.view()).unwrap();
        let histogram = histogram_u8_gpu(runtime, &uploaded, 0).unwrap();
        let cpu_histogram = histogram_u8_gpu(runtime, &uploaded, 0).unwrap();
        runtime.wait_idle();
        Self {
            uploaded,
            histogram,
            cpu_histogram,
            cpu_image,
            bins,
            expected,
            image_bytes: (width * height * 4) as u64,
        }
    }

    pub fn sample(&mut self, runtime: &WgpuRuntime, mode: Mode) -> Sample {
        let (elapsed, device_to_host_bytes) = match mode {
            Mode::CpuScan => {
                let start = Instant::now();
                let counts = inclusive(std::hint::black_box(&self.bins));
                let elapsed = start.elapsed();
                assert_eq!(counts, self.expected);
                (elapsed, 0)
            }
            Mode::Resident | Mode::CdfReadback => {
                let start = Instant::now();
                let mut cdf = histogram_cdf_gpu(runtime, &self.histogram).unwrap();
                let counts = if matches!(mode, Mode::CdfReadback) {
                    Some(cdf.readback_u32(runtime).unwrap())
                } else {
                    runtime.wait_idle();
                    None
                };
                let elapsed = start.elapsed();
                let bytes = cdf.receipt().device_to_host_bytes();
                assert_eq!(
                    counts.unwrap_or_else(|| cdf.readback_u32(runtime).unwrap()),
                    self.expected
                );
                cdf.recycle(runtime);
                (elapsed, bytes)
            }
            Mode::BinsReadbackCpu => {
                let start = Instant::now();
                let bins = self.cpu_histogram.readback_u32(runtime).unwrap();
                let counts = inclusive(&bins);
                let elapsed = start.elapsed();
                assert_eq!(counts, self.expected);
                (elapsed, 1024)
            }
            Mode::ImageChainResident | Mode::ImageChainReadback => {
                let start = Instant::now();
                let histogram = histogram_u8_gpu(runtime, &self.uploaded, 0).unwrap();
                let mut cdf = histogram_cdf_gpu(runtime, &histogram).unwrap();
                let counts = if matches!(mode, Mode::ImageChainReadback) {
                    Some(cdf.readback_u32(runtime).unwrap())
                } else {
                    runtime.wait_idle();
                    None
                };
                let elapsed = start.elapsed();
                let bytes = cdf.receipt().device_to_host_bytes();
                assert_eq!(
                    counts.unwrap_or_else(|| cdf.readback_u32(runtime).unwrap()),
                    self.expected
                );
                cdf.recycle(runtime);
                histogram.recycle(runtime);
                (elapsed, bytes)
            }
            Mode::ImageReadbackCpu => {
                let start = Instant::now();
                let image = self.cpu_image.readback_u8::<1>(runtime).unwrap();
                let bins = spatialrust_vision::histogram_u8(image.view());
                let mut sum = 0u64;
                let counts: [u64; 256] = std::array::from_fn(|i| {
                    sum += bins[i];
                    sum
                });
                let elapsed = start.elapsed();
                assert!(counts.into_iter().eq(self.expected.map(u64::from)));
                (elapsed, self.image_bytes)
            }
        };
        Sample { elapsed, host_to_device_bytes: 0, device_to_host_bytes }
    }

    pub fn recycle(self, runtime: &WgpuRuntime) {
        self.histogram.recycle(runtime);
        self.cpu_histogram.recycle(runtime);
        self.uploaded.recycle(runtime);
        self.cpu_image.recycle(runtime);
    }
}
