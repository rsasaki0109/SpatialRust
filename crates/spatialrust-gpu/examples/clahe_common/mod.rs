use spatialrust_gpu::{clahe_gpu, GpuImage, WgpuRuntime};
use spatialrust_image::Image;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub enum Mode {
    CpuHost,
    GpuResident,
    GpuReadback,
    UploadRoundTrip,
    GpuInputCpuReturnGpu,
    GpuInputCpuOutput,
}

pub const MODES: [Mode; 6] = [
    Mode::CpuHost,
    Mode::GpuResident,
    Mode::GpuReadback,
    Mode::UploadRoundTrip,
    Mode::GpuInputCpuReturnGpu,
    Mode::GpuInputCpuOutput,
];

pub struct Fixture {
    pub uploaded: GpuImage,
    cpu_source: GpuImage,
    image: Image<u8, 1>,
    expected: Image<u8, 1>,
    bytes: u64,
}

pub struct Sample {
    pub elapsed: Duration,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
}

impl Fixture {
    pub fn new(runtime: &WgpuRuntime, width: usize, height: usize, hot: bool) -> Self {
        let image = Image::<u8, 1>::try_new(
            width,
            height,
            (0..width * height)
                .map(|i| if hot { 43 } else { ((i * 37 + i / 997) % 129 + 32) as u8 })
                .collect(),
        )
        .unwrap();
        let expected = spatialrust_vision::clahe(image.view(), 2.0, 8, 8).unwrap();
        let uploaded = GpuImage::upload_u8(runtime, image.view()).unwrap();
        let cpu_source = GpuImage::upload_u8(runtime, image.view()).unwrap();
        runtime.wait_idle();
        Self { uploaded, cpu_source, image, expected, bytes: (width * height * 4) as u64 }
    }

    fn verify(&self, actual: &Image<u8, 1>) {
        assert!(actual.as_slice() == self.expected.as_slice(), "CLAHE byte mismatch");
        assert_eq!(actual.metadata(), self.expected.metadata());
    }

    pub fn sample(&mut self, runtime: &WgpuRuntime, mode: Mode) -> Sample {
        let (elapsed, h2d, d2h) = match mode {
            Mode::CpuHost => {
                let start = Instant::now();
                let output = spatialrust_vision::clahe(self.image.view(), 2.0, 8, 8).unwrap();
                let elapsed = start.elapsed();
                self.verify(&output);
                (elapsed, 0, 0)
            }
            Mode::GpuResident | Mode::GpuReadback => {
                let start = Instant::now();
                let mut output = clahe_gpu(runtime, &self.uploaded, 2.0, 8, 8).unwrap();
                let host = if matches!(mode, Mode::GpuReadback) {
                    Some(output.readback_u8::<1>(runtime).unwrap())
                } else {
                    runtime.wait_idle();
                    None
                };
                let elapsed = start.elapsed();
                let readback = output.receipt().device_to_host_bytes();
                // Resident byte verification is an explicit full read AFTER timing.
                self.verify(&host.unwrap_or_else(|| output.readback_u8::<1>(runtime).unwrap()));
                output.recycle(runtime);
                (elapsed, 0, readback)
            }
            Mode::UploadRoundTrip => {
                let start = Instant::now();
                let source = GpuImage::upload_u8(runtime, self.image.view()).unwrap();
                let mut output = clahe_gpu(runtime, &source, 2.0, 8, 8).unwrap();
                let host = output.readback_u8::<1>(runtime).unwrap();
                let elapsed = start.elapsed();
                self.verify(&host);
                output.recycle(runtime);
                source.recycle(runtime);
                (elapsed, self.bytes, self.bytes)
            }
            Mode::GpuInputCpuReturnGpu | Mode::GpuInputCpuOutput => {
                let start = Instant::now();
                let host = self.cpu_source.readback_u8::<1>(runtime).unwrap();
                let equalized = spatialrust_vision::clahe(host.view(), 2.0, 8, 8).unwrap();
                let uploaded = if matches!(mode, Mode::GpuInputCpuReturnGpu) {
                    let output = GpuImage::upload_u8(runtime, equalized.view()).unwrap();
                    runtime.wait_idle();
                    Some(output)
                } else {
                    None
                };
                let elapsed = start.elapsed();
                self.verify(&equalized);
                if let Some(mut output) = uploaded {
                    self.verify(&output.readback_u8::<1>(runtime).unwrap());
                    output.recycle(runtime);
                }
                (
                    elapsed,
                    if matches!(mode, Mode::GpuInputCpuReturnGpu) { self.bytes } else { 0 },
                    self.bytes,
                )
            }
        };
        Sample { elapsed, host_to_device_bytes: h2d, device_to_host_bytes: d2h }
    }

    pub fn recycle(self, runtime: &WgpuRuntime) {
        self.uploaded.recycle(runtime);
        self.cpu_source.recycle(runtime);
    }
}
