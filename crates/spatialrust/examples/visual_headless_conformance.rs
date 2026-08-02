//! Fail-closed canonical renderer image and transfer conformance.

use std::sync::Arc;

use spatialrust::{
    gpu::WgpuRuntime,
    render_wgpu::{RenderOptions, WgpuRenderer},
    viz::{
        Camera, LinearRgba, PointCloudView, PointColor, PointStyle, PositionColumns3, Projection,
        VisualPrimitive, VisualStyle,
    },
    Vec3,
};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const EXPECTED_RGBA_FNV1A64: u64 = 8_072_337_613_282_577_229;

fn main() {
    let runtime =
        WgpuRuntime::new_headless().expect("Visual conformance requires a headless wgpu adapter");
    let renderer = WgpuRenderer::new(Arc::new(runtime));
    let positions = PositionColumns3::try_new(&[0.0], &[0.0], &[0.0]).unwrap();
    let points = PointCloudView::positions_only(positions);
    let (geometry, upload) = renderer.upload(VisualPrimitive::Points(points)).unwrap();
    assert_eq!(upload.total_bytes().unwrap(), 12);

    let camera = Camera::try_new(
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Projection::Perspective { vertical_fov_radians: 1.0, near: 0.1, far: 10.0 },
    )
    .unwrap();
    let style = VisualStyle::Points(
        PointStyle::try_new(
            9.0,
            PointColor::Uniform(LinearRgba::try_new(1.0, 0.0, 0.0, 1.0).unwrap()),
        )
        .unwrap(),
    );
    let options = RenderOptions::try_new(WIDTH, HEIGHT, camera, style, LinearRgba::BLACK).unwrap();
    let output = renderer.render_headless(&geometry, &options).unwrap();
    assert_eq!(output.transfers.total_bytes().unwrap(), 112);
    assert_eq!(output.receipt.draw_calls, 2);
    assert_eq!(output.receipt.element_count, 1);

    let image = renderer.readback_rgba(&output.target).unwrap();
    assert_eq!(image.transfers.total_bytes().unwrap(), u64::from(WIDTH * HEIGHT * 4));
    assert_eq!(image.rgba.len(), (WIDTH * HEIGHT * 4) as usize);
    assert_eq!(fnv1a64(&image.rgba), EXPECTED_RGBA_FNV1A64, "canonical Visual image changed");
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
