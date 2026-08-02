use spatialrust::{
    math::Vec3,
    viewer::{NativeViewer, NativeViewerOptions, ViewerState, ViewportSize},
    viz::{Camera, Projection},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let camera = Camera::try_new(
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Projection::Perspective {
            vertical_fov_radians: 60.0_f32.to_radians(),
            near: 0.01,
            far: 10_000.0,
        },
    )?;
    let state = ViewerState::try_new(camera, ViewportSize::try_new(1280, 720)?)?;
    NativeViewer::try_new(state, NativeViewerOptions::default())?.run()?;
    Ok(())
}
