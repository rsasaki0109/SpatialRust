//! Minimal owned-image SpatialRust Vision 1 workflow.

use spatialrust::vision::{gray_world_white_balance, resize, ExposureFusionOptions, Interpolation};
use spatialrust::Image;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image =
        Image::<u8, 3>::try_new(2, 2, vec![40, 80, 120, 50, 90, 130, 60, 100, 140, 70, 110, 150])?;
    let balanced = gray_world_white_balance(image.view())?;
    let preview = resize(balanced.view(), 4, 4, Interpolation::Bilinear)?;
    assert_eq!((preview.width(), preview.height()), (4, 4));
    let _documented_defaults = ExposureFusionOptions::default();
    println!("Vision 1 CPU workflow: {} RGB bytes", preview.as_slice().len());
    Ok(())
}
