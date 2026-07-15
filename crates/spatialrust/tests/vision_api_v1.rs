//! Compile-and-behavior contract for the stable SpatialRust Vision 1.x entry surface.

use spatialrust::camera::{CameraIntrinsics, PinholeCamera};
use spatialrust::image::{ColorSpace, Image, ImageMetadata, ImageRegion};
use spatialrust::platform::{ApiStabilityClass, StabilityRegistry};
use spatialrust::vision::{
    filter2d, nms, resize, BorderMode, BoundingBox2, Interpolation, Kernel2D,
};

#[test]
fn stable_image_camera_and_vision_entry_points_compose() {
    let metadata = ImageMetadata { color_space: ColorSpace::Gray, ..ImageMetadata::default() };
    let image = Image::<u8, 1>::try_new_with_metadata(4, 3, (0..12).collect(), metadata).unwrap();
    let roi = image.view().subview(ImageRegion::new(1, 1, 2, 2)).unwrap();
    let resized = resize(roi, 4, 4, Interpolation::Bilinear).unwrap();
    let identity = Kernel2D::try_new(1, 1, vec![1.0]).unwrap();
    let filtered = filter2d(resized.view(), &identity, 0.0, BorderMode::Replicate).unwrap();
    assert_eq!((filtered.width(), filtered.height()), (4, 4));
    assert_eq!(filtered.metadata(), metadata);

    let intrinsics = CameraIntrinsics::try_new(100.0, 100.0, 1.5, 1.0, 4, 3).unwrap();
    let camera = PinholeCamera::new(intrinsics);
    let pixel = spatialrust::Vec2 { x: 1.5, y: 1.0 };
    let point = camera.unproject(pixel, 2.0).unwrap();
    assert_eq!(camera.project(point).unwrap(), pixel);

    let boxes = [
        BoundingBox2::try_new(0.0, 0.0, 3.0, 3.0).unwrap(),
        BoundingBox2::try_new(0.5, 0.5, 2.5, 2.5).unwrap(),
    ];
    assert_eq!(nms(&boxes, &[0.9, 0.8], 0.0, 0.4).unwrap(), vec![0]);

    let registry = StabilityRegistry::vision_v1_surface();
    assert_eq!(
        registry.lookup("spatialrust-vision::resize").unwrap().class,
        ApiStabilityClass::Stable
    );
}
