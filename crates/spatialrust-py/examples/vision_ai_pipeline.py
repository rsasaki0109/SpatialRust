"""End-to-end image preprocessing, post-processing, and spatial AI demo."""

from __future__ import annotations

import numpy as np

import spatialrust as sr


def main() -> None:
    height, width = 48, 64
    yy, xx = np.mgrid[:height, :width]
    image = np.stack(
        [
            (xx * 4).clip(0, 255),
            (yy * 5).clip(0, 255),
            np.full_like(xx, 96),
        ],
        axis=-1,
    ).astype(np.uint8)

    model_image, transform = sr.letterbox_image(image, 64, 64)
    model_tensor = sr.normalize_image_chw(
        model_image,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    boxes = np.array([[8, 8, 30, 28], [10, 9, 29, 27], [38, 30, 58, 52]], np.float32)
    scores = np.array([0.94, 0.83, 0.88], np.float32)
    kept = sr.nms(boxes, scores, score_threshold=0.25, iou_threshold=0.5)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[8:24, 7:26] = 1
    mask[29:43, 39:58] = 1
    labels, components = sr.connected_components_image(mask)
    runs = sr.encode_mask_rle(mask)
    assert np.array_equal(sr.decode_mask_rle(width, height, runs), mask)

    z = np.ones((height, width), dtype=np.float32)
    z[29:43, 39:58] = 0.75
    points = np.stack(
        [
            (xx.astype(np.float32) - width / 2) * z / 60.0,
            (yy.astype(np.float32) - height / 2) * z / 60.0,
            z,
        ],
        axis=-1,
    )
    confidence = np.where(mask != 0, 0.95, 0.8).astype(np.float32)
    cloud = sr.point_map_to_point_cloud(points, confidence, min_confidence=0.5)
    pipeline = sr.run_pipeline(
        cloud,
        leaf_size=0.025,
        cluster_tolerance=0.1,
        min_cluster_size=1,
        plane_distance=0.02,
    )

    print(f"model tensor: {model_tensor.shape}, letterbox={transform}")
    print(f"detections kept: {kept.tolist()}")
    print(f"mask components: {len(components)}, labels={labels.max()}, RLE runs={len(runs)}")
    print(
        f"point cloud: {len(cloud)} points, plane inliers={pipeline.plane_inliers}, "
        f"output={len(pipeline.output)}"
    )


if __name__ == "__main__":
    main()
