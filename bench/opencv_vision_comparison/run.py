"""Numerically compare SpatialRust vision primitives with OpenCV.

Run after `maturin develop` so the native `spatialrust` module is importable.
The script exits non-zero when a compatibility tolerance is exceeded.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

import spatialrust as sr


def max_abs(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def main() -> None:
    rng = np.random.default_rng(75)
    image = rng.integers(0, 256, size=(73, 97, 3), dtype=np.uint8)
    size = (61, 43)
    results: dict[str, object] = {"opencv_version": cv2.__version__}

    resize_cases = {
        "nearest": getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST),
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
    }
    limits = {"nearest": 0, "bilinear": 1, "bicubic": 1, "area": 1}
    for name, flag in resize_cases.items():
        actual = sr.resize_image(image, size[0], size[1], interpolation=name)
        expected = cv2.resize(image, size, interpolation=flag)
        error = max_abs(actual, expected)
        results[f"resize_{name}_max_u8_error"] = error
        if error > limits[name]:
            raise AssertionError(f"{name} resize error {error} > {limits[name]}")

    gray = sr.rgb_to_gray_image(image)
    gray_cv = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_error = max_abs(gray, gray_cv)
    results["rgb_to_gray_max_u8_error"] = gray_error
    if gray_error > 1:
        raise AssertionError(f"gray error {gray_error} > 1")

    hsv = sr.rgb_to_hsv_image(image)
    hsv_cv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue_delta = np.abs(hsv[..., 0].astype(np.int16) - hsv_cv[..., 0].astype(np.int16))
    hue_error = int(np.max(np.minimum(hue_delta, 180 - hue_delta)))
    sv_error = max_abs(hsv[..., 1:], hsv_cv[..., 1:])
    results["rgb_to_hsv_max_hue_error"] = hue_error
    results["rgb_to_hsv_max_sv_error"] = sv_error
    if hue_error > 1 or sv_error > 1:
        raise AssertionError(f"HSV error exceeds tolerance: H={hue_error}, SV={sv_error}")

    grid_x, grid_y = np.meshgrid(
        np.arange(image.shape[1], dtype=np.float32),
        np.arange(image.shape[0], dtype=np.float32),
    )
    map_x = grid_x + np.float32(0.3125)
    map_y = grid_y - np.float32(0.1875)
    remapped = sr.remap_image(image, map_x, map_y, interpolation="bilinear")
    remapped_cv = cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    remap_error = max_abs(remapped, remapped_cv)
    results["remap_bilinear_max_u8_error"] = remap_error
    if remap_error > 1:
        raise AssertionError(f"remap error {remap_error} > 1")

    boxes_xyxy = np.array(
        [[0, 0, 20, 20], [2, 2, 18, 18], [30, 30, 45, 45], [32, 32, 46, 46]],
        dtype=np.float32,
    )
    scores = np.array([0.95, 0.8, 0.9, 0.7], dtype=np.float32)
    actual_indices = sr.nms(boxes_xyxy, scores, 0.1, 0.5).tolist()
    boxes_xywh = boxes_xyxy.copy()
    boxes_xywh[:, 2:] -= boxes_xywh[:, :2]
    cv_indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), 0.1, 0.5)
    expected_indices = np.asarray(cv_indices).reshape(-1).astype(int).tolist()
    results["nms_indices"] = actual_indices
    if actual_indices != expected_indices:
        raise AssertionError(f"NMS mismatch: {actual_indices} != {expected_indices}")

    mask = np.zeros((32, 40), dtype=np.uint8)
    mask[2:10, 3:12] = 1
    mask[16:30, 20:37] = 1
    _, stats = sr.connected_components_image(mask, connectivity=8)
    count_cv, _, stats_cv, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = sorted(stat[1] for stat in stats)
    areas_cv = sorted(int(value) for value in stats_cv[1:count_cv, cv2.CC_STAT_AREA])
    results["connected_component_areas"] = areas
    if areas != areas_cv:
        raise AssertionError(f"component areas mismatch: {areas} != {areas_cv}")

    results["status"] = "pass"
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
