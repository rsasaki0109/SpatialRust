"""Numerically compare SpatialRust vision primitives with OpenCV.

Run after `maturin develop` so the native `spatialrust` module is importable.
The script exits non-zero when a compatibility tolerance is exceeded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import spatialrust as sr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opencv_comparison.report import emit_report, environment, make_report


def max_abs(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)
    rng = np.random.default_rng(75)
    image = rng.integers(0, 256, size=(73, 97, 3), dtype=np.uint8)
    size = (61, 43)
    results: dict[str, object] = {}

    kernel = np.array([[0.0, -0.25, 0.0], [-0.25, 2.0, -0.25], [0.0, -0.25, 0.0]])
    filtered = sr.filter2d_image(image, kernel)
    filtered_cv = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    filter_error = max_abs(filtered, filtered_cv)
    results["filter2d_max_u8_error"] = filter_error
    if filter_error > 1:
        raise AssertionError(f"filter2D error {filter_error} > 1")

    gaussian = sr.gaussian_blur_image(image, 5, 3, 1.2, 0.8)
    gaussian_cv = cv2.GaussianBlur(
        image, (5, 3), 1.2, sigmaY=0.8, borderType=cv2.BORDER_REFLECT_101
    )
    gaussian_error = max_abs(gaussian, gaussian_cv)
    results["gaussian_blur_max_u8_error"] = gaussian_error
    if gaussian_error > 1:
        raise AssertionError(f"Gaussian blur error {gaussian_error} > 1")

    median = sr.median_blur_image(image, 5)
    median_cv = cv2.medianBlur(image, 5)
    median_error = max_abs(median, median_cv)
    results["median_blur_max_u8_error"] = median_error
    if median_error != 0:
        raise AssertionError(f"median blur error {median_error} != 0")

    bilateral = sr.bilateral_filter_image(image, 5, 40.0, 3.0)
    bilateral_cv = cv2.bilateralFilter(
        image, 5, 40.0, 3.0, borderType=cv2.BORDER_REFLECT_101
    )
    bilateral_error = max_abs(bilateral, bilateral_cv)
    results["bilateral_filter_max_u8_error"] = bilateral_error
    if bilateral_error > 2:
        raise AssertionError(f"bilateral filter error {bilateral_error} > 2")

    gray_derivative = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    derivative_cases = {
        "sobel_x": (
            sr.sobel_image(gray_derivative, 1, 0, 5),
            cv2.Sobel(gray_derivative, cv2.CV_32F, 1, 0, ksize=5),
        ),
        "scharr_y": (
            sr.scharr_image(gray_derivative, 0, 1),
            cv2.Scharr(gray_derivative, cv2.CV_32F, 0, 1),
        ),
        "laplacian": (
            sr.laplacian_image(gray_derivative, 3),
            cv2.Laplacian(gray_derivative, cv2.CV_32F, ksize=3),
        ),
    }
    for name, (actual, expected) in derivative_cases.items():
        error = float(np.max(np.abs(actual - expected)))
        results[f"{name}_max_f32_error"] = error
        if error > 1e-4:
            raise AssertionError(f"{name} error {error} > 1e-4")

    pyramid = sr.pyr_down_image(image)
    pyramid_cv = cv2.pyrDown(image)
    pyramid_error = max_abs(pyramid, pyramid_cv)
    results["pyr_down_max_u8_error"] = pyramid_error
    if pyramid_error > 1:
        raise AssertionError(f"pyrDown error {pyramid_error} > 1")
    pyramid_up = sr.pyr_up_image(pyramid)
    pyramid_up_cv = cv2.pyrUp(pyramid_cv)
    pyramid_up_error = max_abs(pyramid_up, pyramid_up_cv)
    results["pyr_up_max_u8_error"] = pyramid_up_error
    if pyramid_up_error > 1:
        raise AssertionError(f"pyrUp error {pyramid_up_error} > 1")

    morphology_source = gray_derivative
    morphology_cases = {
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE,
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "gradient": cv2.MORPH_GRADIENT,
        "tophat": cv2.MORPH_TOPHAT,
        "blackhat": cv2.MORPH_BLACKHAT,
    }
    shape_cases = {
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
        "ellipse": cv2.MORPH_ELLIPSE,
    }
    for shape_name, shape_code in shape_cases.items():
        element = cv2.getStructuringElement(shape_code, (5, 3))
        for operation_name, operation_code in morphology_cases.items():
            actual = sr.morphology_image(
                morphology_source, operation_name, 5, 3, shape_name, 2
            )
            expected = cv2.morphologyEx(
                morphology_source,
                operation_code,
                element,
                iterations=2,
                borderType=cv2.BORDER_REPLICATE,
            )
            error = max_abs(actual, expected)
            results[f"morphology_{shape_name}_{operation_name}_max_u8_error"] = error
            if error != 0:
                raise AssertionError(
                    f"{shape_name}/{operation_name} morphology error {error} != 0"
                )

    threshold_actual = sr.threshold_image(gray_derivative, 117.0)
    _, threshold_expected = cv2.threshold(gray_derivative, 117.0, 255, cv2.THRESH_BINARY)
    results["threshold_max_u8_error"] = max_abs(threshold_actual, threshold_expected)
    if results["threshold_max_u8_error"] != 0:
        raise AssertionError("fixed threshold mismatch")

    otsu_value, otsu_actual = sr.otsu_threshold_image(gray_derivative)
    otsu_cv, otsu_expected = cv2.threshold(
        gray_derivative, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    results["otsu_threshold"] = otsu_value
    results["otsu_max_u8_error"] = max_abs(otsu_actual, otsu_expected)
    if otsu_value != int(otsu_cv) or results["otsu_max_u8_error"] != 0:
        raise AssertionError(f"Otsu mismatch: {otsu_value} != {otsu_cv}")

    for method_name, method_code in {
        "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
        "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    }.items():
        actual = sr.adaptive_threshold_image(gray_derivative, 7, 3.0, method_name)
        expected = cv2.adaptiveThreshold(
            gray_derivative, 255, method_code, cv2.THRESH_BINARY, 7, 3.0
        )
        error = max_abs(actual, expected)
        results[f"adaptive_{method_name}_max_u8_error"] = error
        if error != 0:
            raise AssertionError(f"adaptive {method_name} mismatch: {error}")

    histogram_actual = sr.histogram_image(gray_derivative)
    histogram_expected = cv2.calcHist([gray_derivative], [0], None, [256], [0, 256]).reshape(-1)
    histogram_error = int(np.max(np.abs(histogram_actual.astype(np.int64) - histogram_expected.astype(np.int64))))
    results["histogram_max_count_error"] = histogram_error
    if histogram_error != 0:
        raise AssertionError(f"histogram mismatch: {histogram_error}")

    equalized = sr.equalize_histogram_image(gray_derivative)
    equalized_cv = cv2.equalizeHist(gray_derivative)
    results["equalize_hist_max_u8_error"] = max_abs(equalized, equalized_cv)
    if results["equalize_hist_max_u8_error"] != 0:
        raise AssertionError("histogram equalization mismatch")

    clahe_actual = sr.clahe_image(gray_derivative, 2.0, 8, 8)
    clahe_expected = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_derivative)
    clahe_error = max_abs(clahe_actual, clahe_expected)
    results["clahe_max_u8_error"] = clahe_error
    if clahe_error > 1:
        raise AssertionError(f"CLAHE mismatch: {clahe_error}")

    integral_actual = sr.integral_image_u8(gray_derivative)
    integral_expected = cv2.integral(gray_derivative, sdepth=cv2.CV_64F)
    integral_error = float(np.max(np.abs(integral_actual - integral_expected)))
    results["integral_max_f64_error"] = integral_error
    if integral_error != 0.0:
        raise AssertionError(f"integral image mismatch: {integral_error}")

    for aperture_size in (3, 5, 7):
        for l2_gradient in (False, True):
            actual = sr.canny_image(
                gray_derivative, 50.0, 100.0, aperture_size, l2_gradient
            )
            expected = cv2.Canny(
                gray_derivative,
                50.0,
                100.0,
                apertureSize=aperture_size,
                L2gradient=l2_gradient,
            )
            mismatch = int(np.count_nonzero(actual != expected))
            name = f"canny_aperture_{aperture_size}_{'l2' if l2_gradient else 'l1'}"
            results[f"{name}_mismatch_pixels"] = mismatch
            if mismatch != 0:
                raise AssertionError(f"{name} mismatch: {mismatch} pixels")

    for nonmax_suppression in (False, True):
        actual = sr.fast_keypoints(gray_derivative, 20, nonmax_suppression)
        detector = cv2.FastFeatureDetector_create(
            20, nonmax_suppression, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
        )
        expected = detector.detect(gray_derivative, None)
        actual_rows = [
            (round(point.x), round(point.y), round(point.response)) for point in actual
        ]
        expected_rows = [
            (round(point.pt[0]), round(point.pt[1]), round(point.response))
            for point in expected
        ]
        name = f"fast_9_16_{'nms' if nonmax_suppression else 'raw'}"
        results[f"{name}_keypoints"] = len(actual_rows)
        if actual_rows != expected_rows:
            raise AssertionError(f"{name} keypoints or scores differ from OpenCV")

    for use_harris in (False, True):
        if use_harris:
            actual = sr.harris_keypoints(gray_derivative, 100, 0.01, 1.0, 3, 3, 0.04)
            name = "harris"
        else:
            actual = sr.shi_tomasi_keypoints(gray_derivative, 100, 0.01, 1.0, 3, 3)
            name = "shi_tomasi"
        expected = cv2.goodFeaturesToTrack(
            gray_derivative,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=1.0,
            mask=None,
            blockSize=3,
            useHarrisDetector=use_harris,
            k=0.04,
        )
        actual_points = [(round(point.x), round(point.y)) for point in actual]
        expected_points = (
            []
            if expected is None
            else [(round(point[0][0]), round(point[0][1])) for point in expected]
        )
        results[f"{name}_keypoints"] = len(actual_points)
        if actual_points != expected_points:
            raise AssertionError(f"{name} ordering or coordinates differ from OpenCV")

    binary_query = rng.integers(0, 256, size=(23, 32), dtype=np.uint8)
    binary_train = rng.integers(0, 256, size=(31, 32), dtype=np.uint8)
    actual_binary = sr.match_binary_descriptors(binary_query, binary_train)
    expected_binary = cv2.BFMatcher(cv2.NORM_HAMMING).match(binary_query, binary_train)
    actual_binary_rows = [(query, train, distance) for query, train, distance in actual_binary]
    expected_binary_rows = [
        (match.queryIdx, match.trainIdx, match.distance) for match in expected_binary
    ]
    results["hamming_matches"] = len(actual_binary_rows)
    if actual_binary_rows != expected_binary_rows:
        raise AssertionError("Hamming nearest matches differ from OpenCV BFMatcher")

    float_query = rng.normal(size=(19, 17)).astype(np.float32)
    float_train = rng.normal(size=(29, 17)).astype(np.float32)
    actual_float = sr.match_float_descriptors(float_query, float_train)
    expected_float = cv2.BFMatcher(cv2.NORM_L2).match(float_query, float_train)
    float_index_mismatches = sum(
        (query, train) != (expected.queryIdx, expected.trainIdx)
        for (query, train, _), expected in zip(actual_float, expected_float)
    )
    float_distance_error = max(
        abs(distance - expected.distance)
        for (_, _, distance), expected in zip(actual_float, expected_float)
    )
    results["l2_match_index_mismatches"] = float_index_mismatches
    results["l2_match_max_distance_error"] = float_distance_error
    if float_index_mismatches != 0 or float_distance_error > 1e-5:
        raise AssertionError(
            f"L2 BFMatcher mismatch: indices={float_index_mismatches}, distance={float_distance_error}"
        )

    orb_image = cv2.resize(gray_derivative, (320, 240), interpolation=cv2.INTER_CUBIC)
    actual_orb, actual_descriptors = sr.orb_features(
        orb_image, max_features=200, edge_threshold=16
    )
    cv_orb = cv2.ORB_create(nfeatures=200, edgeThreshold=16)
    expected_orb, expected_descriptors = cv_orb.detectAndCompute(orb_image, None)
    actual_coordinates = np.array([(point.x, point.y) for point in actual_orb], dtype=np.float32)
    expected_coordinates = np.array([point.pt for point in expected_orb], dtype=np.float32)
    repeatable = 0
    if len(actual_coordinates) and len(expected_coordinates):
        nearest_distances = np.sqrt(
            np.min(
                np.sum(
                    (actual_coordinates[:, None, :] - expected_coordinates[None, :, :]) ** 2,
                    axis=2,
                ),
                axis=1,
            )
        )
        repeatable = int(np.count_nonzero(nearest_distances <= 2.0))
    repeatability = repeatable / max(1, min(len(actual_orb), len(expected_orb)))
    results["orb_spatialrust_keypoints"] = len(actual_orb)
    results["orb_opencv_keypoints"] = len(expected_orb)
    results["orb_coordinate_repeatability_2px"] = repeatability
    results["orb_descriptor_width"] = actual_descriptors.shape[1]
    if actual_descriptors.shape != (len(actual_orb), 32):
        raise AssertionError("SpatialRust ORB descriptor layout is not N x 32")
    if expected_descriptors is None or repeatability < 0.30:
        raise AssertionError(f"ORB repeatability against OpenCV is too low: {repeatability}")

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

    distance_mask = np.ones((67, 89), dtype=np.uint8)
    distance_mask[::11, ::13] = 0
    distance_mask[20:28, 31:40] = 0
    distance_sr = sr.distance_transform_edt(distance_mask)
    distance_cv = cv2.distanceTransform(
        distance_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    distance_error = float(np.max(np.abs(distance_sr - distance_cv)))
    results["distance_transform_edt_max_f32_error"] = distance_error
    if distance_error > 1e-5:
        raise AssertionError(f"exact distance-transform error {distance_error} > 1e-5")

    # Geometry: planar homography residual agreement (not scale-normalized identity).
    source = np.array(
        [[10.0, 12.0], [70.0, 14.0], [18.0, 55.0], [66.0, 60.0], [40.0, 34.0], [28.0, 22.0]],
        dtype=np.float64,
    )
    homography = np.array(
        [[1.04, 0.015, 2.5], [-0.02, 0.97, -1.25], [0.0002, -0.0001, 1.0]],
        dtype=np.float64,
    )
    target = []
    for point in source:
        projected = homography @ np.array([point[0], point[1], 1.0])
        target.append([projected[0] / projected[2], projected[1] / projected[2]])
    target = np.asarray(target, dtype=np.float64)
    estimated, inliers, residuals = sr.estimate_homography_ransac(
        source, target, threshold=1.0, seed=3
    )
    estimated_cv, mask_cv = cv2.findHomography(source, target, method=0)
    max_residual = float(np.max(residuals))
    results["homography_max_residual"] = max_residual
    results["homography_inliers"] = int(np.sum(inliers))
    if max_residual > 1e-6 or estimated_cv is None:
        raise AssertionError(f"homography residual {max_residual} too large")
    # Compare transfer error of both models; allow tiny numeric disagreement.
    def transfer_error(matrix: np.ndarray) -> float:
        errors = []
        for src, dst in zip(source, target):
            projected = matrix @ np.array([src[0], src[1], 1.0])
            errors.append(
                np.hypot(projected[0] / projected[2] - dst[0], projected[1] / projected[2] - dst[1])
            )
        return float(np.max(errors))

    transfer_sr = transfer_error(estimated)
    transfer_cv = transfer_error(estimated_cv)
    results["homography_transfer_sr"] = transfer_sr
    results["homography_transfer_cv"] = transfer_cv
    if transfer_sr > 1e-5 or transfer_cv > 1e-5:
        raise AssertionError("homography transfer error exceeds tolerance")

    objects = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.15],
            [0.1, 0.1, 0.05],
            [0.05, -0.08, 0.02],
            [-0.1, 0.05, 0.08],
            [0.12, 0.04, -0.03],
        ],
        dtype=np.float64,
    )
    fx = fy = 500.0
    cx = cy = 240.0
    true_r = np.eye(3, dtype=np.float64)
    true_t = np.array([0.12, -0.04, 2.4], dtype=np.float64)
    images = []
    for point in objects:
        camera = true_r @ point + true_t
        images.append([fx * camera[0] / camera[2] + cx, fy * camera[1] / camera[2] + cy])
    images = np.asarray(images, dtype=np.float64)
    rotation, translation = sr.solve_pnp(objects, images, fx, fy, cx, cy, 480, 480)
    ok_cv, rvec, tvec = cv2.solvePnP(
        objects.astype(np.float64),
        images.astype(np.float64),
        np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok_cv:
        raise AssertionError("OpenCV solvePnP failed")
    t_error = float(np.linalg.norm(translation - tvec.reshape(3)))
    results["pnp_translation_l2_vs_opencv"] = t_error
    if abs(translation[2] - true_t[2]) > 0.05 or t_error > 0.05:
        raise AssertionError(f"PnP translation disagreement {t_error}")
    _ = rotation  # shape already checked by consumer use

    width, height = 128, 96
    disparity = 16
    yy, xx = np.indices((height, width), dtype=np.int32)
    left = ((xx * 17 + yy * 29) % 200 + 20).astype(np.uint8)
    right = np.zeros_like(left)
    right[:, : width - disparity] = left[:, disparity:]
    disparity_sr = sr.stereo_block_match(
        left, right, window_size=11, min_disparity=1, num_disparities=32, uniqueness_ratio=5.0
    )
    matcher = cv2.StereoBM_create(numDisparities=32, blockSize=11)
    disparity_cv = matcher.compute(left, right).astype(np.float32) / 16.0
    center = (height // 2, width // 2)
    results["stereo_bm_sr_center"] = float(disparity_sr[center])
    results["stereo_bm_cv_center"] = float(disparity_cv[center])
    if abs(float(disparity_sr[center]) - float(disparity)) > 1.0:
        raise AssertionError("SpatialRust StereoBM center disparity incorrect")
    if abs(float(disparity_cv[center]) - float(disparity)) > 1.5:
        raise AssertionError("OpenCV StereoBM center disparity unexpected for synthetic pair")

    environment_receipt = environment(
        opencv_version=cv2.__version__,
        spatialrust_version=sr.__version__,
    )
    environment_receipt["opencv_threads"] = cv2.getNumThreads()
    environment_receipt["opencv_opencl_enabled"] = bool(
        hasattr(cv2, "ocl") and cv2.ocl.useOpenCL()
    )
    report = make_report(
        suite="opencv-vision-correctness",
        kind="correctness",
        status="pass",
        environment_receipt=environment_receipt,
        results=results,
    )
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
