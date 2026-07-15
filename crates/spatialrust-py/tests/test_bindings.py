"""Smoke / contract tests for the SpatialRust Python bindings.

These exercise the NumPy <-> Rust boundary for every exported function: the
right shapes and dtypes come back, the documented signatures accept their
keyword arguments, and the algorithms produce sane results on tiny synthetic
clouds. They are intentionally fast (small clouds, loose assertions) so they
can gate every wheel build in CI — the point is "the binding actually imports
and runs", not numerical accuracy (covered by the Rust unit tests).
"""

import math

import numpy as np
import pytest

import spatialrust as sr


# --------------------------------------------------------------------------- #
# Synthetic clouds (deterministic — no RNG, so assertions are stable)
# --------------------------------------------------------------------------- #
def grid_plane(n=20, spacing=0.1, z=0.0):
    """An (n*n, 3) float32 grid in the z-plane (normal +Z)."""
    xs, ys = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing)
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)])
    return pts.astype(np.float32)


def two_blobs(per=60):
    """Two well-separated clusters plus a couple of stray outliers."""
    t = np.linspace(0, 1, per, dtype=np.float32)
    a = np.column_stack([t * 0.3, t * 0.3, np.zeros_like(t)])
    b = np.column_stack([t * 0.3 + 5.0, t * 0.3 + 5.0, np.zeros_like(t)])
    outliers = np.array([[2.5, 2.5, 10.0], [-3.0, -3.0, -8.0]], dtype=np.float32)
    return np.vstack([a, b, outliers]).astype(np.float32)


def sphere_surface(n=400, radius=1.0, center=(0.0, 0.0, 0.0)):
    """Points on a sphere (Fibonacci sampling — deterministic)."""
    i = np.arange(n, dtype=np.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - (i / (n - 1)) * 2.0
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = phi * i
    pts = np.column_stack([np.cos(theta) * r, y, np.sin(theta) * r]) * radius
    return (pts + np.asarray(center, dtype=np.float32)).astype(np.float32)


@pytest.fixture
def plane():
    return sr.PointCloud.from_xyz(grid_plane())


# --------------------------------------------------------------------------- #
# Module surface
# --------------------------------------------------------------------------- #
def test_module_has_version():
    assert isinstance(sr.__version__, str)
    assert sr.__version__


def test_exports_present():
    for name in (
        "PointCloud", "voxel_downsample", "dbscan", "register_icp",
        "voxelize", "knn_graph", "chamfer_distance", "oriented_bounding_box",
        "rgbd_to_point_cloud", "depth_to_xyz",
        "resize_image", "letterbox_image", "normalize_image_chw",
        "rgb_to_gray_image", "rgb_to_hsv_image", "remap_image",
        "nms", "batched_nms", "soft_nms", "connected_components_image", "distance_transform_edt",
        "find_mask_contours", "encode_mask_rle", "decode_mask_rle",
        "point_map_to_point_cloud",
    ):
        assert hasattr(sr, name), f"missing export: {name}"


# --------------------------------------------------------------------------- #
# PointCloud round-trips
# --------------------------------------------------------------------------- #
def test_pointcloud_roundtrip():
    pts = grid_plane(n=8)
    cloud = sr.PointCloud.from_xyz(pts)
    assert len(cloud) == pts.shape[0]
    out = cloud.xyz()
    assert out.shape == (pts.shape[0], 3)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, pts, atol=1e-6)
    assert {"x", "y", "z"}.issubset(set(cloud.field_names()))
    assert "PointCloud(" in repr(cloud)


def test_from_xyz_rejects_bad_shape():
    with pytest.raises(ValueError):
        sr.PointCloud.from_xyz(np.zeros((5, 2), dtype=np.float32))


def test_unlabeled_cloud_has_no_labels(plane):
    assert plane.labels() is None


def test_rgbd_to_point_cloud():
    depth = np.array([[1.0, 0.0], [np.nan, 2.0]], dtype=np.float32)
    color = np.array(
        [[[10, 11, 12], [20, 21, 22]], [[30, 31, 32], [40, 41, 42]]],
        dtype=np.uint8,
    )
    cloud = sr.rgbd_to_point_cloud(depth, color, 2.0, 2.0, 0.0, 0.0)
    assert len(cloud) == 2
    assert {"x", "y", "z", "r", "g", "b"}.issubset(set(cloud.field_names()))
    np.testing.assert_allclose(
        cloud.xyz(), np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float32)
    )


def test_depth_to_xyz_dense():
    depth = np.array([[1.0, 0.0], [np.nan, 2.0]], dtype=np.float32)
    xyz = sr.depth_to_xyz(depth, 2.0, 2.0, 0.0, 0.0)
    assert xyz.shape == (2, 2, 3)
    np.testing.assert_allclose(xyz[0, 0], [0.0, 0.0, 1.0])
    assert np.isnan(xyz[0, 1]).all()
    assert np.isnan(xyz[1, 0]).all()
    np.testing.assert_allclose(xyz[1, 1], [1.0, 1.0, 2.0])
    out = np.empty((2, 2, 3), dtype=np.float32)
    filled = sr.depth_to_xyz(depth, 2.0, 2.0, 0.0, 0.0, out=out)
    assert filled is out
    np.testing.assert_allclose(out[0, 0], [0.0, 0.0, 1.0])
    assert np.isnan(out[0, 1]).all()


def test_image_resize_letterbox_and_normalize():
    image = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
        dtype=np.uint8,
    )
    resized = sr.resize_image(image, 4, 4, interpolation="nearest")
    assert resized.shape == (4, 4, 3)
    np.testing.assert_array_equal(resized[0, 0], image[0, 0])
    resized_out = np.empty((4, 4, 3), dtype=np.uint8)
    assert sr.resize_image(image, 4, 4, interpolation="nearest", out=resized_out) is resized_out
    np.testing.assert_array_equal(resized_out, resized)

    letterboxed, transform = sr.letterbox_image(image, 4, 6, fill=(7, 8, 9))
    assert letterboxed.shape == (6, 4, 3)
    assert transform == (2.0, 0, 1, 4, 4)
    np.testing.assert_array_equal(letterboxed[0, 0], [7, 8, 9])

    chw = sr.normalize_image_chw(image)
    assert chw.shape == (3, 2, 2)
    assert chw.dtype == np.float32
    np.testing.assert_allclose(chw[:, 0, 0], [1.0, 0.0, 0.0], atol=1e-6)
    chw_out = np.empty((3, 2, 2), dtype=np.float32)
    assert sr.normalize_image_chw(image, out=chw_out) is chw_out
    np.testing.assert_allclose(chw_out, chw, atol=1e-6)


def test_image_color_and_remap():
    image = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    gray = sr.rgb_to_gray_image(image)
    assert gray.shape == (1, 2)
    np.testing.assert_allclose(gray, [[76, 150]], atol=1)
    gray_out = np.empty((1, 2), dtype=np.uint8)
    assert sr.rgb_to_gray_image(image, out=gray_out) is gray_out
    np.testing.assert_array_equal(gray_out, gray)
    hsv = sr.rgb_to_hsv_image(image)
    np.testing.assert_array_equal(hsv[0, 0], [0, 255, 255])
    np.testing.assert_array_equal(hsv[0, 1], [60, 255, 255])

    map_x = np.array([[0.0, 1.0]], dtype=np.float32)
    map_y = np.zeros((1, 2), dtype=np.float32)
    remapped = sr.remap_image(image, map_x, map_y, interpolation="nearest")
    np.testing.assert_array_equal(remapped, image)


def test_detection_nms_and_soft_nms():
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    np.testing.assert_array_equal(sr.nms(boxes, scores), [0, 2])
    score_storage = np.empty(scores.size * 2, dtype=np.float32)
    score_storage[::2] = scores
    np.testing.assert_array_equal(sr.nms(boxes, score_storage[::2]), [0, 2])
    np.testing.assert_array_equal(
        sr.batched_nms(boxes, scores, np.array([4, 4, 4], dtype=np.int64)), [0, 2]
    )
    class_storage = np.zeros(scores.size * 2, dtype=np.int64)
    class_storage[::2] = [4, 9, 4]
    np.testing.assert_array_equal(
        sr.batched_nms(boxes, score_storage[::2], class_storage[::2]), [0, 1, 2]
    )
    with pytest.raises(ValueError, match="equal lengths"):
        sr.batched_nms(boxes, scores[:-1], np.array([4, 9, 4], dtype=np.int64))
    indices, updated = sr.soft_nms(boxes, scores, method="linear")
    assert indices[0] == 0
    assert len(indices) == len(updated) == 3
    assert updated[-1] < 0.8


def test_mask_components_contours_and_rle():
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[2:4, 5:7] = 1
    labels, stats = sr.connected_components_image(mask, connectivity=4)
    assert labels.shape == mask.shape
    assert labels.dtype == np.uint32
    assert sorted(stat[1] for stat in stats) == [4, 4]
    contours = sr.find_mask_contours(mask)
    assert len(contours) == 2

    for coco in (False, True):
        counts = sr.encode_mask_rle(mask, coco=coco)
        decoded = sr.decode_mask_rle(7, 5, counts, coco=coco)
        np.testing.assert_array_equal(decoded, mask)


def test_exact_euclidean_distance_transform():
    mask = np.full((3, 4), 255, dtype=np.uint8)
    mask[0, 0] = 0
    actual = sr.distance_transform_edt(mask)
    expected = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, np.sqrt(2.0), np.sqrt(5.0), np.sqrt(10.0)],
            [2.0, np.sqrt(5.0), np.sqrt(8.0), np.sqrt(13.0)],
        ],
        dtype=np.float32,
    )
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=1e-6)

    anisotropic = sr.distance_transform_edt(mask, spacing=(2.0, 3.0))
    assert anisotropic[0, 1] == pytest.approx(2.0)
    assert anisotropic[1, 0] == pytest.approx(3.0)

    out = np.empty_like(actual)
    workspace = sr.DistanceTransformWorkspace()
    reused = sr.distance_transform_edt(mask, out=out, workspace=workspace)
    assert reused is out
    np.testing.assert_array_equal(reused, actual)
    assert workspace.capacity >= mask.size


def test_point_map_to_point_cloud_filters_invalid_and_low_confidence():
    points = np.array(
        [[[0, 0, 1], [1, 0, 1]], [[0, 1, np.nan], [1, 1, 2]]], dtype=np.float32
    )
    confidence = np.array([[0.9, 0.2], [1.0, 0.8]], dtype=np.float32)
    cloud = sr.point_map_to_point_cloud(points, confidence, min_confidence=0.5)
    np.testing.assert_allclose(
        cloud.xyz(), np.array([[0, 0, 1], [1, 1, 2]], dtype=np.float32)
    )


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def test_voxel_downsample_reduces(plane):
    out = sr.voxel_downsample(plane, leaf_size=0.5)
    assert len(out) < len(plane)
    assert len(out) > 0


def test_voxel_downsample_rejects_bad_policy(plane):
    with pytest.raises(ValueError):
        sr.voxel_downsample(plane, leaf_size=0.5, policy="quantum")


def test_statistical_outlier_removal_drops_strays():
    cloud = sr.PointCloud.from_xyz(two_blobs())
    out = sr.statistical_outlier_removal(cloud, k_neighbors=8, std_mul=1.0)
    assert len(out) < len(cloud)


def test_radius_outlier_removal_drops_strays():
    cloud = sr.PointCloud.from_xyz(two_blobs())
    out = sr.radius_outlier_removal(cloud, radius=0.5, min_neighbors=3)
    assert len(out) < len(cloud)


def test_crop_box_and_invert(plane):
    lo, hi = (0.0, 0.0, -1.0), (0.5, 0.5, 1.0)
    inside = sr.crop_box(plane, lo, hi)
    outside = sr.crop_box(plane, lo, hi, invert=True)
    assert len(inside) > 0
    assert len(inside) + len(outside) == len(plane)


def test_pass_through_on_z():
    pts = grid_plane(n=6)
    pts[:10, 2] = 5.0  # lift some points out of the slice
    cloud = sr.PointCloud.from_xyz(pts)
    kept = sr.pass_through(cloud, "z", -0.1, 0.1)
    assert len(kept) == len(pts) - 10


def test_farthest_point_sampling_exact_count(plane):
    out = sr.farthest_point_sampling(plane, sample_size=16)
    assert len(out) == 16


def test_mls_smooth_preserves_count(plane):
    out = sr.mls_smooth(plane, search_radius=0.3, polynomial_order=2, min_neighbors=6)
    assert len(out) == len(plane)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def test_iss_keypoints_are_sparse_subset(plane):
    kp = sr.iss_keypoints(plane, salient_radius=0.3, non_max_radius=0.2, min_neighbors=5)
    assert len(kp) <= len(plane)


def test_orient_normals_attaches_normals(plane):
    out = sr.orient_normals(plane, k_neighbors=10)
    assert {"normal_x", "normal_y", "normal_z"}.issubset(set(out.field_names()))
    assert len(out) == len(plane)


def test_detect_boundary_returns_subcloud(plane):
    boundary = sr.detect_boundary(plane, search_radius=0.25, min_neighbors=3, k_neighbors=8)
    # A finite planar patch has a rim, so some boundary points exist, but they
    # are a strict subset.
    assert 0 < len(boundary) < len(plane)


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def test_dbscan_finds_two_clusters():
    cloud = sr.PointCloud.from_xyz(two_blobs())
    res = sr.dbscan(cloud, eps=0.5, min_points=3)
    assert res.cluster_count >= 2
    labels = res.labels()
    assert labels.shape == (len(cloud),)
    assert labels.dtype == np.int32
    assert res.noise_count >= 0
    assert "DbscanResult(" in repr(res)


def test_multi_plane_extracts_a_plane():
    # Floor + ceiling: two parallel planes.
    floor = grid_plane(n=20, z=0.0)
    ceil = grid_plane(n=20, z=2.0)
    cloud = sr.PointCloud.from_xyz(np.vstack([floor, ceil]).astype(np.float32))
    res = sr.segment_multi_plane(cloud, max_planes=2, distance_threshold=0.05, min_inliers=50)
    assert res.plane_count >= 1
    assert len(res.planes) == res.plane_count
    assert len(res.planes[0]) == 4  # (nx, ny, nz, d)


def test_ground_segmentation_splits():
    floor = grid_plane(n=20, z=0.0)
    bump = grid_plane(n=6, z=1.0)
    cloud = sr.PointCloud.from_xyz(np.vstack([floor, bump]).astype(np.float32))
    res = sr.ground_segmentation(cloud, cell_size=0.5, height_threshold=0.3)
    assert res.ground_count > 0
    assert len(res.ground) + len(res.non_ground) == len(cloud)


def test_ransac_sphere_recovers_radius():
    cloud = sr.PointCloud.from_xyz(sphere_surface(n=500, radius=1.0))
    res = sr.ransac_sphere(cloud, distance_threshold=0.05, max_iterations=2000, min_inliers=50)
    assert res.radius == pytest.approx(1.0, abs=0.2)
    assert len(res.inliers) > 0
    assert "SphereResult(" in repr(res)


def test_region_growing_runs(plane):
    res = sr.region_growing(plane, k_neighbors=10, smoothness_deg=5.0, min_region_size=5)
    assert res.cluster_count >= 1


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_chamfer_zero_for_identical(plane):
    assert sr.chamfer_distance(plane, plane) == pytest.approx(0.0, abs=1e-6)


def test_hausdorff_grows_with_offset():
    a = sr.PointCloud.from_xyz(grid_plane(n=8))
    shifted = grid_plane(n=8)
    shifted[:, 0] += 1.0
    b = sr.PointCloud.from_xyz(shifted)
    assert sr.hausdorff_distance(a, b) > 0.0


# --------------------------------------------------------------------------- #
# Transform utilities
# --------------------------------------------------------------------------- #
def test_apply_transform_translates(plane):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = [1.0, 2.0, 3.0]
    out = sr.apply_transform(plane, m)
    cx, cy, cz = sr.centroid(out)
    ox, oy, oz = sr.centroid(plane)
    assert cx == pytest.approx(ox + 1.0, abs=1e-4)
    assert cy == pytest.approx(oy + 2.0, abs=1e-4)
    assert cz == pytest.approx(oz + 3.0, abs=1e-4)


def test_apply_transform_rejects_bad_shape(plane):
    with pytest.raises(ValueError):
        sr.apply_transform(plane, np.eye(3, dtype=np.float32))


def test_recenter_puts_centroid_at_origin(plane):
    out = sr.recenter(plane)
    cx, cy, cz = sr.centroid(out)
    assert cx == pytest.approx(0.0, abs=1e-5)
    assert cy == pytest.approx(0.0, abs=1e-5)
    assert cz == pytest.approx(0.0, abs=1e-5)


def test_normalize_unit_sphere_bounds(plane):
    out = sr.normalize_unit_sphere(plane)
    xyz = out.xyz()
    assert np.linalg.norm(xyz, axis=1).max() == pytest.approx(1.0, abs=1e-4)


def test_scale_and_bounding_box(plane):
    (lo0, hi0) = sr.bounding_box(plane)
    out = sr.scale(plane, 2.0)
    (lo, hi) = sr.bounding_box(out)
    assert hi[0] == pytest.approx(hi0[0] * 2.0, abs=1e-4)


def test_merge_concatenates(plane):
    merged = sr.merge([plane, plane])
    assert len(merged) == 2 * len(plane)


def test_oriented_bounding_box_shape(plane):
    center, half, axes = sr.oriented_bounding_box(plane)
    assert len(center) == 3 and len(half) == 3
    assert len(axes) == 3 and all(len(a) == 3 for a in axes)


# --------------------------------------------------------------------------- #
# Voxelize / range image / graphs (ML front-ends)
# --------------------------------------------------------------------------- #
def test_voxelize_occupancy_shape_and_values(plane):
    grid, origin, vsize = sr.voxelize(plane, voxel_size=0.2, mode="occupancy")
    assert grid.ndim == 3
    assert grid.dtype == np.float32
    assert set(np.unique(grid)).issubset({0.0, 1.0})
    assert len(origin) == 3
    assert vsize == pytest.approx(0.2)


def test_voxelize_count_mode(plane):
    grid, _, _ = sr.voxelize(plane, voxel_size=0.2, mode="count")
    assert grid.max() >= 1.0


def test_voxelize_rejects_bad_mode(plane):
    with pytest.raises(ValueError):
        sr.voxelize(plane, voxel_size=0.2, mode="bogus")


def test_range_image_shape():
    cloud = sr.PointCloud.from_xyz(sphere_surface(n=500, radius=5.0))
    img = sr.range_image(cloud, width=64, height=16)
    assert img.shape == (16, 64)
    assert img.dtype == np.float32


def test_knn_graph_edge_index(plane):
    edges = sr.knn_graph(plane, k=5)
    assert edges.shape[0] == 2
    assert edges.dtype == np.int32
    assert edges.shape[1] == len(plane) * 5  # k directed edges per node


def test_radius_graph_edge_index(plane):
    edges = sr.radius_graph(plane, radius=0.25)
    assert edges.shape[0] == 2
    assert edges.dtype == np.int32
    assert edges.shape[1] > 0


# --------------------------------------------------------------------------- #
# Pipeline & registration
# --------------------------------------------------------------------------- #
def test_run_pipeline_smoke():
    # The MVP pipeline removes the dominant plane, then clusters the remainder,
    # so the cloud needs a ground plane plus distinct objects sitting above it.
    floor = grid_plane(n=24, spacing=0.1, z=0.0)
    blob_a = grid_plane(n=5, spacing=0.05, z=1.0)
    blob_b = grid_plane(n=5, spacing=0.05, z=1.0) + np.array([3.0, 3.0, 0.0], np.float32)
    pts = np.vstack([floor, blob_a, blob_b]).astype(np.float32)
    cloud = sr.PointCloud.from_xyz(pts)
    res = sr.run_pipeline(cloud, leaf_size=0.05, cluster_tolerance=0.5, min_cluster_size=3)
    assert len(res.output) > 0
    assert res.cluster_count >= 1
    assert res.plane_inliers > 0
    assert len(res.plane_normal) == 3
    assert "PipelineResult(" in repr(res)


def _shift(pts, dx):
    out = pts.copy()
    out[:, 0] += dx
    return out.astype(np.float32)


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (sr.register_icp, {}),
        (sr.register_point_to_plane, {}),
        (sr.register_gicp, {}),
        (sr.register_ndt, {"resolution": 0.5}),
    ],
)
def test_registration_recovers_small_translation(fn, kwargs):
    src_pts = sphere_surface(n=300, radius=1.0)
    tgt = sr.PointCloud.from_xyz(src_pts)
    src = sr.PointCloud.from_xyz(_shift(src_pts, 0.1))
    res = fn(src, tgt, **kwargs)
    t = res.transform()
    assert t.shape == (4, 4)
    assert t.dtype == np.float32
    # The recovered translation should pull source back toward target (negative x).
    assert t[0, 3] < 0.0
    assert "RegistrationResult(" in repr(res)


def test_fpfh_ransac_returns_transform():
    src_pts = sphere_surface(n=300, radius=1.0)
    tgt = sr.PointCloud.from_xyz(src_pts)
    src = sr.PointCloud.from_xyz(_shift(src_pts, 0.05))
    res = sr.register_fpfh_ransac(src, tgt, feature_radius=0.5,
                                  max_correspondence_distance=0.2, ransac_iterations=500)
    assert res.transform().shape == (4, 4)
def test_png_image_io_roundtrip(tmp_path):
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    path = tmp_path / "roundtrip.png"
    sr.write_image(str(path), image[:, ::-1], "png")
    decoded, metadata = sr.read_image(str(path))
    np.testing.assert_array_equal(decoded, image[:, ::-1])
    assert metadata.format == "png"
    assert metadata.color_type == "Rgb8"
    assert metadata.orientation in (0, 1)
    assert "ImageMetadata(" in repr(metadata)


def test_filter2d_and_gaussian_preserve_rgb_shape():
    image = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
    identity = sr.filter2d_image(image[:, ::-1], np.array([[1.0]], dtype=np.float64))
    np.testing.assert_array_equal(identity, image[:, ::-1])
    blurred = sr.gaussian_blur_image(image, 5, 3, 1.2, 0.8)
    assert blurred.shape == image.shape
    assert blurred.dtype == np.uint8


def test_advanced_filters_and_pyramid_shapes():
    image = np.arange(9 * 11 * 3, dtype=np.uint8).reshape(9, 11, 3)
    assert sr.median_blur_image(image[:, ::-1], 3).shape == image.shape
    assert sr.bilateral_filter_image(image, 3, 20.0, 2.0).shape == image.shape
    gray = image[..., 0]
    for derivative in (
        sr.sobel_image(gray, 1, 0),
        sr.scharr_image(gray, 0, 1),
        sr.laplacian_image(gray),
    ):
        assert derivative.shape == gray.shape
        assert derivative.dtype == np.float32
    down = sr.pyr_down_image(image)
    assert down.shape == (5, 6, 3)
    assert sr.pyr_up_image(down).shape == (10, 12, 3)


def test_morphology_operations_and_noncontiguous_input():
    mask = np.zeros((9, 11), dtype=np.uint8)
    mask[2:7, 3:8] = 255
    for operation in ("erode", "dilate", "open", "close", "gradient", "tophat", "blackhat"):
        output = sr.morphology_image(mask[:, ::-1], operation, 3, 3, "ellipse", 2)
        assert output.shape == mask.shape
        assert output.dtype == np.uint8


def test_threshold_histogram_clahe_and_integral_contracts():
    image = np.arange(9 * 11, dtype=np.uint8).reshape(9, 11)[:, ::-1]
    assert sr.threshold_image(image, 40).shape == image.shape
    selected, otsu = sr.otsu_threshold_image(image)
    assert 0 <= selected <= 255 and otsu.shape == image.shape
    for method in ("mean", "gaussian"):
        assert sr.adaptive_threshold_image(image, 5, 2.0, method).shape == image.shape
    histogram = sr.histogram_image(image)
    assert histogram.shape == (256,) and int(histogram.sum()) == image.size
    assert sr.equalize_histogram_image(image).shape == image.shape
    assert sr.clahe_image(image, 2.0, 3, 2).shape == image.shape
    integral = sr.integral_image_u8(image)
    assert integral.shape == (image.shape[0] + 1, image.shape[1] + 1)
    assert integral[-1, -1] == pytest.approx(float(image.sum()))


def test_canny_image_binary_output_and_noncontiguous_input():
    image = np.zeros((17, 19), dtype=np.uint8)
    image[4:13, 6:14] = 255
    edges = sr.canny_image(image[:, ::-1], 50.0, 100.0, 3, True)
    assert edges.shape == image.shape
    assert edges.dtype == np.uint8
    assert set(np.unique(edges)).issubset({0, 255})
    assert np.count_nonzero(edges) > 0


def test_feature2d_corner_detectors_and_keypoint_metadata():
    image = np.zeros((25, 29), dtype=np.uint8)
    image[5:19, 7:22] = 255
    harris = sr.harris_keypoints(image[:, ::-1], 20, 0.01, 1.0, 3, 3, 0.04)
    shi = sr.shi_tomasi_keypoints(image, 20, 0.01, 1.0, 3, 3)
    assert len(harris) >= 4 and len(shi) >= 4
    assert all(point.size == 3.0 and point.angle_degrees is None for point in harris)
    impulse = np.zeros((9, 9), dtype=np.uint8)
    impulse[4, 4] = 255
    fast = sr.fast_keypoints(impulse, 20, True)
    assert len(fast) == 1
    assert (fast[0].x, fast[0].y, fast[0].size) == (4.0, 4.0, 7.0)
    assert "Keypoint2(" in repr(fast[0])


def test_orb_features_and_descriptor_matchers():
    yy, xx = np.indices((96, 112), dtype=np.int32)
    image = ((xx * 37 + yy * 19) ^ (xx * yy * 3) ^ ((xx // 8 + yy // 8) * 127)).astype(np.uint8)
    keypoints, descriptors = sr.orb_features(image[:, ::-1], max_features=60, edge_threshold=16)
    assert 0 < len(keypoints) <= 60
    assert descriptors.shape == (len(keypoints), 32)
    assert descriptors.dtype == np.uint8
    assert all(point.angle_degrees is not None for point in keypoints)

    matches = sr.match_binary_descriptors(descriptors, descriptors, cross_check=True)
    assert len(matches) == len(keypoints)
    assert all(query == train and distance == 0.0 for query, train, distance in matches)

    query = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    train = np.array([[1.0, 0.0], [3.0, 0.0], [10.0, 9.0]], dtype=np.float32)
    assert sr.match_float_descriptors(query, train, cross_check=True, ratio=0.8) == [
        (0, 0, 1.0),
        (1, 2, 1.0),
    ]


def test_geometry_homography_pnp_and_stereo():
    source = np.array(
        [[0.0, 0.0], [40.0, 0.0], [0.0, 30.0], [40.0, 30.0], [20.0, 15.0], [10.0, 5.0]],
        dtype=np.float64,
    )
    h = np.array([[1.05, 0.01, 2.0], [-0.02, 0.98, -1.5], [0.0001, 0.0, 1.0]], dtype=np.float64)
    target = []
    for point in source:
        projected = h @ np.array([point[0], point[1], 1.0])
        target.append([projected[0] / projected[2], projected[1] / projected[2]])
    target = np.asarray(target, dtype=np.float64)
    matrix, inliers, residuals = sr.estimate_homography_ransac(source, target, threshold=1.0)
    assert matrix.shape == (3, 3)
    assert inliers.dtype == np.bool_
    assert residuals.shape == (source.shape[0],)
    assert int(inliers.sum()) >= 4

    objects = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.15, 0.0],
            [0.0, 0.0, 0.1],
            [0.1, 0.05, 0.05],
            [0.05, -0.05, 0.02],
        ],
        dtype=np.float64,
    )
    fx = fy = 500.0
    cx, cy = 320.0, 240.0
    rotation = np.eye(3, dtype=np.float64)
    translation = np.array([0.1, -0.05, 2.5], dtype=np.float64)
    images = []
    for point in objects:
        camera = rotation @ point + translation
        images.append([fx * camera[0] / camera[2] + cx, fy * camera[1] / camera[2] + cy])
    images = np.asarray(images, dtype=np.float64)
    recovered_r, recovered_t = sr.solve_pnp(objects, images, fx, fy, cx, cy)
    assert recovered_r.shape == (3, 3)
    assert recovered_t.shape == (3,)
    assert abs(recovered_t[2] - translation[2]) < 0.05

    width, height = 96, 64
    disparity = 12
    yy, xx = np.indices((height, width), dtype=np.int32)
    left = ((xx * 17 + yy * 29) % 200 + 20).astype(np.uint8)
    right = np.zeros_like(left)
    right[:, : width - disparity] = left[:, disparity:]
    disparity_map = sr.stereo_block_match(
        left, right, window_size=11, min_disparity=1, num_disparities=32, uniqueness_ratio=5.0
    )
    assert disparity_map.shape == (height, width)
    assert abs(float(disparity_map[height // 2, width // 2]) - float(disparity)) <= 1.0


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_tensor_dlpack_zero_copy_export(dtype):
    source = np.arange(3 * 5, dtype=dtype).reshape(3, 5)[:, ::-1]
    tensor = sr.tensor_copy_from_numpy(source)
    first = np.from_dlpack(tensor)
    second = np.from_dlpack(tensor)
    np.testing.assert_array_equal(first, source)
    assert first.dtype == source.dtype
    assert first.shape == source.shape
    assert first.__array_interface__["data"][0] == second.__array_interface__["data"][0]
    assert not first.flags.writeable
    assert tensor.__dlpack_device__() == (1, 0)
    assert "Tensor(shape=" in repr(tensor)


def test_tensor_dlpack_copy_and_device_requests_are_explicit():
    tensor = sr.tensor_copy_from_numpy(np.arange(8, dtype=np.uint8))
    copied = tensor.copy()
    assert np.from_dlpack(copied).__array_interface__["data"][0] != np.from_dlpack(
        tensor
    ).__array_interface__["data"][0]
    with pytest.raises(BufferError):
        tensor.__dlpack__(copy=True)
    with pytest.raises(BufferError):
        tensor.__dlpack__(dl_device=(2, 0))


def test_onnxruntime_dynamic_named_binding_matches_reference(tmp_path):
    model = bytes(
        [
            8, 8, 18, 16, 115, 112, 97, 116, 105, 97, 108, 114, 117, 115, 116, 45, 116,
            101, 115, 116, 58, 106, 10, 27, 10, 5, 105, 110, 112, 117, 116, 10, 5, 105,
            110, 112, 117, 116, 18, 6, 111, 117, 116, 112, 117, 116, 34, 3, 65, 100, 100,
            18, 14, 100, 111, 117, 98, 108, 101, 95, 100, 121, 110, 97, 109, 105, 99, 90,
            28, 10, 5, 105, 110, 112, 117, 116, 18, 19, 10, 17, 8, 1, 18, 13, 10, 7,
            18, 5, 98, 97, 116, 99, 104, 10, 2, 8, 3, 98, 29, 10, 6, 111, 117, 116,
            112, 117, 116, 18, 19, 10, 17, 8, 1, 18, 13, 10, 7, 18, 5, 98, 97, 116,
            99, 104, 10, 2, 8, 3, 66, 4, 10, 0, 16, 13,
        ]
    )
    path = tmp_path / "double_dynamic.onnx"
    path.write_bytes(model)
    try:
        session = sr.OnnxRuntimeSession(str(path), deterministic=True)
    except RuntimeError as error:
        if "without the `onnxruntime` feature" in str(error):
            pytest.skip("extension was intentionally built without ONNX Runtime")
        raise

    source = np.arange(12, dtype=np.float32).reshape(4, 3)
    inputs = {"input": sr.tensor_copy_from_numpy(source)}
    assert session.inputs == [("input", "float32", ["batch", "3"])]
    bound = np.from_dlpack(session.run(inputs)["output"])
    copied = np.from_dlpack(session.run(inputs, copy=True)["output"])
    expected = source * 2.0
    np.testing.assert_array_equal(bound, expected)
    np.testing.assert_array_equal(copied, expected)

    try:
        import onnxruntime as reference_runtime
    except ImportError:
        return
    reference = reference_runtime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    ).run(None, {"input": source})[0]
    np.testing.assert_array_equal(bound, reference)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_tensor_zero_copy_dlpack_import_retains_producer(dtype):
    source = np.arange(12, dtype=dtype).reshape(3, 4)
    imported = sr.tensor_view_from_dlpack(source)
    assert imported.shape == [3, 4]
    assert imported.dtype == np.dtype(dtype).name
    assert imported.version[0] == 1
    del source
    copied = np.from_dlpack(imported.copy())
    np.testing.assert_array_equal(copied, np.arange(12, dtype=dtype).reshape(3, 4))
    assert "DLPackTensorView(" in repr(imported)


def test_calibration_bindings_recover_intrinsics_and_fisheye():
    points = np.array(
        [
            [x * 0.1, y * 0.08, 2.0 + 0.03 * abs(x + y)]
            for y in range(-3, 4)
            for x in range(-4, 5)
        ],
        dtype=np.float64,
    )
    pixels = np.column_stack(
        (
            500.0 * points[:, 0] / points[:, 2] + 320.0,
            510.0 * points[:, 1] / points[:, 2] + 240.0,
        )
    )
    result = sr.calibrate_pinhole_camera(points, pixels, 640, 480)
    np.testing.assert_allclose(result[:4], [500.0, 510.0, 320.0, 240.0], atol=1e-9)
    assert result[4] < 1e-9

    expected = np.array([0.03, -0.004, 0.0005, -0.00003])
    theta = np.linspace(0.08, 1.0, 12, dtype=np.float64)
    theta2 = theta * theta
    radius = theta * (
        1.0
        + theta2
        * (
            expected[0]
            + theta2 * (expected[1] + theta2 * (expected[2] + theta2 * expected[3]))
        )
    )
    fitted = sr.calibrate_fisheye_angles(theta, radius)
    np.testing.assert_allclose(fitted[:4], expected, atol=1e-9)
    assert fitted[4] < 1e-12


def test_dense_flow_binding_recovers_translation_and_marks_border_invalid():
    height, width = 32, 40
    yy, xx = np.indices((height, width), dtype=np.int32)
    previous = ((xx * 17 + yy * 29 + xx * yy * 3) % 251).astype(np.uint8)
    next_frame = np.zeros_like(previous)
    next_frame[1:, 2:] = previous[:-1, :-2]
    flow = sr.dense_flow_image(previous, next_frame)
    assert flow.shape == (height, width, 2)
    np.testing.assert_array_equal(flow[16, 20], [2.0, 1.0])
    assert np.isnan(flow[0, 0]).all()
