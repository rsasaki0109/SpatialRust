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
