"""One native ICP update compared to an independent NumPy SVD fit of its pairs."""
import numpy as np
import pytest
import spatialrust as sr


@pytest.mark.parametrize('scale', [1e-4, 1., 1e4])
@pytest.mark.parametrize('planar', [False, True])
@pytest.mark.parametrize('large_origin', [False, True])
@pytest.mark.parametrize('noisy', [False, True])
def test_native_update_matches_numpy_svd_across_scales(scale, planar, large_origin, noisy):
    for seed in range(5):
        base = np.random.default_rng(seed).uniform(-1, 1, (120, 3))
        if planar:
            base[:, 2] = 0
        angle = np.deg2rad(3)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        origin = np.array([1e4, -2e4, 3e4]) if large_origin else np.zeros(3)
        source = ((base + origin)*scale).astype(np.float32)
        moved = base @ rotation.T + [.015, -.008, .004]
        if noisy:
            moved += np.random.default_rng(100+seed).normal(0, .002, moved.shape)
        target = ((moved + origin)*scale).astype(np.float32)
        gate = np.float32(.2*scale)
        # Match the native f32 metric, then fit the paired coordinates in f64.
        difference = source[:, None, :] - target[None, :, :]
        d2 = (difference*difference).sum(axis=2)
        # f32 quantization can create exact nearest-distance ties at a large
        # origin. ICP's traversal tie order differs from NumPy's input order;
        # compare the solvers only on queries with a unique nearest point.
        unique = (d2 == d2.min(axis=1)[:, None]).sum(axis=1) == 1
        source, d2 = source[unique], d2[unique]
        assert len(source) >= 100
        indices = d2.argmin(axis=1)
        accepted = d2[np.arange(len(source)), indices] <= gate*gate
        paired_source = source[accepted].astype(np.float64)
        paired_target = target[indices[accepted]].astype(np.float64)
        mean_source, mean_target = paired_source.mean(axis=0), paired_target.mean(axis=0)
        u, _, vt = np.linalg.svd((paired_source-mean_source).T @ (paired_target-mean_target))
        if np.linalg.det(vt.T @ u.T) < 0:
            vt[-1] *= -1
        reference_rotation = vt.T @ u.T
        trace = sr.register_icp_diagnostics(sr.PointCloud.from_xyz(source), sr.PointCloud.from_xyz(target),
                                           float(gate), 1, fitness_epsilon=0,
                                           translation_epsilon=0, rotation_epsilon=0)
        assert trace.history[0].correspondences == accepted.sum()
        matrix = trace.result.transform().astype(np.float64)
        np.testing.assert_allclose(matrix[:3, :3], reference_rotation, atol=2e-6, rtol=0)
        np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=2e-6)
        assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1, abs=2e-6)
        # Translation must use the actual returned f32 rotation, avoiding large
        # origin amplification of a rotation-rounding inconsistency.
        corrected_centroid = (mean_target - matrix[:3, :3] @ mean_source).astype(np.float32)
        np.testing.assert_allclose(matrix[:3, 3], corrected_centroid, rtol=2e-6, atol=1e-6*scale)
