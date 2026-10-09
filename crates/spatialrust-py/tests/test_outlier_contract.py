"""Neighborhood filters reject invalid settings and bound neighbor requests."""
import numpy as np
import pytest
import spatialrust as sr


def cloud():
    return sr.PointCloud.from_xyz(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32))


@pytest.mark.parametrize('std_mul', [float('nan'), float('inf'), -1])
def test_invalid_sor_multiplier(std_mul):
    with pytest.raises(ValueError, match='std_mul'):
        sr.statistical_outlier_removal(cloud(), k_neighbors=2, std_mul=std_mul)


@pytest.mark.parametrize('radius', [float('inf'), 1e30, 1e-30])
def test_invalid_radius_square(radius):
    with pytest.raises(ValueError, match='radius'):
        sr.radius_outlier_removal(cloud(), radius=radius)


def test_extreme_neighbor_counts_are_safe():
    import struct
    maximum = (1 << (8 * struct.calcsize('P'))) - 1
    points = cloud()
    huge = sr.statistical_outlier_removal(points, k_neighbors=maximum)
    available = sr.statistical_outlier_removal(points, k_neighbors=2)
    np.testing.assert_array_equal(huge.xyz(), available.xyz())
    assert len(sr.radius_outlier_removal(points, radius=1, min_neighbors=maximum)) == 0
