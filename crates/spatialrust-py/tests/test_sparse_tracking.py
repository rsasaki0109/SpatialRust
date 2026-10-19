"""Known image motion, invalid tracks and owned/strided Python tracking inputs."""
import numpy as np
import pytest
import spatialrust as sr


def texture(dx=0.0, dy=0.0):
    y, x = np.mgrid[:96, :128].astype(float)
    x -= dx
    y -= dy
    pixels = 120 + 35*np.sin(x*.17) + 25*np.cos(y*.21) + 20*np.sin(x*.11+y*.13)
    return np.round(pixels).clip(0, 255).astype(np.uint8)


def test_tracks_known_translation_and_preserves_rows_and_ownership():
    previous, following = texture(), texture(1.25, -.75)
    points = np.array([[40., 40.], [65., 55.], [85., 45.]], dtype=np.float64)
    inputs = [value.copy() for value in (previous, following, points)]
    next_points, valid, errors = sr.track_points_forward_backward(
        previous, following, points, window_radius=4, pyramid_levels=2,
        min_eigenvalue=1e-6, max_forward_backward_error=.3)
    np.testing.assert_array_equal(valid, True)
    np.testing.assert_allclose(next_points, points + [1.25, -.75], atol=.2)
    assert np.all(errors <= .3)
    for original, current in zip(inputs, (previous, following, points)):
        np.testing.assert_array_equal(current, original)
    saved = next_points.copy()
    previous.fill(0)
    following.fill(0)
    points.fill(0)
    np.testing.assert_array_equal(next_points, saved)


def test_strided_inputs_match_contiguous_and_empty_shapes_are_owned():
    image = texture()
    padded = np.zeros((96, 256), np.uint8)
    padded[:, ::2] = image
    coords = np.array([[40., 7., 40., 7.], [65., 7., 55., 7.]], np.float64)
    left = sr.track_points_forward_backward(padded[:, ::2], padded[:, ::2], coords[:, ::2])
    right = sr.track_points_forward_backward(image, image, np.ascontiguousarray(coords[:, ::2]))
    for actual, expected in zip(left, right):
        np.testing.assert_array_equal(actual, expected)
    empty = sr.track_points_forward_backward(image, image, np.empty((0, 2), np.float64))
    assert [value.shape for value in empty] == [(0, 2), (0,), (0,)]


def test_flat_outside_and_excessive_windows_preserve_invalid_rows():
    image = np.zeros((32, 32), np.uint8)
    points = np.array([[16., 16.], [-1., 16.]], np.float64)
    for radius in [4, 1000000]:
        coordinates, valid, errors = sr.track_points_forward_backward(
            image, image, points, window_radius=radius, pyramid_levels=1)
        assert coordinates.shape == (2, 2)
        np.testing.assert_array_equal(valid, False)
        np.testing.assert_array_equal(errors, np.inf)


@pytest.mark.parametrize('control,value', [
    ('pyramid_levels', 0), ('pyramid_levels', 33), ('window_radius', 0),
    ('max_iterations', 0), ('epsilon', 0.), ('epsilon', np.nan),
    ('min_eigenvalue', np.inf), ('max_forward_backward_error', -1.),
    ('max_forward_backward_error', np.nan),
])
def test_invalid_controls_rejected(control, value):
    image = texture()
    with pytest.raises(ValueError):
        sr.track_points_forward_backward(image, image, np.array([[40., 40.]]), **{control: value})


@pytest.mark.parametrize('points', [np.zeros((2, 3)), np.array([[np.nan, 2.]]), np.array([[2., np.inf]])])
def test_bad_coordinate_arrays_rejected(points):
    image = texture()
    with pytest.raises(ValueError):
        sr.track_points_forward_backward(image, image, points)


def test_geometry_and_dtype_rejected():
    points = np.array([[10., 10.]])
    with pytest.raises(ValueError):
        sr.track_points_forward_backward(texture(), texture()[:-1], points)
    with pytest.raises((ValueError, TypeError)):
        sr.track_points_forward_backward(texture().astype(np.float32), texture(), points)
