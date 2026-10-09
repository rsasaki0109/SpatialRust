"""Filtering selects complete attribute rows, including through file export."""
import numpy as np
import pytest
import spatialrust as sr
from test_alignment_attributes import write_rich_pcd


@pytest.mark.parametrize('method', ['statistical', 'radius', 'remove_all'])
def test_outlier_filter_preserves_attribute_rows_through_pcd(tmp_path, method):
    pa = pytest.importorskip('pyarrow')
    dense = np.random.default_rng(93).uniform(-.02, .02, (80, 3)).astype(np.float32)
    # Nonmatching points at the start, middle and end expose index shifts.
    xyz = np.vstack(([20, 20, 20], dense[:40], [-20, -20, -20], dense[40:], [30, -30, 30])).astype(np.float32)
    normals = np.random.default_rng(94).normal(size=xyz.shape).astype(np.float32)
    path = tmp_path / 'attributes.pcd'
    write_rich_pcd(path, xyz, normals)
    original = sr.read(str(path))
    original_columns = pa.array(original)
    if method == 'statistical':
        filtered = sr.statistical_outlier_removal(original, k_neighbors=8, std_mul=1)
    elif method == 'radius':
        filtered = sr.radius_outlier_removal(original, radius=.1, min_neighbors=3)
    else:
        filtered = sr.radius_outlier_removal(original, radius=.1, min_neighbors=len(original))
    expected_indices = np.array([i for i in range(len(xyz)) if i not in [0, 41, 82]], dtype=np.int64)
    if method == 'remove_all':
        expected_indices = np.array([], dtype=np.int64)
    assert len(filtered) == len(expected_indices)
    saved_path = tmp_path / 'filtered.pcd'
    sr.write(str(saved_path), filtered)
    saved = sr.read(str(saved_path))
    for result in [filtered, saved]:
        columns = pa.array(result)
        assert columns.type == original_columns.type
        for name in original.field_names():
            np.testing.assert_array_equal(columns.field(name).to_numpy(),
                                          original_columns.field(name).to_numpy()[expected_indices])
    # The original remains intact and owns its own storage.
    np.testing.assert_array_equal(original.xyz(), xyz)
    for name in original.field_names():
        np.testing.assert_array_equal(pa.array(original).field(name).to_numpy(),
                                      original_columns.field(name).to_numpy())
