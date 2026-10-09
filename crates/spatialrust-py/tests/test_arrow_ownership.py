"""Actual PyArrow consumption and ownership/lifetime contracts."""
import gc

import numpy as np
import pytest
import spatialrust as sr

pa = pytest.importorskip('pyarrow')


def test_owned_exports_outlive_cloud_and_each_other():
    xyz = np.array([[1,2,3], [4,5,6], [7,8,9]], dtype=np.float32)
    cloud = sr.PointCloud.from_xyz(xyz)
    first, second = pa.array(cloud), pa.array(cloud)
    for axis in ('x', 'y', 'z'):
        # Current native export creates independent owned snapshots.
        assert first.field(axis).buffers()[1].address != second.field(axis).buffers()[1].address
    del cloud
    gc.collect()
    for column, axis in enumerate(('x', 'y', 'z')):
        np.testing.assert_array_equal(first.field(axis).to_numpy(), xyz[:,column])
    del first
    gc.collect()
    np.testing.assert_array_equal(second.field('z').to_numpy(), xyz[:,2])


def test_stream_reader_owns_consumed_stream_and_batches_outlive_reader(tmp_path):
    xyz = np.arange(90, dtype=np.float32).reshape(-1,3)
    path = tmp_path / 'cloud.pcd'
    sr.write(str(path), sr.PointCloud.from_xyz(xyz))
    stream = sr.open_point_cloud_stream(str(path), chunk_points=7)
    reader = pa.RecordBatchReader.from_stream(stream)
    with pytest.raises(RuntimeError, match='consumed'):
        pa.RecordBatchReader.from_stream(stream)
    del stream
    gc.collect()
    batches = list(reader)
    reader.close()
    del reader
    gc.collect()
    assert [b.num_rows for b in batches] == [7,7,7,7,2]
    for column, axis in enumerate(('x', 'y', 'z')):
        actual = pa.concat_arrays([b.column(axis) for b in batches]).to_numpy()
        np.testing.assert_array_equal(actual, xyz[:,column])
