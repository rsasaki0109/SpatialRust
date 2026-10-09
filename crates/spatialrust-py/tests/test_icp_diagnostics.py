"""Independent checks of post-update metrics and owned opt-in history."""
from concurrent.futures import ThreadPoolExecutor
import gc

import numpy as np
import pytest
import spatialrust as sr


def clouds():
    xyz = np.random.default_rng(71).uniform(-1, 1, (200, 3)).astype(np.float32)
    return (sr.PointCloud.from_xyz(xyz + np.array([.035, -.012, .007], np.float32)),
            sr.PointCloud.from_xyz(xyz))


def test_trace_matches_plain_result_and_independent_distances():
    source, target = clouds()
    plain = sr.register_icp(source, target, .15, 1)
    trace = sr.register_icp_diagnostics(source, target, .15, 1)
    np.testing.assert_array_equal(trace.result.transform(), plain.transform())
    assert (trace.result.fitness, trace.result.iterations, trace.result.converged) == (
        plain.fitness, plain.iterations, plain.converged)
    row, = trace.history
    before = np.sum((source.xyz()[:, None, :] - target.xyz()[None, :, :])**2, axis=2).min(axis=1)
    moved = sr.apply_transform(source, trace.result.transform()).xyz()
    distances = np.sum((moved[:, None, :].astype(np.float64) - target.xyz()[None, :, :])**2, axis=2).min(axis=1)
    assert row.correspondences == np.count_nonzero(before <= np.float32(.15)**2)
    assert row.evaluated_correspondences == np.count_nonzero(distances <= np.float32(.15)**2)
    assert row.fitness == pytest.approx(distances.mean(), abs=1e-14)
    assert row.fitness_change is None and row.iteration == 1
    assert row.translation_delta == pytest.approx(np.linalg.norm(plain.transform()[:3, 3]), rel=1e-6)
    assert 0 <= row.rotation_delta_radians <= np.pi
    assert trace.stop_reason == 'fitness_threshold'
    with pytest.raises(AttributeError):
        row.fitness = 1
    history = trace.history
    history.clear()
    del source, target
    gc.collect()
    assert len(trace.history) == 1
    assert trace.result.fitness == plain.fitness


def test_history_fitness_changes_and_iteration_limit():
    xyz = np.random.default_rng(18).uniform(-1, 1, (300, 3)).astype(np.float32)
    source = sr.PointCloud.from_xyz(xyz + np.array([.12, -.09, .04], np.float32))
    target = sr.PointCloud.from_xyz(xyz)
    trace = sr.register_icp_diagnostics(source, target, .3, 2)
    assert trace.stop_reason == 'iteration_limit'
    assert not trace.result.converged
    assert len(trace.history) == trace.result.iterations == 2
    first, last = trace.history
    assert last.fitness == trace.result.fitness
    assert last.fitness_change == first.fitness - last.fitness


@pytest.mark.parametrize('gate', [0, -1, np.nan, np.inf, 1e30, 1e-30])
def test_trace_rejects_invalid_gate(gate):
    source, target = clouds()
    with pytest.raises(ValueError, match='max_correspondence_distance'):
        sr.register_icp_diagnostics(source, target, gate, 10)


def test_trace_input_validation_and_concurrent_shared_reads():
    source, target = clouds()
    with pytest.raises(ValueError, match='max_iterations'):
        sr.register_icp_diagnostics(source, target, .1, 0)
    invalid = sr.PointCloud.from_xyz(np.array([[np.nan, 0, 0]]*3, np.float32))
    with pytest.raises(ValueError, match='finite XYZ'):
        sr.register_icp_diagnostics(invalid, target)
    with ThreadPoolExecutor(max_workers=2) as pool:
        traces = list(pool.map(lambda _: sr.register_icp_diagnostics(source, target, .15, 3), range(4)))
    for trace in traces[1:]:
        np.testing.assert_array_equal(trace.result.transform(), traces[0].result.transform())
        assert trace.history[-1].fitness == traces[0].history[-1].fitness
