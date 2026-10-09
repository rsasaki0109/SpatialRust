"""Trimmed correspondences, deterministic ties and full-population diagnostics."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import spatialrust as sr


def tied_clouds():
    target = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0],
                       [0, 0, 2], [2, 2, 0], [2, 0, 2]], np.float32)
    source = target.copy()
    source[:3, 0] += .125
    source[3:, 1] += .125
    return sr.PointCloud.from_xyz(source), sr.PointCloud.from_xyz(target)


def test_equal_distance_trim_ties_follow_source_order_and_keep_fitness_denominator():
    source, target = tied_clouds()
    before_source, before_target = source.xyz(), target.xyz()
    trace = sr.register_icp_diagnostics(source, target, .3, 1, trim_fraction=.5, fitness_epsilon=0)
    np.testing.assert_allclose(trace.result.transform(),
                               [[1, 0, 0, -.125], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], atol=1e-7)
    assert trace.history[0].correspondences == 3
    assert trace.history[0].evaluated_correspondences == 6
    assert trace.result.fitness == pytest.approx(.015625, abs=1e-10)
    np.testing.assert_array_equal(source.xyz(), before_source)
    np.testing.assert_array_equal(target.xyz(), before_target)
    result = sr.register_icp(source, target, .3, 1, trim_fraction=.5, fitness_epsilon=0)
    np.testing.assert_array_equal(result.transform(), trace.result.transform())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: sr.register_icp(source, target, .3, 1, trim_fraction=.5), range(8)))
    for result in results:
        np.testing.assert_array_equal(result.transform(), trace.result.transform())


def test_default_and_explicit_untrimmed_results_are_exact():
    source, target = tied_clouds()
    default = sr.register_icp_diagnostics(source, target, .3, 5)
    explicit = sr.register_icp_diagnostics(source, target, .3, 5, trim_fraction=1)
    np.testing.assert_array_equal(default.result.transform(), explicit.result.transform())
    assert default.result.fitness == explicit.result.fitness
    assert default.stop_reason == explicit.stop_reason
    assert [(r.correspondences, r.fitness) for r in default.history] == [
        (r.correspondences, r.fitness) for r in explicit.history]


@pytest.mark.parametrize('fraction', [0, -.1, 1.01, np.nan, np.inf, -np.inf])
def test_invalid_trim_fractions_are_named_errors(fraction):
    source, target = tied_clouds()
    for function in (sr.register_icp, sr.register_icp_diagnostics):
        with pytest.raises(ValueError, match='trim_fraction'):
            function(source, target, trim_fraction=fraction)


def test_floor_retention_and_too_few_pairs_fail_without_silent_clamping():
    source, target = tied_clouds()
    source = sr.PointCloud.from_xyz(source.xyz()[:5])
    with pytest.raises(ValueError, match='trim_fraction retains only 2 of 5'):
        sr.register_icp(source, target, .3, 1, trim_fraction=.5)
    trace = sr.register_icp_diagnostics(source, target, .3, 1, trim_fraction=.6)
    assert trace.history[0].correspondences == 3
