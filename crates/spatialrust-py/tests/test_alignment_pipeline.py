"""Real bindings: file -> voxel -> ICP -> full-resolution save/read roundtrip."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tempfile
from unittest.mock import patch
import spatialrust as sr

EXAMPLE = Path(__file__).parents[1] / 'examples' / 'align_point_clouds.py'
spec = importlib.util.spec_from_file_location('alignment_example', EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def fixture_files(tmp_path):
    rng = np.random.default_rng(42)
    target = rng.uniform(-1, 1, (120, 3)).astype(np.float32)
    source = target + np.array([.012, -.008, .005], np.float32)
    paths = [tmp_path / 'source.pcd', tmp_path / 'target.pcd']
    for path, xyz in zip(paths, (source, target)):
        sr.write(str(path), sr.PointCloud.from_xyz(xyz))
    return paths, source, target


def test_distinct_coarse_and_fine_gates_reach_correct_stages(tmp_path):
    paths, _, _ = fixture_files(tmp_path)
    with patch.object(sr, 'register_icp', wraps=sr.register_icp) as register:
        _, report = example.align_files(*paths, max_distance=.1, fine_distance=.02)
    assert [call.args[2] for call in register.call_args_list] == [.1, .02]
    assert [stage['max_correspondence_distance_metres'] for stage in report['stages']] == [.1, .02]
    assert report['evaluation_distance_metres'] == .1  # Diagnostic gate is independent.
    assert report['fine_distance_metres'] == .02
    default_cloud, default_report = example.align_files(*paths, max_distance=.1)
    explicit_cloud, explicit_report = example.align_files(*paths, max_distance=.1, fine_distance=.1)
    assert default_report == explicit_report
    np.testing.assert_array_equal(default_cloud.xyz(), explicit_cloud.xyz())


def test_opt_in_history_preserves_alignment_and_reaches_cli_html(tmp_path):
    paths, _, _ = fixture_files(tmp_path)
    plain_cloud, plain = example.align_files(*paths, fine_distance=.02)
    traced_cloud, traced = example.align_files(*paths, fine_distance=.02, trace=True)
    np.testing.assert_array_equal(plain_cloud.xyz(), traced_cloud.xyz())
    for stage in traced['stages']:
        history = stage.pop('icp_history')
        reason = stage.pop('stop_reason')
        assert len(history) == stage['iterations']
        assert history[-1]['fitness_metres_squared'] == stage['kernel_fitness_metres_squared']
        assert (reason != 'iteration_limit') == stage['converged']
    assert traced == plain
    output = tmp_path / 'traced'
    result = subprocess.run([sys.executable, str(EXAMPLE), *map(str, paths),
                             '--output-dir', str(output), '--trace', '--html-report'],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / 'alignment.json').read_text())
    assert all('icp_history' in stage for stage in report['stages'])
    rendered = (output / 'report.html').read_text()
    assert rendered.count('class="trace-chart"') == 8
    assert 'Rotation update (degrees)' in rendered
    assert 'Stop reason:' in rendered


def test_invalid_fine_gate_is_rejected_before_file_io():
    for gate in (0, -1, True, float('nan'), float('inf'), 1e30, 1e-30):
        with patch.object(sr, 'read', side_effect=AssertionError('IO before validation')):
            try:
                example.align_files('missing', 'missing', fine_distance=gate)
            except ValueError as error:
                assert 'fine_distance' in str(error)
            else:
                raise AssertionError('invalid fine gate accepted')


def test_full_resolution_roundtrip_and_transform_direction(tmp_path):
    paths, source, target = fixture_files(tmp_path)
    output = tmp_path / 'run'
    result = subprocess.run([sys.executable, str(EXAMPLE), *map(str, paths), '--output-dir', str(output), '--leaf', '.2'], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / 'alignment.json').read_text())
    saved = sr.read(str(output / 'aligned.pcd')).xyz()
    transform = np.asarray(report['transform_source_to_target'])
    assert len(saved) == len(source)
    assert report['registration_source_points'] < len(source)
    assert [s['name'] for s in report['stages']] == ['voxel', 'full_resolution']
    assert report['stages'][1]['source_points'] == len(source)
    assert report['aligned_support']['distance_gated_points'] == len(source)
    assert report['aligned_reverse_support']['distance_gated_points'] == len(target)
    assert report['aligned_support']['gated_rmse_metres'] < 1e-5
    assert report['iterations'] == report['stages'][1]['iterations']
    assert report['converged'] == report['stages'][1]['converged']
    assert report['initial_transform_supplied'] is False
    assert report['initial_support'] == report['before_support']
    np.testing.assert_allclose(report['stages'][1]['transform_source_to_target'], report['transform_source_to_target'])
    expected = source @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(saved, expected, atol=2e-6)
    assert np.linalg.norm(saved-target, axis=1).mean() < np.linalg.norm(source-target, axis=1).mean()
    original = (output / 'aligned.pcd').read_bytes()
    result = subprocess.run([sys.executable, str(EXAMPLE), *map(str, paths), '--output-dir', str(output)], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert (output / 'aligned.pcd').read_bytes() == original


def test_cli_html_report_and_failed_html_write_cleanup(tmp_path, monkeypatch):
    paths, _, _ = fixture_files(tmp_path)
    output = tmp_path / 'with-html'
    result = subprocess.run([sys.executable, str(EXAMPLE), *map(str, paths),
                             '--output-dir', str(output), '--html-report', '--evaluation-distance', '.001'],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    rendered = (output / 'report.html').read_text()
    assert 'Alignment support report' in rendered
    assert 'After: target → source' in rendered
    assert 'Evaluation distance gate: 0.001 m' in rendered
    assert (output / 'aligned.pcd').exists() and (output / 'alignment.json').exists()
    monkeypatch.syspath_prepend(str(EXAMPLE.parent))
    write_text = Path.write_text
    failed = tmp_path / 'failed-html'
    def fail_html(path, *args, **kwargs):
        if path.name == 'report.html':
            raise OSError('injected HTML write failure')
        return write_text(path, *args, **kwargs)
    with patch.object(sys, 'argv', [str(EXAMPLE), *map(str, paths), '--output-dir', str(failed), '--html-report']), patch.object(Path, 'write_text', fail_html):
        try:
            example.main()
        except OSError as error:
            assert 'injected HTML' in str(error)
        else:
            raise AssertionError('expected HTML write failure')
    assert not failed.exists()


def test_independent_evaluation_distance_does_not_change_estimated_pose(tmp_path):
    paths, _, _ = fixture_files(tmp_path)
    _, baseline = example.align_files(*paths, max_distance=.1)
    _, evaluated = example.align_files(*paths, max_distance=.1, evaluation_distance=.001)
    np.testing.assert_array_equal(baseline['transform_source_to_target'], evaluated['transform_source_to_target'])
    assert evaluated['max_distance_metres'] == .1
    assert evaluated['evaluation_distance_metres'] == .001
    assert evaluated['before_support']['query_fraction'] == 0
    assert evaluated['aligned_support']['query_fraction'] == 1
    for value in [0, -1, float('nan'), float('inf'), True, 1e-30, 1e30]:
        try:
            example.align_files('missing', 'missing', evaluation_distance=value)
        except ValueError as error:
            assert 'evaluation_distance' in str(error)
        else:
            raise AssertionError('invalid evaluation distance accepted')


def test_loaded_cloud_alignment_matches_files_and_preserves_inputs(tmp_path):
    paths, source_xyz, target_xyz = fixture_files(tmp_path)
    source, target = [sr.read(str(path)) for path in paths]
    expected, file_report = example.align_files(*paths)
    with patch.object(sr, 'read', side_effect=AssertionError('loaded alignment must not read files')):
        actual, loaded_report = example.align_clouds(source, target, source_name=str(paths[0]), target_name=str(paths[1]))
    np.testing.assert_array_equal(actual.xyz(), expected.xyz())
    assert loaded_report == file_report
    np.testing.assert_array_equal(source.xyz(), source_xyz)
    np.testing.assert_array_equal(target.xyz(), target_xyz)


def test_invalid_settings_before_io():
    for settings in ({'leaf': 0}, {'max_distance': float('nan')}, {'iterations': 0}, {'iterations': True}, {'max_distance': 1e30}):
        try:
            example.align_files('missing-source.pcd', 'missing-target.pcd', **settings)
        except ValueError:
            pass
        else:
            raise AssertionError(settings)


def test_disjoint_inputs_do_not_publish_results(tmp_path):
    paths, source, _ = fixture_files(tmp_path)
    sr.write(str(paths[0]), sr.PointCloud.from_xyz(source + 100))
    output = tmp_path / 'failed'
    result = subprocess.run([sys.executable, str(EXAMPLE), *map(str, paths), '--output-dir', str(output)], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert not output.exists()


def test_coarse_cloud_too_small_has_actionable_error(tmp_path):
    paths, _, _ = fixture_files(tmp_path)
    try:
        example.align_files(*paths, leaf=100)
    except ValueError as error:
        assert 'reduce leaf' in str(error)
    else:
        raise AssertionError('expected coarse-cloud rejection')


def test_nonfinite_inputs_rejected(tmp_path):
    paths, _, _ = fixture_files(tmp_path)
    xyz = np.array([[0,0,0], [1,0,0], [0,np.nan,0]], dtype=np.float32)
    sr.write(str(paths[0]), sr.PointCloud.from_xyz(xyz))
    try:
        example.align_files(*paths)
    except ValueError as error:
        assert 'finite' in str(error)
    else:
        raise AssertionError('expected nonfinite input rejection')


def test_write_failure_cleans_reserved_output(tmp_path):
    paths, _, _ = fixture_files(tmp_path)
    output = tmp_path / 'broken-save'
    def fail(path, cloud):
        Path(path).write_bytes(b'partial')
        raise OSError('injected write failure')
    with patch.object(sys, 'argv', [str(EXAMPLE), *map(str, paths), '--output-dir', str(output)]), patch.object(sr, 'write', side_effect=fail):
        try:
            example.main()
        except OSError as error:
            assert 'injected' in str(error)
        else:
            raise AssertionError('expected write failure')
    assert not output.exists()


def test_rotated_initial_pose_cli_and_composition(tmp_path):
    paths, _, target = fixture_files(tmp_path)
    angle = np.pi / 3
    forward = np.eye(4)
    forward[:3,:3] = [[np.cos(angle),-np.sin(angle),0], [np.sin(angle),np.cos(angle),0], [0,0,1]]
    forward[:3,3] = [10,-2,.5]
    source = (target @ forward[:3,:3].T + forward[:3,3]).astype(np.float32)
    sr.write(str(paths[0]), sr.PointCloud.from_xyz(source))
    initial = np.linalg.inv(forward)
    pose = tmp_path / 'initial.json'
    pose.write_text(json.dumps(initial.tolist()))
    output = tmp_path / 'seeded'
    result = subprocess.run([sys.executable, str(EXAMPLE), *map(str, paths), '--initial-transform', str(pose), '--leaf', '.01', '--output-dir', str(output)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / 'alignment.json').read_text())
    np.testing.assert_allclose(report['transform_source_to_target'], initial, atol=3e-5)
    np.testing.assert_allclose(sr.read(str(output / 'aligned.pcd')).xyz(), target, atol=3e-5)
    np.testing.assert_allclose(report['initial_transform_source_to_target'], initial, atol=1e-6)
    assert report['initial_transform_supplied'] is True
    assert report['before_support']['query_fraction'] == 0
    assert report['initial_support']['query_fraction'] == 1
    assert report['initial_support']['gated_rmse_metres'] < 1e-5


def test_invalid_initial_poses_before_io():
    reflection = np.diag([-1,1,1,1])
    scaled = np.diag([2,1,1,1])
    projective = np.eye(4); projective[3,0] = .1
    for pose in (reflection, scaled, projective, np.zeros((3,3)), np.full((4,4), np.nan), [['1']*4]*4):
        try:
            example.align_files('missing', 'missing', initial_transform=pose)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid pose accepted')


def test_full_resolution_refines_voxel_bias(tmp_path):
    paths, source, target = fixture_files(tmp_path)
    angle = .03
    rotation = np.array([[np.cos(angle),-np.sin(angle),0], [np.sin(angle),np.cos(angle),0], [0,0,1]])
    source = (target @ rotation.T + np.array([.012,-.008,.005])).astype(np.float32)
    sr.write(str(paths[0]), sr.PointCloud.from_xyz(source))
    aligned, report = example.align_files(*paths, leaf=.4, max_distance=.2)
    coarse = np.asarray(report['stages'][0]['transform_source_to_target'])
    coarse_xyz = source @ coarse[:3,:3].T + coarse[:3,3]
    before = np.sqrt(np.mean(np.sum((coarse_xyz-target)**2, axis=1)))
    after = np.sqrt(np.mean(np.sum((aligned.xyz()-target)**2, axis=1)))
    assert before > 1e-4
    assert after < 1e-5 and after < before / 10


def test_native_support_direction_missing_and_invalid():
    target = sr.PointCloud.from_xyz(np.array([[0,0,0],[1,0,0],[0,1,0],[5,0,0]], dtype=np.float32))
    source = sr.PointCloud.from_xyz(target.xyz()[:3])
    assert sr.distance_gated_support(source, target, .1) == (3, 1., 0.)
    assert sr.distance_gated_support(target, source, .1) == (3, .75, 0.)
    far = sr.PointCloud.from_xyz(source.xyz()+100)
    assert sr.distance_gated_support(far, target, .1) == (0, 0., None)
    for distance in (0, -1, float('nan'), float('inf'), 1e30, 1e-30):
        try:
            sr.distance_gated_support(source, target, distance)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid distance accepted')
    for xyz in (np.empty((0,3), dtype=np.float32), np.array([[np.nan,0,0]], dtype=np.float32)):
        try:
            sr.distance_gated_support(sr.PointCloud.from_xyz(xyz), target, .1)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid cloud accepted')


def test_support_matches_brute_force_including_gate_boundary():
    rng = np.random.default_rng(43)
    target = rng.uniform(-1, 1, (77, 3)).astype(np.float32)
    query = rng.uniform(-1, 1, (91, 3)).astype(np.float32)
    delta = query[:, None, :] - target[None, :, :]
    nearest = (delta[..., 0]**2 + delta[..., 1]**2 + delta[..., 2]**2).min(axis=1)
    gate = np.float32(.25)
    accepted = nearest[nearest <= gate*gate]
    result = sr.distance_gated_support(sr.PointCloud.from_xyz(query), sr.PointCloud.from_xyz(target), float(gate))
    assert result[0] == len(accepted)
    assert result[1] == len(accepted) / len(query)
    np.testing.assert_allclose(result[2], np.sqrt(accepted.astype(np.float64).mean()), rtol=1e-7)
    # Exact gate, tied/duplicate references, outside gate, then an exact match:
    # every query must clear the reusable neighbor buffer.
    references = sr.PointCloud.from_xyz(np.array([[0, 0, 0], [0, 0, 0]], np.float32))
    queries = sr.PointCloud.from_xyz(np.array([[.25, 0, 0], [2, 0, 0], [0, 0, 0]], np.float32))
    count, fraction, rmse = sr.distance_gated_support(queries, references, .25)
    assert (count, fraction) == (2, 2 / 3)
    assert rmse == np.sqrt(.25**2 / 2)


def test_support_releases_gil_and_shared_queries_are_consistent():
    xyz = np.random.default_rng(7).uniform(-1, 1, (120000, 3)).astype(np.float32)
    cloud = sr.PointCloud.from_xyz(xyz)
    ready, start = threading.Event(), threading.Event()
    progress = []
    def observer():
        ready.set()
        start.wait()
        progress.append(True)
    worker = threading.Thread(target=observer)
    worker.start()
    ready.wait(timeout=5)
    previous = sys.getswitchinterval()
    try:
        # Prevent interpreter time-slicing from masquerading as a native GIL release.
        sys.setswitchinterval(60)
        start.set()
        result = sr.distance_gated_support(cloud, cloud, .1)
        progressed_during_call = bool(progress)
    finally:
        sys.setswitchinterval(previous)
        worker.join(timeout=5)
    assert progressed_during_call
    assert result == (len(xyz), 1., 0.)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(sr.distance_gated_support, cloud, cloud, .1) for _ in range(2)]
        assert all(future.result(timeout=30) == result for future in futures)


def test_icp_releases_gil_and_shared_inputs_are_consistent():
    xyz = np.random.default_rng(18).uniform(-1, 1, (20000, 3)).astype(np.float32)
    cloud = sr.PointCloud.from_xyz(xyz)
    ready, start = threading.Event(), threading.Event()
    progress = []
    def observer():
        ready.set()
        start.wait()
        progress.append(True)
    worker = threading.Thread(target=observer)
    worker.start()
    assert ready.wait(timeout=5)
    previous = sys.getswitchinterval()
    try:
        sys.setswitchinterval(60)
        start.set()
        result = sr.register_icp(cloud, cloud, .1, 2)
        progressed_during_call = bool(progress)
    finally:
        sys.setswitchinterval(previous)
        worker.join(timeout=5)
    assert progressed_during_call
    np.testing.assert_allclose(result.transform(), np.eye(4), atol=1e-6)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(sr.register_icp, cloud, cloud, .1, 2) for _ in range(2)]
        for future in futures:
            other = future.result(timeout=30)
            np.testing.assert_array_equal(other.transform(), result.transform())
            assert (other.iterations, other.converged, other.fitness) == (result.iterations, result.converged, result.fitness)


def test_public_icp_invalid_parameters_and_clouds():
    valid = sr.PointCloud.from_xyz(np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32))
    for distance in (0, -1, float('nan'), float('inf'), 1e30, 1e-30):
        try:
            sr.register_icp(valid, valid, distance, 10)
        except ValueError as error:
            assert 'max_correspondence_distance' in str(error)
        else:
            raise AssertionError('invalid distance accepted')
    try:
        sr.register_icp(valid, valid, .1, 0)
    except ValueError as error:
        assert 'max_iterations' in str(error)
    else:
        raise AssertionError('zero iterations accepted')
    for xyz in (np.empty((0,3), dtype=np.float32), np.zeros((2,3), dtype=np.float32),
                np.array([[0,0,0],[1,0,0],[0,np.nan,0]], dtype=np.float32)):
        invalid = sr.PointCloud.from_xyz(xyz)
        for source, target, name in ((invalid,valid,'source'), (valid,invalid,'target')):
            try:
                sr.register_icp(source, target, .1, 10)
            except ValueError as error:
                assert name in str(error)
            else:
                raise AssertionError('invalid cloud accepted')
    # Rejection must not damage later valid calls or input ownership.
    result = sr.register_icp(valid, valid, .1, 10)
    np.testing.assert_allclose(result.transform(), np.eye(4), atol=1e-6)


if __name__ == '__main__':
    test_invalid_settings_before_io()
    test_invalid_initial_poses_before_io()
    test_native_support_direction_missing_and_invalid()
    test_support_releases_gil_and_shared_queries_are_consistent()
    test_icp_releases_gil_and_shared_inputs_are_consistent()
    test_public_icp_invalid_parameters_and_clouds()
    for test in (test_full_resolution_roundtrip_and_transform_direction,
                 test_disjoint_inputs_do_not_publish_results, test_coarse_cloud_too_small_has_actionable_error,
                 test_nonfinite_inputs_rejected, test_write_failure_cleans_reserved_output,
                 test_rotated_initial_pose_cli_and_composition, test_full_resolution_refines_voxel_bias):
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory))
    print('Python alignment pipeline: PASS (13 groups, real bindings and subprocess CLI)')
