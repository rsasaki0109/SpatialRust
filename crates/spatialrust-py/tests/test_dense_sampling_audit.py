"""Fixed operator expectations, strict provenance and explicit audit limitations."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_dense_sampling as audit


def native_record():
    plan = audit.load_plan(); point, gray, depth = audit.row_inputs(plan)
    return dict(schema='spatialrust.native-sampling-probe.v1', even_dimensions=[4, 4],
        coarse_gray=23/255., odd_dimensions=[4, 3], hole_coarse_depth=2.,
        hole_admitted=False, border_admitted=False, subpixel=[26.25/255., 3/255., 5/255.],
        row_point=point.tolist(), row_gray_gradient=gray.tolist(), row_depth_gradient=depth.tolist(),
        row_jacobian=audit.analytic_jacobian(point, plan['row']['intrinsics'], gray, depth).tolist())


@pytest.mark.parametrize('crlf', [False, True])
def test_fixed_plan_accepts_only_its_exact_uniform_line_endings(tmp_path, crlf):
    raw = audit.PLAN.read_bytes().replace(b'\r\n', b'\n')
    path = tmp_path / 'plan.json'; path.write_bytes(raw.replace(b'\n', b'\r\n') if crlf else raw)
    assert audit.load_plan(path)['slice'] == '146O'


@pytest.mark.parametrize('change', ['stray-cr', 'mixed-lines', 'whitespace', 'numeric'])
def test_plan_rejects_source_or_model_byte_changes(tmp_path, change):
    raw = audit.PLAN.read_bytes().replace(b'\r\n', b'\n')
    if change == 'stray-cr': raw = b'\r' + raw
    elif change == 'mixed-lines': raw = raw.replace(b'\n', b'\r\n', 1)
    elif change == 'whitespace': raw += b' '
    else: raw = raw.replace(b'"depth_m": 2.0', b'"depth_m": 3.0')
    path = tmp_path / 'changed.json'; path.write_bytes(raw)
    with pytest.raises(ValueError): audit.load_plan(path)


def test_analytic_chain_rule_matches_closed_form_at_optical_axis():
    observed = audit.analytic_jacobian([0., 0., 2.], [64., 80., 0., 0.], [.002, -.001], [.003, .004])
    expected = [[.08, .128, 0., .064, -.04, 0.], [-.32, .192, 0., .096, .16, -1.]]
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-15)
    with pytest.raises(ValueError): audit.analytic_jacobian([0., 0., 0.], [64., 80., 0., 0.], [.002, -.001], [.003, .004])


def test_native_record_checks_all_expected_values():
    assert max(audit.validate_native(native_record(), audit.load_plan()).values()) == 0.


@pytest.mark.parametrize('axis', range(12))
def test_each_native_jacobian_entry_is_independently_checked(axis):
    observed = native_record(); observed['row_jacobian'][axis//6][axis%6] += 1e-5
    with pytest.raises(ValueError, match='expectation'): audit.validate_native(observed, audit.load_plan())


@pytest.mark.parametrize('key,value', [
    ('odd_dimensions', [3, 2]), ('hole_admitted', True), ('border_admitted', 0),
    ('coarse_gray', float('nan')), ('subpixel', [0., 0.]), ('row_point', ['x', 0., 2.])])
def test_bad_native_geometry_types_missingness_or_values_reject(key, value):
    observed = native_record(); observed[key] = value
    with pytest.raises(ValueError): audit.validate_native(observed, audit.load_plan())


def test_probe_log_accepts_one_inline_test_record_and_rejects_duplicates(tmp_path):
    path = tmp_path / 'probe.log'; raw = 'test probe ... '+audit.MARKER+json.dumps(native_record())+'\n'
    path.write_text(raw, encoding='utf-8')
    assert audit.native_observation(path) == native_record()
    path.write_text(raw+raw, encoding='utf-8')
    with pytest.raises(ValueError, match='exactly one'): audit.native_observation(path)


def reference_record():
    plan = audit.load_plan(); row = plan['row']; weights = np.sqrt([1-row['depth_weight'], row['depth_weight']])
    residual = [float(np.float32(row['target_gray']))-float(np.float32(row['source_gray'])),
                float(np.float32(row['target_depth']))-float(np.float32(row['point'][2]))]
    return dict(even_dimensions=[4, 4], coarse_gray=27/255., odd_dimensions=[3, 2],
        hole_coarse_valid=False, integer_hole_valid=True, integer_border_valid=True,
        rounded_coordinates=[2, 3], rounded_gray=28/255., gradient=[3/255., 5/255.],
        row_residual=(weights*np.asarray(residual)).tolist(), row_weights=[1., 1.],
        jacobian_observable=False, row_jacobian=None)


def test_unobservable_reference_jacobian_is_explicit_not_a_zero_success():
    observed = reference_record(); audit.validate_reference(observed, audit.load_plan())
    observed['row_jacobian'] = np.zeros((2, 6)).tolist()
    with pytest.raises(ValueError, match='unobserved'): audit.validate_reference(observed, audit.load_plan())


def test_observable_reference_jacobian_and_actual_residuals_must_match():
    observed = reference_record(); observed['jacobian_observable'] = True
    plan = audit.load_plan(); point, gray, depth = audit.row_inputs(plan)
    scale = np.sqrt([1-plan['row']['depth_weight'], plan['row']['depth_weight']])
    observed['row_jacobian'] = (scale[:, None]*audit.analytic_jacobian(point, plan['row']['intrinsics'], gray, depth)).tolist()
    audit.validate_reference(observed, plan)
    observed['row_residual'][1] += .01
    with pytest.raises(ValueError): audit.validate_reference(observed, plan)


def test_pinned_source_bytes_cannot_be_replaced_by_equivalent_metadata(tmp_path):
    file = tmp_path / 'source.cpp'; file.write_bytes(b'pinned source\n')
    expected = dict(upstream_path='source.cpp', bytes=file.stat().st_size, sha256=audit.sha(file))
    plan = copy.deepcopy(audit.load_plan()); plan['reference']['sources'] = [expected]
    row = dict(expected, path=str(file.resolve()))
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps(dict(commit=plan['reference']['commit'], files=[row])), encoding='utf-8')
    audit.verify_sources(path, plan)
    file.write_bytes(b'changed data!\n')
    with pytest.raises(ValueError): audit.verify_sources(path, plan)


def test_low_disk_refuses_before_native_inputs_or_reference_import(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    monkeypatch.setattr(audit.importlib, 'import_module', lambda _: pytest.fail('backend imported before storage admission'))
    with pytest.raises(ValueError, match='20GiB'):
        audit.run(tmp_path/'missing.log', '0'*64, tmp_path/'missing.exe', '0'*64, tmp_path/'missing.json', tmp_path/'out')
    assert not (tmp_path/'out').exists()
