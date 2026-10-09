import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'scripts'))
    import validate_3dmatch_holdout_plan
    return validate_3dmatch_holdout_plan


def fixture():
    cases = [dict(case_id=f'heldout-{i}', scene='heldout', pair=[i*4, i*4+2],
                  source=f'source-{i}', target=f'target-{i}', reference_file=f'reference-{i}') for i in range(3)]
    manifest = dict(schema='spatialrust.3dmatch-cases.v1', cases=cases,
        selection_rule='integer_quantiles_of_sorted_nonconsecutive_gt_headers_before_fitting')
    plan = dict(schema='spatialrust.3dmatch-holdout-plan.v1', discovery_scenes=['discovery'],
                heldout_scenes=['heldout'], cases_per_scene=3, cases=copy.deepcopy(cases),
                seeds=[7, 8, 9], ransac_iterations=10000,
                primary_rules=dict(trim_fractions=[1., .8], motion_limit_metres=.2,
                                   publisher_score_squared_threshold=.04),
                comparison_controls=dict(coarse_leaf=.1, feature_radius=.5, normal_neighbors=20,
                    global_gate=.2, stages=[dict(leaf=.05, gate=.2, max_iterations=50),
                                           dict(leaf=None, gate=.05, max_iterations=50)], evaluation_gate=.05))
    return plan, manifest


@pytest.mark.parametrize('change', ['scene_leak', 'case_order', 'pair', 'seed', 'trim', 'motion', 'gate', 'count'])
def test_changed_holdout_design_rejected(module, change):
    plan, manifest = fixture()
    if change == 'scene_leak':
        plan['discovery_scenes'].append('heldout')
    elif change == 'case_order':
        manifest['cases'].reverse()
    elif change == 'pair':
        manifest['cases'][0]['pair'] = [0, 3]
    elif change == 'seed':
        plan['seeds'] = [7, 8, 10]
    elif change == 'trim':
        plan['primary_rules']['trim_fractions'] = [1., .6]
    elif change == 'motion':
        plan['primary_rules']['motion_limit_metres'] = .1
    elif change == 'gate':
        plan['comparison_controls']['global_gate'] = .3
    else:
        manifest['cases'].pop()
    with pytest.raises(ValueError):
        module.validate_plan(plan, manifest)


def test_file_binding_and_native_validation(module, tmp_path, monkeypatch):
    plan, manifest = fixture()
    for case, binding in zip(manifest['cases'], plan['cases']):
        binding['file_sha256'] = {}
        for role in ('source', 'target', 'reference_file'):
            path = tmp_path / case[role]
            path.write_bytes((case[role]*100).encode())
            case[role] = str(path)
            binding['file_sha256'][role] = module.digest(path)
    native = tmp_path / 'native'
    native.write_bytes(b'controlled native fixture')
    helper = tmp_path / 'helper.py'
    helper.write_text('frozen helper', encoding='utf-8')
    plan.update(native_sha256=module.digest(native), open3d_native_sha256=module.digest(native),
                algorithm_source_sha256={'helper.py': module.digest(helper)})
    data = json.dumps(manifest).encode()
    plan['cases_manifest_sha256'] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(module.importlib, 'import_module', lambda name: SimpleNamespace(__file__=str(native)))
    result = module.verify_files(plan, manifest, data, tmp_path)
    assert result['planned_baseline_registrations'] == 18
    assert result['planned_refinement_replays'] == 18
    native.write_bytes(b'changed native fixture')
    with pytest.raises(ValueError, match='native'):
        module.verify_files(plan, manifest, data, tmp_path)
    native.write_bytes(b'controlled native fixture')
    Path(manifest['cases'][0]['source']).write_bytes(b'changed input')
    with pytest.raises(ValueError, match='input'):
        module.verify_files(plan, manifest, data, tmp_path)
