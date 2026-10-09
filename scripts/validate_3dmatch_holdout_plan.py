"""Verify a precommitted scene-holdout plan and frozen local fitting inputs."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

from compare_3dmatch_cases import validate_cases


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def validate_plan(plan, manifest):
    if plan.get('schema') != 'spatialrust.3dmatch-holdout-plan.v1':
        raise ValueError('unsupported holdout plan')
    cases = validate_cases(manifest)
    scenes = {case['scene'] for case in cases}
    if scenes & set(plan['discovery_scenes']):
        raise ValueError('holdout scenes intersect discovery scenes')
    if scenes != set(plan['heldout_scenes']) or len(plan['heldout_scenes']) != len(scenes):
        raise ValueError('planned scene set differs')
    count = plan['cases_per_scene']
    if type(count) is not int or not 1 <= count <= 64 or any(sum(c['scene'] == scene for c in cases) != count for scene in scenes):
        raise ValueError('planned per-scene case count differs')
    if [case['case_id'] for case in cases] != [case['case_id'] for case in plan['cases']]:
        raise ValueError('case order or membership differs')
    if manifest.get('selection_rule') != 'integer_quantiles_of_sorted_nonconsecutive_gt_headers_before_fitting':
        raise ValueError('pair selection rule differs')
    for case, binding in zip(cases, plan['cases']):
        if case['scene'] != binding['scene'] or case['pair'] != binding['pair']:
            raise ValueError('case scene or pair differs')
    if plan['seeds'] != [7, 8, 9] or plan['ransac_iterations'] != 10000:
        raise ValueError('frozen sampler settings differ')
    if plan['primary_rules'] != dict(trim_fractions=[1., .8], motion_limit_metres=.2,
                                     publisher_score_squared_threshold=.04):
        raise ValueError('primary intervention or evaluation settings differ')
    if plan['comparison_controls'] != dict(coarse_leaf=.1, feature_radius=.5, normal_neighbors=20,
        global_gate=.2, stages=[dict(leaf=.05, gate=.2, max_iterations=50),
                               dict(leaf=None, gate=.05, max_iterations=50)], evaluation_gate=.05):
        raise ValueError('comparison controls differ')
    return cases


def verify_files(plan, manifest, manifest_bytes, root):
    cases = validate_plan(plan, manifest)
    if hashlib.sha256(manifest_bytes).hexdigest() != plan['cases_manifest_sha256']:
        raise ValueError('manifest bytes changed')
    for case, binding in zip(cases, plan['cases']):
        for role in ('source', 'target', 'reference_file'):
            if digest(case[role]) != binding['file_sha256'][role]:
                raise ValueError('holdout input or reference changed')
    for relative, expected in plan['algorithm_source_sha256'].items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('helper path escapes repository')
        if digest(root / path) != expected:
            raise ValueError('frozen algorithm helper changed')
    native = Path(importlib.import_module('spatialrust.spatialrust').__file__)
    open3d_native = Path(importlib.import_module('open3d.cpu.pybind').__file__)
    if digest(native) != plan['native_sha256'] or digest(open3d_native) != plan['open3d_native_sha256']:
        raise ValueError('frozen native build changed')
    return dict(planned_cases=len(cases), planned_native_seeds=len(cases)*len(plan['seeds']),
                planned_baseline_registrations=len(cases)*len(plan['seeds'])*2,
                planned_refinement_replays=len(cases)*len(plan['seeds'])*2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.cases.read_bytes()
    result = verify_files(json.loads(args.plan.read_bytes()), json.loads(manifest_bytes),
                          manifest_bytes, Path(__file__).resolve().parents[1])
    print(json.dumps(result))


if __name__ == '__main__':
    main()
