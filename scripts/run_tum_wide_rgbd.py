"""Freeze one wider LK capture setting for both observed RGB-D development scenes."""
import argparse
import json
from pathlib import Path

from run_tum_adaptive_rgbd import AdaptiveRgbdTracker, validate_adaptive_rgbd_plan
from run_tum_rgbd_odometry import DEFAULT_CONTROLS, _align_rgbd, validate_rgbd_plan
from run_tum_sequence import generate
from tum_sequence_reference import file_sha256
from wide_rgbd_protocol import CAPTURE_OVERRIDES, validate_capture_transition

PLAN_ROOT = Path(__file__).resolve().parents[1] / 'benchmarks'
WIDE_CONTROLS = dict(DEFAULT_CONTROLS, **CAPTURE_OVERRIDES)
ESTIMATORS = dict(rgbd='spatialrust_wide_rgbd_pnp_cpu',
                  adaptive='spatialrust_wide_adaptive_rgbd_pnp_cpu')
SCENES = dict(rgbd_dataset_freiburg1_xyz='xyz', rgbd_dataset_freiburg1_rpy='rpy')


def parent_plan(scene, method):
    return PLAN_ROOT / ('tum-fr1-'+scene+'-'+method+'-pnp-consensus-plan.json')


def validate_wide_rgbd_plan(plan):
    scene = SCENES.get(plan.get('sequence'))
    method = next((key for key,value in ESTIMATORS.items() if value == plan.get('estimator')), None)
    if scene is None or method is None:
        raise ValueError('require one fixed scene and explicit wide RGB-D estimator')
    path = parent_plan(scene,method)
    if plan.get('prior_consensus_plan_sha256') != file_sha256(path):
        raise ValueError('wide RGB-D protocol requires its exact consensus-preserving parent')
    parent = json.loads(path.read_text(encoding='utf-8'))
    validate_capture_transition(parent,plan)
    normalized = dict(plan, estimator=parent['estimator'], controls=parent['controls'])
    (validate_rgbd_plan if method == 'rgbd' else validate_adaptive_rgbd_plan)(normalized)
    return method


def align_wide_rgbd(previous_depth, previous_gray, current_gray, camera, controls):
    return _align_rgbd(previous_depth,previous_gray,current_gray,camera,controls,WIDE_CONTROLS)


class WideAdaptiveRgbdTracker(AdaptiveRgbdTracker):
    def tracking_settings(self):
        return WIDE_CONTROLS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    plan = json.loads((args.prepared_dir/'plan.json').read_text(encoding='utf-8'))
    method = validate_wide_rgbd_plan(plan)
    adapter = align_wide_rgbd if method == 'rgbd' else WideAdaptiveRgbdTracker().align
    files = [__file__]+[Path(__file__).with_name(name) for name in (
        'run_tum_rgbd_odometry.py','run_tum_keyframe_rgbd.py','run_tum_adaptive_rgbd.py',
        'wide_rgbd_protocol.py')]
    files += [parent_plan(SCENES[plan['sequence']],method),
              PLAN_ROOT/'tum-fr1-xyz-plan.json',PLAN_ROOT/'tum-fr1-xyz-rgbd-pnp-plan.json']
    if method == 'adaptive':
        files += [PLAN_ROOT/('tum-fr1-'+SCENES[plan['sequence']]+'-keyframe-rgbd-pnp-plan.json')]
    result = generate(args.prepared_dir,args.manifest_sha256,args.output_dir,
        rgbd_aligner=adapter,method=plan['estimator'],additional_bindings=files)
    print('Frozen wide '+method+' {}/{} poses; estimates SHA-256 {}'.format(
        result['generated_poses'],result['planned_poses'],result['estimates_sha256']))


if __name__ == '__main__':
    main()
