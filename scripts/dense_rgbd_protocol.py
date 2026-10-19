"""Data-only fixed-plan identity for two explicit dense RGB-D development runs."""
import copy
import json
from pathlib import Path

from tum_sequence_reference import file_sha256, validate_plan
from wide_rgbd_protocol import exact_json

ESTIMATOR = 'spatialrust_dense_projective_rgbd_cpu'
FINAL_ESTIMATOR = 'spatialrust_final_overlap_dense_rgbd_cpu'
ROOT = Path(__file__).resolve().parents[1]
PARENTS = {name: ROOT/'benchmarks'/('tum-fr1-'+short+'-rgbd-wide-capture-plan.json')
           for name, short in [('rgbd_dataset_freiburg1_xyz', 'xyz'),
                               ('rgbd_dataset_freiburg1_rpy', 'rpy')]}
CONTROLS = dict(iterations=[20, 10, 5], min_depth=.1, max_depth=5.,
    max_depth_difference=.03, depth_weight=.968, min_correspondences=30,
    min_correspondence_fraction=.5, epsilon=1e-7, max_working_bytes=128*1024**2)


def dense_plan(parent, parent_sha256):
    plan = copy.deepcopy(parent)
    plan['estimator'] = ESTIMATOR
    plan['prior_dense_source_plan_sha256'] = parent_sha256
    plan['controls']['dense_rgbd_odometry'] = copy.deepcopy(CONTROLS)
    plan.pop('capture_revision', None)
    plan['native_refinement_policy'] = ('Dense joint photometric/projective depth Gauss-Newton; '
        'explicit source/target images and depths with identity initialization. '
        'Inherited sparse controls are source lineage only and are not consumed.')
    plan['generation'] = ('Every indexed registered RGB-D frame; bounded native dense '
        'projective photometric/depth estimation on adjacent frames. Invert source-to-target '
        'once into the original world chain; retain full-cloud support and every failure. '
        'No reference initialization, scale fit, future-frame selection or lost-chain reset.')
    plan['scope'] = ('Two already observed development scenes; independently implemented dense '
        'CPU model, not an equivalent Open3D implementation or a holdout. Preserve calibration, '
        'geometric support, completeness and original accuracy gates; no settings search or '
        'post-score criterion changes. Sparse controls retained only to bind reused source inputs.')
    return plan


def validate_dense_plan(plan):
    validate_plan(plan)
    parent_path = PARENTS.get(plan.get('sequence'))
    if parent_path is None:
        raise ValueError('require either fixed dense development scene')
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    expected = dense_plan(parent, file_sha256(parent_path))
    if exact_json(plan) != exact_json(expected):
        raise ValueError('dense model, source lineage and every original acceptance gate are fixed')
    return plan


def final_overlap_plan(parent, parent_sha256):
    plan = copy.deepcopy(parent)
    plan['estimator'] = FINAL_ESTIMATOR
    plan['prior_dense_protocol_sha256'] = parent_sha256
    plan['dense_support_policy'] = ('Minimum count and a nonsingular system for initial/intermediate hypotheses; '
        'the unchanged 0.5 full-resolution correspondence fraction applies to the returned pose. '
        'The independent full-cloud support and complete trajectory accuracy gates are unchanged.')
    plan['native_refinement_policy'] = ('Same bounded dense Gauss-Newton, pyramid, objective and numeric controls; '
        'separate numerical hypothesis admission from final estimate acceptance. '
        'Require the native final_support_only receipt flag; no reference initializer or reset.')
    plan['scope'] = ('Explicit solver-admission revision after preserving both complete strict-iteration failures. '
        'Same two observed development scenes and exact controls; no held-out or general accuracy claim. '
        'All final support, completeness and accuracy thresholds remain unchanged.')
    return plan


def validate_final_overlap_plan(plan):
    validate_plan(plan)
    source = PARENTS.get(plan.get('sequence'))
    if source is None:
        raise ValueError('require the fixed dense development scenes')
    parent_path = source.with_name(source.name.replace('-rgbd-wide-capture-plan', '-dense-rgbd-plan'))
    parent = validate_dense_plan(json.loads(parent_path.read_text(encoding='utf-8')))
    expected = final_overlap_plan(parent, file_sha256(parent_path))
    if exact_json(plan) != exact_json(expected):
        raise ValueError('final-only support revision cannot change numeric controls or acceptance')
    return plan


def validate_dense_experiment_plan(plan):
    return validate_final_overlap_plan(plan) if plan.get('estimator') == FINAL_ESTIMATOR else validate_dense_plan(plan)
