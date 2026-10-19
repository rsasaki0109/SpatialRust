"""Data-only fixed previous-motion initialization and unchanged output gates."""
import copy
import json
from pathlib import Path

from dense_rgbd_protocol import validate_final_overlap_plan
from tum_sequence_reference import file_sha256, validate_plan
from wide_rgbd_protocol import exact_json

ESTIMATOR = 'spatialrust_causal_dense_rgbd_cpu'
ROOT = Path(__file__).resolve().parents[1]
PARENTS = {name: ROOT/'benchmarks'/('tum-fr1-'+short+'-dense-final-overlap-plan.json')
           for name, short in [('rgbd_dataset_freiburg1_xyz', 'xyz'),
                               ('rgbd_dataset_freiburg1_rpy', 'rpy')]}
INITIALIZATION = dict(first_pair='identity', subsequent_pairs='previous returned source-to-target increment',
    time_rescaling=False, reference_initialization=False, retries=False, reset_after_loss=False)


def causal_dense_plan(parent, parent_sha256):
    plan = copy.deepcopy(parent)
    plan['estimator'] = ESTIMATOR
    plan['prior_final_overlap_plan_sha256'] = parent_sha256
    plan['dense_initialization'] = copy.deepcopy(INITIALIZATION)
    plan['generation'] = ('Every indexed registered RGB-D frame with the same dense solver and numeric controls. '
        'Identity for the first pair; the exact previous returned source-to-target increment initializes '
        'the next adjacent pair. Retain only one proper 4x4 seed, with no time rescaling, retries, '
        'reference initialization, future-frame selection or lost-chain reset. '
        'Every subsequent call follows the original full-cloud and world-chain acceptance.')
    plan['native_refinement_policy'] = ('Same frozen final-overlap dense numerical binary and all controls; '
        'the explicit initial pose comes from the preceding accepted chain increment only. '
        'Require final_support_only and independently verify every reported initial pose.')
    plan['scope'] = ('Two already observed development scenes, with both parent outcomes preserved. '
        'A separately fixed causal previous-increment hypothesis, not a settings search, held-out '
        'result or general accuracy claim. All final support, completeness and accuracy gates remain.')
    return plan


def validate_causal_dense_plan(plan):
    validate_plan(plan)
    parent_path = PARENTS.get(plan.get('sequence'))
    if parent_path is None:
        raise ValueError('require either fixed causal dense development scene')
    parent = validate_final_overlap_plan(json.loads(parent_path.read_text(encoding='utf-8')))
    expected = causal_dense_plan(parent, file_sha256(parent_path))
    if exact_json(plan) != exact_json(expected):
        raise ValueError('causal initialization, numeric controls and all acceptance gates are fixed')
    return plan
