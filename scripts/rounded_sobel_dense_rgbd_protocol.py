"""Fixed complete rounded-pixel/Sobel fitting model; original raw final admission."""
import copy
import hashlib
import json
from pathlib import Path

from pair_mean_dense_rgbd_protocol import validate_pair_mean_dense_plan
from tum_sequence_reference import validate_plan
from wide_rgbd_protocol import exact_json

ESTIMATOR = 'spatialrust_rounded_sobel_dense_rgbd_cpu'
POLICY = {'model': 'rounded-sobel', 'initial_filter': {'gray': 'Gaussian3', 'depth': 'Gaussian3; any nonfinite tap propagates', 'kernel': [1, 2, 1], 'divisor': 16, 'border': 'replicate'}, 'intensity_normalization': {'method': 'initial-rounded-correspondence-pair-mean', 'target_mean': 0.5, 'ordering': 'once; full-resolution row-major', 'admission': 'minimum count and finite positive means/gains before mutation', 'clipping': False}, 'pyramid': {'gray': 'Gaussian3 then 2x2 arithmetic mean', 'depth': '2x2 arithmetic mean; nonfinite propagates', 'dimensions': 'floor(width/2), floor(height/2); all scheduled levels at least 3x3', 'intrinsics': 'halve fx, fy, cx, cy per level'}, 'lookup': {'coordinate': 'trunc(projected_coordinate + 0.5); test finite and signed bounds before usize cast', 'point': 'finite transformed point with positive z', 'depth': 'selected target depth finite and original metric difference gate', 'uniqueness': 'one row per valid source pixel; no target uniqueness claim'}, 'gradient': {'method': 'replicated-edge Sobel3, multiplied by 1/8', 'depth_nonfinite': 'replace each nonfinite derivative with zero', 'jacobian': 'local Sobel surrogate; not an exact derivative of the discontinuous rounded-pixel objective'}, 'iterations': 'all scheduled updates; no epsilon early exit', 'pose_update': 'existing native f64 proper left Rodrigues update and solver', 'arithmetic': 'native f64; no external binary or bitwise-equivalence claim', 'final_statistics': 'original raw full-resolution images; original bilinear final admission, count/fraction/error gates', 'ownership': 'borrowed native input; owned Python input copies before GIL release; explicit CPU/device transfers', 'working_bytes': {'formula': '32*sum(floor-pyramid pixel counts) + 64*base_pixels + ceil(height/16)*sizeof(Normal)', 'meaning': 'both fitting pyramids; both raw levels; max of four current-level target gradient planes and two temporary smoothing planes; ordered reductions', 'excludes': 'caller inputs/Python input copies, allocator metadata, stacks', 'checked_before_allocation': True}, 'workers': 4, 'wasm': 'serial'}
ROOT = Path(__file__).resolve().parents[1]
PARENTS = {name: ROOT/'benchmarks'/('tum-fr1-'+short+'-pair-mean-dense-plan.json')
    for name, short in [('rgbd_dataset_freiburg1_xyz', 'xyz'), ('rgbd_dataset_freiburg1_rpy', 'rpy')]}


def parent_plan(path):
    raw = Path(path).read_bytes(); lf = raw.replace(b'\r\n', b'\n')
    if b'\r' in lf or raw not in (lf, lf.replace(b'\n', b'\r\n')):
        raise ValueError('parent requires exact content with uniform LF or CRLF')
    return validate_pair_mean_dense_plan(json.loads(lf)), hashlib.sha256(lf).hexdigest()


def rounded_sobel_dense_plan(parent, digest):
    plan = copy.deepcopy(parent)
    plan['estimator'] = ESTIMATOR
    plan['prior_pair_mean_dense_lf_sha256'] = digest
    del plan['controls']['dense_fitting_preprocessing']
    plan['controls']['dense_fitting_model'] = copy.deepcopy(POLICY)
    plan['generation'] = ('All original adjacent RGB-D frames; explicit rounded-sobel native model. '
        'Identity then the preceding returned relative increment. No reference, rescale, retry, selection or loss reset.')
    plan['native_refinement_policy'] = ('Joint initial Gaussian3, rounded-pair means, floor 2x2 pyramids, '
        'integer target lookup and Sobel3 surrogate gradients form one fixed opt-in model. Execute '
        'the complete original iteration schedule. Retain the native f64 solver/proper left update, '
        'all original raw final gates and bounded ownership; no external-estimator equivalence claim.')
    plan['scope'] = ('Prospective development comparison after source-bound operator audit. Preserve '
        'all controls/results. Freeze both full timelines before either score. No post-score tuning, '
        'held-out, speed or physical-calibration claim; every operational 80% requirement remains.')
    return plan


def validate_rounded_sobel_dense_plan(plan):
    validate_plan(plan); path = PARENTS.get(plan.get('sequence'))
    if path is None:
        raise ValueError('require one of the two fixed rounded-sobel scenes')
    parent, digest = parent_plan(path)
    if exact_json(plan) != exact_json(rounded_sobel_dense_plan(parent, digest)):
        raise ValueError('rounded-sobel model, source lineage and all gates are fixed')
    return plan
