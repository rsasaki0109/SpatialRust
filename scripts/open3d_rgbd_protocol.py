"""Fixed CPU RGB-D comparator; no Open3D import or estimator execution."""
import copy
import hashlib
import json
from pathlib import Path

from causal_dense_rgbd_protocol import INITIALIZATION, validate_causal_dense_plan
from tum_sequence_reference import validate_plan
from wide_rgbd_protocol import exact_json

ESTIMATOR = 'open3d_causal_rgbd_cpu'
ROOT = Path(__file__).resolve().parents[1]
PARENTS = {name: ROOT/'benchmarks'/('tum-fr1-'+short+'-causal-dense-plan.json')
           for name, short in [('rgbd_dataset_freiburg1_xyz', 'xyz'),
                               ('rgbd_dataset_freiburg1_rpy', 'rpy')]}
CONTROLS = dict(iterations=[20, 10, 5], depth_min=.1, depth_max=5., depth_diff_max=.03,
                jacobian='RGBDOdometryJacobianFromHybridTerm', package_version='0.19.0')
CPU_MODULE_SHA256 = 'ab1b2f4a1126a11886a2a9f3eeab5ceaeac0bab1b53ee319eb363df7ef629d88'


def comparator_plan(parent, parent_sha256):
    plan = copy.deepcopy(parent)
    plan['estimator'] = ESTIMATOR
    plan['prior_causal_dense_plan_sha256'] = parent_sha256
    plan['controls']['open3d_rgbd_odometry'] = copy.deepcopy(CONTROLS)
    plan['dense_initialization'] = copy.deepcopy(INITIALIZATION)
    plan['open3d_cpu_module_sha256'] = CPU_MODULE_SHA256
    plan['generation'] = ('All indexed original RGB-D frames. Use owned float32 grayscale/metric-depth '
        'CPU images with the pinned Open3D legacy hybrid estimator; initialize identity, then the '
        'preceding returned source-to-target increment. No reference read, retry, rescaling, '
        'frame selection or reset. Retain all rows and stop estimator calls after any loss.')
    plan['native_refinement_policy'] = ('Separate Open3D CPU comparator, not the native dense solver. '
        'The inherited dense controls define unchanged raw full-resolution final validation only. '
        'Apply the original full-cloud/world/typed IO gates to every admitted estimate. '
        'Open3D information matrices are diagnostics, not physical calibration uncertainty.')
    plan['comparator_preprocessing'] = ('Pinned Open3D full-resolution Gaussian3 gray/depth and '
        'initial-correspondence mean normalization to 0.5, followed by its own pyramids, sampling '
        'and derivatives. No equivalent implementation, stopping, memory or speed claim.')
    plan['scope'] = ('Full runs on the same two observed development scenes, with native controls '
        'and every failed result preserved. Freeze both complete timelines before either new '
        'reference score. Completeness, reference coverage, final support, ATE/RPE and calibration '
        'are unchanged; no post-score tuning or held-out generalization claim.')
    return plan


def validate_comparator_plan(plan):
    validate_plan(plan)
    path = PARENTS.get(plan.get('sequence'))
    if path is None:
        raise ValueError('require either fixed full-scene RGB-D comparator input')
    raw = path.read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    crlf = lf.replace(b'\n', b'\r\n')
    declared = plan.get('prior_causal_dense_plan_sha256')
    # The recorded parent was a pre-existing CRLF working copy despite the LF
    # checkout rule. Retain that exact identity and admit only its LF equivalent;
    # never canonicalize JSON content, mixed endings, whitespace or field values.
    if (b'\r' in lf or raw not in (lf, crlf)
            or declared not in {hashlib.sha256(lf).hexdigest(), hashlib.sha256(crlf).hexdigest()}):
        raise ValueError('require the exact parent bytes with uniform LF or CRLF endings')
    parent = validate_causal_dense_plan(json.loads(raw))
    expected = comparator_plan(parent, declared)
    if exact_json(plan) != exact_json(expected):
        raise ValueError('comparator package, model, causal seed and all final gates are fixed')
    return plan
