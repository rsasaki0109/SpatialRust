"""Prospective native Gaussian3 fitting policy with unchanged raw final gates."""
import copy
import hashlib
import json
from pathlib import Path

from causal_dense_rgbd_protocol import validate_causal_dense_plan
from tum_sequence_reference import validate_plan
from wide_rgbd_protocol import exact_json

ESTIMATOR = 'spatialrust_gaussian3_dense_rgbd_cpu'
POLICY = dict(preprocessing='gaussian3', kernel=[1, 2, 1], divisor=16,
    border='replicate', depth_missing='any invalid tap makes filtered depth missing',
    intensity_normalization=False, final_statistics='original raw full-resolution images')
ROOT = Path(__file__).resolve().parents[1]
PARENTS = {name: ROOT/'benchmarks'/('tum-fr1-'+short+'-causal-dense-plan.json')
    for name, short in [('rgbd_dataset_freiburg1_xyz','xyz'),('rgbd_dataset_freiburg1_rpy','rpy')]}


def parent_plan(path):
    raw = Path(path).read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    if b'\r' in lf or raw not in (lf, lf.replace(b'\n', b'\r\n')):
        raise ValueError('parent source requires uniform LF or CRLF')
    return validate_causal_dense_plan(json.loads(raw)), hashlib.sha256(lf).hexdigest()


def gaussian_dense_plan(parent, canonical_parent_sha256):
    plan = copy.deepcopy(parent)
    plan['estimator'] = ESTIMATOR
    plan['prior_causal_dense_lf_sha256'] = canonical_parent_sha256
    plan['controls']['dense_fitting_preprocessing'] = copy.deepcopy(POLICY)
    plan['generation'] = ('All original adjacent frames with the previous returned source-to-target '
        'increment, starting identity. Native fitting uses owned float64 Gaussian3 gray/depth '
        'at full resolution before the unchanged pyramid. Final support/residuals use original '
        'raw images. No reference, rescale, retries, selection or lost-chain reset.')
    plan['native_refinement_policy'] = ('Only the explicit full-resolution fitting input policy changes. '
        'No brightness normalization, derivative, update, stopping, downsample, numeric-control or '
        'final admission change. Charge both retained raw levels before allocating fit buffers.')
    plan['scope'] = ('Prospective preprocessing block on two observed development scenes; preserve '
        'native causal and external comparator controls and all failures. Freeze both complete '
        'timelines before either reference score. No held-out, equivalence or speed claim. '
        'Canonical clock/physical calibration, raw Linux classes and full mapping/release remain open.')
    return plan


def validate_gaussian_dense_plan(plan):
    validate_plan(plan)
    path = PARENTS.get(plan.get('sequence'))
    if path is None:
        raise ValueError('require one of the two fixed full-scene native inputs')
    parent, digest = parent_plan(path)
    if exact_json(plan) != exact_json(gaussian_dense_plan(parent,digest)):
        raise ValueError('Gaussian3 model, causal seed and all raw/accuracy gates are fixed')
    return plan
