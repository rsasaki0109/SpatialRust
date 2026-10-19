"""Prospective gray-only Gaussian3 fitting with original depth and final gates."""
import copy
import hashlib
import json
from pathlib import Path

from gaussian_dense_rgbd_protocol import POLICY as JOINT_POLICY, validate_gaussian_dense_plan
from tum_sequence_reference import validate_plan
from wide_rgbd_protocol import exact_json

ESTIMATOR='spatialrust_gray_gaussian3_dense_rgbd_cpu'
POLICY=copy.deepcopy(JOINT_POLICY)
POLICY.update(preprocessing='gaussian3-gray-only',
    depth_missing='original raw depth validity; no neighborhood expansion')
ROOT=Path(__file__).resolve().parents[1]
PARENTS={name:ROOT/'benchmarks'/('tum-fr1-'+short+'-gaussian3-dense-plan.json')
    for name,short in [('rgbd_dataset_freiburg1_xyz','xyz'),('rgbd_dataset_freiburg1_rpy','rpy')]}


def parent_plan(path):
    raw=Path(path).read_bytes();lf=raw.replace(b'\r\n',b'\n')
    if b'\r' in lf or raw not in (lf,lf.replace(b'\n',b'\r\n')):
        raise ValueError('parent requires exact content with uniform LF or CRLF')
    return validate_gaussian_dense_plan(json.loads(raw)),hashlib.sha256(lf).hexdigest()


def gray_gaussian_dense_plan(parent,digest):
    plan=copy.deepcopy(parent)
    plan['estimator']=ESTIMATOR
    plan['prior_joint_gaussian_lf_sha256']=digest
    plan['controls']['dense_fitting_preprocessing']=copy.deepcopy(POLICY)
    plan['generation']=('All original adjacent RGB-D frames; identical Gaussian3 gray inputs, '
        'original unsmoothed metric depth and the prior returned relative seed, starting identity. '
        'Original raw final statistics and full-cloud/world/IO admission remain. '
        'No reference read, rescale, retries, selection or lost-chain reset.')
    plan['native_refinement_policy']=('Only full-resolution fitting depth changes from the joint '
        'Gaussian3 control: copy original valid depth, keeping missing pixels at their original '
        'locations. Gray kernel, border, pyramid, derivatives, weight, iterations, stopping and '
        'all raw final/trajectory gates remain fixed. Charge the same retained raw planes.')
    plan['scope']=('Prospective component comparison on two observed development scenes after '
        'preserving joint Gaussian3 xyz success and rpy loss at index 349. Preserve raw, joint '
        'and external controls and every failure; freeze both complete timelines before either '
        'new score. No post-score tuning, held-out, speed or physical-calibration claim.')
    return plan


def validate_gray_gaussian_dense_plan(plan):
    validate_plan(plan);path=PARENTS.get(plan.get('sequence'))
    if path is None:raise ValueError('require one of the two fixed gray-only scenes')
    parent,digest=parent_plan(path)
    if exact_json(plan)!=exact_json(gray_gaussian_dense_plan(parent,digest)):
        raise ValueError('gray-only policy, source lineage and all gates are fixed')
    return plan
