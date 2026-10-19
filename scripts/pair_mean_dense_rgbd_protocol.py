"""Fixed initial-correspondence intensity normalization with original depth."""
import copy
import hashlib
import json
from pathlib import Path

from gray_gaussian_dense_rgbd_protocol import POLICY as GRAY_POLICY, validate_gray_gaussian_dense_plan
from tum_sequence_reference import validate_plan
from wide_rgbd_protocol import exact_json

ESTIMATOR='spatialrust_pair_mean_dense_rgbd_cpu'
POLICY=copy.deepcopy(GRAY_POLICY)
POLICY.update(preprocessing='gaussian3-gray-only-pair-mean', intensity_normalization=dict(
    method='initial-correspondence-pair-mean', target_mean=0.5,
    correspondence='original fitting depth; native bilinear target depth and filtered gray; initial pose',
    ordering='full-resolution row-major accumulation once before pyramid construction',
    admission='minimum correspondence count; finite positive means and gains', clipping=False))
ROOT=Path(__file__).resolve().parents[1]
PARENTS={name:ROOT/'benchmarks'/('tum-fr1-'+short+'-gray-gaussian3-dense-plan.json')
    for name,short in [('rgbd_dataset_freiburg1_xyz','xyz'),('rgbd_dataset_freiburg1_rpy','rpy')]}


def parent_plan(path):
    raw=Path(path).read_bytes();lf=raw.replace(b'\r\n',b'\n')
    if b'\r' in lf or raw not in (lf,lf.replace(b'\n',b'\r\n')):
        raise ValueError('parent requires exact content with uniform LF or CRLF')
    return validate_gray_gaussian_dense_plan(json.loads(raw)),hashlib.sha256(lf).hexdigest()


def pair_mean_dense_plan(parent,digest):
    plan=copy.deepcopy(parent)
    plan['estimator']=ESTIMATOR
    plan['prior_gray_gaussian_lf_sha256']=digest
    plan['controls']['dense_fitting_preprocessing']=copy.deepcopy(POLICY)
    plan['generation']=('All original adjacent RGB-D frames; Gaussian3 gray and original fitting '
        'depth, normalized once from the initial native full-resolution correspondence means '
        'to 0.5 independently in each gray image, without clipping. Identity then the preceding '
        'returned relative increment. No reference, rescale, retry, selection or loss reset.')
    plan['native_refinement_policy']=('Only explicit initial-correspondence intensity scaling '
        'changes from the gray-only control. Require the existing minimum count and finite '
        'positive means/gains before mutation. Ordered streaming mean accumulation owns no '
        'pixel-sized buffer; independent scaling shares the bounded CPU pool. Preserve original '
        'raw final statistics, depth validity, camera, pyramid, derivatives, solver and all gates.')
    plan['scope']=('Prospective component comparison on two observed development scenes after '
        'preserving complete gray-only xyz success and rpy ATE failure. Preserve all controls '
        'and failures; freeze both full timelines before either new score. No post-score tuning, '
        'held-out, speed, external-estimator equivalence or physical-calibration claim.')
    return plan


def validate_pair_mean_dense_plan(plan):
    validate_plan(plan);path=PARENTS.get(plan.get('sequence'))
    if path is None:raise ValueError('require one of the two fixed pair-mean scenes')
    parent,digest=parent_plan(path)
    if exact_json(plan)!=exact_json(pair_mean_dense_plan(parent,digest)):
        raise ValueError('pair-mean policy, source lineage and all gates are fixed')
    return plan
