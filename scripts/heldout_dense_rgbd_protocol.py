"""Exact candidate/source selection before acquiring previously unscored recordings."""
import copy
import hashlib
import json
from pathlib import Path

from rounded_sobel_dense_rgbd_protocol import POLICY, parent_plan, validate_rounded_sobel_dense_plan
from tum_sequence_reference import validate_plan
from wide_rgbd_protocol import exact_json

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT/'benchmarks/frozen-heldout-selection.json'
SELECTION_LF_SHA256 = '3e17aa6668697743e62c75116814466989e528dc58a451567b450bd9d6d1f12b'
PARENT = ROOT/'benchmarks/tum-fr1-xyz-rounded-sobel-dense-plan.json'
ESTIMATOR = 'spatialrust_heldout_rounded_sobel_dense_rgbd_cpu'


def selection(path=SELECTION):
    raw = Path(path).read_bytes(); lf = raw.replace(b'\r\n',b'\n')
    if len(raw)>65536 or b'\r' in lf or raw not in (lf,lf.replace(b'\n',b'\r\n')) or hashlib.sha256(lf).hexdigest()!=SELECTION_LF_SHA256:
        raise ValueError('require the exact fixed candidate/source/resource selection with uniform LF or CRLF')
    return json.loads(lf)


def heldout_plan(sequence):
    selected = selection()
    source = next((row for row in selected['sources'] if row['sequence']==sequence),None)
    if source is None:
        raise ValueError('only the two fixed previously unscored recordings are admitted')
    raw = PARENT.read_bytes(); lf = raw.replace(b'\r\n',b'\n')
    if b'\r' in lf or raw not in (lf,lf.replace(b'\n',b'\r\n')):
        raise ValueError('fixed parent requires uniform LF or CRLF')
    parent = validate_rounded_sobel_dense_plan(json.loads(lf))
    plan = copy.deepcopy(parent)
    plan.update(sequence=sequence,url=source['url'],observed_redirect=source['url'],estimator=ESTIMATOR,
        prior_rounded_sobel_lf_sha256=hashlib.sha256(lf).hexdigest(),
        heldout_validation=dict(selection_lf_sha256=SELECTION_LF_SHA256,candidate=copy.deepcopy(selected['candidate']),source=copy.deepcopy(source)),
        generation='Every original adjacent RGB-D frame with the frozen rounded-sobel native candidate; identity then preceding returned increment; no reference, rescale, retry, reset or selection.',
        scope=selected['scope'],native_refinement_policy='No estimator, preprocessing, sampling, gradients, solver, pose update or production/native dependency change. Freeze both full timelines before either score. Original complete-row, raw-final, world/full-cloud/IO and accuracy requirements remain fixed.')
    plan['limits']['max_total_bytes'] = selected['resources']['prepared_max_bytes']//2
    return plan


def validate_heldout_plan(plan, *, check_base=True):
    if check_base:
        validate_plan(plan)
    if exact_json(plan)!=exact_json(heldout_plan(plan.get('sequence'))):
        raise ValueError('held-out candidate, source identity, calibration, controls and acceptance are fixed')
    return plan
