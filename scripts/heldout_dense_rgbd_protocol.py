"""Exact candidate/source selection before acquiring previously unscored recordings."""
import copy
import hashlib
import json
import re
from pathlib import Path
from datetime import datetime

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


def source_identity(path, expected_lf_sha256, sequence, archive_sha256):
    """Require the complete post-acquisition hash pin before generation/auditing."""
    path=Path(path)
    if not path.is_absolute() or type(expected_lf_sha256) is not str or not re.fullmatch('[0-9a-f]{64}',expected_lf_sha256):
        raise ValueError('absolute source identity file and frozen SHA-256 required')
    raw=path.read_bytes();lf=raw.replace(b'\r\n',b'\n')
    if len(raw)>65536 or b'\r' in lf or raw not in (lf,lf.replace(b'\n',b'\r\n')) or hashlib.sha256(lf).hexdigest()!=expected_lf_sha256:
        raise ValueError('source identity file changed')
    pin=json.loads(lf);expected={r['sequence']:r for r in selection()['sources']}
    if type(pin) is not dict or set(pin)!={'schema','selection_lf_sha256','frozen_at_utc','sources'} or pin['schema']!='spatialrust.heldout-source-identities.v1' or pin['selection_lf_sha256']!=SELECTION_LF_SHA256 or type(pin['sources']) is not list or len(pin['sources'])!=len(expected) or any(type(r) is not dict for r in pin['sources']) or {r.get('sequence') for r in pin['sources']}!=set(expected):
        raise ValueError('exact complete source identity set required')
    try:
        frozen=datetime.fromisoformat(pin['frozen_at_utc'].replace('Z','+00:00'))
        if frozen.utcoffset() is None:raise ValueError('timestamp must carry an offset')
    except (TypeError,AttributeError,ValueError):
        raise ValueError('source pin requires an explicit timezone timestamp') from None
    for row in pin['sources']:
        wanted=expected[row['sequence']]
        if set(row)!={'sequence','url','etag','bytes','sha256'} or row['url']!=wanted['url'] or row['etag']!=wanted['etag'] or type(row['bytes']) is not int or row['bytes']!=wanted['content_length'] or type(row.get('sha256')) is not str or not re.fullmatch('[0-9a-f]{64}',row['sha256']):
            raise ValueError('acquired source identity differs from fixed official metadata')
    row=next((r for r in pin['sources'] if r['sequence']==sequence),None)
    if row is None or row['sha256']!=archive_sha256:
        raise ValueError('prepared source does not match the pre-generation archive pin')
    return row
