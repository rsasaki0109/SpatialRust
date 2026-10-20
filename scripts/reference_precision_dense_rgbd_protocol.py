"""Hash-bound complete reference-precision development model and unchanged gates."""
import copy
import hashlib
import json
from pathlib import Path

from tum_sequence_reference import validate_plan
from wide_rgbd_protocol import exact_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'benchmarks/reference-precision-dense-plan.json'
PLAN_LF_SHA256 = '0a73e2c475e7f5e732809576ba2764909cf3377717888eee2ba4468aa22a702f'
ESTIMATOR = 'spatialrust_reference_precision_dense_rgbd_cpu'


def fixed_json(path, expected=None):
    raw=Path(path).read_bytes();lf=raw.replace(b'\r\n',b'\n')
    if b'\r' in lf or raw not in (lf,lf.replace(b'\n',b'\r\n')) or len(raw)>65536:
        raise ValueError('fixed model/source plan requires bounded uniform LF or CRLF')
    digest=hashlib.sha256(lf).hexdigest()
    if expected is not None and digest!=expected:
        raise ValueError('fixed reference-precision protocol changed')
    return json.loads(lf),digest


SELECTION,_=fixed_json(PLAN,PLAN_LF_SHA256)
POLICY=SELECTION['fitting_model']
PARENTS={s['sequence']:ROOT/s['parent_plan'] for s in SELECTION['scenes']}


def reference_precision_plan(sequence):
    path=PARENTS.get(sequence)
    if path is None:raise ValueError('only four fixed observed development recordings are admitted')
    parent,digest=fixed_json(path);validate_plan(parent)
    plan=copy.deepcopy(parent)
    plan.pop('heldout_validation',None)
    plan.update(estimator=ESTIMATOR,reference_precision_protocol_lf_sha256=PLAN_LF_SHA256,
        prior_fitting_plan_lf_sha256=digest,observed_development=True,
        generation=SELECTION['real_protocol'],scope=SELECTION['motivation'],
        native_refinement_policy='Fixed complete reference-precision model; original raw final admission and every original accuracy gate. All four full timelines frozen before scoring; observed development data only.')
    plan['controls']['dense_fitting_model']=copy.deepcopy(POLICY)
    return plan


def validate_reference_precision_plan(plan, *, check_base=True):
    if check_base:validate_plan(plan)
    if exact_json(plan)!=exact_json(reference_precision_plan(plan.get('sequence'))):
        raise ValueError('reference-precision model, source lineage, calibration and every acceptance gate are fixed')
    return plan


def validate_transition(parent, plan):
    validate_reference_precision_plan(plan)
    expected,_=fixed_json(PARENTS[plan['sequence']])
    if exact_json(parent)!=exact_json(expected):
        raise ValueError('require the exact prepared parent for this model transition')


def source_identity(sequence, digest):
    row=next((s for s in SELECTION['scenes'] if s['sequence']==sequence),None)
    if row is None or digest!=row['source_sha256']:
        raise ValueError('require the original fixed recording source identity')
    return row
