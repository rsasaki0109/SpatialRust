"""One hash-bound complete room reference run; no estimator import/execution."""
import copy
from pathlib import Path

from open3d_rgbd_protocol import CONTROLS, CPU_MODULE_SHA256
from reference_precision_dense_rgbd_protocol import fixed_json, validate_reference_precision_plan
from tum_sequence_reference import validate_plan
from wide_rgbd_protocol import exact_json

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'benchmarks/room-reference-audit-plan.json'
PLAN_LF_SHA256='524b3150dfd04efca96ca72272d94ff4a35d5cfff7ffba8d82ef6bdd999817a0'
SELECTION,_=fixed_json(PLAN,PLAN_LF_SHA256)
ESTIMATOR='open3d_room_reference_rgbd_cpu'
PARENT=ROOT/SELECTION['parent_plan']


def room_reference_plan():
    parent,_=fixed_json(PARENT,SELECTION['parent_plan_lf_sha256'])
    validate_reference_precision_plan(parent)
    if exact_json(CONTROLS)!=exact_json(SELECTION['external']['controls']) or CPU_MODULE_SHA256!=SELECTION['external']['cpu_module_sha256']:
        raise ValueError('frozen reference package and complete model changed')
    plan=copy.deepcopy(parent)
    for key in ('reference_precision_protocol_lf_sha256','prior_fitting_plan_lf_sha256'):
        plan.pop(key,None)
    plan['controls'].pop('dense_fitting_model')
    plan['controls']['open3d_rgbd_odometry']=copy.deepcopy(CONTROLS)
    plan.update(estimator=ESTIMATOR,room_reference_protocol_lf_sha256=PLAN_LF_SHA256,
        prior_native_plan_lf_sha256=SELECTION['parent_plan_lf_sha256'],
        open3d_cpu_module_sha256=CPU_MODULE_SHA256,observed_development=True,
        generation=SELECTION['real_protocol'],scope=SELECTION['selection'],
        native_refinement_policy='External frozen CPU reference comparator; original dense controls define independent raw final admission. No native estimate or acceptance is replaced.')
    return plan


def validate_room_reference_plan(plan,*,check_base=True):
    if check_base:validate_plan(plan)
    if exact_json(plan)!=exact_json(room_reference_plan()):
        raise ValueError('complete room reference lineage, calibration, controls and original gates are fixed')
    return plan


def validate_transition(parent,plan):
    validate_room_reference_plan(plan)
    expected,_=fixed_json(PARENT,SELECTION['parent_plan_lf_sha256'])
    if exact_json(parent)!=exact_json(expected):
        raise ValueError('require exact prepared native reference-precision room parent')


def source_identity(sequence,digest):
    if sequence!=SELECTION['sequence'] or digest!=SELECTION['source_sha256']:
        raise ValueError('require exact original room recording source identity')
