"""Post-fit TUM pose association with integer times and fixed SE(3) gauge.

Reference interpolation and first-pose alignment are evaluation operations only.
This module does not estimate motion, select registrations or infer calibration.
"""
from bisect import bisect_left
from decimal import Decimal, InvalidOperation, localcontext
import math
import re

import numpy as np

from evaluate_pose_reference import rigid


def decimal_timestamp_ns(token):
    if not isinstance(token,str) or not 0 < len(token) <= 64:
        raise ValueError('timestamp requires a bounded decimal string')
    try:
        with localcontext() as context:
            context.prec = 90
            seconds = Decimal(token)
            value = seconds * 10**9
            if not seconds.is_finite() or value < 0 or value > 2**63-1 or value != value.to_integral_value():
                raise ValueError('timestamp is not an exact nonnegative int64 nanosecond value')
            return int(value)
    except (InvalidOperation, TypeError) as error:
        raise ValueError('invalid decimal timestamp') from error


def quaternion_matrix(values):
    q = np.asarray(values, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-3:
        raise ValueError('reference quaternion must be finite and approximately unit length')
    q = q / np.linalg.norm(q)
    x,y,z,w = q
    matrix = np.eye(4)
    matrix[:3,:3] = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    return matrix, q


def read_tum(data):
    records = []
    for line in data.decode('ascii').splitlines():
        line = line.split('#',1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) != 8:
            raise ValueError('TUM pose rows require time, xyz and xyzw quaternion')
        ns = decimal_timestamp_ns(tokens[0])
        xyz = np.array([float(v) for v in tokens[1:4]])
        matrix,q = quaternion_matrix([float(v) for v in tokens[4:]])
        if not np.isfinite(xyz).all() or (records and ns <= records[-1]['timestamp_ns']):
            raise ValueError('reference poses must be finite and strictly time ordered')
        matrix[:3,3] = xyz
        records.append(dict(timestamp_ns=ns, world_from_sensor=rigid(matrix), quaternion=q))
    if not records:
        raise ValueError('empty reference trajectory')
    return records


def associate(records, timestamps, ns, max_gap_ns):
    """Interpolate only inside a bounded bracket, without extrapolating."""
    index = bisect_left(timestamps,ns)
    if index < len(records) and timestamps[index] == ns:
        return records[index]['world_from_sensor']
    if index == 0 or index == len(records):
        return None
    before,after = records[index-1],records[index]
    lo,hi = before['timestamp_ns'],after['timestamp_ns']
    if max(ns-lo,hi-ns) > max_gap_ns:
        return None
    alpha = (ns-lo)/(hi-lo)  # Subtract integer epochs before conversion to f64.
    q0,q1 = before['quaternion'],after['quaternion']
    dot = float(q0 @ q1)
    if dot < 0:
        q1,dot = -q1,-dot
    dot = min(1.,dot)
    if dot > .9995:
        q = (1-alpha)*q0+alpha*q1
    else:
        angle = math.acos(dot)
        q = (math.sin((1-alpha)*angle)*q0+math.sin(alpha*angle)*q1)/math.sin(angle)
    pose,_ = quaternion_matrix(q)
    pose[:3,3] = (1-alpha)*before['world_from_sensor'][:3,3]+alpha*after['world_from_sensor'][:3,3]
    return rigid(pose)


def errors(estimate, truth):
    rotation = truth[:3,:3].T @ estimate[:3,:3]
    skew = np.array([rotation[2,1]-rotation[1,2],rotation[0,2]-rotation[2,0],rotation[1,0]-rotation[0,1]])
    distance=math.hypot(*(float(v) for v in estimate[:3,3]-truth[:3,3]))
    if not math.isfinite(distance):
        raise ValueError('translation error overflows')
    return dict(translation_m=distance,
                rotation_degrees=math.degrees(math.atan2(float(np.linalg.norm(skew))/2,
                                                       float(np.clip((np.trace(rotation)-1)/2,-1,1)))))


def evaluate(frozen, reference, reference_frame, max_gap_ns=20000000, rpe_delta_ns=1000000000):
    if frozen.get('schema') != 'spatialrust.timestamped-trajectory.v1' or frozen.get('length_unit') != 'm':
        raise ValueError('unsupported trajectory schema or units')
    if not reference_frame or frozen.get('sensor_frame') != reference_frame:
        raise ValueError('reference and estimates require the same explicitly declared sensor frame')
    if frozen.get('reference_used_for_generation') is not False:
        raise ValueError('estimates must declare reference-free generation')
    for name in ('source_sha256','calibration_sha256','plan_sha256'):
        if not isinstance(frozen.get(name),str) or not re.fullmatch('[0-9a-f]{64}',frozen[name]):
            raise ValueError('trajectory requires source, calibration and plan SHA-256 bindings')
    if any(type(v) is not int or v <= 0 for v in (max_gap_ns,rpe_delta_ns)):
        raise ValueError('association and RPE controls must be positive integer nanoseconds')
    poses = frozen['poses']
    if not poses:
        raise ValueError('empty planned trajectory')
    times, estimates = [],[]
    for item in poses:
        ns = item['timestamp_ns']
        if type(ns) is not int or not 0 <= ns <= 2**63-1 or (times and ns <= times[-1]):
            raise ValueError('estimated timestamps must be strictly ordered int64 nanoseconds')
        times.append(ns)
        if item['status']=='success':
            estimates.append(rigid(item['world_from_sensor']))
        elif item['status']=='error' and isinstance(item.get('error'),str) and item['error']:
            estimates.append(None)
        else:
            raise ValueError('every planned pose requires success or a retained error')
    references = read_tum(reference)
    reference_times = [r['timestamp_ns'] for r in references]
    truth = [associate(references,reference_times,t,max_gap_ns) for t in times]
    matched = [i for i in range(len(times)) if truth[i] is not None and estimates[i] is not None]
    matched_set = set(matched)
    anchor = matched[0] if matched else None
    alignment = truth[anchor] @ np.linalg.inv(estimates[anchor]) if anchor is not None else None
    rows = []
    for i,item in enumerate(poses):
        row = dict(timestamp_ns=times[i], generation_status=item['status'], reference_matched=truth[i] is not None)
        if i in matched_set:
            row['raw_error'] = errors(estimates[i],truth[i])
            row['first_pose_aligned_error'] = errors(rigid(alignment @ estimates[i]),truth[i])
        elif item['status']=='error':
            row['error'] = item['error']
        rows.append(row)
    pairs = []
    for i,t in enumerate(times):
        desired = t+rpe_delta_ns
        index = bisect_left(times,desired)
        candidates = [j for j in (index-1,index) if i < j < len(times)]
        if not candidates:
            continue
        j = min(candidates,key=lambda k:(abs(times[k]-desired),k))
        if abs(times[j]-desired) > max_gap_ns:
            continue
        pair = dict(source_index=i,target_index=j,delta_ns=times[j]-t,evaluated=False)
        if all(values[k] is not None for values in (truth,estimates) for k in (i,j)):
            expected = rigid(np.linalg.inv(truth[i]) @ truth[j])
            actual = rigid(np.linalg.inv(estimates[i]) @ estimates[j])
            pair.update(evaluated=True,error=errors(actual,expected))
        pairs.append(pair)
    def rmse(values):
        if not values:
            return None
        scale=max(values)
        return scale * (math.hypot(*(v/scale for v in values))/math.sqrt(len(values))) if scale else 0.
    return dict(schema='spatialrust.timestamped-accuracy.v1', planned_poses=len(poses),
                generated_poses=sum(e is not None for e in estimates), evaluated_poses=len(matched),
                unmatched_reference_poses=sum(t is None for t in truth),
                first_pose_alignment_anchor_index=anchor, first_pose_alignment=alignment.tolist() if anchor is not None else None,
                alignment='first jointly valid pose; SE3 gauge only; no scale fitting',
                max_reference_gap_ns=max_gap_ns,rpe_requested_delta_ns=rpe_delta_ns,
                ate_raw_rmse_m=rmse([rows[i]['raw_error']['translation_m'] for i in matched]),
                ate_first_pose_aligned_rmse_m=rmse([rows[i]['first_pose_aligned_error']['translation_m'] for i in matched]),
                rpe_planned_pairs=len(pairs),rpe_evaluated_pairs=sum(p['evaluated'] for p in pairs),
                rpe_translation_rmse_m=rmse([p['error']['translation_m'] for p in pairs if p['evaluated']]),
                rows=rows,rpe_pairs=pairs)
