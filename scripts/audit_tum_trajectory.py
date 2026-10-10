"""Independently recompute frozen TUM scores using SciPy SLERP/rotations.

Optional post-fit audit: imports neither the generator nor its trajectory scorer.
SciPy is used only here; it is not added to the native SDK's dependencies.
"""
import argparse
from bisect import bisect_left
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(estimates_path, estimates_sha, evaluation_path, evaluation_sha, reference_path, reference_sha, output):
    import scipy
    from scipy.spatial.transform import Rotation, Slerp
    raw = Path(estimates_path).read_bytes()
    if sha(raw) != estimates_sha:
        raise ValueError('estimates differ from frozen hash')
    estimates = json.loads(raw)
    scored_raw = Path(evaluation_path).read_bytes()
    if sha(scored_raw) != evaluation_sha:
        raise ValueError('evaluation differs from frozen hash')
    scored = json.loads(scored_raw)
    if scored['estimates_sha256'] != estimates_sha or scored['reference_sha256'] != reference_sha:
        raise ValueError('evaluation does not bind these inputs')
    if estimates['reference_used_for_generation'] is not False or estimates['length_unit'] != 'm' or estimates['sensor_frame'] != 'tum_rgb_optical':
        raise ValueError('require declared reference-free metre optical trajectory')
    reference_raw = Path(reference_path).read_bytes()
    if sha(reference_raw) != reference_sha:
        raise ValueError('reference differs from frozen hash')
    if Path(output).exists():
        raise FileExistsError('preserve previous independent audit')
    ref_times, ref_xyz, ref_q = [], [], []
    for line in reference_raw.decode('ascii').splitlines():
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        if len(fields) != 8:
            raise ValueError('invalid reference row')
        with localcontext() as context:
            context.prec = 90
            ns = Decimal(fields[0])*Decimal(1000000000)
            if ns != ns.to_integral_value():
                raise ValueError('reference time loses integer nanoseconds')
            ref_times.append(int(ns))
        ref_xyz.append([float(v) for v in fields[1:4]])
        ref_q.append([float(v) for v in fields[4:]])
    if not ref_times or any(a >= b for a, b in zip(ref_times, ref_times[1:])):
        raise ValueError('reference must be strictly ordered')
    ref_xyz, ref_q = np.array(ref_xyz), np.array(ref_q)
    max_gap = scored['max_reference_gap_ns']
    times, actual, truth = [], [], []
    for pose in estimates['poses']:
        ns = pose['timestamp_ns']
        if type(ns) is not int or (times and ns <= times[-1]):
            raise ValueError('estimate timestamps are not ordered integers')
        times.append(ns)
        actual.append(np.array(pose['world_from_sensor'], float) if pose['status'] == 'success' else None)
        index = bisect_left(ref_times, ns)
        expected = None
        if index < len(ref_times) and ref_times[index] == ns:
            expected = np.eye(4)
            expected[:3, :3] = Rotation.from_quat(ref_q[index]).as_matrix()
            expected[:3, 3] = ref_xyz[index]
        elif 0 < index < len(ref_times) and max(ns-ref_times[index-1], ref_times[index]-ns) <= max_gap:
            before, after = ref_times[index-1], ref_times[index]
            interval, offset = (after-before)/1e9, (ns-before)/1e9
            expected = np.eye(4)
            expected[:3, :3] = Slerp([0., interval], Rotation.from_quat(ref_q[index-1:index+1]))([offset]).as_matrix()[0]
            alpha = (ns-before)/(after-before)
            expected[:3, 3] = ref_xyz[index-1]+alpha*(ref_xyz[index]-ref_xyz[index-1])
        truth.append(expected)
    matched = [i for i, (a, b) in enumerate(zip(actual, truth)) if a is not None and b is not None]
    anchor = matched[0] if matched else None
    gauge = truth[anchor] @ np.linalg.inv(actual[anchor]) if anchor is not None else None
    if (gauge is None and scored['first_pose_alignment'] is not None) or (gauge is not None and
            not np.allclose(gauge, scored['first_pose_alignment'], rtol=0, atol=1e-9)):
        raise ValueError('exported first-pose gauge differs')
    differences = []
    def compare_errors(a, b, record):
        translation = float(np.linalg.norm(a[:3, 3]-b[:3, 3]))
        degrees = float(np.degrees(Rotation.from_matrix(b[:3, :3].T @ a[:3, :3]).magnitude()))
        dt, da = abs(translation-record['translation_m']), abs(degrees-record['rotation_degrees'])
        if dt > 1e-8 or da > 1e-6:
            raise ValueError('independent translation/rotation score differs')
        differences.append((dt, da))
        return translation
    raw_errors, aligned_errors = [], []
    if len(scored['rows']) != len(times):
        raise ValueError('evaluation silently drops planned poses')
    for i, row in enumerate(scored['rows']):
        if row['timestamp_ns'] != times[i] or row['generation_status'] != estimates['poses'][i]['status'] or row['reference_matched'] != (truth[i] is not None):
            raise ValueError('coverage/association differs')
        if actual[i] is not None and truth[i] is not None:
            raw_errors.append(compare_errors(actual[i], truth[i], row['raw_error']))
            aligned_errors.append(compare_errors(gauge @ actual[i], truth[i], row['first_pose_aligned_error']))
    expected_pairs = []
    delta = scored['rpe_requested_delta_ns']
    for i, ns in enumerate(times):
        index = bisect_left(times, ns+delta)
        candidates = [j for j in (index-1, index) if i < j < len(times)]
        if candidates:
            j = min(candidates, key=lambda j: (abs(times[j]-ns-delta), j))
            if abs(times[j]-ns-delta) <= max_gap:
                expected_pairs.append((i, j))
    if [(p['source_index'], p['target_index']) for p in scored['rpe_pairs']] != expected_pairs:
        raise ValueError('RPE planned denominator differs')
    relative_errors = []
    for (i, j), row in zip(expected_pairs, scored['rpe_pairs']):
        evaluated = all(v[k] is not None for v in (actual, truth) for k in (i, j))
        if row['evaluated'] != evaluated or row['delta_ns'] != times[j]-times[i]:
            raise ValueError('RPE coverage/time differs')
        if evaluated:
            relative_errors.append(compare_errors(np.linalg.inv(actual[i]) @ actual[j],
                                                   np.linalg.inv(truth[i]) @ truth[j], row['error']))
    aggregates = {}
    for key, values in (('ate_raw_rmse_m', raw_errors), ('ate_first_pose_aligned_rmse_m', aligned_errors),
                        ('rpe_translation_rmse_m', relative_errors)):
        value = float(np.sqrt(np.mean(np.square(values)))) if values else None
        if (value is None) != (scored[key] is None) or (value is not None and abs(value-scored[key]) > 1e-8):
            raise ValueError('aggregate error differs')
        aggregates[key] = value
    counts = dict(planned_poses=len(times), generated_poses=sum(a is not None for a in actual),
                  evaluated_poses=len(matched), unmatched_reference_poses=sum(t is None for t in truth),
                  rpe_planned_pairs=len(expected_pairs), rpe_evaluated_pairs=len(relative_errors))
    if any(scored[k] != v for k, v in counts.items()) or scored['first_pose_alignment_anchor_index'] != anchor:
        raise ValueError('aggregate coverage differs')
    for path, content in ((estimates_path, raw), (evaluation_path, scored_raw), (reference_path, reference_raw)):
        if Path(path).read_bytes() != content:
            raise ValueError('audit input changed')
    result = dict(schema='spatialrust.tum-independent-audit.v1', estimates_sha256=estimates_sha,
                  evaluation_sha256=evaluation_sha, reference_sha256=reference_sha,
                  audit_sha256=sha(Path(__file__).read_bytes()), scipy=scipy.__version__, numpy=np.__version__,
                  counts=counts, aggregates=aggregates,
                  max_translation_score_difference_m=max((d[0] for d in differences), default=0.),
                  max_rotation_score_difference_degrees=max((d[1] for d in differences), default=0.),
                  translation_tolerance_m=1e-8, rotation_tolerance_degrees=1e-6,
                  independent_method='SciPy quaternion SLERP/rotation magnitude; independently parsed Decimal times and pose coverage',
                  scope='Post-fit score audit; does not independently certify physical calibration or publisher trajectory accuracy')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'audit.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('estimates', 'evaluation', 'reference'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.estimates, args.estimates_sha256, args.evaluation, args.evaluation_sha256,
                 args.reference, args.reference_sha256, args.output_dir)
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
