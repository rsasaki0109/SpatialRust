"""Read TUM reference contents only after checking hash-frozen estimates."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from timestamped_trajectory_reference import evaluate
from tum_sequence_reference import file_sha256, json_bytes, require_sha256, validate_plan


def score(archive, estimates, estimates_sha256, plan_path, output):
    frozen_raw = Path(estimates).read_bytes()
    if hashlib.sha256(frozen_raw).hexdigest() != require_sha256(estimates_sha256):
        raise ValueError('estimates differ from supplied frozen SHA-256')
    frozen = json.loads(frozen_raw)
    plan_raw = Path(plan_path).read_bytes()
    plan = validate_plan(json.loads(plan_raw))
    if hashlib.sha256(plan_raw).hexdigest() != frozen['plan_sha256']:
        raise ValueError('evaluation plan differs from generation plan')
    source_sha256 = require_sha256(frozen['source_sha256'])
    if file_sha256(archive) != source_sha256:
        raise ValueError('reference archive differs from prepared source')
    if frozen.get('reference_used_for_generation') is not False:
        raise ValueError('generation must declare reference-free estimates')
    if Path(output).exists():
        raise FileExistsError('preserve previous evaluation')
    reference = None
    expected = plan['sequence']+'/groundtruth.txt'
    with tarfile.open(archive, mode='r|gz') as stream:
        for i, member in enumerate(stream):
            if i >= plan['limits']['max_members']:
                raise ValueError('reference archive exceeds member bound')
            if member.name == expected:
                if reference is not None or not member.isfile() or not 0 < member.size <= plan['limits']['max_member_bytes']:
                    raise ValueError('duplicate, linked or over-budget reference')
                with stream.extractfile(member) as incoming:
                    reference = incoming.read(member.size+1)
                if len(reference) != member.size:
                    raise ValueError('truncated reference')
    if reference is None:
        raise ValueError('archive lacks independently distributed groundtruth.txt')
    c = plan['controls']
    result = evaluate(frozen, reference, plan['calibration']['sensor_frame'],
                      c['evaluation_max_gap_ns'], c['rpe_delta_ns'])
    result.update(estimates_sha256=estimates_sha256, reference_sha256=hashlib.sha256(reference).hexdigest(),
                  source_sha256=source_sha256, plan_sha256=frozen['plan_sha256'],
                  reference_member=expected,
                  evaluator_bindings={str(p.resolve()):file_sha256(p) for p in
                    (Path(__file__), Path(__file__).with_name('timestamped_trajectory_reference.py'),
                     Path(__file__).with_name('tum_sequence_reference.py'),
                     Path(__file__).with_name('evaluate_pose_reference.py'))})
    if file_sha256(archive) != source_sha256 or Path(estimates).read_bytes() != frozen_raw or Path(plan_path).read_bytes() != plan_raw:
        raise ValueError('input changed during post-fit evaluation')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (('groundtruth.txt', reference), ('accuracy.json', json_bytes(result))):
        with (output/name).open('xb') as stream:
            stream.write(data)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--estimates', type=Path, required=True)
    parser.add_argument('--estimates-sha256', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    r = score(args.archive, args.estimates, args.estimates_sha256, args.plan, args.output_dir)
    print('Evaluated {}/{} poses; first-pose aligned ATE RMSE {}'.format(
        r['evaluated_poses'], r['planned_poses'], r['ate_first_pose_aligned_rmse_m']))


if __name__ == '__main__':
    main()
