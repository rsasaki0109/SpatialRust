"""Evaluate hash-frozen estimates against a separately supplied TUM trajectory."""
import argparse
import hashlib
import json
from pathlib import Path

from timestamped_trajectory_reference import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--estimates',type=Path,required=True)
    parser.add_argument('--estimates-sha256',required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--reference-sha256',required=True)
    parser.add_argument('--reference-frame',required=True)
    parser.add_argument('--max-gap-ns',type=int,default=20000000)
    parser.add_argument('--rpe-delta-ns',type=int,default=1000000000)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; preserve previous evaluation')
    raw=args.estimates.read_bytes()
    reference=args.reference.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.estimates_sha256 or hashlib.sha256(reference).hexdigest()!=args.reference_sha256:
        parser.error('estimates or reference differ from the supplied frozen hashes')
    result=evaluate(json.loads(raw),reference,args.reference_frame,args.max_gap_ns,args.rpe_delta_ns)
    result.update(estimates_sha256=args.estimates_sha256,reference_sha256=args.reference_sha256,
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  helper_sha256=hashlib.sha256(Path(__file__).with_name('timestamped_trajectory_reference.py').read_bytes()).hexdigest())
    if args.estimates.read_bytes()!=raw or args.reference.read_bytes()!=reference:
        raise ValueError('input files changed during evaluation')
    with args.output.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'Evaluated {result["evaluated_poses"]}/{result["planned_poses"]} poses; '
          f'RPE {result["rpe_evaluated_pairs"]}/{result["rpe_planned_pairs"]} pairs')


if __name__=='__main__':
    main()
