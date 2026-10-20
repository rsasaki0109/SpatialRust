"""Run the preselected frozen native candidate; never select using held-out scores."""
import argparse
import importlib
import json
from pathlib import Path

from heldout_dense_rgbd_protocol import ESTIMATOR, selection, source_identity, validate_heldout_plan
from run_tum_rounded_sobel_dense_rgbd import PreviousMotionRoundedSobelRgbd
from run_tum_sequence import generate
from tum_sequence_reference import file_sha256, require_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--native-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--source-identities',type=Path,required=True)
    parser.add_argument('--source-identities-sha256',required=True)
    args=parser.parse_args()
    validate_heldout_plan(json.loads((args.prepared_dir/'plan.json').read_text(encoding='utf-8')))
    manifest=json.loads((args.prepared_dir/'manifest.json').read_text(encoding='utf-8'))
    source_identity(args.source_identities,args.source_identities_sha256,manifest['sequence'],manifest['source_sha256'])
    expected=selection()['candidate']['native_sha256']
    native=Path(importlib.import_module('spatialrust.spatialrust').__file__)
    if require_sha256(args.native_sha256)!=expected or file_sha256(native)!=expected:
        raise ValueError('require the native binary selected before held-out acquisition')
    result=generate(args.prepared_dir,args.manifest_sha256,args.output_dir,
        dense_rgbd_aligner=PreviousMotionRoundedSobelRgbd().align,method=ESTIMATOR,
        additional_bindings=[__file__,args.source_identities]+[Path(__file__).with_name(name) for name in (
            'heldout_dense_rgbd_protocol.py','rounded_sobel_dense_rgbd_protocol.py',
            'run_tum_rounded_sobel_dense_rgbd.py','run_tum_dense_rgbd.py','audit_tum_dense_rgbd.py')])
    print('Frozen held-out {}/{} poses; SHA-256 {}'.format(result['generated_poses'],result['planned_poses'],result['estimates_sha256']))


if __name__=='__main__':main()
