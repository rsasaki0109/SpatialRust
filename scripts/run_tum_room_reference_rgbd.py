"""Freeze one complete room trajectory using the separately frozen public CPU API."""
import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path

from open3d_rgbd_protocol import CONTROLS,CPU_MODULE_SHA256
from room_reference_rgbd_protocol import ESTIMATOR,SELECTION,source_identity,validate_room_reference_plan
from run_tum_open3d_rgbd import PreviousMotionOpen3dRgbd
from run_tum_sequence import generate
from tum_sequence_reference import file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    validate_room_reference_plan(json.loads((args.prepared_dir/'plan.json').read_text(encoding='utf-8')))
    manifest=json.loads((args.prepared_dir/'manifest.json').read_text(encoding='utf-8'))
    source_identity(manifest['sequence'],manifest['source_sha256'])
    if metadata.version('open3d')!=CONTROLS['package_version']:
        raise ValueError('frozen public CPU package version required')
    cpu=Path(importlib.import_module('open3d.cpu.pybind').__file__)
    native=Path(importlib.import_module('spatialrust.spatialrust').__file__)
    if file_sha256(cpu)!=CPU_MODULE_SHA256 or file_sha256(native)!=SELECTION['native_io']['sha256']:
        raise ValueError('separately frozen CPU reference and native IO modules required')
    for row in SELECTION['external']['sources']['files']:
        p=Path(row['path'])
        if p.stat().st_size!=row['bytes'] or file_sha256(p)!=row['sha256']:
            raise ValueError('pinned upstream source changed')
    adapter=PreviousMotionOpen3dRgbd()
    result=generate(args.prepared_dir,args.manifest_sha256,args.output_dir,
        dense_rgbd_aligner=adapter.align,method=ESTIMATOR,additional_bindings=[__file__,cpu,native,
            Path(__file__).with_name('room_reference_rgbd_protocol.py'),
            Path(__file__).with_name('run_tum_open3d_rgbd.py'),
            Path(__file__).with_name('open3d_rgbd_protocol.py'),
            Path(__file__).with_name('open3d_rgbd_validation.py'),
            Path(__file__).with_name('audit_tum_dense_rgbd.py'),
            *[Path(row['path']) for row in SELECTION['external']['sources']['files']]])
    print('Frozen room CPU reference {}/{} poses; SHA-256 {}'.format(
        result['generated_poses'],result['planned_poses'],result['estimates_sha256']))


if __name__=='__main__':main()
