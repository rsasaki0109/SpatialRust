"""Bind original fragment files to an explicitly selected Redwood/3DMatch pose."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'crates/spatialrust-py/examples'))
from align_point_clouds import read_bound_cloud
from redwood_reference import load_records, select_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fragments-dir', type=Path, required=True)
    parser.add_argument('--gt-log', type=Path, required=True)
    parser.add_argument('--source-id', type=int, required=True)
    parser.add_argument('--target-id', type=int, required=True)
    parser.add_argument('--extension', choices=['ply', 'pcd'], default='ply')
    parser.add_argument('--provenance-url', required=True)
    parser.add_argument('--description', required=True)
    parser.add_argument('--project-reference-rotation', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists')
    if not args.provenance_url.startswith('https://') or not args.description.strip():
        parser.error('declare an HTTPS provenance URL and nonempty description')
    records, log_hash = load_records(args.gt_log, project_rotation=args.project_reference_rotation)
    pose, pair, inverted = select_pose(records, args.source_id, args.target_id)
    paths, fingerprints, counts = {}, {}, {}
    for name, index in [('source', args.source_id), ('target', args.target_id)]:
        path = args.fragments_dir / f'cloud_bin_{index}.{args.extension}'
        cloud, digest = read_bound_cloud(path)
        if len(cloud) < 3 or not np.isfinite(cloud.xyz()).all():
            raise ValueError('fragment requires at least three finite XYZ points')
        paths[name], fingerprints[name], counts[name] = str(path.resolve()), digest, len(cloud)
    reference = dict(schema='spatialrust.pose-reference.v1', length_unit='m',
                     input_file_sha256=fingerprints, transform_source_to_target=pose.tolist(),
                     provenance=dict(kind='dataset_reference', url=args.provenance_url, description=args.description,
                         source_fragment_id=args.source_id, target_fragment_id=args.target_id,
                         gt_log_sha256=log_hash, stored_pair=list(pair), stored_direction='second_header_id_to_first',
                         stored_pose_inverted=inverted, fragment_count=records[pair]['fragments'],
                         reference_rotation_correction=records[pair]['rotation_correction'],
                         original_fragment_files=paths, original_fragment_points=counts))
    serialized = json.dumps(reference, indent=2, allow_nan=False) + '\n'
    args.output_dir.mkdir()
    try:
        (args.output_dir / 'reference.json').write_text(serialized)
    except Exception:
        shutil.rmtree(args.output_dir)
        raise
    print(f'Bound source {args.source_id} to target {args.target_id}; stored pose inverted={inverted}')


if __name__ == '__main__':
    main()
