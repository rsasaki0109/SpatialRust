"""Prepare fixed, evenly spaced eligible pairs from official fragment archives."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'crates/spatialrust-py/examples'))
from align_point_clouds import file_sha256, read_bound_cloud
from redwood_reference import load_records, parse_records


def selected_pairs(records, count):
    pairs = sorted(pair for pair in records if pair[1] - pair[0] > 1)
    if type(count) is not int or not 1 <= count <= 64 or count > len(pairs):
        raise ValueError('require 1..64 cases and enough eligible pairs')
    # Exact integer quantiles of header order; no pose error or solver outcome.
    indices = [0] if count == 1 else [i * (len(pairs) - 1) // (count - 1) for i in range(count)]
    return [pairs[index] for index in indices]


def member_bytes(archive, name, limit=64 * 1024 * 1024):
    matches = [info for info in archive.infolist() if info.filename == name]
    if len(matches) != 1 or matches[0].is_dir() or matches[0].file_size > limit:
        raise ValueError('archive member absent, duplicated or too large')
    # No paths supplied by the archive are extracted; exact expected names only.
    return archive.read(matches[0])  # zipfile verifies its CRC.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+', required=True)
    parser.add_argument('--cases-per-scene', type=int, default=4)
    parser.add_argument('--cross-check-metadata-root', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists')
    if not 1 <= len(args.scenes) <= 16 or len(set(args.scenes)) != len(args.scenes) or any(
        not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*', name) for name in args.scenes
    ):
        parser.error('require 1..16 distinct safe scene names')
    if not 1 <= args.cases_per_scene <= 64:
        parser.error('cases-per-scene must be 1..64')
    scenes, cases = [], []
    args.output_dir.mkdir()
    try:
        for scene in args.scenes:
            archive_path = args.archive_dir / (scene + '.zip')
            evaluation_path = args.archive_dir / (scene + '-evaluation.zip')
            hashes = dict(fragments=file_sha256(archive_path), evaluation=file_sha256(evaluation_path))
            with zipfile.ZipFile(evaluation_path) as evaluation:
                pose_bytes = member_bytes(evaluation, scene + '-evaluation/gt.log')
                info_bytes = member_bytes(evaluation, scene + '-evaluation/gt.info')
            poses = parse_records(pose_bytes.decode(), project_rotation=True)
            information = parse_records(info_bytes.decode(), 6)
            if set(poses) != set(information) or any(poses[p]['fragments'] != information[p]['fragments'] for p in poses):
                raise ValueError('GT pose/information pairs or fragment counts differ')
            checked = False
            if args.cross_check_metadata_root is not None:
                pinned = args.cross_check_metadata_root / (scene + '-evaluation')
                old_poses, _ = load_records(pinned / 'gt.log', project_rotation=True)
                old_info, _ = load_records(pinned / 'gt.info', 6)
                if set(poses) != set(old_poses) or set(information) != set(old_info):
                    raise ValueError('archive/reference repository pair sets differ')
                for pair in poses:
                    if poses[pair]['fragments'] != old_poses[pair]['fragments']:
                        raise ValueError('archive/reference repository fragment count differs')
                    np.testing.assert_array_equal(poses[pair]['original_matrix'], old_poses[pair]['original_matrix'])
                    np.testing.assert_array_equal(information[pair]['original_matrix'], old_info[pair]['original_matrix'])
                checked = True
            pairs = selected_pairs(poses, args.cases_per_scene)
            directory = args.output_dir / scene
            directory.mkdir()
            (directory / 'gt.log').write_bytes(pose_bytes)
            (directory / 'gt.info').write_bytes(info_bytes)
            inputs = {}
            with zipfile.ZipFile(archive_path) as fragments:
                for index in sorted({i for pair in pairs for i in pair}):
                    name = f'cloud_bin_{index}.ply'
                    path = directory / name
                    path.write_bytes(member_bytes(fragments, scene + '/' + name))
                    cloud, digest = read_bound_cloud(path)
                    if len(cloud) < 3 or not np.isfinite(cloud.xyz()).all():
                        raise ValueError('fragment requires at least three finite XYZ points')
                    inputs[index] = dict(path=str(path.resolve()), sha256=digest, points=len(cloud))
            for target_id, source_id in pairs:
                source, target = inputs[source_id], inputs[target_id]
                case_id = f'{scene}-{source_id}-to-{target_id}'
                reference = dict(schema='spatialrust.pose-reference.v1', length_unit='m',
                    input_file_sha256=dict(source=source['sha256'], target=target['sha256']),
                    transform_source_to_target=poses[(target_id, source_id)]['matrix'].tolist(),
                    provenance=dict(kind='dataset_reference', url='https://3dmatch.cs.princeton.edu/',
                        description='Official fragment/evaluation archives; GT second ID maps to first',
                        scene=scene, source_fragment_id=source_id, target_fragment_id=target_id,
                        archive_sha256=hashes, gt_log_sha256=hashlib.sha256(pose_bytes).hexdigest(),
                        gt_info_sha256=hashlib.sha256(info_bytes).hexdigest(),
                        metadata_repository_matches_exact_values=checked,
                        reference_rotation_correction=poses[(target_id, source_id)]['rotation_correction']))
                path = directory / (case_id + '-reference.json')
                path.write_text(json.dumps(reference, indent=2, allow_nan=False), encoding='utf-8')
                cases.append(dict(case_id=case_id, scene=scene, source=source['path'], target=target['path'],
                                  source_points=source['points'], target_points=target['points'],
                                  reference_file=str(path.resolve()), pair=[target_id, source_id]))
            if hashes != dict(fragments=file_sha256(archive_path), evaluation=file_sha256(evaluation_path)):
                raise ValueError('archives changed during preparation')
            scenes.append(dict(scene=scene, archive_sha256=hashes, selected_pairs=[list(p) for p in pairs],
                               metadata_repository_matches_exact_values=checked))
        manifest = dict(schema='spatialrust.3dmatch-cases.v1', cases=cases, scenes=scenes,
                        selection_rule='integer_quantiles_of_sorted_nonconsecutive_gt_headers_before_fitting',
                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (args.output_dir / 'cases.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    except Exception:
        shutil.rmtree(args.output_dir)
        raise
    print(f'Prepared {len(cases)} fixed cases from {len(scenes)} scenes')


if __name__ == '__main__':
    main()
