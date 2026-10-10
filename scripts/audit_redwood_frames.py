"""Independently audit saved Redwood geometry and poses without refitting."""
import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import open3d as o3d
import spatialrust as sr

from redwood_frame_reference import checked_bytes, decode_depth, digest, load_plan, read_log, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--frozen-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('audit output directory already exists')
    plan, plan_bytes = load_plan(args.plan)
    raw = (args.frozen_dir / 'registrations.json').read_bytes()
    accuracy_bytes = (args.frozen_dir / 'accuracy.json').read_bytes()
    frozen, accuracy = json.loads(raw), json.loads(accuracy_bytes)
    if digest(raw) != accuracy['registrations_sha256'] or digest(plan_bytes) != accuracy['plan_sha256'] or frozen['plan_sha256'] != accuracy['plan_sha256']:
        raise ValueError('saved result or plan hash mismatch')
    trajectory = checked_bytes(args.data_dir / plan['trajectory']['name'], plan['trajectory']['md5'])
    archive_bytes = checked_bytes(args.data_dir / plan['depth_archive']['name'], plan['depth_archive']['md5'])
    if digest(trajectory) != accuracy['trajectory_sha256'] or digest(archive_bytes) != frozen['preparation']['archive_sha256']:
        raise ValueError('dataset hash mismatch')
    if score(frozen, trajectory, plan)['rows'] != accuracy['rows']:
        raise ValueError('saved accuracy differs from score recomputation')
    args.output_dir.mkdir()
    # Open3D dispatches by suffix: .txt means TUM, .log means Redwood LOG.
    log_path = args.output_dir / 'trajectory.log'
    log_path.write_bytes(trajectory)
    oracle = o3d.io.read_pinhole_camera_trajectory(str(log_path))
    poses = read_log(trajectory, plan['frame_count'])
    np.testing.assert_allclose(poses, [np.linalg.inv(x.extrinsic) for x in oracle.parameters], rtol=0, atol=2e-14)
    c = plan['camera']
    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    np.testing.assert_array_equal(intrinsic.intrinsic_matrix, [[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1]])
    frames = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for key, frame in frozen['preparation']['frames'].items():
            png = archive.read(frame['member'])
            path = args.frozen_dir / 'prepared' / f'{int(key):05d}.pcd'
            if digest(png) != frame['png_sha256'] or digest(path.read_bytes()) != frame['pcd_sha256']:
                raise ValueError('prepared geometry hash mismatch')
            actual = sr.read(str(path)).xyz()
            expected = np.asarray(o3d.geometry.PointCloud.create_from_depth_image(
                o3d.geometry.Image(decode_depth(png)), intrinsic,
                depth_scale=c['depth_scale'], depth_trunc=c['depth_trunc_m']).points)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-7)
            frames.append(dict(frame=int(key), points=len(actual), max_abs_error_m=float(np.max(np.abs(actual - expected)))))
    angle_deltas, corrections = [], []
    for saved, row in zip(frozen['rows'], accuracy['rows']):
        if row['status'] != 'success':
            continue
        s, t = row['pair']
        expected = oracle.parameters[t].extrinsic @ np.linalg.inv(oracle.parameters[s].extrinsic)
        estimate = np.array(saved['report']['transform_source_to_target'])
        if saved['report']['initial_transform_supplied'] is not False:
            raise ValueError('saved registration used an initial pose')
        translation = float(np.linalg.norm(estimate[:3, 3] - expected[:3, 3]))
        # Project only the audit rotation onto SO(3) for trace/acos. Raw f32
        # matrices have ~1e-7 orthogonality noise; saved poses/scores are untouched.
        relative = expected[:3, :3].T @ estimate[:3, :3]
        u, _, vt = np.linalg.svd(relative)
        projected = u @ vt
        angle = float(np.rad2deg(np.arccos(np.clip((np.trace(projected) - 1) / 2, -1, 1))))
        np.testing.assert_allclose(translation, row['accuracy']['translation_error'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(angle, row['accuracy']['rotation_error_degrees'], rtol=0, atol=1e-7)
        angle_deltas.append(abs(angle - row['accuracy']['rotation_error_degrees']))
        corrections.append(float(np.linalg.norm(projected - relative)))
    receipt = dict(schema='spatialrust.redwood-independent-audit.v1',
                   audit_source_sha256=digest(Path(__file__).read_bytes()),
                   registrations_sha256=digest(raw), accuracy_sha256=digest(accuracy_bytes),
                   plan_sha256=digest(plan_bytes), trajectory_sha256=digest(trajectory),
                   open3d=o3d.__version__, log_frames=len(poses), frames=frames,
                   evaluated_poses=len(angle_deltas), rotation_audit_projection='nearest_SO3_svd_for_trace_acos_only',
                   max_rotation_audit_correction_frobenius=max(corrections, default=0),
                   max_rotation_audit_difference_degrees=max(angle_deltas, default=0),
                   totals=accuracy['totals'])
    (args.output_dir / 'audit.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(f'Audited {len(frames)} depth frames, {len(poses)} LOG frames and {len(angle_deltas)} saved poses')


if __name__ == '__main__':
    main()
