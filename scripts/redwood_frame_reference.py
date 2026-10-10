"""Hash-bound preparation and post-fit scoring for the frozen Redwood frame plan.

Optional benchmark dependencies: numpy, Pillow, spatialrust, open3d. The data is
noisy synthetic augmented ICL-NUIM, not measured multi-sensor calibration.
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from evaluate_pose_reference import evaluate, rigid


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_bytes(path, expected_md5):
    value = Path(path).read_bytes()
    if hashlib.md5(value).hexdigest() != expected_md5:
        raise ValueError('input does not match the official dataset MD5')
    return value


def load_plan(path):
    data = Path(path).read_bytes()
    plan = json.loads(data)
    if plan.get('schema') != 'spatialrust.redwood-frame-plan.v1':
        raise ValueError('unsupported frame plan')
    expected = dict(coarse_leaf_m=.1, coarse_size_limit=5000, normal_neighbors=20,
                    feature_radius_m=.5, global_gate_m=.2,
                    icp_stages=[dict(leaf_m=.05, gate_m=.2, iterations=50),
                                dict(leaf_m=None, gate_m=.05, iterations=50)],
                    evaluation_gate_m=.05, threads=1)
    if plan['controls'] != expected:
        raise ValueError('plan controls differ from the existing comparison runner')
    pairs = plan['pairs']
    if not pairs or len({tuple(p) for p in pairs}) != len(pairs) or any(
        len(p) != 2 or p[0] == p[1] or any(type(i) is not int or not 0 <= i < plan['frame_count'] for i in p)
        for p in pairs
    ):
        raise ValueError('invalid, duplicate, or out-of-range frame pairs')
    seeds = plan['seeds']
    if not 1 <= len(seeds) <= 16 or len(set(seeds)) != len(seeds) or any(
        type(s) is not int or not 0 <= s < 2**31 for s in seeds
    ) or not 1 <= plan['ransac_iterations'] <= 1000000:
        raise ValueError('invalid seeds or iteration budget')
    limits = plan['accuracy_limits']
    if set(limits) != {'translation_m', 'rotation_degrees'} or any(
        not np.isfinite(v) or v <= 0 for v in limits.values()
    ):
        raise ValueError('invalid accuracy limits')
    return plan, data


def depth_points(depth, camera):
    """Backproject uint16 millimetres into optical xyz metres in raster order."""
    depth = np.asarray(depth)
    if depth.dtype != np.uint16 or depth.shape != (camera['height'], camera['width']):
        raise ValueError('depth must be a camera-sized uint16 array')
    if any(not np.isfinite(camera[k]) for k in ('fx', 'fy', 'cx', 'cy', 'depth_scale', 'depth_trunc_m')) or any(
        camera[k] <= 0 for k in ('fx', 'fy', 'depth_scale', 'depth_trunc_m')
    ):
        raise ValueError('invalid camera intrinsics or depth scale')
    z = depth.astype(np.float64) / camera['depth_scale']
    valid = (z > 0) & (z < camera['depth_trunc_m'])
    v, u = np.nonzero(valid)
    z = z[valid]
    return np.column_stack(((u - camera['cx']) * z / camera['fx'],
                            (v - camera['cy']) * z / camera['fy'], z)).astype(np.float32)


def decode_depth(data):
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'PNG':
            raise ValueError('expected a PNG depth image')
        image.load()
        array = np.asarray(image)
        if array.dtype != np.uint16 or array.ndim != 2:
            raise ValueError('expected a single-channel uint16 depth PNG')
        return array


def prepare(archive_path, plan, output_dir):
    """Read only declared members, with no ZIP filesystem extraction or truth."""
    import spatialrust as sr
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError('prepared output already exists')
    archive_bytes = checked_bytes(archive_path, plan['depth_archive']['md5'])
    frames = {}
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        expected_names = {f'{i:05d}.png' for i in range(plan['frame_count'])}
        names = archive.namelist()
        if len(names) != len(expected_names) or set(names) != expected_names:
            raise ValueError('archive members do not match the declared frame set')
        output_dir.mkdir()
        for index in sorted({i for pair in plan['pairs'] for i in pair}):
            member = f'{index:05d}.png'
            if archive.getinfo(member).file_size > 4 * 1024 * 1024:
                raise ValueError('depth member exceeds bounded image size')
            png = archive.read(member)
            xyz = depth_points(decode_depth(png), plan['camera'])
            if len(xyz) < 3:
                raise ValueError('insufficient finite depth points')
            path = output_dir / f'{index:05d}.pcd'
            sr.write(str(path), sr.PointCloud.from_xyz(xyz))
            np.testing.assert_array_equal(sr.read(str(path)).xyz(), xyz)
            frames[str(index)] = dict(member=member, png_sha256=digest(png),
                                     pcd_sha256=digest(path.read_bytes()), points=len(xyz))
    receipt = dict(schema='spatialrust.redwood-preparation.v1', archive_sha256=digest(archive_bytes),
                   camera=plan['camera'], frames=frames)
    (output_dir / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def read_log(data, count):
    """Require contiguous LOG frame identities and proper world_from_camera poses."""
    lines = data.decode('ascii').splitlines()
    if len(lines) != count * 5:
        raise ValueError('trajectory must contain exactly five lines per frame')
    poses = []
    for i in range(count):
        if [int(x) for x in lines[5*i].split()] != [i, i, i+1]:
            raise ValueError('trajectory frame identity/order mismatch')
        rows = [[float(x) for x in line.split()] for line in lines[5*i+1:5*i+5]]
        poses.append(rigid(rows))
    return poses


def relative_pose(source, target):
    return rigid(np.linalg.inv(rigid(target)) @ rigid(source))


def score(frozen, trajectory_bytes, plan):
    """Score every saved row; generation failures remain in each denominator."""
    if frozen.get('schema') != 'spatialrust.redwood-frozen-registrations.v1' or frozen['plan'] != plan:
        raise ValueError('saved registration plan mismatch')
    poses = read_log(trajectory_bytes, plan['frame_count'])
    expected = [(pair, seed, method) for pair in plan['pairs'] for seed in plan['seeds']
                for method in ('spatialrust', 'open3d')]
    rows = frozen['rows']
    if [(r['pair'], r['seed'], r['method']) for r in rows] != expected:
        raise ValueError('saved rows do not match the full plan denominator/order')
    results = []
    for row in rows:
        item = dict(pair=row['pair'], seed=row['seed'], method=row['method'],
                    status=row['status'], accurate=False)
        if row['status'] == 'success':
            s, t = row['pair']
            hashes = {role: frozen['preparation']['frames'][str(i)]['pcd_sha256']
                      for role, i in zip(('source', 'target'), row['pair'])}
            if row['report']['input_file_sha256'] != hashes:
                raise ValueError('saved registration inputs mismatch the preparation receipt')
            reference = dict(schema='spatialrust.pose-reference.v1', length_unit='m',
                             input_file_sha256=hashes,
                             transform_source_to_target=relative_pose(poses[s], poses[t]).tolist(),
                             provenance=dict(kind='dataset_reference', url=plan['download_prefix'] + plan['trajectory']['name'],
                                             description=plan['dataset']))
            accuracy = evaluate(row['report'], reference)
            item.update(accuracy=accuracy, accurate=(
                accuracy['translation_error'] <= plan['accuracy_limits']['translation_m'] and
                accuracy['rotation_error_degrees'] <= plan['accuracy_limits']['rotation_degrees']))
        elif row['status'] == 'error':
            item['error'] = row['error']
        else:
            raise ValueError('unsupported saved row status')
        results.append(item)
    totals = {method: dict(denominator=sum(r['method'] == method for r in results),
                          generated=sum(r['method'] == method and r['status'] == 'success' for r in results),
                          accurate=sum(r['method'] == method and r['accurate'] for r in results))
              for method in ('spatialrust', 'open3d')}
    return dict(schema='spatialrust.redwood-accuracy.v1', accuracy_limits=plan['accuracy_limits'],
                scope=plan['scope'], totals=totals, rows=results)
