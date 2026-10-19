"""Freeze three full public RGB-D frames for explicit multi-adapter conformance."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE_SHA256 = 'a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98'
FRAME_INDICES = (0, 399, 797)
CAMERA = dict(width=640, height=480, fx=525., fy=525., cx=319.5, cy=239.5,
              depth_unit_m=.0002, distortion_applied=False, extrinsic_applied=False)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_image(prepared, record, kind):
    if not isinstance(record, dict):
        raise ValueError('require a timestamp-associated source image')
    relative = Path(record.get('path', ''))
    if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != kind or relative.suffix != '.png' or '..' in relative.parts:
        raise ValueError('require a relative {}/image.png path'.format(kind))
    path = (prepared/relative).resolve()
    path.relative_to(prepared.resolve())
    if type(record.get('bytes')) is not int or not 0 < record['bytes'] <= 16*1024*1024:
        raise ValueError('source image size outside fixed bounds')
    if path.stat().st_size != record['bytes']:
        raise ValueError('source image size changed')
    raw = path.read_bytes()
    if sha256(raw) != record.get('sha256'):
        raise ValueError('source image checksum changed')
    return raw


def decode_rgb(raw, camera):
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'PNG' or image.mode != 'RGB' or image.size != (camera['width'], camera['height']) or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('require original RGB uint8 PNG geometry')
        image.load()
        return np.array(image, dtype=np.uint8, copy=True)


def decode_depth(raw, camera):
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'PNG' or image.size != (camera['width'], camera['height']) or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('require original depth PNG geometry')
        image.load()
        depth = np.asarray(image)
        if depth.ndim != 2 or depth.dtype.kind not in 'iu' or depth.dtype.itemsize < 2 or np.any(depth < 0) or np.any(depth > 65535):
            raise ValueError('require original unsigned 16-bit depth counts')
        return np.array(depth, dtype=np.uint16, copy=True)


def project_reference(depth, camera):
    """Publisher optical projection in float64, then one explicit float32 cast."""
    if depth.dtype != np.uint16 or depth.shape != (camera['height'], camera['width']):
        raise ValueError('projection requires original depth count geometry')
    z = depth.astype(np.float64)*camera['depth_unit_m']
    valid = (z >= .1) & (z <= 5.)
    rows, columns = np.nonzero(valid)
    xyz = np.column_stack(((columns-camera['cx'])*z[valid]/camera['fx'],
                           (rows-camera['cy'])*z[valid]/camera['fy'], z[valid]))
    # Row-major valid-pixel order is retained; no voxel or point selection here.
    return np.ascontiguousarray(xyz, dtype='<f4')


def prepare(prepared, expected_sha, output):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    raw_manifest = (prepared/'manifest.json').read_bytes()
    if sha256(raw_manifest) != expected_sha:
        raise ValueError('prepared manifest SHA-256 mismatch')
    manifest = json.loads(raw_manifest)
    if manifest.get('schema') != 'spatialrust.tum-prepared.v1' or manifest.get('source_sha256') != SOURCE_SHA256 or len(manifest.get('frames', [])) != 798:
        raise ValueError('require all 798 indexed frames of the fixed TUM archive')
    if manifest.get('reference_used_for_generation') is not False:
        raise ValueError('input preparation must withhold reference contents')
    calibration_bytes = (prepared/'calibration.json').read_bytes()
    if sha256(calibration_bytes) != manifest.get('calibration_sha256'):
        raise ValueError('prepared calibration checksum changed')
    camera = json.loads(calibration_bytes)
    if any(type(camera.get(key)) != type(value) or camera[key] != value for key, value in CAMERA.items()):
        raise ValueError('require fixed publisher registered-depth calibration')
    loaded = []
    for index in FRAME_INDICES:
        source = manifest['frames'][index]
        rgb_entry, depth_entry = source.get('rgb'), source.get('depth')
        for entry in (rgb_entry, depth_entry):
            if not isinstance(entry, dict) or type(entry.get('timestamp_ns')) is not int:
                raise ValueError('source image requires integer timestamp')
        delta = rgb_entry['timestamp_ns']-depth_entry['timestamp_ns']
        if type(source.get('rgb_delta_ns')) is not int or delta != source['rgb_delta_ns'] or abs(delta) > 20_000_000 or source.get('timestamp_ns') != depth_entry['timestamp_ns']:
            raise ValueError('source RGB/depth association differs from fixed timestamp contract')
        rgb_raw = checked_image(prepared, rgb_entry, 'rgb')
        depth_raw = checked_image(prepared, depth_entry, 'depth')
        rgb = decode_rgb(rgb_raw, camera)
        positions = project_reference(decode_depth(depth_raw, camera), camera)
        if not 64 <= len(positions) <= 640*480:
            raise ValueError('full projected point count outside fixed bounds')
        loaded.append((index, source, rgb.tobytes(order='C'), positions.tobytes(order='C'), len(positions)))
    if output.exists():
        raise ValueError('refuse to overwrite a frozen GPU input directory')
    output.mkdir(parents=True)
    frames = []
    for index, source, rgb, positions, count in loaded:
        record = dict(index=index, timestamp_ns=source['timestamp_ns'], rgb_delta_ns=source['rgb_delta_ns'],
                      source_rgb=source['rgb'], source_depth=source['depth'], point_count=count)
        for label, suffix, payload in [('rgb','rgb-u8',rgb),('positions','xyz-f32le',positions)]:
            name = 'frame-{:06d}.{}'.format(index,suffix)
            (output/name).write_bytes(payload)
            record[label] = dict(path=name, bytes=len(payload), sha256=sha256(payload))
        frames.append(record)
    result = dict(schema='spatialrust.public-gpu-inputs.v1', sequence='TUM RGB-D freiburg1_xyz',
                  source_sha256=SOURCE_SHA256, source_bytes=448204271,
                  source_url='https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz',
                  source_license='CC BY 4.0', attribution='Sturm et al., RGB-D SLAM benchmark, IROS 2012',
                  prepared_manifest_sha256=expected_sha, calibration_sha256=sha256(calibration_bytes),
                  calibration=camera, width=640, height=480, frames=frames,
                  preparation_sha256=sha256(Path(__file__).read_bytes()),
                  reference_used=False,
                  projection='Independent publisher optical formula in float64; keep every valid .1..5m pixel in row order; cast coordinates once to little-endian float32.')
    raw = (json.dumps(result,indent=2,sort_keys=True)+'\n').encode('utf-8')
    (output/'manifest.json').write_bytes(raw)
    if (prepared/'manifest.json').read_bytes() != raw_manifest or (prepared/'calibration.json').read_bytes() != calibration_bytes:
        raise ValueError('source metadata changed during preparation')
    return sha256(raw), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    digest, result = prepare(args.prepared_dir,args.manifest_sha256,args.output_dir)
    print(json.dumps(dict(manifest_sha256=digest,indices=FRAME_INDICES,point_counts=[frame['point_count'] for frame in result['frames']])))


if __name__ == '__main__':
    main()
