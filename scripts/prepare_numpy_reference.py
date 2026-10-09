"""Convert a declared public NumPy pair and reference into hash-bound PCD inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import spatialrust as sr

from evaluate_pose_reference import rigid


def load_array(path):
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    value = np.load(path, allow_pickle=False)
    if hashlib.sha256(path.read_bytes()).hexdigest() != before:
        raise ValueError('NumPy input changed while reading')
    return value, before


def points(value):
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 3 or value.dtype.kind not in 'fiu':
        raise ValueError('point array must be numeric Nx3 with at least three rows')
    if not np.isfinite(value).all() or np.max(np.abs(value.astype(np.float64))) > np.finfo(np.float32).max:
        raise ValueError('points must be finite and representable as f32')
    converted = value.astype(np.float32)
    return converted, float(np.max(np.abs(value.astype(np.float64) - converted.astype(np.float64))))


def reference_pose(value, project=False):
    if not project:
        return rigid(value), None
    raw = np.asarray(value)
    if raw.shape != (4, 4) or raw.dtype.kind not in 'fiu':
        raise ValueError('reference must be a numeric 4x4 matrix')
    matrix = raw.astype(np.float64)
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError('reference must be finite and homogeneous')
    rotation = matrix[:3, :3]
    u, singular, vt = np.linalg.svd(rotation)
    if np.linalg.det(rotation) <= 0 or np.max(np.abs(singular - 1)) > 1e-3:
        raise ValueError('reference correction exceeds near-rigid limit or contains reflection')
    corrected = u @ vt
    detail = dict(operation='nearest_SO3_svd', original_rotation=rotation.tolist(),
                  singular_values=singular.tolist(),
                  correction_frobenius=float(np.linalg.norm(corrected - rotation)),
                  singular_value_deviation_limit=1e-3)
    matrix[:3, :3] = corrected
    return rigid(matrix), detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'target', 'pose'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--length-unit', choices=['m', 'cm', 'mm'], required=True)
    parser.add_argument('--provenance-url', required=True)
    parser.add_argument('--description', required=True)
    parser.add_argument('--project-reference-rotation', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists')
    if not args.provenance_url.startswith('https://') or not args.description.strip():
        parser.error('declare an HTTPS provenance URL and nonempty description')
    loaded = {name: load_array(getattr(args, name)) for name in ('source', 'target', 'pose')}
    source, source_error = points(loaded['source'][0])
    target, target_error = points(loaded['target'][0])
    pose, correction = reference_pose(loaded['pose'][0], args.project_reference_rotation)
    args.output_dir.mkdir()
    try:
        fingerprints = {}
        for name, xyz in [('source', source), ('target', target)]:
            path = args.output_dir / (name + '.pcd')
            sr.write(str(path), sr.PointCloud.from_xyz(xyz))
            np.testing.assert_array_equal(sr.read(str(path)).xyz(), xyz)
            fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        reference = dict(schema='spatialrust.pose-reference.v1', length_unit=args.length_unit,
                         input_file_sha256=fingerprints, transform_source_to_target=pose.tolist(),
                         provenance=dict(kind='dataset_reference', url=args.provenance_url,
                             description=args.description,
                             original_file_sha256={name: pair[1] for name, pair in loaded.items()},
                             f32_conversion_max_abs_error=dict(source=source_error, target=target_error),
                             reference_rotation_correction=correction))
        (args.output_dir / 'reference.json').write_text(json.dumps(reference, indent=2, allow_nan=False) + '\n')
    except Exception:
        shutil.rmtree(args.output_dir)
        raise
    print(f'Prepared {len(source)} source and {len(target)} target points; correction={correction is not None}')


if __name__ == '__main__':
    main()
