"""Evaluate saved registrations against declared reference poses, after fitting."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np


def rigid(value):
    """Reject non-rigid or non-finite reference/estimate matrices."""
    raw = np.asarray(value)
    if raw.shape != (4, 4) or raw.dtype.kind not in 'fiu':
        raise ValueError('pose must be a numeric 4x4 matrix')
    matrix = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError('pose must be finite')
    if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError('pose must have a homogeneous last row')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6) or abs(np.linalg.det(rotation) - 1) > 1e-6:
        raise ValueError('pose rotation must be proper and orthonormal')
    return matrix


def validate_reference(reference):
    if reference.get('schema') != 'spatialrust.pose-reference.v1':
        raise ValueError('unsupported reference schema')
    fingerprints = reference.get('input_file_sha256')
    if not isinstance(fingerprints, dict) or set(fingerprints) != {'source', 'target'}:
        raise ValueError('reference requires source and target file hashes')
    if any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in fingerprints.values()):
        raise ValueError('invalid reference file hash')
    if reference.get('length_unit') not in ('m', 'cm', 'mm'):
        raise ValueError('reference length_unit must be m, cm or mm')
    provenance = reference.get('provenance')
    if not isinstance(provenance, dict) or provenance.get('kind') not in ('dataset_reference', 'synthetic'):
        raise ValueError('reference requires declared dataset_reference or synthetic provenance')
    if not isinstance(provenance.get('description'), str) or not provenance['description'].strip():
        raise ValueError('reference provenance requires a description')
    if provenance['kind'] == 'dataset_reference' and (
        not isinstance(provenance.get('url'), str) or not provenance['url'].startswith('https://')
    ):
        raise ValueError('dataset reference requires an HTTPS source URL')
    return rigid(reference.get('transform_source_to_target'))


def evaluate(report, reference):
    expected = validate_reference(reference)
    if report.get('schema_version') != 'spatialrust.python-alignment.v1':
        raise ValueError('unsupported alignment report schema')
    if report.get('input_file_sha256') != reference['input_file_sha256']:
        raise ValueError('registration inputs do not match reference file hashes')
    estimate = rigid(report.get('transform_source_to_target'))
    relative = expected[:3, :3].T @ estimate[:3, :3]
    skew = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]])
    angle = math.degrees(math.atan2(float(np.linalg.norm(skew)) / 2,
                                  float(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
    with np.errstate(over='ignore', invalid='ignore'):
        delta = estimate[:3, 3] - expected[:3, 3]
    # hypot avoids overflow from squaring otherwise representable errors.
    translation = math.hypot(*(float(v) for v in delta))
    if not math.isfinite(translation):
        raise ValueError('translation error overflows')
    return dict(rotation_error_degrees=angle, translation_error=translation,
                length_unit=reference['length_unit'],
                initial_transform_supplied=report.get('initial_transform_supplied'),
                transform_source_to_target=estimate.tolist())


def render(result):
    rows = []
    for item in result['evaluations']:
        name = html.escape(item['report_file'])
        angle = item['rotation_error_degrees']
        translation = item['translation_error']
        # Fixed 0..180 degree scale; tables retain exact measurements.
        bar = f'<svg width="180" height="16" aria-label="{angle:.6g} degrees"><rect width="{angle:.6g}" height="14" fill="#3978b5"/></svg>'
        rows.append(f'<tr><td>{name}</td><td>{angle:.9g}° {bar}</td><td>{translation:.9g} {result["length_unit"]}</td></tr>')
    provenance = html.escape(json.dumps(result['reference_provenance'], ensure_ascii=False))
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Reference pose accuracy</title>'
            '<style>body{font:15px system-ui;margin:2rem}td,th{padding:.6rem;text-align:left}svg{vertical-align:middle}</style>'
            '<h1>Reference pose accuracy</h1><p>Both poses map source coordinates to target coordinates. '
            'Translation error is the distance between translations in the target frame. '
            'Reference poses enter evaluation only; this tool does not run or choose registrations.</p>'
            f'<p>Declared provenance: {provenance}</p><table><tr><th>Saved report</th><th>Rotation error</th><th>Translation error</th></tr>'
            + ''.join(rows) + '</table><p>Matching byte hashes bind the input pair; they do not independently '
            'certify dataset ground truth, units, or whether a reference was used in an earlier initialization. '
            'Proximity support and optimizer convergence do not certify pose accuracy.</p></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists')
    reference_bytes = args.reference.read_bytes()
    reference = json.loads(reference_bytes)
    validate_reference(reference)
    evaluations = []
    for path in args.reports:
        data = path.read_bytes()
        item = evaluate(json.loads(data), reference)
        item.update(report_file=str(path.resolve()), report_sha256=hashlib.sha256(data).hexdigest())
        evaluations.append(item)
    result = dict(schema='spatialrust.pose-reference-evaluation.v1',
                  reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                  reference_provenance=reference['provenance'], length_unit=reference['length_unit'],
                  input_file_sha256=reference['input_file_sha256'], evaluations=evaluations)
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    rendered = render(result)
    args.output_dir.mkdir()
    try:
        (args.output_dir / 'accuracy.json').write_text(serialized)
        (args.output_dir / 'report.html').write_text(rendered)
    except Exception:
        shutil.rmtree(args.output_dir)
        raise
    print(f'Evaluated {len(evaluations)} saved registrations; unit={reference["length_unit"]}')


if __name__ == '__main__':
    main()
