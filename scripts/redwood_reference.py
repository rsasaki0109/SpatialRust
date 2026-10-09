"""Strict Redwood/3DMatch log records: header i,j,N maps fragment j into i."""
import hashlib
from pathlib import Path

import numpy as np

from prepare_numpy_reference import reference_pose


def parse_records(text, dimension=4, project_rotation=False):
    if dimension not in (4, 6):
        raise ValueError('matrix dimension must be 4 (pose) or 6 (information)')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    width = dimension + 1
    if not lines or len(lines) % width:
        raise ValueError('empty or truncated reference log')
    records, total = {}, None
    for offset in range(0, len(lines), width):
        header = lines[offset].split()
        if len(header) != 3:
            raise ValueError('reference header must contain i, j, N')
        try:
            i, j, count = [int(value) for value in header]
        except ValueError as error:
            raise ValueError('reference header requires integer IDs') from error
        if not 0 <= i < j < count or count < 2:
            raise ValueError('reference requires 0 <= i < j < N')
        if total is not None and total != count:
            raise ValueError('fragment count changes within reference log')
        total = count
        if (i, j) in records:
            raise ValueError('duplicate reference pair')
        rows = [line.split() for line in lines[offset + 1:offset + width]]
        if any(len(row) != dimension for row in rows):
            raise ValueError('reference matrix row width mismatch')
        try:
            matrix = np.array(rows, dtype=np.float64)
        except ValueError as error:
            raise ValueError('reference matrix must be numeric') from error
        original = matrix.copy()
        correction = None
        if dimension == 4:
            matrix, correction = reference_pose(matrix, project_rotation)
        else:
            if not np.isfinite(matrix).all() or matrix[0, 0] <= 0:
                raise ValueError('information must be finite with positive normalization')
            scale = max(float(np.max(np.abs(matrix))), 1.)
            if not np.allclose(matrix / scale, matrix.T / scale, rtol=0, atol=1e-8):
                raise ValueError('information matrix must be symmetric')
            # A quadratic form depends only on the symmetric part.
            matrix = matrix * .5 + matrix.T * .5
            if np.linalg.eigvalsh(matrix / scale)[0] < -1e-8:
                raise ValueError('information matrix must be positive semidefinite')
        records[(i, j)] = dict(matrix=matrix, original_matrix=original, fragments=count, rotation_correction=correction)
    return records


def load_records(path, dimension=4, project_rotation=False):
    snapshot = Path(path).read_bytes()
    return parse_records(snapshot.decode('utf-8'), dimension, project_rotation), hashlib.sha256(snapshot).hexdigest()


def select_pose(records, source_id, target_id):
    if type(source_id) is not int or type(target_id) is not int or min(source_id, target_id) < 0 or source_id == target_id:
        raise ValueError('source/target IDs must be distinct nonnegative integers')
    pair = tuple(sorted((source_id, target_id)))
    if pair not in records:
        raise ValueError('requested pair is absent from reference log')
    pose = records[pair]['matrix']
    # Stored matrix maps the larger second ID to the smaller first ID.
    invert = source_id == pair[0]
    return (np.linalg.inv(pose) if invert else pose.copy()), pair, invert
