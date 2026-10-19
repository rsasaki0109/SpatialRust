"""Explicit storage conditions for tiny synthetic audit fixtures only."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def synthetic_dense_audit_space(monkeypatch):
    import audit_tum_dense_rgbd

    usage = SimpleNamespace(total=64 * 1024**3, used=32 * 1024**3, free=32 * 1024**3)
    monkeypatch.setattr(audit_tum_dense_rgbd.shutil, 'disk_usage', lambda _path: usage)
    return usage
