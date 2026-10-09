"""Study receipts bind the imported binary instead of scanning wrapper siblings."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest
import spatialrust as sr


@pytest.mark.parametrize('name,case,extra', [
    ('study_icp_convergence', 'clean_volume', ['--scales', '1', '--angles', '0']),
    ('study_trimmed_icp', 'clean', []),
])
def test_pyd_only_wrapper_directory_keeps_real_loaded_binary_hash(tmp_path, monkeypatch, name, case, extra):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/'scripts'))
    module = importlib.import_module(name)
    native = Path(importlib.import_module('spatialrust.spatialrust').__file__)
    expected = hashlib.sha256(native.read_bytes()).hexdigest()
    wrapper = tmp_path/'package'
    wrapper.mkdir()
    (wrapper/'spatialrust.pyd').write_bytes(b'controlled non-loaded sibling fixture')
    monkeypatch.setattr(sr, '__file__', str(wrapper/'__init__.py'))
    monkeypatch.setattr(module, 'CASES', (case,))
    output = tmp_path/'study'
    monkeypatch.setattr('sys.argv', [name, '--output-dir', str(output), '--seeds', '1', '--iterations', '2']+extra)
    module.main()
    result = json.loads((output/'study.json').read_bytes())
    assert result['native_sha256'] == expected
    assert result['native_sha256'] != hashlib.sha256((wrapper/'spatialrust.pyd').read_bytes()).hexdigest()
