"""A denied subtree must never become a valid partial capacity observation."""
import copy
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
import regular_file_inventory as storage
import acquire_heldout_rgbd as download
import acquire_heldout_rgbd_ranges as ranges
from heldout_dense_rgbd_protocol import selection


def test_complete_nested_logical_counts_and_filename_filter(tmp_path):
    root = tmp_path/'owned space'; root.mkdir()
    nested = root/'nested'; nested.mkdir()
    (root/'first.tgz').write_bytes(b'123')
    (nested/'.hidden.tgz.part').write_bytes(b'4567')
    (nested/'other.bin').write_bytes(b'12345')
    (nested/'empty').write_bytes(b'')
    all_files = storage.inventory(root)
    assert all_files == storage.FileInventory(12, 4, 4, 2)
    assert storage.inventory(root, filename_pattern='*.tgz*') == storage.FileInventory(7, 4, 2, 2)
    with pytest.raises(FrozenInstanceError):
        all_files.logical_bytes = 0


def test_hard_links_are_charged_once_per_pathname(tmp_path):
    first = tmp_path/'first'; first.write_bytes(b'123')
    os.link(first, tmp_path/'second')
    assert storage.inventory(tmp_path) == storage.FileInventory(6, 2, 2, 1)


@pytest.mark.parametrize('pattern', ['', None, 7])
def test_invalid_filter_is_not_an_empty_success(tmp_path, pattern):
    with pytest.raises(ValueError, match='pattern'):
        storage.inventory(tmp_path, filename_pattern=pattern)


def test_root_must_be_an_existing_absolute_directory(tmp_path):
    with pytest.raises(ValueError, match='absolute'):
        storage.inventory('relative')
    with pytest.raises(FileNotFoundError):
        storage.inventory(tmp_path/'missing')
    file = tmp_path/'file'; file.write_bytes(b'x')
    with pytest.raises(ValueError, match='directories'):
        storage.inventory(file)


def deny_scandir(monkeypatch, blocked):
    original = storage.os.scandir
    @contextmanager
    def scandir(path):
        if Path(path) == blocked:
            raise PermissionError('synthetic protected subtree')
        with original(path) as entries:
            yield entries
    monkeypatch.setattr(storage.os, 'scandir', scandir)


@pytest.mark.parametrize('pattern', ['*', '*.tgz*'])
def test_filtered_and_unfiltered_denied_subtrees_never_return_totals(tmp_path, monkeypatch, pattern):
    (tmp_path/'known.tgz').write_bytes(b'123')
    blocked = tmp_path/'not-an-archive'; blocked.mkdir()
    deny_scandir(monkeypatch, blocked)
    with pytest.raises(PermissionError, match='protected'):
        storage.inventory(tmp_path, filename_pattern=pattern)
    assert (tmp_path/'known.tgz').read_bytes() == b'123'


@pytest.mark.parametrize('error_type', [PermissionError, FileNotFoundError, OSError])
def test_failed_stat_after_a_valid_entry_cannot_return_partial_counts(tmp_path, monkeypatch, error_type):
    (tmp_path/'known.tgz').write_bytes(b'123')
    original = storage.os.scandir
    with original(tmp_path) as entries:
        known = next(iter(entries))
    def fail(**kwargs):
        raise error_type('synthetic stat failure')
    broken = SimpleNamespace(name='unmatched.bin', path=str(tmp_path/'unmatched.bin'), stat=fail)
    @contextmanager
    def scandir(path):
        assert Path(path) == tmp_path
        yield [known, broken]
    monkeypatch.setattr(storage.os, 'scandir', scandir)
    with pytest.raises(error_type, match='stat failure'):
        storage.inventory(tmp_path, filename_pattern='*.tgz*')


@pytest.mark.parametrize('mode,attributes', [
    (stat.S_IFLNK, 0), (stat.S_IFIFO, 0),
    (stat.S_IFDIR, 0x400), (stat.S_IFREG, 0x400),
])
def test_link_reparse_or_unsupported_entries_reject(tmp_path, monkeypatch, mode, attributes):
    info = SimpleNamespace(st_mode=mode, st_file_attributes=attributes, st_size=99)
    entry = SimpleNamespace(name='entry', path=str(tmp_path/'entry'), stat=lambda **kwargs: info)
    @contextmanager
    def scandir(path):
        yield [entry]
    monkeypatch.setattr(storage.os, 'scandir', scandir)
    with pytest.raises(ValueError):
        storage.inventory(tmp_path)


@pytest.mark.parametrize('where', ['root', 'ancestor'])
def test_reparse_root_and_ancestors_reject_before_scandir(tmp_path, monkeypatch, where):
    original = Path.lstat
    blocked = tmp_path if where == 'root' else tmp_path.parent
    def lstat(path, *args, **kwargs):
        if path == blocked:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return original(path, *args, **kwargs)
    def unexpected(path):
        raise AssertionError('scandir must not be admitted')
    monkeypatch.setattr(Path, 'lstat', lstat)
    monkeypatch.setattr(storage.os, 'scandir', unexpected)
    with pytest.raises(ValueError, match='reparse'):
        storage.inventory(tmp_path)


@pytest.mark.parametrize('caller', ['single', 'ranges'])
def test_acquisition_denied_inventory_refuses_before_request_and_keeps_prefix(tmp_path, monkeypatch, caller):
    root = tmp_path/'inputs'; root.mkdir()
    blocked = root/'not-an-archive'; blocked.mkdir()
    source = copy.deepcopy(selection()['sources'][0]); source['content_length'] = 42
    limits = copy.deepcopy(selection()['resources']); limits['archives_max_bytes'] = 1024
    monkeypatch.setattr(download.shutil, 'disk_usage', lambda _: SimpleNamespace(free=32*1024**3))
    destination = root/source['sequence']; partial = destination/(source['sequence']+'.tgz.part')
    prefix = b'\x1f\x8b12345'
    if caller == 'ranges':
        destination.mkdir(); partial.write_bytes(prefix)
        (destination/'acquisition.json').write_text(json.dumps(dict(source=source,status='incomplete')),encoding='utf-8')
    calls = []
    def unexpected(*args, **kwargs):
        calls.append(args); raise AssertionError('network must not be admitted')
    opener = SimpleNamespace(open=unexpected)
    deny_scandir(monkeypatch, blocked)
    operation = download.acquire if caller == 'single' else ranges.acquire_ranges
    with pytest.raises(PermissionError, match='protected'):
        operation(source, root, limits, opener=opener)
    assert calls == [] and not (destination/(source['sequence']+'.tgz')).exists()
    if caller == 'ranges':
        assert partial.read_bytes() == prefix
        assert json.loads((destination/'acquisition.json').read_text(encoding='utf-8'))['status'] == 'incomplete'
    else:
        failed = json.loads((destination/'acquisition.json').read_text(encoding='utf-8'))
        assert failed['status'] == 'failed' and failed['error_type'] == 'PermissionError'
