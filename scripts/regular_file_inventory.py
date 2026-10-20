"""Complete logical-byte observations for explicitly owned regular-file trees."""
from dataclasses import dataclass
import fnmatch
import os
from pathlib import Path
import stat


@dataclass(frozen=True)
class FileInventory:
    logical_bytes: int
    regular_files: int
    matched_files: int
    directories: int


def _reject_link(info):
    if (stat.S_ISLNK(info.st_mode)
            or getattr(info, 'st_file_attributes', 0)
            & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
        raise ValueError('inventory cannot traverse links or reparse points')


def _directory(path):
    info = path.lstat()
    _reject_link(info)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError('inventory requires regular directories')


def inventory(root, *, filename_pattern='*'):
    """Return counts only after every directory and entry has been inspected.

    IO errors propagate. A filename filter changes the charged regular files,
    never directory traversal or error admission. Hard links are charged per
    pathname. The observation does not promise an atomic filesystem snapshot.
    """
    root = Path(root)
    if not root.is_absolute():
        raise ValueError('inventory requires an existing absolute directory')
    if type(filename_pattern) is not str or not filename_pattern:
        raise ValueError('inventory requires a nonempty filename pattern')
    for ancestor in [root] + list(root.parents):
        _directory(ancestor)
    pending = [root]
    logical_bytes = regular_files = matched_files = directories = 0
    while pending:
        directory = pending.pop()
        _directory(directory)
        # Unlike recursive glob iteration, scandir failures are never omitted.
        with os.scandir(directory) as entries:
            directories += 1
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                _reject_link(info)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    regular_files += 1
                    if fnmatch.fnmatch(entry.name, filename_pattern):
                        logical_bytes += info.st_size
                        matched_files += 1
                else:
                    raise ValueError('inventory accepts only regular entries')
    return FileInventory(logical_bytes, regular_files, matched_files, directories)
