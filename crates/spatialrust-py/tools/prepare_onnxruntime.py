"""Stage the pinned official Linux runtime wheel's native libraries for ort-sys.

Only for local/CI tests; not a redistribution or production installation tool.
"""
import argparse
from pathlib import Path
import shutil
import sys

import onnxruntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    if not sys.platform.startswith('linux') or onnxruntime.__version__ != '1.24.2':
        parser.error('requires Linux and official onnxruntime==1.24.2')
    source = Path(onnxruntime.__file__).parent / 'capi'
    library = source / 'libonnxruntime.so.1.24.2'
    if not library.is_file():
        parser.error('runtime wheel does not contain the expected native library')
    args.directory.mkdir(parents=True, exist_ok=True)
    for path in source.glob('libonnxruntime*.so*'):
        shutil.copy2(path, args.directory / path.name)
    for name in ('libonnxruntime.so', 'libonnxruntime.so.1'):
        path = args.directory / name
        if path.is_symlink() and path.readlink() == Path(library.name):
            continue
        path.symlink_to(library.name)
    print(args.directory.resolve())


if __name__ == '__main__':
    main()
