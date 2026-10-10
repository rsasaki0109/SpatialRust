"""Read the wasm-bindgen CLI version required by the built workspace lockfile.

Requires Python 3.11+ (the Web/WASM CI job uses Python 3.12).
"""
import argparse
from pathlib import Path
import re
import tomllib


def required_version(lockfile):
    with Path(lockfile).open('rb') as stream:
        packages = tomllib.load(stream).get('package', [])
    versions = {p['version'] for p in packages if p.get('name') == 'wasm-bindgen'}
    if len(versions) != 1:
        raise ValueError('require exactly one resolved wasm-bindgen version')
    version = versions.pop()
    if not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('unsupported wasm-bindgen version')
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lockfile', type=Path)
    args = parser.parse_args()
    print(required_version(args.lockfile))


if __name__ == '__main__':
    main()
