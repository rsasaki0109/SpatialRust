"""Reuse frozen TUM PNGs with hard links under a new, compatible fixed plan.

The source directory is never edited. Link failure preserves a partial file
inventory; it never silently falls back to payload copies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from tum_sequence_reference import file_sha256, json_bytes, require_sha256, safe_relative, validate_plan


MIN_FREE_BYTES = 20 * 1024**3
WORKSPACE = Path(__file__).resolve().parents[1]


def external(path):
    original = Path(path)
    if not original.is_absolute():
        raise ValueError('require absolute external data paths')
    resolved = original.resolve()
    if resolved == WORKSPACE or WORKSPACE in resolved.parents:
        raise ValueError('prepared sensor data must remain outside the workspace')
    return resolved


def reuse(prepared, manifest_sha256, plan_path, output):
    if Path(output).is_symlink():
        raise ValueError('output cannot be a filesystem link')
    prepared, output = external(prepared), external(output)
    if prepared in output.parents:
        raise ValueError('output must not mutate the source directory')
    if output.exists() or not output.parent.is_dir():
        raise ValueError('require a new output directory with an existing parent')
    free_before = shutil.disk_usage(output.parent).free
    if free_before < MIN_FREE_BYTES:
        raise ValueError('output filesystem has less than 20 GiB free')
    if not 0 < (prepared/'manifest.json').stat().st_size <= 16*1024**2:
        raise ValueError('prepared manifest size outside bounds')
    source_raw = (prepared/'manifest.json').read_bytes()
    if hashlib.sha256(source_raw).hexdigest() != require_sha256(manifest_sha256):
        raise ValueError('prepared manifest identity changed')
    manifest = json.loads(source_raw)
    if manifest.get('schema') != 'spatialrust.tum-prepared.v1' or manifest.get('reference_used_for_generation') is not False:
        raise ValueError('require complete reference-free prepared inputs')
    old_plan_raw = (prepared/'plan.json').read_bytes()
    calibration_raw = (prepared/'calibration.json').read_bytes()
    if hashlib.sha256(old_plan_raw).hexdigest() != manifest['plan_sha256'] or hashlib.sha256(calibration_raw).hexdigest() != manifest['calibration_sha256']:
        raise ValueError('prepared plan or calibration changed')
    old_plan = validate_plan(json.loads(old_plan_raw))
    plan_raw = Path(plan_path).read_bytes()
    plan = validate_plan(json.loads(plan_raw))
    if any(plan[key] != old_plan[key] for key in ('sequence','calibration','limits')) or json.loads(calibration_raw) != plan['calibration']:
        raise ValueError('reused sequence and calibration must stay identical')
    if any(plan['controls'].get(key) != value for key, value in old_plan['controls'].items()):
        raise ValueError('original preparation controls cannot change')
    files = manifest.get('files')
    if not isinstance(files, dict) or not 2 <= len(files) <= old_plan['limits']['max_members']:
        raise ValueError('invalid prepared file inventory')
    sources = []
    for name, entry in files.items():
        relative = safe_relative(name)
        if str(relative) not in ('depth.txt','rgb.txt') and not (len(relative.parts)==2 and relative.parts[0] in ('rgb','depth') and relative.suffix=='.png'):
            raise ValueError('only original indexes and PNGs may be linked')
        source = (prepared / str(relative)).resolve()
        source.relative_to(prepared)
        if entry.get('path') != str(relative) or type(entry.get('bytes')) is not int or not 0 < entry['bytes'] <= old_plan['limits']['max_member_bytes']:
            raise ValueError('invalid prepared file identity or size')
        if not source.is_file() or source.stat().st_size != entry['bytes'] or file_sha256(source) != require_sha256(entry['sha256']):
            raise ValueError('prepared file checksum changed')
        sources.append((relative, source, entry))
    if (prepared/'manifest.json').read_bytes() != source_raw or Path(plan_path).read_bytes() != plan_raw:
        raise ValueError('metadata changed before linking')
    if shutil.disk_usage(output.parent).free < MIN_FREE_BYTES + 64*1024**2:
        raise ValueError('free-space floor no longer met before creation')
    output.mkdir()
    try:
        for relative, source, entry in sources:
            destination = output / str(relative)
            destination.parent.mkdir(exist_ok=True)
            os.link(str(source), str(destination))
            if not destination.samefile(source) or destination.stat().st_size != entry['bytes'] or file_sha256(destination) != entry['sha256']:
                raise ValueError('reused hard link identity changed')
        manifest['plan_sha256'] = hashlib.sha256(plan_raw).hexdigest()
        manifest['bindings'] = dict(manifest.get('bindings', {}), **{str(Path(__file__).resolve()):file_sha256(__file__)})
        manifest['reused_input'] = dict(path=str(prepared), manifest_sha256=manifest_sha256,
            storage='hard links; original PNG/index payloads shared, source never edited',
            linked_files=len(sources), linked_logical_bytes=sum(e['bytes'] for _, _, e in sources),
            new_image_payload_bytes=0, free_before_bytes=free_before,
            free_after_links_bytes=shutil.disk_usage(output).free)
        if (prepared/'manifest.json').read_bytes() != source_raw or (prepared/'plan.json').read_bytes() != old_plan_raw or (prepared/'calibration.json').read_bytes() != calibration_raw or Path(plan_path).read_bytes() != plan_raw:
            raise ValueError('source metadata changed during linking')
        for name, raw in [('calibration.json',calibration_raw),('plan.json',plan_raw),('manifest.json',json_bytes(manifest))]:
            with (output/name).open('xb') as stream: stream.write(raw)
    except (OSError, ValueError):
        survivors = [dict(path=str(p.relative_to(output)), bytes=p.stat().st_size, sha256=file_sha256(p)) for p in sorted(output.rglob('*')) if p.is_file()]
        (output/'reuse.partial.json').write_bytes(json_bytes(dict(accepted=False, files=survivors)))
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = reuse(args.prepared_dir, args.manifest_sha256, args.plan, args.output_dir)
    print(json.dumps(dict(manifest_sha256=file_sha256(args.output_dir/'manifest.json'), reused=manifest['reused_input'])))


if __name__ == '__main__':
    main()
