"""Build the camera Pages directory with size/SHA-pinned, same-origin assets.

The Rust WASM module is built by the caller. No package manager or mutable CDN
imports run in the published application. Failed downloads remain as .part files.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "site" / "camera"
MAX_BYTES = 64 * 1024 * 1024
HOSTS = {"cdn.jsdelivr.net", "storage.googleapis.com", "raw.githubusercontent.com"}
APP_FILES = ("index.html", "camera.css", "app.mjs", "session.mjs", "tracking-worker.mjs", "person-mask.mjs", "assets.json")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_assets(document):
    if document.get("schema") != "spatialrust.camera-assets.v1":
        raise ValueError("unsupported camera asset manifest")
    limit = document.get("max_total_bytes")
    if type(limit) is not int or not 0 < limit <= MAX_BYTES:
        raise ValueError("invalid asset byte limit")
    assets = document.get("assets")
    if not isinstance(assets, list) or not 1 <= len(assets) <= 16:
        raise ValueError("invalid asset cardinality")
    seen = set()
    total = 0
    for asset in assets:
        path = asset.get("path", "")
        parts = PurePosixPath(path)
        url = urllib.parse.urlsplit(asset.get("url", ""))
        count = asset.get("bytes")
        if (not path.startswith("vendor/") or "\\" in path or parts.is_absolute()
                or any(value in {"", ".", ".."} for value in path.split("/")) or path in seen):
            raise ValueError("invalid or duplicate asset path")
        if url.scheme != "https" or url.hostname not in HOSTS or url.username or url.password or url.fragment:
            raise ValueError("asset origin must be an approved HTTPS publisher")
        if type(count) is not int or count <= 0 or not re.fullmatch(r"[0-9a-f]{64}", asset.get("sha256", "")):
            raise ValueError("invalid asset size or SHA256")
        total += count
        seen.add(path)
    if total > limit:
        raise ValueError("asset total exceeds fixed byte limit")
    return assets


def download(asset, output):
    target = output / asset["path"]
    if target.is_symlink():
        raise ValueError("symlink asset target is not permitted")
    if target.exists():
        if target.stat().st_size != asset["bytes"] or digest(target) != asset["sha256"]:
            raise ValueError("existing asset is not hash-identical: " + asset["path"])
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() not in target.parent.resolve().parents and target.parent.resolve() != output.resolve():
        raise ValueError("asset parent escapes owned output")
    partial = target.with_name(target.name + f".{time.time_ns()}.part")
    value = hashlib.sha256()
    written = 0
    request = urllib.request.Request(asset["url"], headers={"User-Agent": "SpatialRust-camera-assets/1"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as stream:
        if urllib.parse.urlsplit(response.geturl()).hostname not in HOSTS:
            raise ValueError("asset redirected outside approved publishers")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) != asset["bytes"]:
            raise ValueError("asset Content-Length differs from manifest")
        while True:
            block = response.read(min(1024 * 1024, asset["bytes"] - written + 1))
            if not block:
                break
            written += len(block)
            if written > asset["bytes"]:
                raise ValueError("asset body exceeds manifest size")
            value.update(block)
            stream.write(block)
    if written != asset["bytes"] or value.hexdigest() != asset["sha256"]:
        raise ValueError("downloaded asset size/SHA256 mismatch: " + asset["path"])
    partial.replace(target)


def build(output, wasm, bindgen):
    output = Path(output).absolute()
    wasm = Path(wasm).resolve()
    if output.name != "camera" or output.resolve() == SOURCE.resolve() or output.is_symlink():
        raise ValueError("use a separate owned output directory named camera")
    manifest = json.loads((SOURCE / "assets.json").read_text(encoding="utf-8"))
    assets = validate_assets(manifest)
    if not wasm.is_file():
        raise ValueError("build the Rust camera-tracking WASM module first")
    expected = subprocess.check_output([sys.executable, str(ROOT / "scripts" / "wasm_bindgen_version.py"), str(ROOT / "Cargo.lock")], text=True).strip()
    version = subprocess.check_output([bindgen, "--version"], text=True).strip()
    if version != "wasm-bindgen " + expected:
        raise ValueError("wasm-bindgen CLI does not match the resolved Rust dependency")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("pkg", "vendor"):
        if (output / name).is_symlink():
            raise ValueError("symlink output subdirectory is not permitted")
    for name in APP_FILES:
        if (output / name).is_symlink():
            raise ValueError("symlink app destination is not permitted")
        shutil.copyfile(SOURCE / name, output / name)
    for source in sorted((SOURCE.parent).glob("*.html")) + [SOURCE.parent / "styles.css"]:
        if (output.parent / source.name).is_symlink():
            raise ValueError("symlink guide destination is not permitted")
        shutil.copyfile(source, output.parent / source.name)
    subprocess.run([bindgen, "--target", "web", "--out-dir", str(output / "pkg"), str(wasm)], check=True, timeout=180)
    for asset in assets:
        download(asset, output)
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "build.json":
            files.append({"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
    if sum(value["bytes"] for value in files) > 96 * 1024 * 1024:
        raise ValueError("complete camera site exceeds 96MiB")
    report = {"schema": "spatialrust.camera-build.v1", "package_version": manifest["version"], "bindgen": version, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "files": files, "asset_bytes": sum(asset["bytes"] for asset in assets)}
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "files": len(files), "asset_bytes": report["asset_bytes"], "site_bytes": sum(value["bytes"] for value in files)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wasm", required=True)
    parser.add_argument("--wasm-bindgen", default=shutil.which("wasm-bindgen") or "wasm-bindgen")
    arguments = parser.parse_args()
    build(arguments.output, arguments.wasm, arguments.wasm_bindgen)
