# Epic 83 image IO

Date: 2026-07-14

## Scope

- Canonical roadmap: `C:\Users\rsasa\Workspace\SpatialRust\docs\ROADMAP.md`
- Codec crate: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-image-io`
- Python API: `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-py`

## Reproduction

From `C:\Users\rsasa\Workspace\SpatialRust`:

```powershell
& "$env:USERPROFILE\.cargo\bin\cargo.exe" test -p spatialrust-image-io --features full
& "$env:USERPROFILE\.cargo\bin\cargo.exe" clippy -p spatialrust-image-io --features full --all-targets --no-deps -- -D warnings
& "$env:USERPROFILE\.cargo\bin\cargo.exe" bench -p spatialrust-image-io --bench decode
& "$env:USERPROFILE\.cargo\bin\cargo.exe" check --manifest-path crates/spatialrust-py/Cargo.toml
```

The decode benchmark builds deterministic RGB fixtures once, then measures PNG
and JPEG decoding at 640×480, 1920×1080, and 3840×2160. It reports pixels per
second and records the compressed fixture size in each benchmark identifier.

## Boundaries

Reader input is staged into a bounded CPU buffer because the upstream codec API
requires buffered detection and seekable decoding. Encoding similarly creates a
documented CPU staging copy. Neither operation performs an implicit device
transfer. TIFF and OpenEXR are excluded from default and standard-only builds.
