# Isolated Gallium GLES public-data validation

Operational slice 146J. Commit this contract and
`benchmarks/tum-fr1-xyz-isolated-gallium-plan.json` before the first context
using the new driver, including any capability-only EGL probe. CPU-only source,
dependency and library preparation can precede this execution contract.

The [original explicit GL study](EXPLICIT_GLES_GPU_VALIDATION.md) preserves both
Linux device-creation failures: the selected desktop GLSL 4.20 cannot compile
wgpu's internal dynamic-array validation pipeline. Pinned Mesa 26.2.5 source
advertises GLSL feature level 460 and ESSL 310. This source observation motivates
a separate driver/WSI combination; it does not establish the actual runtime
version, successful shader execution or hardware coverage.

## Execution and ownership contract

Reuse the existing public-data executable exactly:
`E:/SpatialRustBuild/explicit-gles-v1/linux-target/release/examples/gpu_public_rgbd_conformance`,
SHA-256 `b1c23f36687d808329afda90bebfcf466cd4097501be0c7f0c49ea9c6bcc9860`,
from source `ebf117dc781500bcdacb24e1afb1f95855155740`. Keeping the executable
fixed separates this driver/WSI experiment from a new numerical implementation.
The later compiled-backend/compute preflights in the framework have separately
bound tests; do not relabel this older measured executable as their build.

Build the already checked Mesa archive and source in a new private E directory,
without another source download or source-tree copy. Keep the earlier isolated
Dozen prefix, installed system libraries and all original results unchanged.
The fixed Meson configuration enables Gallium D3D12 and the softpipe loader
support, GL/EGL/GLES, GLVND and surfaceless EGL; Vulkan, LLVM, video and driver
tests are disabled. Install only through an E-contained `--destdir`; its logical
`/opt/spatialrust-gles-v1` prefix is staged under that destination.

Two prior configurations fail before any new-driver context: the first does
not select the existing Mako/YAML tool environment, and the second lacks parser
generators. Preserve their logs and metadata. Reuse the existing Python tools
and extract checksum-pinned Ubuntu bison/flex packages only in
`E:/SpatialRustBuild/mesa-gallium-tools-v1/`. Their compressed total is 1,054,910
bytes, with 2,519,998 regular extracted bytes. Validate archive paths, skip links
and special entries, and retain package/hash identities. This performs no
system package installation; use private PATH and BISON_PKGDATADIR for the build.

Before the first new context, freeze the clean protocol commit, plan bytes,
input/executable identities, new vendor/Gallium library hashes, private vendor
selection JSON, build/tool receipts and protected old files. Select the new
libraries only in each child process with the plan's explicit GLVND/driver/library
paths, `EGL_PLATFORM=surfaceless` and `GALLIUM_DRIVER=d3d12`. The pinned driver's
dispatch returns failure when that explicit driver fails, rather than trying
another renderer. Preserve raw adapter classes and actual selected backend.

Reject inherited version, extension, forced-software or noncompliant-adapter
overrides. Do not disable wgpu validation, invent capabilities, adopt an unsafe
external context, mutate production defaults or select an alternate backend.
Classify private renderer strings without recording/publishing local model names.
Reported D3D12 selection and a capability probe cannot replace application runs
or the original hardware-class gate.

The initial contract used a hardcoded v3 logical suffix for the staged prefix.
Before any new context, derive it from the unchanged build configuration instead:
`/opt/spatialrust-gles-v1` inside the v3 DESTDIR. Preserve the initial protocol
commit; this corrects paths without a rebuild, data observation or numeric change.

## Fixed data and gates

Reuse
`E:/SpatialRustData/maturity-80/tum-fr1-xyz/gpu-conformance-input-v1/`, manifest
SHA-256 `f0c2fe90f845de3bb434b7d0e24b10b4ddfe8754bb39455841bcaa654c98a7ae`.
Do not download or copy image payloads. Attempt both high/low preferences once
with explicit `--backend gl`, preserving ordered frames 0/399/797, all process
failures, stderr and unobserved frames. Do not use the capability-probe result
to drop an unfavorable application attempt.

Keep the original independent image/point references and exact grayscale,
11x11/259x3 blur, histogram, voxel membership and radius counts. Centroid error
stays <=1e-5m; radius stays 0.03125m. All original exact transfer receipts,
zero intermediate host image reads and 64MiB retained-pool budgets are required.
Use the source-bound existing auditor with Linux/actual `Gl`/requested `gl`.

Audit operation correctness separately from both distinct raw `IntegratedGpu`
and `DiscreteGpu` classes. `Other`, CPU, virtual, duplicate classes and missing
frames cannot satisfy hardware coverage. A passing operation on raw `Other`
therefore leaves the hardware gate open. No class inference from renderer
strings, requested preferences or a source capability can promote that result.

## Storage and scope

Use at most four compiler jobs, a 4GiB new build/prefix/temp cap, a 16GiB combined
controlled-root cap and a 20GiB free-space floor. Account for existing source,
archive, tools, failed preparations and shared GPU build/cache roots, without
following WSL reparse aliases in the Windows capacity inventory. Preserve the
baseline inventory failure record as well. The private parser-tool root has a
32MiB cap. Keep original archives, wheels/extensions, old/new executables, logs,
test XMLs and scientific outputs.

Write new application outputs only in
`E:/SpatialRustData/maturity-80/tum-fr1-xyz/isolated-gallium-conformance-v1/`,
with a 128MiB cap and ten minutes per preference. Bound shader cache/temp there,
record before/after storage and protected hashes, and stop owned processes on
their bounds. This contract was committed before any new-driver result; the
separate frozen outcomes follow below.

These are three already observed development frames on WSL2 and a new
driver/WSI combination. This contract changes no framework numerical code or
CPU/GPU ownership. It does not certify bare-metal Linux, broad GL support,
performance, motion accuracy, canonical measured clock/physical calibration,
full mapping/semantic/release acceptance or the operational 80% objective.

## Frozen outcome (2026-10-20)

Initial protocol `bbac4d155eec785327ac813f3854f1cb8b30dba2` and prefix correction
`70e48c8c6bf0fb7dcc45f1d56a4b08b48388e364` both precede the first new context.
The [evidence receipt](receipts/2026-10-20_isolated_gallium_gles.json) binds the
execution freeze, input/executable/source identities, exact libraries and links,
private vendor selection, process environment, raw outputs and fresh audit.

| Preference | Actual backend/class | Observed frames | Operation gates | Two-class hardware coverage |
| --- | --- | --- | --- | --- |
| High | Gl / Other | 0/399/797, all accepted | Pass | Unaccepted |
| Low | Gl / Other | 0/399/797, all accepted | Pass | Unaccepted |

All six unchanged image/voxel/radius/transfer/pool cases pass. Maximum centroid
coordinate error is `3.817541470718666e-6 m`, below the original `1e-5 m` limit.
Both complete raw reports retain `Other`; the independent auditor accepts
operation correctness and rejects hardware coverage with no accepted raw
hardware classes. No preference or private renderer classification promotes
either run to IntegratedGpu or DiscreteGpu.

The separate capability probe observes desktop OpenGL 4.6 / GLSL 4.60 for both
3.3 and 4.3 requests, and ES 3.1 / ESSL 3.10. All three probes classify the
renderer as D3D12 with no software-renderer marker, and report 1,024 compute
invocations, 32KiB shared storage and
16 compute storage blocks, with no GL error. These capability observations
remain separate from the actual application reports. No version or extension
override is used. The combination passes on the same frozen executable that
fails with the original system driver; preserve both original failures.

CPU build preparation, compilation and E-contained DESTDIR installation finish
with all original protected files unchanged. The build-time new-root snapshot
is 184,037,018 bytes, below 4GiB. A fresh post-execution check verifies 698
protected hash entries and eight library links, before documenting outcomes.
The new regression feeds the actual passing GL reports to the existing auditor
and requires raw Other to remain outside the two-class hardware gate.
All 35 report/backend cases pass; the full Python suite passes 974 cases with
three optional ONNX skips. Python 3.8 syntax checking also passes. No framework
kernel or Rust numerical source changes; the existing native binary is retained.

Finalized application files total 462,370 bytes, below 128MiB. Preserve the
worker's earlier 306,151-byte sample as a pre-final-summary observation and bind
the separate final count in the receipt. After checking the passing 977-case
XML, exact parent and every child for reparse points, remove only
`E:/SpatialRustBuild/mesa-gallium-gles-v3/pytest-full-v1/`: 275,674,789 bytes.
The recorded controlled regular-file total is 6,907,095,864 bytes, below 16GiB;
capacity inventories omit reparse aliases while retaining
their actual owned files. No image payload is downloaded or copied. Only the
two small parser packages are fetched; all prior preparation failures, archives,
wheels/extensions, binaries, logs, XMLs and scientific records remain.

This is accepted operation on the isolated D3D12/surfaceless GL combination.
Production selection and installed system drivers retain their original state.
Both raw Linux hardware classes and the other operational 80% gates remain open.
