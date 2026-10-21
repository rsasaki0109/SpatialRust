# Renderer test runtime ownership

Native renderer tests use the existing `WgpuRuntime::shared()` fixture. They
retain distinct `WgpuRenderer` identities, buffer pools and render pipelines;
sharing the device does not make geometry interchangeable between renderers.
Production constructors, runtime lifetime, explicit transfers and public APIs
are unchanged. Wasm test compilation retains the previous owned-constructor path.

## Motivation and limits

PR 329's first Linux Visual conformance process ended with SIGSEGV after two of
five renderer test results. The unchanged-source rerun passed. The exact crashing
test, driver root cause and causal relationship to parallel initialization were
not established. The three GPU tests independently initialized and destroyed
devices even though their assertions did not require independent devices.

The fixture removes those redundant native initializations. It is not evidence
that the original driver fault has been identified or that arbitrary concurrent
owned runtime creation is safe. No production lock, backend override, implicit
copy or blanket test serialization is introduced. Runtime construction/drop
stress outside the shared fixture remains a separate qualification task.

## Parallel and real-adapter admission

A native control starts four callers behind a barrier. Each requests the same
fixture, creates its own renderer, uploads one point, explicitly submits and
waits, then recycles its geometry. The control requires identical runtime Arcs,
four different renderer identities and a 12-byte upload receipt for every caller.
Existing image bytes, point picking, topology/color modes, transfer totals,
recycling and cross-renderer rejection assertions remain intact.

Four GPU tests emit completion markers only after their assertions succeed:
`upload`, `runtime-identity`, `render` and `shared-concurrency`. The concurrent
control reports backend and device class without the adapter model name.
Ordinary library tests retain their existing no-adapter skip behavior.

Linux Visual conformance runs the complete six-test library in three separate
processes with four test threads. Each process has a 180-second limit and must
emit all four GPU completion markers. Cargo failure, a native crash, a missing
adapter, a skipped GPU path or a missing marker fails the step. There is no
automatic retry. Fresh logs are stored outside the Rust cache and uploaded even
when a repeated process fails. This is software Vulkan CI coverage, not physical
Linux GPU or performance acceptance.

## Evidence and resource contract

The contract is
`C:/Users/rsasa/Workspace/SpatialRust/benchmarks/render-test-runtime-plan.json`.
The [local receipt](receipts/2026-10-21_render_test_runtime.json) binds the
unchanged-parent five-test baseline, candidate processes, executable hashes and
related validation to their exact sources. Original PR 329 failure/retry records
are preserved. Linux repeat results are bound to the final CI source before
merge; successful repetition does not establish the cause of the prior crash.

Local checks reuse installed Rust and locked offline dependencies. Separate
baseline/candidate executables and logs live on E: under a 64MiB owned-output
limit; positive compiler-cache growth is capped at 512MiB. Frozen native examples,
the raw WASM, resolved lockfile, 493 scientific JSON and all earlier camera
evidence are hash-protected. No sensor, SDK/model or toolchain acquisition occurs.
The registered local WSL environment could not mount its virtual disk, so Linux
validation uses CI. Operational 80% qualification remains open.

The local receipt records eight terminal commands: seven successes and one
retained capacity-sampling stop while a compiler temporary directory vanished.
Only that interrupted base-feature check was repeated. A bounded sampler retry
restarts the complete cache inventory up to three times for a vanished in-cache
entry; persistent absence, paths outside that cache and permission errors still
fail. Four controls verify this admission. The production file-inventory helper
is unchanged. Local owned outputs are about **12.25MB** and positive compiler-cache
growth about **328.53MB**, within the fixed limits. Preserve **594** prior camera
files, 493 scientific JSON and 42 frozen native artifacts.
