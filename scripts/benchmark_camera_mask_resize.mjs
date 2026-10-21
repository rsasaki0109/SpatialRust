// Fixed synthetic mask-only workload; no camera, SDK, GPU or asset acquisition.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

function argument(name) { const i = process.argv.indexOf(name);assert.ok(i >= 0 && process.argv[i + 1]);return resolve(process.argv[i + 1]); }
const modulePath = argument("--module"), referencePath = argument("--reference");
const { personMask } = await import(pathToFileURL(modulePath));
const { personMask: reference } = await import(pathToFileURL(referencePath));
const plan = JSON.parse(readFileSync(new URL("../benchmarks/camera-mask-resize-plan.json", import.meta.url)));
const rows = [];
for (const [sw, sh, w, h] of plan.benchmark.shapes) {
  const values = [0, .4999999701976776, .5, .5000000596046448, 1];
  const source = Float32Array.from({ length: sw * sh }, (_, i) => values[((Math.imul(i, 73) ^ (i >>> 3)) >>> 0) % values.length]);
  const sourceHash = createHash("sha256").update(new Uint8Array(source.buffer)).digest("hex");
  const expected = reference(source, sw, sh, w, h);
  for (let i = 0; i < plan.benchmark.warmup_calls; i++) assert.deepEqual(personMask(source, sw, sh, w, h), expected);
  const samples = [];
  for (let sample = 0; sample < plan.benchmark.samples; sample++) {
    const outputs = new Array(plan.benchmark.calls_per_sample);
    const start = process.hrtime.bigint();
    for (let i = 0; i < outputs.length; i++) outputs[i] = personMask(source, sw, sh, w, h);
    samples.push(Number(process.hrtime.bigint() - start));
    for (const output of outputs) assert.deepEqual(output, expected);
  }
  assert.equal(createHash("sha256").update(new Uint8Array(source.buffer)).digest("hex"), sourceHash);
  rows.push({ source_width: sw, source_height: sh, width: w, height: h, source_sha256: sourceHash, mask_bytes: expected.length, mask_sha256: createHash("sha256").update(expected).digest("hex"), samples_ns: samples });
}
console.log(JSON.stringify({ schema: "spatialrust.camera-mask-resize-benchmark.v1", synthetic_only: true, node: process.version, v8: process.versions.v8, module_sha256: createHash("sha256").update(readFileSync(modulePath)).digest("hex"), reference_sha256: createHash("sha256").update(readFileSync(referencePath)).digest("hex"), warmup_calls: plan.benchmark.warmup_calls, calls_per_sample: plan.benchmark.calls_per_sample, rows }));
