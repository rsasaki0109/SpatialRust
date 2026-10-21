// Prospective in-process follow-up; retain the original four-process surveys.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
function argument(name) { const i = process.argv.indexOf(name);assert.ok(i >= 0 && process.argv[i + 1]);return resolve(process.argv[i + 1]); }
const modulePath = argument("--module"), referencePath = argument("--reference");
const { personMask: candidate } = await import(pathToFileURL(modulePath));
const { personMask: baseline } = await import(pathToFileURL(referencePath));
const original = JSON.parse(readFileSync(new URL("../benchmarks/camera-mask-resize-plan.json", import.meta.url)));
const plan = JSON.parse(readFileSync(new URL("../benchmarks/camera-mask-resize-pair-plan.json", import.meta.url)));
const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const functions = { baseline, candidate }, rows = [];
for (const [sw, sh, w, h] of original.benchmark.shapes) {
  const values = [0, .4999999701976776, .5, .5000000596046448, 1];
  const source = Float32Array.from({ length: sw * sh }, (_, i) => values[((Math.imul(i, 73) ^ (i >>> 3)) >>> 0) % values.length]);
  const sourceHash = digest(new Uint8Array(source.buffer)), expected = baseline(source, sw, sh, w, h);
  for (let i = 0; i < plan.warmup_calls_per_function_per_shape; i++) for (const fn of [baseline, candidate]) assert.deepEqual(fn(source, sw, sh, w, h), expected);
  const samples = [];
  for (let round = 0; round < plan.rounds_per_shape; round++) for (const flavor of plan.order_per_round) {
    const fn = functions[flavor], outputs = new Array(plan.calls_per_sample), start = process.hrtime.bigint();
    for (let i = 0; i < outputs.length; i++) outputs[i] = fn(source, sw, sh, w, h);
    const ns = Number(process.hrtime.bigint() - start);
    for (const output of outputs) assert.deepEqual(output, expected);
    samples.push({ round, flavor, ns });
  }
  assert.equal(digest(new Uint8Array(source.buffer)), sourceHash);
  rows.push({ source_width: sw, source_height: sh, width: w, height: h, source_sha256: sourceHash, mask_bytes: expected.length, mask_sha256: digest(expected), samples });
}
console.log(JSON.stringify({ schema: "spatialrust.camera-mask-resize-pair-benchmark.v1", synthetic_only: true, node: process.version, v8: process.versions.v8, module_sha256: digest(readFileSync(modulePath)), reference_sha256: digest(readFileSync(referencePath)), rows }));
