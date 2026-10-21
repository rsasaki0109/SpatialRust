import { test } from "node:test";
import assert from "node:assert/strict";
import { CameraSession } from "../../../docs/site/camera/session.mjs";
import { FrameRate } from "../../../docs/site/camera/frame-rate.mjs";

test("frame rate starts without an invented rate and averages image intervals", () => {
  const rate = new FrameRate();assert.equal(rate.observe(0), null);
  assert.equal(rate.observe(125), 8);assert.equal(rate.observe(250), 8);
  assert.equal(rate.observe(500), 6);
});

test("frame rate uses an inclusive two-second image window", () => {
  const rate = new FrameRate();rate.observe(0);rate.observe(1000);
  assert.equal(rate.observe(2000), 1);
  assert.equal(rate.observe(2001), 2000 / 1001);
});

test("frame rate retains at most seventeen timestamps", () => {
  const rate = new FrameRate();for (const t of [0, 100, 200, 300]) rate.observe(t);
  for (let t = 301; t < 317; t++) rate.observe(t);
  assert.equal(rate.observe(317), 1000);
});

test("a long image gap and an explicit reset need fresh rate samples", () => {
  const rate = new FrameRate();rate.observe(10);rate.observe(135);
  assert.equal(rate.observe(3000), null);assert.equal(rate.observe(3250), 4);
  rate.reset();assert.equal(rate.observe(1), null);assert.equal(rate.observe(501), 2);
});

test("invalid frame clocks reject without changing retained rate state", () => {
  const rate = new FrameRate();rate.observe(0);rate.observe(100);
  for (const t of [NaN, Infinity, -Infinity, -1, 0, 50, 100, "200", null, undefined]) assert.throws(() => rate.observe(t), RangeError);
  assert.equal(rate.observe(200), 10);
});

test("frame rate never reports an infinite numerical result", () => {
  const rate = new FrameRate();rate.observe(0);
  assert.equal(rate.observe(Number.MIN_VALUE), null);
  assert.equal(rate.observe(Number.MAX_VALUE), null);
});

function deferred() { let resolve, reject; const promise = new Promise((a,b) => { resolve=a; reject=b; }); return { promise, resolve, reject }; }
function harness(acquire = async () => ({ getTracks: () => [] })) {
  const workers = [], states = [], errors = [], results = [];
  const session = new CameraSession({ acquire, createWorker() {
    const w = { messages: [], terminated: false, disposed: false, postMessage(m) { this.messages.push(m); }, terminate() { this.terminated=true; }, dispose() { this.disposed=true; } };
    workers.push(w); return w;
  }, onState: (s) => states.push(s), onError: (e) => errors.push(e), onResult: (r) => results.push(r), timeoutMs: 5000 });
  function send(w, kind, extra={}) { w.onmessage({ data: { kind, session: w.messages[0].session, ...extra } }); }
  return { session, workers, states, errors, results, send };
}

test("sample mode never requests a camera", async () => {
  let requested=0;const h=harness(async () => { requested++; throw new Error("camera"); });
  const p=h.session.start("sample",false);h.send(h.workers[0],"ready",{backend:"OFF"});assert.equal(await p,true);assert.equal(requested,0);h.session.stop();
});
test("late camera permission is stopped after cancellation", async () => {
  const gate=deferred();let stopped=0;const h=harness(() => gate.promise);const p=h.session.start("camera");
  await Promise.resolve();h.session.stop();gate.resolve({getTracks:()=>[{stop(){stopped++;}}]});assert.equal(await p,false);await Promise.resolve();assert.equal(stopped,1);assert.equal(h.workers[0].terminated,true);
});
test("camera denial terminates pending worker initialization", async () => {
  const h=harness(async () => { throw new Error("denied"); });assert.equal(await h.session.start("camera"),false);assert.equal(h.workers[0].terminated,true);assert.equal(h.errors.length,1);assert.equal(h.session.token,0);
});
test("worker failure stops an already acquired stream", async () => {
  let stopped=0;const h=harness(async()=>({getTracks:()=>[{stop(){stopped++;}}]}));const p=h.session.start("camera");await Promise.resolve();await Promise.resolve();h.send(h.workers[0],"error",{message:"model unavailable"});assert.equal(await p,false);assert.equal(stopped,1);
});
test("old messages do not affect a newer session", async () => {
  const h=harness();const a=h.session.start("sample");const old=h.workers[0];const b=h.session.start("sample");h.send(old,"ready");h.send(old,"error",{message:"late"});h.send(h.workers[1],"ready",{backend:"CPU"});assert.equal(await a,false);assert.equal(await b,true);assert.equal(h.errors.length,0);h.session.stop();
});
test("only one bitmap is in flight and dropped bitmaps close", async () => {
  const h=harness();const p=h.session.start("sample");h.send(h.workers[0],"ready");await p;let closed=0;
  const first={close(){closed++;}},second={close(){closed++;}};assert.equal(h.session.submit(first,1),true);assert.equal(h.session.submit(second,2),false);assert.equal(closed,1);h.send(h.workers[0],"result",{report:{}});assert.equal(h.session.busy,false);h.session.stop();
});
test("stop releases worker and every stream track exactly once", async () => {
  let stopped=0;const h=harness(async()=>({getTracks:()=>[{stop(){stopped++;}},{stop(){stopped++;}}]}));const p=h.session.start("camera");h.send(h.workers[0],"ready");await p;h.session.stop();h.session.stop();assert.equal(stopped,2);assert.equal(h.workers[0].disposed,true);
});
test("a failed bitmap transfer stops the session", async () => {
  const h=harness();const p=h.session.start("sample");h.send(h.workers[0],"ready");await p;h.workers[0].postMessage=()=>{throw new Error("transfer");};let closed=0;assert.equal(h.session.submit({close(){closed++;}},1),false);assert.equal(closed,1);assert.equal(h.errors.length,1);
});
test("late onReady completion cannot restart after stop", async () => {
  const h=harness();const gate=deferred();h.session.onReady=()=>gate.promise;const p=h.session.start("sample");h.send(h.workers[0],"ready");await Promise.resolve();await Promise.resolve();h.session.stop();gate.resolve();assert.equal(await p,false);assert.notEqual(h.states.at(-1),"running");
});

test("stop before the acquisition microtask does not prompt for a camera", async () => {
 let calls=0;const h=harness(async()=>{calls++;return{getTracks:()=>[]};});const p=h.session.start("camera");h.session.stop();assert.equal(await p,false);assert.equal(calls,0);
});
test("worker failure after ready also cancels an unfinished onReady", async () => {
 const h=harness();const gate=deferred();h.session.onReady=()=>gate.promise;const p=h.session.start("sample");h.send(h.workers[0],"ready");await Promise.resolve();await Promise.resolve();h.send(h.workers[0],"error",{message:"late startup"});gate.resolve();assert.equal(await p,false);assert.equal(h.errors.length,1);assert.equal(h.session.token,0);
});

// Confidence-to-binary conversion is independent of SDK-specific category IDs.
import { personMask } from "../../../docs/site/camera/person-mask.mjs";
test("person confidence uses a strict finite half threshold", () => {
  assert.deepEqual([...personMask(new Float32Array([0, .49, .5, .51, 1]), 5, 1, 5, 1)], [0, 0, 0, 1, 1]);
});
test("person confidence resamples rows and columns without transposition", () => {
  assert.deepEqual([...personMask(new Float32Array([1, 0, 0, 1]), 2, 2, 4, 2)], [1, 1, 0, 0, 0, 0, 1, 1]);
});
test("person confidence rejects malformed shape and nonfinite values", () => {
  for (const values of [[NaN], [Infinity], [-.1], [1.1]]) assert.throws(() => personMask(new Float32Array(values), 1, 1, 1, 1));
  for (const dimensions of [[0, 1, 1, 1], [1, 1, 321, 1], [1.5, 1, 1, 1], [2, 1, 1, 1]]) assert.throws(() => personMask(new Float32Array([0]), ...dimensions));
  assert.throws(() => personMask(new Uint8Array([1]), 1, 1, 1, 1));
});

import { cameraFrameSize } from "../../../docs/site/camera/frame-size.mjs";
test("camera dimensions follow landscape and portrait source coordinates", () => {
  assert.deepEqual(cameraFrameSize(640, 480), { width: 256, height: 192 });
  assert.deepEqual(cameraFrameSize(480, 640), { width: 192, height: 256 });
});
test("camera dimensions preserve widescreen and square proportions", () => {
  assert.deepEqual(cameraFrameSize(1920, 1080), { width: 256, height: 144 });
  assert.deepEqual(cameraFrameSize(1080, 1920), { width: 144, height: 256 });
  assert.deepEqual(cameraFrameSize(720, 720), { width: 256, height: 256 });
});
test("camera dimensions round pixel coordinates consistently", () => {
  assert.deepEqual(cameraFrameSize(1001, 777), { width: 256, height: 199 });
  assert.deepEqual(cameraFrameSize(777, 1001), { width: 199, height: 256 });
});
test("camera dimensions admit the minimum tracker side", () => {
  assert.deepEqual(cameraFrameSize(1024, 256), { width: 256, height: 64 });
  assert.deepEqual(cameraFrameSize(256, 1024), { width: 64, height: 256 });
});
test("camera dimensions reject invalid or unsafe intrinsic numbers", () => {
  for (const n of [0, -1, NaN, Infinity, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => cameraFrameSize(n, 480), RangeError);
    assert.throws(() => cameraFrameSize(640, n), RangeError);
  }
});
test("camera dimensions reject extreme ratios without stretching", () => {
  for (const pair of [[1000, 100], [100, 1000], [256, 63], [63, 256]]) assert.throws(() => cameraFrameSize(...pair), RangeError);
});
test("camera dimensions remain bounded for large valid metadata", () => {
  assert.deepEqual(cameraFrameSize(Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER), { width: 256, height: 256 });
});

test("light preset preserves camera and sample proportions in both orientations", () => {
  for (const [w, h, expected] of [
    [640, 480, { width: 192, height: 144 }], [480, 640, { width: 144, height: 192 }],
    [1920, 1080, { width: 192, height: 108 }], [1080, 1920, { width: 108, height: 192 }],
    [4, 3, { width: 192, height: 144 }], [720, 720, { width: 192, height: 192 }],
  ]) assert.deepEqual(cameraFrameSize(w, h, 192), expected);
});

test("light preset rounds and admits the minimum tracker side", () => {
  assert.deepEqual(cameraFrameSize(1001, 777, 192), { width: 192, height: 149 });
  assert.deepEqual(cameraFrameSize(777, 1001, 192), { width: 149, height: 192 });
  assert.deepEqual(cameraFrameSize(3, 1, 192), { width: 192, height: 64 });
  assert.deepEqual(cameraFrameSize(1, 3, 192), { width: 64, height: 192 });
  assert.deepEqual(cameraFrameSize(Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER, 192), { width: 192, height: 192 });
});

test("light preset rejects unsupported aspect without stretching", () => {
  for (const pair of [[4, 1], [1, 4], [192, 63], [63, 192]]) assert.throws(() => cameraFrameSize(...pair, 192), RangeError);
  for (const n of [0, -1, NaN, Infinity, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => cameraFrameSize(n, 480, 192), RangeError);
    assert.throws(() => cameraFrameSize(640, n, 192), RangeError);
  }
});

test("only explicit bounded processing presets are admitted", () => {
  for (const n of [0, 64, 128, 193, 320, NaN, Infinity, null, "192", {}]) assert.throws(() => cameraFrameSize(640, 480, n), RangeError);
  const light = cameraFrameSize(640, 480, 192), standard = cameraFrameSize(640, 480);
  assert.equal(light.width * light.height, 27648);
  assert.equal(1 - light.width * light.height / (standard.width * standard.height), .4375);
});
