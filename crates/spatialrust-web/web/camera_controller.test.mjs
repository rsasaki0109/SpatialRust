import { test } from "node:test";
import assert from "node:assert/strict";
import { CameraSession } from "../../../docs/site/camera/session.mjs";
import { FrameRate } from "../../../docs/site/camera/frame-rate.mjs";
import { CameraProgress } from "../../../docs/site/camera/camera-progress.mjs";
import { captureBitmap } from "../../../docs/site/camera/capture-bitmap.mjs";

test("a completed bitmap remains caller-owned after its deadline or signal abort", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const c = new AbortController();let closed = 0;const bitmap = { close() { closed++; } };
  assert.equal(await captureBitmap(() => bitmap, { signal: c.signal }), bitmap);
  c.abort();t.mock.timers.tick(10000);assert.equal(closed, 0);bitmap.close();assert.equal(closed, 1);
});

test("a capture expires at exactly five seconds and closes its eventual bitmap once", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const gate = deferred();let closed = 0, done = false;
  const pending = captureBitmap(() => gate.promise);pending.then(() => { done = true; }, () => { done = true; });
  const rejected = assert.rejects(pending, { name: "FrameCaptureTimeoutError" });
  t.mock.timers.tick(4999);await Promise.resolve();assert.equal(done, false);
  t.mock.timers.tick(1);await rejected;assert.equal(done, true);
  gate.resolve({ close() { closed++; } });await Promise.resolve();assert.equal(closed, 1);t.mock.timers.tick(10000);assert.equal(closed, 1);
});

test("an already stopped capture never invokes native bitmap creation", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });const c = new AbortController();c.abort();let calls = 0;
  await assert.rejects(captureBitmap(() => { calls++; }, { signal: c.signal }), { name: "AbortError" });
  t.mock.timers.tick(10000);assert.equal(calls, 0);
});

test("Stop rejects a pending capture promptly and closes a later bitmap once", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });const c = new AbortController(), gate = deferred();let closed = 0;
  const rejected = assert.rejects(captureBitmap(() => gate.promise, { signal: c.signal }), { name: "AbortError" });
  t.mock.timers.tick(4999);c.abort();await rejected;gate.resolve({ close() { closed++; } });await Promise.resolve();
  c.abort();t.mock.timers.tick(10000);assert.equal(closed, 1);
});

test("late native rejection after timeout or Stop is consumed", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  for (const kind of ["timeout", "stop"]) {
    const c = new AbortController(), gate = deferred();const rejected = assert.rejects(captureBitmap(() => gate.promise, { signal: c.signal }));
    if (kind === "timeout") t.mock.timers.tick(5000);else c.abort();await rejected;
    gate.reject(new Error("late native failure"));await new Promise((resolve) => setImmediate(resolve));
  }
});

test("synchronous and asynchronous native failures keep their error and cancel the timer", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  for (const kind of ["sync", "async"]) {
    const failure = new Error(kind), c = new AbortController();
    await assert.rejects(captureBitmap(() => { if (kind === "sync") throw failure;return Promise.reject(failure); }, { signal: c.signal }), (error) => error === failure);
    c.abort();t.mock.timers.tick(10000);
  }
});

test("each capture gets a fresh interval independent of the previous signal and deadline", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });const first = deferred(), c = new AbortController();
  const old = captureBitmap(() => first.promise, { signal: c.signal });t.mock.timers.tick(4999);first.resolve({ close() {} });await old;
  const next = captureBitmap(() => new Promise(() => {}));const rejected = assert.rejects(next, { name: "FrameCaptureTimeoutError" });
  c.abort();t.mock.timers.tick(1);let done = false;next.catch(() => { done = true; });await Promise.resolve();assert.equal(done, false);
  t.mock.timers.tick(4998);await Promise.resolve();assert.equal(done, false);t.mock.timers.tick(1);await rejected;
});

test("capture timer admission rejects invalid durations before invoking native work", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });let calls = 0;
  for (const timeoutMs of [0, -1, 1.5, NaN, Infinity, null, "5000", 2147483648]) assert.throws(() => captureBitmap(() => { calls++; }, { timeoutMs }), RangeError);
  assert.equal(calls, 0);assert.throws(() => captureBitmap(null), TypeError);
  for (const timeoutMs of [1, 5000, 2147483647]) { const bitmap = { close() {} };assert.equal(await captureBitmap(() => bitmap, { timeoutMs }), bitmap); }
});

test("every capture completion path removes its abort listener", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  for (const kind of ["success", "failure", "timeout", "abort"]) {
    const c = new AbortController(), active = new Set(), gate = deferred(), signal = c.signal;
    const add = signal.addEventListener.bind(signal), remove = signal.removeEventListener.bind(signal);
    signal.addEventListener = (type, fn, options) => { active.add(fn);add(type, fn, options); };
    signal.removeEventListener = (type, fn) => { active.delete(fn);remove(type, fn); };
    const pending = captureBitmap(() => gate.promise, { signal });assert.equal(active.size, 1);
    const checked = kind === "success" ? pending : assert.rejects(pending);
    if (kind === "success") gate.resolve({ close() {} });else if (kind === "failure") gate.reject(new Error("native"));else if (kind === "timeout") t.mock.timers.tick(5000);else c.abort();
    await checked;assert.equal(active.size, 0);
  }
});

test("session Stop and failure abort old capture ownership while replacement stays live", async () => {
  for (const kind of ["stop", "failure"]) {
    const h = harness(), p = h.session.start("sample");h.send(h.workers[0], "ready");await p;
    const old = h.session.run;let aborted = 0;old.captureAbort.signal.addEventListener("abort", () => { aborted++; });
    if (kind === "stop") h.session.stop();else h.session.fail(old, new Error("failure"));
    const next = h.session.start("sample");h.send(h.workers[1], "ready");await next;
    assert.equal(aborted, 1);assert.equal(old.captureAbort.signal.aborted, true);assert.equal(h.session.run.captureAbort.signal.aborted, false);
    old.captureAbort.abort();assert.equal(h.session.run.captureAbort.signal.aborted, false);h.session.stop();
  }
});

test("camera progress expires at the exact five-second boundary", () => {
  const progress = new CameraProgress(100);assert.equal(progress.stalled(1, 100), false);
  assert.equal(progress.stalled(1, 5099.999), false);assert.equal(progress.stalled(1, 5100), true);
});

test("advancing video time renews progress without any pixel comparison", () => {
  const progress = new CameraProgress(0);
  for (let i = 0; i < 20; i++) assert.equal(progress.stalled(i / 24, i * 4999), false);
});

test("a video timeline reset also renews observed progress", () => {
  const progress = new CameraProgress(0);progress.stalled(120, 0);
  assert.equal(progress.stalled(0, 4999), false);assert.equal(progress.stalled(0, 9998), false);assert.equal(progress.stalled(0, 9999), true);
});

test("unusable video data cannot renew the progress clock", () => {
  const progress = new CameraProgress(0);assert.equal(progress.stalled(NaN, 4999, false), false);
  assert.equal(progress.stalled(100, 5000, false), true);
  const recovery = new CameraProgress(0);recovery.stalled(NaN, 4998, false);
  assert.equal(recovery.stalled(0, 4999), false);assert.equal(recovery.stalled(0, 9998), false);assert.equal(recovery.stalled(0, 9999), true);
});

test("the first usable frame establishes its own progress deadline", () => {
  const progress = new CameraProgress(100);assert.equal(progress.stalled(0, 1000), false);
  assert.equal(progress.stalled(0, 5999), false);assert.equal(progress.stalled(0, 6000), true);
});

test("invalid progress clocks reject before any retained state changes", () => {
  for (const value of [-1, NaN, Infinity, "0", null]) assert.throws(() => new CameraProgress(value), RangeError);
  const progress = new CameraProgress(100);progress.stalled(1, 100);
  for (const args of [[1, 99], [1, NaN], [1, Infinity], [1, "200"], [NaN, 6000], [-1, 6000], [1, 6000, "true"]]) assert.throws(() => progress.stalled(...args), RangeError);
  assert.equal(progress.stalled(1, 5099), false);assert.equal(progress.stalled(1, 5100), true);
});

test("a replacement session owns a fresh progress clock", () => {
  const old = new CameraProgress(0);old.stalled(2, 0);assert.equal(old.stalled(2, 5000), true);
  const current = new CameraProgress(10000);assert.equal(current.stalled(2, 10000), false);
  assert.equal(current.stalled(2, 14999), false);assert.equal(current.stalled(3, 15000), false);
  assert.equal(old.stalled(2, 15000), true);assert.equal(current.stalled(3, 19999), false);
});

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
function harness(acquire = async () => ({ getTracks: () => [] }), options = {}) {
  const workers = [], states = [], errors = [], results = [];
  const session = new CameraSession({ acquire, createWorker() {
    const w = { messages: [], terminated: false, disposed: false, postMessage(m) { this.messages.push(m); }, terminate() { this.terminated=true; }, dispose() { this.disposed=true; } };
    workers.push(w); return w;
  }, onState: (s) => states.push(s), onError: (e) => errors.push(e), onResult: (r) => results.push(r), timeoutMs: 5000, ...options });
  function send(w, kind, extra={}) { w.onmessage({ data: { kind, session: w.messages[0].session, frameId: w.messages.findLast((m) => m.kind === "frame")?.frameId, ...extra } }); }
  return { session, workers, states, errors, results, send };
}


// Completion must not depend on releasing either native operation.
for (const phase of ["acquire", "presenter"]) for (const action of ["stop", "timeout", "failure", "replacement"]) {
  test(`${action} settles startup while ${phase} stays pending`, async (t) => {
    t.mock.timers.enable({ apis: ["setTimeout"] });
    const gate = deferred(), entered = deferred();let stopped = 0;
    const stream = { getTracks: () => [{ stop() { stopped++; } }] };
    const h = harness(() => {
      if (phase === "acquire") { entered.resolve();return gate.promise; }
      return stream;
    }, { onReady: (run) => { if (phase === "presenter" && run.mode === "camera") { entered.resolve();return gate.promise; } } });
    const pending = h.session.start("camera");h.send(h.workers[0], "ready");await entered.promise;
    let settled = false, value;pending.then((result) => { settled = true;value = result; });
    t.mock.timers.tick(4999);await Promise.resolve();assert.equal(settled, false);
    let replacement;
    if (action === "stop") h.session.stop();
    else if (action === "timeout") t.mock.timers.tick(1);
    else if (action === "failure") h.send(h.workers[0], "error", { message: "startup failed" });
    else { replacement = h.session.start("sample");h.send(h.workers[1], "ready");assert.equal(await replacement, true); }
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(settled, true, "cancelled start remained pending");assert.equal(value, false);
    assert.equal(h.workers[0].terminated, true);assert.equal(h.workers[0].disposed, true);
    assert.equal(h.errors.length, ["timeout", "failure"].includes(action) ? 1 : 0);
    const current = h.session.run;gate.resolve(phase === "acquire" ? stream : undefined);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(stopped, 1);assert.equal(h.session.run, current);assert.equal(await pending, false);
    if (action === "replacement") { assert.equal(current.started, true);assert.equal(h.workers[1].terminated, false); }
    else assert.equal(h.session.run, null);
    h.session.stop();t.mock.timers.tick(30000);
    assert.equal(h.errors.length, ["timeout", "failure"].includes(action) ? 1 : 0);
  });
}

for (const phase of ["acquire", "presenter"]) test(`late ${phase} rejection is consumed after cancelled startup settles`, async () => {
  const gate = deferred(), entered = deferred();
  const h = harness(() => {
    if (phase === "acquire") { entered.resolve();return gate.promise; }
    return { getTracks: () => [] };
  }, { onReady: (run) => { if (phase === "presenter" && run.mode === "camera") { entered.resolve();return gate.promise; } } });
  const pending = h.session.start("camera");h.send(h.workers[0], "ready");await entered.promise;h.session.stop();
  let settled = false;pending.then(() => { settled = true; });await new Promise((resolve) => setImmediate(resolve));assert.equal(settled, true);
  const next = h.session.start("sample");h.send(h.workers[1], "ready");assert.equal(await next, true);
  gate.reject(new Error("late native rejection"));await new Promise((resolve) => setImmediate(resolve));
  assert.equal(h.errors.length, 0);assert.equal(h.session.run.started, true);assert.equal(h.workers[1].terminated, false);h.session.stop();
});

test("synchronous worker creation failure settles startup and allows a fresh run", async () => {
  const h = harness(), create = h.session.createWorker;
  h.session.createWorker = () => { throw new Error("worker creation failed"); };
  assert.equal(await h.session.start("camera"), false);assert.equal(h.session.run, null);assert.equal(h.errors.length, 1);
  h.session.createWorker = create;const next = h.session.start("sample");h.send(h.workers[0], "ready");assert.equal(await next, true);h.session.stop();
});

for (const mode of ["camera", "sample"]) test(`${mode} init-send failure is fully observed and releases the worker`, async () => {
  let acquired = 0;const h = harness(async () => { acquired++;return { getTracks: () => [] }; });
  const create = h.session.createWorker, failure = new DOMException("cannot clone initialization", "DataCloneError");
  h.session.createWorker = () => { const worker = create();worker.postMessage = () => { throw failure; };return worker; };
  assert.equal(await h.session.start(mode), false);await settleSource();
  const old = h.workers[0];assert.equal(old.terminated, true);assert.equal(old.disposed, true);
  assert.equal(h.session.run, null);assert.equal(acquired, 0);assert.deepEqual(h.errors, [failure]);
  h.session.createWorker = create;const next = h.session.start("sample");h.send(h.workers[1], "ready");assert.equal(await next, true);
  old.onerror({ message: "late failed initialization" });old.onmessageerror();await settleSource();
  assert.equal(h.errors.length, 1);assert.equal(h.workers[1].terminated, false);h.session.stop();
});

for (const phase of ["waiting-ready", "pending-camera", "pending-presenter", "pending-result", "pending-capture"]) {
  test(`messageerror cleans ${phase} and cannot affect its replacement`, async (t) => {
    t.mock.timers.enable({ apis: ["setTimeout"] });
    const source = cameraSource(), gate = deferred(), entered = deferred();let closed = 0;
    const h = harness(() => {
      if (phase === "pending-camera") { entered.resolve();return gate.promise; }
      return source.stream;
    }, { onReady: (run) => { if (phase === "pending-presenter" && run.mode === "camera") { entered.resolve();return gate.promise; } } });
    const pending = h.session.start("camera"), run = h.session.run, old = h.workers[0];
    if (phase === "waiting-ready") await settleSource();
    else {
      h.send(old, "ready");
      if (["pending-camera", "pending-presenter"].includes(phase)) await entered.promise;
      else assert.equal(await pending, true);
    }
    let capture;
    if (phase === "pending-result") assert.equal(h.session.submit({ close() {} }, 1), true);
    if (phase === "pending-capture") capture = assert.rejects(captureBitmap(() => gate.promise, { signal: run.captureAbort.signal }), { name: "AbortError" });
    old.onmessageerror();await capture;await settleSource();
    assert.equal(await pending, ["pending-result", "pending-capture"].includes(phase));
    assert.equal(h.session.run, null);assert.equal(run.captureAbort.signal.aborted, true);
    assert.equal(old.terminated, true);assert.equal(old.disposed, true);assert.equal(h.errors.length, 1);assert.equal(h.errors[0].name, "WorkerMessageError");
    if (phase !== "pending-camera") assert.ok(source.tracks.every((track) => track.stops === 1));
    assert.ok(source.signals.every((signal) => signal.aborted));
    const next = h.session.start("sample");h.send(h.workers[1], "ready");assert.equal(await next, true);
    const current = h.session.run;old.onmessageerror();old.onerror({ message: "late error" });h.send(old, "ready");h.send(old, "result");
    gate.resolve(phase === "pending-camera" ? source.stream : phase === "pending-capture" ? { close() { closed++; } } : undefined);
    await settleSource();t.mock.timers.tick(90000);await settleSource();
    assert.equal(h.session.run, current);assert.equal(h.workers[1].terminated, false);assert.equal(h.errors.length, 1);assert.equal(h.results.length, 0);
    assert.ok(source.tracks.every((track) => track.stops === 1));assert.equal(closed, phase === "pending-capture" ? 1 : 0);h.session.stop();
  });
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

test("a sample frame expires exactly fifteen seconds after submission, never while idle", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const h = harness(), p = h.session.start("sample", false);h.send(h.workers[0], "ready");await p;
  t.mock.timers.tick(30000);assert.equal(h.session.run.started, true);assert.equal(h.errors.length, 0);
  h.session.submit({ close() {} }, 1);t.mock.timers.tick(14999);assert.equal(h.session.busy, true);assert.equal(h.errors.length, 0);
  t.mock.timers.tick(1);assert.equal(h.session.run, null);assert.equal(h.workers[0].terminated, true);assert.equal(h.workers[0].disposed, true);
  assert.equal(h.errors.length, 1);assert.equal(h.errors[0].name, "FrameResultTimeoutError");t.mock.timers.tick(30000);assert.equal(h.errors.length, 1);
});

test("a missing camera result stops all tracks and removes source listeners", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const s = cameraSource(), h = harness(async () => s.stream), p = h.session.start("camera");h.send(h.workers[0], "ready");await p;
  h.session.submit({ close() {} }, 1);t.mock.timers.tick(15000);
  assert.ok(s.tracks.every((track) => track.stops === 1));assert.ok(s.signals.every((signal) => signal.aborted));
  assert.equal(h.workers[0].terminated, true);assert.equal(h.errors[0].name, "FrameResultTimeoutError");
});

test("a matching result cancels its deadline and the next frame gets a full interval", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const h = harness(), p = h.session.start("sample");h.send(h.workers[0], "ready");await p;
  h.session.submit({ close() {} }, 1);t.mock.timers.tick(14999);h.send(h.workers[0], "result");assert.equal(h.session.busy, false);
  h.session.submit({ close() {} }, 2);t.mock.timers.tick(1);assert.equal(h.errors.length, 0);assert.equal(h.session.busy, true);
  t.mock.timers.tick(14998);assert.equal(h.errors.length, 0);t.mock.timers.tick(1);assert.equal(h.errors.length, 1);
});

test("Stop cancels the pending deadline and a replacement gets its own timer", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const h = harness(), p = h.session.start("sample");h.send(h.workers[0], "ready");await p;
  h.session.submit({ close() {} }, 1);t.mock.timers.tick(14999);h.session.stop();
  const next = h.session.start("sample");h.send(h.workers[1], "ready");await next;h.session.submit({ close() {} }, 2);
  t.mock.timers.tick(1);assert.equal(h.errors.length, 0);assert.equal(h.workers[1].terminated, false);
  t.mock.timers.tick(14999);assert.equal(h.errors.length, 1);assert.equal(h.workers[1].terminated, true);
});

test("an expired session result cannot complete a replacement frame with the same sequence", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const h = harness(), p = h.session.start("sample");h.send(h.workers[0], "ready");await p;
  h.session.submit({ close() {} }, 1);t.mock.timers.tick(15000);
  const next = h.session.start("sample");h.send(h.workers[1], "ready");await next;h.session.submit({ close() {} }, 2);
  h.send(h.workers[0], "result");assert.equal(h.results.length, 0);assert.equal(h.session.busy, true);
  t.mock.timers.tick(15000);assert.equal(h.errors.length, 2);
});

test("unsolicited, duplicate and obsolete-frame results cannot release current ownership", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const h = harness(), p = h.session.start("sample");h.send(h.workers[0], "ready");await p;
  h.send(h.workers[0], "result", { frameId: 1 });assert.equal(h.results.length, 0);
  h.session.submit({ close() {} }, 1);assert.equal(h.workers[0].messages.at(-1).frameId, 1);
  for (const frameId of [undefined, null, "1", 0, 2]) h.send(h.workers[0], "result", { frameId });
  assert.equal(h.results.length, 0);assert.equal(h.session.busy, true);
  h.send(h.workers[0], "result");h.send(h.workers[0], "result");assert.equal(h.results.length, 1);
  h.session.submit({ close() {} }, 2);assert.equal(h.workers[0].messages.at(-1).frameId, 2);
  t.mock.timers.tick(14999);h.send(h.workers[0], "result", { frameId: 1 });assert.equal(h.results.length, 1);assert.equal(h.session.busy, true);
  t.mock.timers.tick(1);assert.equal(h.errors.length, 1);
});

test("worker and presentation errors each cancel their pending result timer", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  for (const kind of ["worker", "presentation"]) {
    const h = harness(), p = h.session.start("sample");h.send(h.workers[0], "ready");await p;h.session.submit({ close() {} }, 1);
    if (kind === "worker") h.send(h.workers[0], "error", { message: "failure" });
    else { h.session.onResult = () => { throw new Error("presentation"); };h.send(h.workers[0], "result"); }
    assert.equal(h.errors.length, 1);assert.equal(h.workers[0].disposed, true);t.mock.timers.tick(30000);assert.equal(h.errors.length, 1);
  }
});

test("a transfer error clears the newly installed result timer", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const h = harness(), p = h.session.start("sample");h.send(h.workers[0], "ready");await p;
  h.workers[0].postMessage = () => { throw new Error("transfer"); };let closed = 0;
  assert.equal(h.session.submit({ close() { closed++; } }, 1), false);assert.equal(closed, 1);assert.equal(h.errors.length, 1);
  t.mock.timers.tick(30000);assert.equal(h.errors.length, 1);assert.equal(h.session.run, null);
});

test("result timeout configuration rejects clamped or disabled timer durations", () => {
  for (const resultTimeoutMs of [0, -1, 1.5, NaN, Infinity, "15000", null, 2147483648]) assert.throws(() => harness(undefined, { resultTimeoutMs }), RangeError);
  for (const resultTimeoutMs of [1, 15000, 2147483647]) assert.equal(harness(undefined, { resultTimeoutMs }).session.resultTimeoutMs, resultTimeoutMs);
});

test("initialization time does not consume a submitted frame's result deadline", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const h = harness(), p = h.session.start("sample");t.mock.timers.tick(4999);h.send(h.workers[0], "ready");await p;
  h.session.submit({ close() {} }, 1);t.mock.timers.tick(14999);assert.equal(h.errors.length, 0);
  h.send(h.workers[0], "result");t.mock.timers.tick(30000);assert.equal(h.errors.length, 0);assert.equal(h.results.length, 1);h.session.stop();
});

function cameraSource() {
  const signals = [], callbacks = [];
  const tracks = ["video", "audio"].map((kind) => {
    const track = new EventTarget();track.kind = kind;track.readyState = "live";track.stops = 0;
    track.stop = () => { track.stops++;track.readyState = "ended"; };
    const add = track.addEventListener.bind(track);
    track.addEventListener = (kind, fn, options) => { signals.push(options.signal);callbacks.push(fn);add(kind, fn, options); };
    return track;
  });
  const stream = new EventTarget();stream.getTracks = () => tracks;
  Object.defineProperty(stream, "active", { configurable: true, get: () => tracks.some((t) => t.readyState === "live") });
  return { stream, video: tracks[0], audio: tracks[1], tracks, signals, callbacks };
}
const settleSource = () => new Promise((resolve) => setImmediate(resolve));

test("ended video releases the current worker and every owned track", async () => {
  const s = cameraSource(), h = harness(async () => s.stream), p = h.session.start("camera");
  h.send(h.workers[0], "ready");assert.equal(await p, true);
  s.video.readyState = "ended";s.video.dispatchEvent(new Event("ended"));
  assert.equal(h.session.run, null);assert.equal(h.workers[0].terminated, true);assert.equal(h.workers[0].disposed, true);
  assert.ok(s.tracks.every((t) => t.stops === 1));assert.ok(s.signals.every((signal) => signal.aborted));
  assert.equal(h.errors.length, 1);assert.equal(h.errors[0].name, "CameraEndedError");
});

test("an inactive source releases its worker even with live track metadata", async () => {
  const s = cameraSource(), h = harness(async () => s.stream), p = h.session.start("camera");h.send(h.workers[0], "ready");await p;
  Object.defineProperty(s.stream, "active", { get: () => false });s.stream.dispatchEvent(new Event("inactive"));
  assert.equal(h.session.run, null);assert.equal(h.errors[0].name, "CameraEndedError");assert.ok(s.tracks.every((t) => t.stops === 1));
});

test("already ended or inactive acquisition cannot reach onReady", async () => {
  for (const condition of ["ended", "inactive"]) {
    const s = cameraSource();if (condition === "ended") s.video.readyState = "ended";
    else Object.defineProperty(s.stream, "active", { get: () => false });
    const h = harness(async () => s.stream);let ready = 0;h.session.onReady = () => { ready++; };
    const p = h.session.start("camera");h.send(h.workers[0], "ready");assert.equal(await p, false);
    assert.equal(ready, 0);assert.equal(h.workers[0].terminated, true);assert.equal(h.errors.length, 1);
  }
});

test("source termination cancels pending worker initialization", async () => {
  const s = cameraSource(), h = harness(async () => s.stream), p = h.session.start("camera");await settleSource();
  assert.equal(h.session.run.stream, s.stream);s.video.readyState = "ended";s.video.dispatchEvent(new Event("ended"));
  assert.equal(await p, false);assert.equal(h.workers[0].terminated, true);assert.equal(h.errors.length, 1);
});

test("a source ending during onReady cannot finish startup", async () => {
  const s = cameraSource(), h = harness(async () => s.stream), gate = deferred();h.session.onReady = () => gate.promise;
  const p = h.session.start("camera");h.send(h.workers[0], "ready");await settleSource();
  s.video.readyState = "ended";s.video.dispatchEvent(new Event("ended"));assert.equal(h.workers[0].terminated, true);
  gate.resolve();assert.equal(await p, false);assert.notEqual(h.states.at(-1), "running");
});

test("stopped source callbacks and state checks cannot affect a replacement", async () => {
  const s = cameraSource(), h = harness(async () => s.stream), p = h.session.start("camera");h.send(h.workers[0], "ready");await p;
  const old = h.session.run;h.session.stop();const sample = h.session.start("sample", false);h.send(h.workers[1], "ready");await sample;
  s.video.dispatchEvent(new Event("ended"));s.stream.dispatchEvent(new Event("inactive"));for (const callback of s.callbacks) callback();
  assert.equal(h.session.checkCamera(old), false);assert.equal(h.session.run.mode, "sample");assert.equal(h.errors.length, 0);
  assert.equal(h.workers[1].terminated, false);assert.ok(s.signals.every((signal) => signal.aborted));h.session.stop();
});

test("state polling clears a silently ended video while a result is pending", async () => {
  const s = cameraSource(), h = harness(async () => s.stream), p = h.session.start("camera");h.send(h.workers[0], "ready");await p;
  const run = h.session.run;h.session.submit({ close() {} }, 1);assert.equal(h.session.busy, true);
  s.video.readyState = "ended";assert.equal(h.session.checkCamera(run), false);
  assert.equal(h.session.run, null);assert.equal(h.workers[0].terminated, true);assert.equal(h.errors.length, 1);
});

test("an audio-only ended event does not stop a healthy video source", async () => {
  const s = cameraSource(), h = harness(async () => s.stream), p = h.session.start("camera");h.send(h.workers[0], "ready");await p;
  s.audio.readyState = "ended";s.audio.dispatchEvent(new Event("ended"));
  assert.equal(h.session.checkCamera(h.session.run), true);assert.equal(h.errors.length, 0);assert.equal(h.workers[0].terminated, false);h.session.stop();
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
