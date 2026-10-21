import { test } from "node:test";
import assert from "node:assert/strict";
import { CameraSession } from "../../../docs/site/camera/session.mjs";

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
