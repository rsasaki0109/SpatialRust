// Real browser/WASM/model checks using synthetic capture only. Never opens a real camera.
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const assert = require("node:assert/strict");
const { chromium } = require("playwright");

function argument(name) { const i = process.argv.indexOf(name); if (i < 0 || !process.argv[i + 1]) throw new Error(`missing ${name}`); return path.resolve(process.argv[i + 1]); }
const site = argument("--site"), output = argument("--output");
const caseIndex = process.argv.indexOf("--regression-case"), regressionCase = caseIndex < 0 ? null : process.argv[caseIndex + 1];
if (regressionCase && !["startup", "rotation", "cadence", "rate", "source-ended", "source-inactive", "source-silent", "progress-idle", "progress-busy"].includes(regressionCase)) throw new Error("unknown regression case");
if (fs.existsSync(output)) throw new Error("fresh browser evidence directory required");
fs.mkdirSync(output, { recursive: true });
const report = { schema: "spatialrust.camera-browser.v1", syntheticCaptureOnly: true, physicalPhoneTested: false, checks: [], screenshots: [], externalResponses: [], nonGetRequests: [], consoleErrors: [], failedRequests: [], httpErrors: [], workerErrors: [], cspViolations: [] };
const types = { ".html": "text/html; charset=utf-8", ".css": "text/css", ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".txt": "text/plain" };
const server = http.createServer((req, res) => {
  if (req.method !== "GET" && req.method !== "HEAD") { report.nonGetRequests.push(req.method);res.writeHead(405);res.end();return; }
  try {
    let relative = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    if (relative === "/favicon.ico") { res.writeHead(204);res.end();return; }
    let file = path.resolve(site, "." + relative);
    if (file !== site && !file.startsWith(site + path.sep)) { res.writeHead(403);res.end();return; }
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
    if (!fs.existsSync(file)) { res.writeHead(404);res.end();return; }
    const real = fs.realpathSync(file);
    if (!real.startsWith(fs.realpathSync(site) + path.sep)) throw new Error("escaped site");
    res.writeHead(200, { "Content-Type": types[path.extname(file)] ?? "application/octet-stream", "Content-Length": fs.statSync(file).size, "Cache-Control": "no-cache" });
    if (req.method === "HEAD") res.end(); else fs.createReadStream(file).pipe(res);
  } catch { res.writeHead(500);res.end(); }
});

let browser, activePage;
const check = (name) => report.checks.push(name);
async function waitFrames(page, before = 0) {
  await page.waitForFunction((n) => document.documentElement.dataset.sessionState === "error" || Number(document.documentElement.dataset.trackedFrames ?? 0) > n, before, { timeout: 120000 });
  assert.notEqual(await page.evaluate(() => document.documentElement.dataset.sessionState), "error", await page.locator("#status").textContent());
}
const frameCount = (page) => page.evaluate(() => Number(document.documentElement.dataset.trackedFrames ?? 0));
async function screenshot(page, name) { const file = path.join(output, name + ".png"); await page.screenshot({ path: file, fullPage: true });report.screenshots.push(file); }
async function stopped(page) {
  await page.waitForFunction(() => document.documentElement.dataset.sessionState === "idle");
  assert.equal(await page.evaluate(() => document.querySelector("video").srcObject), null);
  assert.equal(await page.evaluate(() => window.__cameraStreams.every((s) => s.getTracks().every((t) => t.readyState === "ended"))), true);
}
async function instrument(context) {
  await context.addInitScript(() => {
    window.__workerErrors = [];
    window.__workerCount = 0;
    window.__terminatedWorkers = 0;
    window.__cameraReports = [];
    window.__holdNextResult = false;
    window.__submittedFrames = [];
    window.__timedSubmissions = [];
    window.__cspViolations = [];
    document.addEventListener("securitypolicyviolation", (e) => window.__cspViolations.push({ directive: e.effectiveDirective, uri: e.blockedURI }));
    const OriginalWorker = window.Worker;
    window.Worker = class extends OriginalWorker {
      constructor(...args) { super(...args); window.__workerCount++;
        this.addEventListener("message", (event) => {
          if (event.data.kind !== "result") return;
          const r = event.data.report;
          window.__cameraReports.push({ width: r.width, height: r.height, status: r.status, motion: r.background_flow_px, maskBytes: event.data.mask.byteLength, timestampMs: r.timestamp_ms });
          if (window.__cameraReports.length > 32) window.__cameraReports.shift();
          if (window.__holdNextResult) {
            window.__holdNextResult = false; event.stopImmediatePropagation();
            window.__heldResult = () => this.dispatchEvent(new MessageEvent("message", { data: event.data }));
          }
        });
        this.addEventListener("error", (e) => window.__workerErrors.push({ message: e.message, filename: e.filename, line: e.lineno })); }
      postMessage(data, transfer) {
        if (data.kind === "frame") { window.__timedSubmissions.push({ sentAt: performance.now(), timestampMs: data.timestampMs });if (window.__timedSubmissions.length > 64) window.__timedSubmissions.shift(); }
        if (data.kind === "frame") { window.__submittedFrames.push([data.bitmap.width, data.bitmap.height]);if (window.__submittedFrames.length > 64) window.__submittedFrames.shift(); }
        return super.postMessage(data, transfer);
      }
      terminate() { window.__terminatedWorkers++;return super.terminate(); }
    };
    window.__cameraCalls = [];
    window.__cameraStreams = [];
    const original = navigator.mediaDevices?.getUserMedia.bind(navigator.mediaDevices);
    if (original) navigator.mediaDevices.getUserMedia = async (constraints) => {
      window.__cameraCalls.push(constraints);
      const stream = await original(constraints);
      window.__cameraStreams.push(stream);
      return stream;
    };
  });
}

async function staleStartup(context, origin) {
  const page = await context.newPage(); activePage = page;
  await page.addInitScript(() => {
    const original = HTMLMediaElement.prototype.play;
    HTMLMediaElement.prototype.play = async function () {
      await original.call(this);
      if (window.__gateNextPlay) {
        window.__gateNextPlay = false;
        await new Promise((resolve) => { window.__releasePlay = resolve; });
        window.__playReleased = true;
      }
    };
  });
  await page.goto(origin + "/camera/");await page.locator("#ai").uncheck();
  await page.evaluate(() => { window.__gateNextPlay = true; });
  await page.locator("#start").click();
  await page.waitForFunction(() => typeof window.__releasePlay === "function");
  await page.evaluate(() => {
    // Some engines retain intrinsic metadata after srcObject is cleared.
    const video = document.querySelector("video");
    Object.defineProperty(video, "videoWidth", { configurable: true, get: () => 640 });
    Object.defineProperty(video, "videoHeight", { configurable: true, get: () => 480 });
    document.querySelector("#stop").click();window.__releasePlay();
  });
  await page.waitForFunction(() => window.__playReleased);
  await page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 0)));
  assert.equal(await page.locator("#placeholder").isVisible(), true, "stopped placeholder changed after stale play completion");
  assert.equal(await page.locator("#overlay").isHidden(), true);
  assert.equal(await page.locator("#mode-label").textContent(), "CAMERA OFF");
  await stopped(page);check("stale-video-play-does-not-restore-stopped-presentation");
  await page.evaluate(() => { window.__gateNextPlay = true;window.__playReleased = false;delete window.__releasePlay; });
  await page.locator("#start").click();
  await page.waitForFunction(() => typeof window.__releasePlay === "function");
  await page.locator("#stop").click();await page.locator("#sample").click();await waitFrames(page, 2);
  const before = await frameCount(page);
  await page.evaluate(() => {
    const video = document.querySelector("video");
    Object.defineProperty(video, "videoWidth", { configurable: true, get: () => 480 });
    Object.defineProperty(video, "videoHeight", { configurable: true, get: () => 640 });
    window.__releasePlay();
  });
  await page.waitForFunction(() => window.__playReleased);await waitFrames(page, before + 1);
  assert.equal(await page.locator("#overlay").evaluate((c) => c.width), 256);
  assert.equal(await page.locator("#overlay").evaluate((c) => c.height), 192);
  assert.ok((await page.locator("#mode-label").textContent()).startsWith("SAMPLE"));
  assert.equal(await page.evaluate(() => document.documentElement.dataset.sessionState), "running");
  check("stale-camera-start-preserves-replacement-sample");
  await page.locator("#stop").click();await stopped(page);await page.close();
}

async function videoProgress(context, origin, busy) {
  const page = await context.newPage(); activePage = page;
  await page.goto(origin + "/camera/");await page.locator("#ai").uncheck();await page.locator("#resolution").selectOption("192");
  await page.locator("#start").click();await waitFrames(page, 1);
  if (busy) {
    await page.evaluate(() => { window.__holdNextResult = true; });
    await page.waitForFunction(() => typeof window.__heldResult === "function");
  }
  const frozen = await page.evaluate(() => {
    const video = document.querySelector("video");video.pause();
    return { active: video.srcObject.active, tracksLive: video.srcObject.getVideoTracks().every((t) => t.readyState === "live"), ready: video.readyState, time: video.currentTime, paused: video.paused };
  });
  assert.ok(frozen.active && frozen.tracksLive && frozen.ready >= 2 && frozen.paused);
  (report.progressCases ??= []).push({ busy, frozen });
  await page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 5400)));
  assert.equal(await page.evaluate(() => document.documentElement.dataset.sessionState), "error", "live video clock stall retained stale presentation");
  assert.equal(await page.evaluate(() => document.querySelector("video").srcObject), null);
  assert.equal(await page.evaluate(() => window.__cameraStreams.every((s) => s.getTracks().every((t) => t.readyState === "ended"))), true);
  assert.equal(await page.evaluate(() => window.__terminatedWorkers), 1);assert.equal(await page.locator("#overlay").isHidden(), true);
  for (const id of ["points", "fps", "person", "backend", "frame-size"]) assert.equal(await page.locator("#" + id).textContent(), "—");
  const status = await page.locator("#status").textContent();assert.ok(status.includes("更新") && status.includes("もう一度") && !status.includes("AIをオフ"));
  check(`live-video-${busy ? "busy" : "idle"}-stall-clears-presentation-and-resources`);
  if (!busy) await screenshot(page, "mobile-progress-timeout");
  const before = await frameCount(page);await page.locator("#start").click();await waitFrames(page, before + 1);
  if (busy) {
    const late = await page.evaluate(() => {
      const before = document.documentElement.dataset.trackedFrames;
      const release = window.__heldResult;delete window.__heldResult;release();
      return { before, after: document.documentElement.dataset.trackedFrames, state: document.documentElement.dataset.sessionState };
    });
    assert.equal(late.after, late.before);assert.equal(late.state, "running");
  }
  assert.ok((await page.locator("#mode-label").textContent()).startsWith("LIVE"));
  assert.equal(await page.evaluate(() => window.__cameraCalls.length), 2);check(`live-video-${busy ? "busy" : "idle"}-restart-has-fresh-progress-state`);
  if (busy) {
    await page.evaluate(() => { window.__holdNextResult = true; });await page.waitForFunction(() => typeof window.__heldResult === "function");
    const before = await page.evaluate(() => ({ time: document.querySelector("video").currentTime, submissions: window.__timedSubmissions.length }));
    await page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 5400)));
    const after = await page.evaluate(() => ({ time: document.querySelector("video").currentTime, submissions: window.__timedSubmissions.length, state: document.documentElement.dataset.sessionState }));
    assert.equal(after.state, "running");assert.ok(after.time > before.time);assert.equal(after.submissions, before.submissions);
    report.liveResultWait = { before, after };check("advancing-video-does-not-timeout-while-result-is-held");
    const frames = await frameCount(page);await page.evaluate(() => { const release = window.__heldResult;delete window.__heldResult;release(); });await waitFrames(page, frames + 1);
    await screenshot(page, "mobile-progress-restart");
  }
  await page.locator("#stop").click();await stopped(page);await page.close();
}

async function sourceTermination(context, origin, kind) {
  const page = await context.newPage(); activePage = page;
  await page.goto(origin + "/camera/");await page.locator("#ai").uncheck();await page.locator("#resolution").selectOption("192");
  await page.locator("#start").click();await waitFrames(page, 1);
  await page.evaluate(() => { window.__holdNextResult = true; });
  await page.waitForFunction(() => typeof window.__heldResult === "function");
  await page.evaluate((kind) => {
    const stream = window.__cameraStreams.at(-1), track = stream.getVideoTracks()[0];
    window.__oldSource = { stream, track };track.stop();
    // stop() itself does not emit ended; these cases model external source events.
    if (kind === "ended") track.dispatchEvent(new Event("ended"));
    if (kind === "inactive") stream.dispatchEvent(new Event("inactive"));
  }, kind);
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  assert.equal(await page.evaluate(() => document.documentElement.dataset.sessionState), "error", "terminated source left stale camera presentation");
  assert.equal(await page.evaluate(() => document.querySelector("video").srcObject), null);
  assert.equal(await page.evaluate(() => window.__cameraStreams.every((s) => s.getTracks().every((t) => t.readyState === "ended"))), true);
  assert.equal(await page.evaluate(() => window.__terminatedWorkers), 1);
  assert.equal(await page.locator("#overlay").isHidden(), true);assert.equal(await page.locator("#placeholder").isVisible(), true);
  for (const id of ["points", "fps", "person", "backend", "frame-size"]) assert.equal(await page.locator("#" + id).textContent(), "—");
  for (const id of ["start", "sample", "ai", "facing", "resolution"]) assert.equal(await page.locator("#" + id).isEnabled(), true);
  const status = await page.locator("#status").textContent();assert.ok(status.includes("カメラ") && status.includes("もう一度"));assert.ok(!status.includes("AIをオフ"));
  check(`source-${kind}-releases-capture-worker-and-stale-presentation`);
  const before = await frameCount(page);await page.locator("#sample").click();await waitFrames(page, before + 1);
  const late = await page.evaluate(() => {
    const before = document.documentElement.dataset.trackedFrames;
    window.__oldSource.track.dispatchEvent(new Event("ended"));window.__oldSource.stream.dispatchEvent(new Event("inactive"));
    const release = window.__heldResult;delete window.__heldResult;release();
    return { before, after: document.documentElement.dataset.trackedFrames, state: document.documentElement.dataset.sessionState };
  });
  assert.equal(late.after, late.before);assert.equal(late.state, "running");
  assert.ok((await page.locator("#mode-label").textContent()).startsWith("SAMPLE"));assert.equal(await page.evaluate(() => window.__cameraCalls.length), 1);
  await waitFrames(page, await frameCount(page));check(`source-${kind}-late-events-cannot-stop-replacement-sample`);
  if (kind === "ended") await screenshot(page, "mobile-source-recovery");
  await page.locator("#stop").click();await stopped(page);await page.close();
}

async function captureCadence(context, origin) {
  const page = await context.newPage(); activePage = page;
  await page.addInitScript(() => {
    const original = globalThis.createImageBitmap;
    globalThis.createImageBitmap = async (...args) => {
      const started = performance.now(), bitmap = await original(...args);
      if (window.__delayCapture) {
        window.__delayCapture = false;window.__captureStarted = started;
        await new Promise((resolve) => { window.__releaseTimedCapture = resolve; });
      }
      return bitmap;
    };
  });
  await page.goto(origin + "/camera/");await page.locator("#ai").uncheck();await page.locator("#resolution").selectOption("192");
  await page.evaluate(() => { window.__delayCapture = true; });await page.locator("#sample").click();
  await page.waitForFunction(() => typeof window.__releaseTimedCapture === "function");
  await page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 600)));
  await page.evaluate(() => window.__releaseTimedCapture());
  await page.waitForFunction(() => window.__timedSubmissions.length >= 3);
  const snapshot = await page.evaluate(() => ({ submissions: window.__timedSubmissions.slice(0, 3), captureStarted: window.__captureStarted }));
  report.cadence = snapshot;
  for (let i = 1; i < snapshot.submissions.length; i++) assert.ok(snapshot.submissions[i].sentAt - snapshot.submissions[i - 1].sentAt >= 125, "capture delay consumed submission interval");
  check("delayed-capture-preserves-submission-interval");
  assert.ok(snapshot.submissions[0].timestampMs <= snapshot.captureStarted && snapshot.captureStarted - snapshot.submissions[0].timestampMs < 125, "capture timestamp came from an old animation callback");
  const frames = await frameCount(page);await waitFrames(page, frames);
  const reports = await page.evaluate(() => window.__cameraReports);
  assert.ok(reports.some((r) => r.timestampMs === snapshot.submissions[0].timestampMs));check("capture-start-time-survives-rust-report");
  await page.locator("#stop").click();await stopped(page);await page.close();
}

async function captureRate(context, origin) {
  const page = await context.newPage(); activePage = page;
  await page.goto(origin + "/camera/");await page.locator("#ai").uncheck();await page.locator("#resolution").selectOption("192");
  await page.evaluate(() => { window.__holdNextResult = true; });await page.locator("#sample").click();
  await page.waitForFunction(() => typeof window.__heldResult === "function");
  await page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 600)));
  await page.evaluate(() => { const release = window.__heldResult;delete window.__heldResult;release(); });
  await waitFrames(page, 1);
  const snapshot = await page.evaluate(() => ({ fps: document.querySelector("#fps").textContent, completed: window.__cameraReports.map((r) => r.timestampMs), submissions: window.__timedSubmissions }));
  const completed = new Set(snapshot.completed), newest = Math.max(...snapshot.completed);
  const stamps = snapshot.submissions.map((r) => r.timestampMs).filter((t) => completed.has(t) && t >= newest - 2000).slice(-17);
  assert.ok(stamps.length >= 2);
  const expected = 1000 * (stamps.length - 1) / (stamps.at(-1) - stamps[0]);
  report.captureRate = { ...snapshot, expected };
  assert.ok(Math.abs(Number.parseFloat(snapshot.fps) - expected) <= .051, "result delivery jitter inflated displayed capture frequency");
  check("completion-jitter-does-not-inflate-capture-rate");await screenshot(page, "mobile-capture-rate");
  await page.locator("#stop").click();await stopped(page);
  assert.equal(await page.locator("#fps").textContent(), "—");check("stop-clears-capture-rate");await page.close();
}

async function lightPreset(context, origin) {
  const page = await context.newPage(); activePage = page;
  await page.goto(origin + "/camera/");await page.locator("#ai").uncheck();
  assert.equal(await page.locator("#resolution").inputValue(), "256");check("standard-processing-default");
  await page.locator("#resolution").selectOption("192");await page.locator("#sample").click();
  assert.equal(await page.locator("#resolution").isDisabled(), true);
  await page.waitForFunction(() => document.documentElement.dataset.trackingStatus === "tracking", null, { timeout: 120000 });
  assert.equal(await page.locator("#frame-size").textContent(), "192 × 144 px");
  assert.ok(Number.parseInt(await page.locator("#points").textContent()) >= 8);
  assert.equal(await page.evaluate(() => window.__cameraCalls.length), 0);check("light-sample-real-wasm-motion-without-camera");
  await screenshot(page, "mobile-light-sample");
  const before = await frameCount(page);
  // Even programmatic edits to a disabled control cannot change active coordinates.
  await page.locator("#resolution").evaluate((select) => { select.value = "256"; });await waitFrames(page, before + 2);
  assert.equal(await page.locator("#overlay").evaluate((c) => c.width), 192);
  assert.equal(await page.locator("#frame-size").textContent(), "192 × 144 px");check("processing-size-is-frozen-during-session");
  await page.locator("#stop").click();await stopped(page);
  assert.equal(await page.locator("#resolution").isEnabled(), true);
  assert.equal(await page.locator("#frame-size").textContent(), "—");check("stop-reenables-processing-preset-and-clears-size");
  await page.locator("#sample").click();
  await page.waitForFunction(() => document.documentElement.dataset.trackingStatus === "tracking", null, { timeout: 120000 });
  assert.equal(await page.locator("#frame-size").textContent(), "256 × 192 px");
  assert.equal(await page.evaluate(() => window.__cameraCalls.length), 0);check("standard-preset-restarts-after-light-session");
  await page.locator("#stop").click();await stopped(page);await page.close();
}

async function intrinsicRotation(context, origin, longEdge = 256) {
  const page = await context.newPage(); activePage = page;
  await page.addInitScript(() => {
    const originalBitmap = globalThis.createImageBitmap;
    globalThis.createImageBitmap = async (...args) => {
      const bitmap = await originalBitmap(...args);
      if (window.__holdNextCapture) {
        window.__holdNextCapture = false;
        await new Promise((resolve) => { window.__releaseCapture = resolve; });
      }
      return bitmap;
    };
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      window.__cameraCalls.push(constraints);
      const canvas = document.createElement("canvas");canvas.width = 320;canvas.height = 240;
      window.__syntheticCamera = canvas;
      const ctx = canvas.getContext("2d"), stream = canvas.captureStream(12);
      window.__cameraStreams.push(stream);
      function draw() {
        if (stream.getTracks().every((t) => t.readyState === "ended")) return;
        for (let y = 0; y < canvas.height; y += 16) for (let x = 0; x < canvas.width; x += 16) {
          ctx.fillStyle = ((x / 16 + y / 16) % 2) ? "#65d6bb" : "#163d58";ctx.fillRect(x, y, 16, 16);
        }
        ctx.fillStyle = "white";ctx.font = "bold 20px sans-serif";ctx.fillText("ROTATION", 42, 92);
        requestAnimationFrame(draw);
      }
      draw();return stream;
    };
  });
  await page.goto(origin + "/camera/");
  await page.locator("#resolution").selectOption(String(longEdge));
  await page.locator("#start").click();await waitFrames(page, 2);
  const workers = await page.evaluate(() => window.__workerCount);
  await page.evaluate(() => { window.__holdNextResult = true; });
  await page.waitForFunction(() => typeof window.__heldResult === "function");
  const before = await frameCount(page);
  await page.evaluate(() => { window.__syntheticCamera.width = 240;window.__syntheticCamera.height = 320; });
  await page.waitForFunction(() => document.querySelector("video").videoWidth === 240 && document.querySelector("video").videoHeight === 320);
  assert.equal(await page.locator("#overlay").evaluate((c) => c.width), longEdge, "in-flight geometry changed before its result");
  await page.evaluate(() => { const release = window.__heldResult;delete window.__heldResult;release(); });
  await waitFrames(page, before + 2);
  const shortEdge = longEdge * 3 / 4;
  assert.equal(await page.locator("#overlay").evaluate((c) => c.width), shortEdge, "camera width did not follow intrinsic resize");
  assert.equal(await page.locator("#overlay").evaluate((c) => c.height), longEdge);
  assert.equal(await page.locator("#frame-size").textContent(), `${shortEdge} × ${longEdge} px`);
  const firstPortrait = await page.evaluate(([w, h]) => window.__cameraReports.find((r) => r.width === w && r.height === h), [shortEdge, longEdge]);
  assert.equal(firstPortrait.maskBytes, shortEdge * longEdge);
  assert.equal(firstPortrait.status, "warming");assert.equal(firstPortrait.motion, null);check("portrait-reseeds-after-in-flight-result");
  assert.equal(await page.evaluate(() => window.__workerCount), workers);check("rotation-keeps-initialized-ai-worker");
  assert.ok(["GPU", "CPU"].includes(await page.locator("#backend").textContent()));
  await screenshot(page, `mobile-portrait-camera-${longEdge}`);
  await page.evaluate(() => { window.__holdNextCapture = true; });
  await page.waitForFunction(() => typeof window.__releaseCapture === "function");
  const portraitFrames = await frameCount(page);
  const submitted = await page.evaluate(() => window.__submittedFrames.length);
  await page.evaluate(() => { window.__syntheticCamera.width = 320;window.__syntheticCamera.height = 240; });
  await page.waitForFunction(() => document.querySelector("video").videoWidth === 320);
  await page.evaluate(() => { const release = window.__releaseCapture;delete window.__releaseCapture;release(); });
  await waitFrames(page, portraitFrames + 2);
  const afterResize = await page.evaluate((n) => window.__submittedFrames.slice(n), submitted);
  assert.ok(afterResize.length > 0 && afterResize.every(([w,h]) => w === longEdge && h === shortEdge), "bitmap captured in obsolete geometry was submitted");
  check("capture-from-old-geometry-is-dropped");
  assert.equal(await page.locator("#overlay").evaluate((c) => c.width), longEdge);
  assert.equal(await page.locator("#overlay").evaluate((c) => c.height), shortEdge);
  assert.equal(await page.evaluate(() => window.__workerCount), workers);check("landscape-resize-recovers-without-ai-reload");
  await page.locator("#stop").click();await stopped(page);check("rotated-synthetic-stream-stops");
  await page.close();
}

(async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ headless: true, executablePath: process.env.CAMERA_CHROMIUM_PATH || chromium.executablePath(), args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
  report.browser = browser.version();
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1, isMobile: true, hasTouch: true });
  await instrument(context);
  if (regressionCase) {
    if (regressionCase === "startup") await staleStartup(context, origin);
    else if (regressionCase === "rotation") await intrinsicRotation(context, origin);
    else if (regressionCase === "cadence") await captureCadence(context, origin);
    else if (regressionCase.startsWith("source-")) await sourceTermination(context, origin, regressionCase.slice(7));
    else if (regressionCase.startsWith("progress-")) await videoProgress(context, origin, regressionCase === "progress-busy");
    else await captureRate(context, origin);
    report.passed = true;return;
  }
  const page = await context.newPage(); activePage = page;
  page.on("requestfailed", (r) => report.failedRequests.push({ url: r.url(), failure: r.failure() }));
  page.on("response", (r) => { if (r.status() >= 400) report.httpErrors.push({ url: r.url(), status: r.status() }); if (/^https?:/.test(r.url()) && !r.url().startsWith(origin + "/")) report.externalResponses.push(r.url()); });
  page.on("console", (m) => { if (m.type() === "error") report.consoleErrors.push(m.text()); });
  page.on("pageerror", (e) => report.consoleErrors.push(e.message));
  await page.goto(origin + "/camera/", { waitUntil: "networkidle" });
  assert.equal(await page.evaluate(() => window.__cameraCalls.length), 0);check("no-camera-on-load");
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), true);check("390px-no-horizontal-overflow");
  await screenshot(page, "mobile-idle");
  await page.locator("#ai").uncheck();
  await page.locator("#sample").click();
  await page.waitForFunction(() => document.documentElement.dataset.sessionState === "error" || document.documentElement.dataset.trackingStatus === "tracking", null, { timeout: 120000 });
  assert.notEqual(await page.evaluate(() => document.documentElement.dataset.sessionState), "error", await page.locator("#status").textContent());
  assert.equal(await page.evaluate(() => window.__cameraCalls.length), 0);
  assert.equal(await page.locator("#backend").textContent(), "OFF");check("real-wasm-sample-motion-without-camera");
  await screenshot(page, "mobile-tracking");
  await page.locator("#stop").click();await stopped(page);check("sample-stop");
  let frames = await frameCount(page);
  await page.locator("#start").click();await waitFrames(page, frames);
  assert.equal(await page.evaluate(() => window.__cameraCalls.length), 1);
  assert.equal(await page.evaluate(() => window.__cameraCalls.every((v) => v.audio === false)), true);check("fake-camera-real-frame-processing-no-microphone");
  await page.locator("#stop").click();await stopped(page);check("stop-ends-capture-tracks");
  frames = await frameCount(page);await page.locator("#start").click();await waitFrames(page, frames);
  await page.evaluate(() => window.dispatchEvent(new Event("pagehide")));await stopped(page);check("pagehide-stops-tracks-and-worker");
  frames = await frameCount(page);await page.locator("#start").click();await waitFrames(page, frames);
  await page.evaluate(() => { Object.defineProperty(document, "hidden", { configurable: true, get: () => true });document.dispatchEvent(new Event("visibilitychange")); });
  await stopped(page);check("hidden-event-stops-tracks-and-worker");
  await page.evaluate(() => { delete document.hidden; });
  await page.locator("#ai").check();frames = await frameCount(page);
  await page.locator("#sample").click();await waitFrames(page, frames + 2);
  report.aiFramesProcessed = (await frameCount(page)) - frames;
  assert.ok(report.aiFramesProcessed >= 3);
  report.aiBackend = await page.locator("#backend").textContent();
  assert.ok(["GPU", "CPU"].includes(report.aiBackend));check("pinned-real-model-inference-and-mask-through-wasm");
  await screenshot(page, "mobile-ai");await page.locator("#stop").click();await stopped(page);
  const blocked = await page.evaluate(() => new Promise((resolve, reject) => {
    const code = `self.addEventListener('securitypolicyviolation',e=>self.postMessage({directive:e.effectiveDirective,uri:e.blockedURI}));fetch('https://example.invalid/camera-network-must-be-blocked').catch(()=>{});`;
    const url = URL.createObjectURL(new Blob([code], { type: "text/javascript" }));
    const worker = new Worker(url, { type: "module" });
    const timer = setTimeout(() => {worker.terminate();URL.revokeObjectURL(url);reject(new Error("worker CSP violation not observed"));}, 10000);
    worker.onmessage = ({ data }) => {clearTimeout(timer);worker.terminate();URL.revokeObjectURL(url);resolve(data);};
  }));
  assert.equal(blocked.directive, "connect-src");assert.ok(blocked.uri.startsWith("https://example.invalid"));check("blob-worker-inherits-self-only-network-csp");
  await page.setViewportSize({ width: 320, height: 780 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), true);check("320px-no-horizontal-overflow");
  await page.setViewportSize({ width: 1280, height: 900 });await screenshot(page, "desktop-idle");
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), true);check("desktop-layout");
  const deniedContext = await browser.newContext({ viewport: { width: 390, height: 844 } });
  await deniedContext.addInitScript(() => {navigator.mediaDevices.getUserMedia = async () => {throw new DOMException("Permission denied", "NotAllowedError");};});
  const denied = await deniedContext.newPage();await denied.goto(origin + "/camera/");await denied.locator("#ai").uncheck();await denied.locator("#start").click();
  await denied.waitForFunction(() => document.documentElement.dataset.sessionState === "error");
  assert.ok((await denied.locator("#status").textContent()).includes("許可されていません"));check("camera-denial-shows-recovery");
  await denied.locator("#sample").click();await waitFrames(denied);check("sample-works-after-denial");await denied.locator("#stop").click();
  await deniedContext.close();
  await staleStartup(context, origin);
  await intrinsicRotation(context, origin);
  await lightPreset(context, origin);
  await intrinsicRotation(context, origin, 192);
  await captureCadence(context, origin);
  await captureRate(context, origin);
  for (const kind of ["ended", "inactive", "silent"]) await sourceTermination(context, origin, kind);
  await videoProgress(context, origin, false);await videoProgress(context, origin, true);
  assert.equal(report.externalResponses.length, 0);assert.equal(report.nonGetRequests.length, 0);check("same-origin-static-get-responses-only");
  report.passed = true;
})().catch(async (error) => {
  report.passed = false;report.error = error.stack || String(error);process.exitCode = 1;
  if (activePage && !activePage.isClosed()) {
    report.workerErrors = await activePage.evaluate(() => window.__workerErrors ?? []).catch(() => []);
    report.cspViolations = await activePage.evaluate(() => window.__cspViolations ?? []).catch(() => []);
    report.lastStatus = await activePage.locator("#status").textContent().catch(() => "unavailable");
    await screenshot(activePage, "failure").catch(() => {});
  }
}).finally(async () => {
  await browser?.close();await new Promise((resolve) => server.close(resolve));
  fs.writeFileSync(path.join(output, "report.json"), JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify({ passed: report.passed, checks: report.checks, screenshots: report.screenshots, aiBackend: report.aiBackend, error: report.error, lastStatus: report.lastStatus, externalResponseCount: report.externalResponses.length, consoleErrorCount: report.consoleErrors.length }));
});
