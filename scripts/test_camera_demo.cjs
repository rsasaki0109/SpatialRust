// Real browser/WASM/model checks using synthetic capture only. Never opens a real camera.
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const assert = require("node:assert/strict");
const { chromium } = require("playwright");

function argument(name) { const i = process.argv.indexOf(name); if (i < 0 || !process.argv[i + 1]) throw new Error(`missing ${name}`); return path.resolve(process.argv[i + 1]); }
const site = argument("--site"), output = argument("--output");
const caseIndex = process.argv.indexOf("--regression-case"), regressionCase = caseIndex < 0 ? null : process.argv[caseIndex + 1];
if (regressionCase && !["startup", "rotation"].includes(regressionCase)) throw new Error("unknown regression case");
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
    window.__cameraReports = [];
    window.__holdNextResult = false;
    window.__cspViolations = [];
    document.addEventListener("securitypolicyviolation", (e) => window.__cspViolations.push({ directive: e.effectiveDirective, uri: e.blockedURI }));
    const OriginalWorker = window.Worker;
    window.Worker = class extends OriginalWorker {
      constructor(...args) { super(...args); window.__workerCount++;
        this.addEventListener("message", (event) => {
          if (event.data.kind !== "result") return;
          const r = event.data.report;
          window.__cameraReports.push({ width: r.width, height: r.height, status: r.status, motion: r.background_flow_px });
          if (window.__cameraReports.length > 32) window.__cameraReports.shift();
          if (window.__holdNextResult) {
            window.__holdNextResult = false; event.stopImmediatePropagation();
            window.__heldResult = () => this.dispatchEvent(new MessageEvent("message", { data: event.data }));
          }
        });
        this.addEventListener("error", (e) => window.__workerErrors.push({ message: e.message, filename: e.filename, line: e.lineno })); }
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
  await page.close();
}

async function intrinsicRotation(context, origin) {
  const page = await context.newPage(); activePage = page;
  await page.addInitScript(() => {
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
  await page.locator("#start").click();await waitFrames(page, 2);
  const workers = await page.evaluate(() => window.__workerCount);
  await page.evaluate(() => { window.__holdNextResult = true; });
  await page.waitForFunction(() => typeof window.__heldResult === "function");
  const before = await frameCount(page);
  await page.evaluate(() => { window.__syntheticCamera.width = 240;window.__syntheticCamera.height = 320; });
  await page.waitForFunction(() => document.querySelector("video").videoWidth === 240 && document.querySelector("video").videoHeight === 320);
  assert.equal(await page.locator("#overlay").evaluate((c) => c.width), 256, "in-flight geometry changed before its result");
  await page.evaluate(() => { const release = window.__heldResult;delete window.__heldResult;release(); });
  await waitFrames(page, before + 2);
  assert.equal(await page.locator("#overlay").evaluate((c) => c.width), 192, "camera width did not follow intrinsic resize");
  assert.equal(await page.locator("#overlay").evaluate((c) => c.height), 256);
  const firstPortrait = await page.evaluate(() => window.__cameraReports.find((r) => r.width === 192 && r.height === 256));
  assert.equal(firstPortrait.status, "warming");assert.equal(firstPortrait.motion, null);check("portrait-reseeds-after-in-flight-result");
  assert.equal(await page.evaluate(() => window.__workerCount), workers);check("rotation-keeps-initialized-ai-worker");
  assert.ok(["GPU", "CPU"].includes(await page.locator("#backend").textContent()));
  await screenshot(page, "mobile-portrait-camera");
  const portraitFrames = await frameCount(page);
  await page.evaluate(() => { window.__syntheticCamera.width = 320;window.__syntheticCamera.height = 240; });
  await page.waitForFunction(() => document.querySelector("video").videoWidth === 320);
  await waitFrames(page, portraitFrames + 2);
  assert.equal(await page.locator("#overlay").evaluate((c) => c.width), 256);
  assert.equal(await page.locator("#overlay").evaluate((c) => c.height), 192);
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
    else await intrinsicRotation(context, origin);
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
