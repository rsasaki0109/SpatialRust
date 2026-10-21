// Compare frozen parent and candidate WASM reports without camera access.
const fs = require("node:fs"), path = require("node:path"), http = require("node:http"), assert = require("node:assert/strict"), { createHash } = require("node:crypto");
const { chromium } = require("playwright");
const arg = (name) => { const i = process.argv.indexOf(name);assert.ok(i >= 0 && process.argv[i + 1]);return path.resolve(process.argv[i + 1]); };
const root = arg("--root"), output = arg("--output");assert.ok(!fs.existsSync(output));fs.mkdirSync(output, { recursive: true });
const files = new Map();
for (const [prefix, directory] of [["baseline", "baseline-pkg"], ["candidate", "site/camera/pkg"]]) for (const name of ["spatialrust_web.js", "spatialrust_web_bg.wasm"]) files.set(`/${prefix}/${name}`, path.join(root, directory, name));
const server = http.createServer((req, res) => {
  if (req.method !== "GET") { res.writeHead(405);res.end();return; }
  const url = new URL(req.url, "http://localhost");
  if (url.pathname === "/") { res.writeHead(200, { "Content-Type": "text/html", "Content-Security-Policy": "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'" });res.end("<!doctype html><title>Mask support parity</title>");return; }
  if (url.pathname === "/favicon.ico") { res.writeHead(204);res.end();return; }
  const file = files.get(url.pathname);if (!file) { res.writeHead(404);res.end();return; }
  res.writeHead(200, { "Content-Type": file.endsWith(".wasm") ? "application/wasm" : "text/javascript", "Content-Length": fs.statSync(file).size });fs.createReadStream(file).pipe(res);
});
let browser;const report = { schema: "spatialrust.camera-mask-support-wasm-parity.v1", syntheticCaptureOnly: true, physicalPhoneTested: false, cameraCalls: 0, consoleErrors: [], externalRequests: [], files: [] };
(async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));const origin = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ headless: true });report.browser = browser.version();const page = await browser.newPage();
  page.on("console", (message) => { if (message.type() === "error") report.consoleErrors.push(message.text()); });page.on("pageerror", (error) => report.consoleErrors.push(error.message));
  page.on("request", (request) => { if (!request.url().startsWith(origin + "/")) report.externalRequests.push(request.url()); });
  await page.addInitScript(() => { window.__cameraCalls = 0;if (navigator.mediaDevices) navigator.mediaDevices.getUserMedia = () => { window.__cameraCalls++;throw new Error("camera is forbidden in parity checks"); }; });
  await page.goto(origin + "/");
  report.result = await page.evaluate(async () => {
    const parent = await import("/baseline/spatialrust_web.js"), candidate = await import("/candidate/spatialrust_web.js");await parent.default();await candidate.default();
    const cases = [];let comparisons = 0, rejectedComparisons = 0;
    const texture = (w, h, frame) => Uint8Array.from({ length: w * h }, (_, i) => { const x = i % w - frame % 3, y = Math.floor(i / w) - frame % 2;const hash = (Math.imul(x, 73856093) ^ Math.imul(y, 19349663)) >>> 0;return (hash ^ (hash >>> 13)) & 255; });
    const mask = (w, h, kind, frame) => {
      if (kind === "changing") kind = ["half", "zero", "sparse", "full", "disabled"][frame];
      if (kind === "disabled") return new Uint8Array(0);
      return Uint8Array.from({ length: w * h }, (_, i) => kind === "full" ? 1 : kind === "half" ? Number(i % w >= Math.floor(w / 2)) : kind === "sparse" ? Number(i === Math.floor(h / 2) * w + Math.floor(w / 2)) : kind === "checker" ? Number((i % w + Math.floor(i / w)) % 2 === 0) : 0);
    };
    for (const [w, h] of [[64,64],[65,79],[160,128],[192,144],[256,192],[320,320]]) for (const kind of ["disabled","zero","full","half","sparse","checker","changing"]) {
      const a = new parent.BrowserCameraTracker(w, h), b = new candidate.BrowserCameraTracker(w, h);let first, last;
      try {
        for (let frame = 0; frame < 5; frame++) {
          const image = texture(w, h, frame), exclusion = mask(w, h, kind, frame);
          if (frame === 3) for (const invalid of ["label", "clock"]) {
            const bad = new Uint8Array(w * h);bad[0] = 2;const errors = [];
            for (const tracker of [a, b]) { try { tracker.track_gray_copy(image, invalid === "label" ? bad : exclusion, invalid === "clock" ? NaN : frame * 125);throw new Error("invalid call accepted"); } catch (error) { errors.push(String(error)); } }
            if (errors[0] !== errors[1] || errors[0].includes("invalid call accepted")) throw new Error(`invalid admission differs: ${w}x${h} ${kind} ${invalid}`);rejectedComparisons++;
          }
          const left = a.track_gray_copy(image, exclusion, frame * 125), right = b.track_gray_copy(image, exclusion, frame * 125);
          if (left !== right) throw new Error(`full report differs: ${w}x${h} ${kind} frame ${frame}`);
          const parsed = JSON.parse(left);if (frame === 0) first = parsed.status;last = parsed.status;comparisons++;
        }
        a.reset();b.reset();const image = texture(w, h, 0), exclusion = mask(w, h, kind, 0);
        if (a.track_gray_copy(image, exclusion, 0) !== b.track_gray_copy(image, exclusion, 0)) throw new Error("reset report differs");comparisons++;
        cases.push({ width: w, height: h, mask: kind, matchedReports: 6, rejectedCalls: 2, first, last });
      } finally { a.free();b.free(); }
    }
    return { cases, comparisons, rejectedComparisons, cameraCalls: window.__cameraCalls };
  });
  assert.equal(report.result.comparisons, 252);assert.equal(report.result.rejectedComparisons, 84);assert.equal(report.result.cameraCalls, 0);assert.equal(report.result.cases.length, 42);assert.deepEqual(report.consoleErrors, []);assert.deepEqual(report.externalRequests, []);
  for (const [url, file] of files) { const bytes = fs.readFileSync(file);report.files.push({ url, bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") }); }
  report.passed = true;
})().catch((error) => { report.passed = false;report.error = error.stack || String(error);process.exitCode = 1; }).finally(async () => { await browser?.close();await new Promise((resolve) => server.close(resolve));fs.writeFileSync(path.join(output, "report.json"), JSON.stringify(report, null, 2) + "\n");console.log(JSON.stringify({ passed: report.passed, comparisons: report.result?.comparisons, rejectedComparisons: report.result?.rejectedComparisons, error: report.error })); });
