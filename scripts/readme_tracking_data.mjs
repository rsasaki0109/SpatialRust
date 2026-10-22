// Export actual Rust/WASM reports for the README's synthetic playback visual.
// No camera, AI model, browser, network request or timing benchmark is used.
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";

function arg(name) {
  const i = process.argv.indexOf(name);
  if (i < 0 || !process.argv[i + 1]) throw new Error(`missing ${name}`);
  return path.resolve(process.argv[i + 1]);
}
const pkg = arg("--pkg"), output = arg("--output");
if (fs.existsSync(output)) throw new Error("output already exists");
const js = fs.readFileSync(path.join(pkg, "spatialrust_web.js"));
const wasm = fs.readFileSync(path.join(pkg, "spatialrust_web_bg.wasm"));
const { initSync, BrowserCameraTracker } = await import(`data:text/javascript;base64,${js.toString("base64")}`);
initSync({ module: wasm });
const width = 256, height = 192, frameCount = 48;
const tracker = new BrowserCameraTracker(width, height);
const frames = [];
try {
  for (let frame = 0; frame < frameCount; frame++) {
    const angle = 2 * Math.PI * frame / frameCount;
    const dx = Math.round(8 * Math.sin(angle)), dy = Math.round(5 * Math.cos(angle));
    const gray = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const sx = x - dx + 32, sy = y - dy + 32;
      const h = (Math.imul(sx, 73856093) ^ Math.imul(sy, 19349663)) >>> 0;
      gray[y * width + x] = ((((sx >> 4) + (sy >> 4)) % 2) ? 125 : 35) + ((h ^ (h >>> 13)) & 31);
    }
    const report = JSON.parse(tracker.track_gray_copy(gray, new Uint8Array(0), frame * 125));
    if (report.schema !== 1 || report.tracks.length > 80) throw new Error("unexpected tracking report");
    if (frame && (report.status !== "tracking" || !report.background_flow_px)) throw new Error(`tracking lost at frame ${frame}`);
    frames.push({ shift: [dx, dy], gray_sha256: createHash("sha256").update(gray).digest("hex"), report });
  }
} finally { tracker.free(); }
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, JSON.stringify({
  schema: "spatialrust.readme-tracking.v1", synthetic_only: true, ai_enabled: false,
  execution: "CPU Rust/WASM", width, height, frame_duration_ms: 125,
  wasm_sha256: createHash("sha256").update(wasm).digest("hex"),
  bindings_sha256: createHash("sha256").update(js).digest("hex"), frames,
}, null, 2) + "\n", { flag: "wx" });
console.log(JSON.stringify({ frames: frames.length, tracked: frames.filter(f => f.report.status === "tracking").length, output }));
