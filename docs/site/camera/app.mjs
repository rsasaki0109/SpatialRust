import { CameraSession } from "./session.mjs";
import { cameraFrameSize } from "./frame-size.mjs";

const $ = (id) => document.getElementById(id);
const video = $("video"), sample = $("sample-frame"), overlay = $("overlay");
const ctx = overlay.getContext("2d"), sampleCtx = sample.getContext("2d");
const motionCtx = $("motion").getContext("2d");
let frameHandle = 0, capturingToken = 0, lastSubmitted = -Infinity, lastVideoTime = -1, lastResult = 0, phase = 0;
let mode = null, width = 256, height = 192, longEdge = 256;

const session = new CameraSession({
  acquire: () => {
    if (!isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error("HTTPS対応のブラウザで開き直すか、サンプルをお試しください。");
    return navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: $("facing").value }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 24, max: 30 } } });
  },
  createWorker: () => {
    if (!globalThis.Worker || !globalThis.OffscreenCanvas || !globalThis.createImageBitmap) throw new Error("このブラウザは必要な画像処理機能に対応していません。最新版のChromeやSafariでお試しください。");
    // Blob module workers inherit this document's self-only connection CSP.
    const source = `import ${JSON.stringify(new URL("./tracking-worker.mjs", import.meta.url).href)};`;
    const url = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
    try {
      const worker = new Worker(url, { type: "module", name: "SpatialRust camera" });
      worker.dispose = () => URL.revokeObjectURL(url);
      return worker;
    } catch (error) { URL.revokeObjectURL(url); throw error; }
  },
  onReady: async (run) => {
    if (session.run !== run) return;
    let size = cameraFrameSize(4, 3, longEdge);
    if (run.mode === "camera") {
      video.srcObject = run.stream;
      await video.play();
      // Stop or a replacement session may win while play() is pending.
      if (session.run !== run) return;
      size = cameraFrameSize(video.videoWidth, video.videoHeight, longEdge);
    }
    mode = run.mode;
    applyGeometry(size);
    video.hidden = mode !== "camera";
    sample.hidden = mode !== "sample";
    overlay.hidden = false;
    $("placeholder").hidden = true;
    $("mode-label").textContent = mode === "sample" ? "SAMPLE · サンプル映像" : "LIVE · カメラ映像";
    $("backend").textContent = run.backend;
    phase = 0;
  },
  onResult: ({ report, elapsedMs, backend, mask }) => {
    if (report.schema !== 1 || report.width !== width || report.height !== height || report.tracks.length > 80) throw new Error("追跡結果の形が想定と異なります。");
    ctx.clearRect(0, 0, width, height);
    const person = new Uint8Array(mask);
    if (person.length) {
      if (person.length !== width * height) throw new Error("人物マスクのサイズが想定と異なります。");
      const tint = ctx.createImageData(width, height);
      for (let i = 0; i < person.length; i++) if (person[i]) { tint.data[4 * i] = 244; tint.data[4 * i + 1] = 113; tint.data[4 * i + 2] = 143; tint.data[4 * i + 3] = 65; }
      ctx.putImageData(tint, 0, 0);
    }
    ctx.strokeStyle = "#57efd0"; ctx.fillStyle = "#b5fff0"; ctx.lineWidth = .9;
    for (const track of report.tracks) {
      ctx.beginPath(); ctx.moveTo(...track.from); ctx.lineTo(...track.to); ctx.stroke();
      ctx.beginPath(); ctx.arc(...track.to, 1.5, 0, Math.PI * 2); ctx.fill();
    }
    const now = performance.now();
    $("points").textContent = `${report.tracks.length} 点`;
    $("fps").textContent = lastResult ? `${Math.min(99, Math.round(1000 / (now - lastResult)))} fps` : "—";
    lastResult = now;
    $("person").textContent = backend === "OFF" ? "オフ" : `${Math.round(report.person_fraction * 100)} %`;
    $("engine").textContent = `処理 / ${Math.round(elapsedMs)} ms`;
    $("backend").textContent = backend;
    const messages = { tracking: "背景を追跡中です。ゆっくり動かしてみてください。", warming: "背景の特徴点を準備しています。", reseeded: "間隔が空いたため、特徴点を取り直しています。", lost: "模様のある背景に向け、ゆっくり動かしてください。" };
    setStatus(messages[report.status] ?? messages.lost);
    drawMotion(report.background_flow_px);
    document.documentElement.dataset.trackingStatus = report.status;
    document.documentElement.dataset.trackedFrames = String(Number(document.documentElement.dataset.trackedFrames ?? 0) + 1);
  },
  onState: (state) => {
    const active = state === "loading" || state === "running";
    $("start").disabled = active; $("sample").disabled = active;
    $("stop").disabled = !active; $("ai").disabled = active; $("facing").disabled = active; $("resolution").disabled = active;
    document.documentElement.dataset.sessionState = state;
    if (state === "loading") setStatus("準備しています。初回はAI用ファイルを読み込みます。停止はいつでもできます。");
    if (state === "running") { setStatus("映像から特徴点を探しています。"); frameHandle = requestAnimationFrame(pump); }
  },
  onError: (error) => {
    const denied = error.name === "NotAllowedError" || /permission|denied/i.test(error.message);
    setStatus(denied ? "カメラの使用が許可されていません。ブラウザの権限を確認するか、サンプルでお試しください。" : `開始できませんでした。${error.message}${error.name === "RangeError" ? "" : " AIをオフにして試すこともできます。"}`);
  },
  onStop: () => {
    cancelAnimationFrame(frameHandle); frameHandle = 0;
    video.pause(); video.srcObject = null;
    video.hidden = sample.hidden = overlay.hidden = true;
    $("placeholder").hidden = false; $("mode-label").textContent = "CAMERA OFF";
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    mode = null; capturingToken = 0; lastResult = 0; lastSubmitted = -Infinity; lastVideoTime = -1;
    for (const id of ["points", "fps", "person", "backend", "frame-size"]) $(id).textContent = "—";
    $("engine").textContent = "待機中"; drawMotion(null);
    delete document.documentElement.dataset.trackingStatus;
  },
});

function start(mode) {
  longEdge = Number($("resolution").value);
  session.start(mode, $("ai").checked);
}
$("start").addEventListener("click", () => start("camera"));
$("sample").addEventListener("click", () => start("sample"));
$("stop").addEventListener("click", () => { session.stop(); setStatus("停止しました。カメラも処理もオフです。"); });
document.addEventListener("visibilitychange", () => { if (document.hidden) { session.stop(); setStatus("画面を離れたため停止しました。再開するには開始ボタンを押してください。"); } });
window.addEventListener("pagehide", () => session.stop());
drawMotion(null);

async function pump(now) {
  const run = session.run;
  const token = run?.token;
  if (!token || !mode) return;
  frameHandle = requestAnimationFrame(pump);
  if (session.busy || capturingToken === token || now - lastSubmitted < 125) return;
  if (mode === "camera" && (video.readyState < 2 || video.currentTime === lastVideoTime)) return;
  const sourceWidth = video.videoWidth, sourceHeight = video.videoHeight;
  if (mode === "camera" && (!sourceWidth || !sourceHeight)) return;
  capturingToken = token;
  try {
    if (mode === "camera") {
      const size = cameraFrameSize(sourceWidth, sourceHeight, longEdge);
      // One in-flight result is fully consumed before its coordinate system changes.
      if (size.width !== width || size.height !== height) applyGeometry(size);
    }
    const source = mode === "sample" ? renderSample() : video;
    const bitmap = await createImageBitmap(source, { resizeWidth: width, resizeHeight: height, resizeQuality: "low" });
    if (session.token !== token || (mode === "camera" && (video.videoWidth !== sourceWidth || video.videoHeight !== sourceHeight))) { bitmap.close(); return; }
    lastVideoTime = video.currentTime; lastSubmitted = now;
    session.submit(bitmap, now);
  } catch (error) { session.fail(run, error); }
  finally { if (capturingToken === token) capturingToken = 0; }
}

function applyGeometry(size) {
  width = size.width; height = size.height;
  for (const canvas of [sample, overlay]) { canvas.width = width; canvas.height = height; }
  $("stage").style.aspectRatio = `${width}/${height}`;
  $("frame-size").textContent = `${width} × ${height} px`;
  for (const id of ["points", "fps", "person"]) $(id).textContent = "—";
  lastResult = 0; lastVideoTime = -1;
  delete document.documentElement.dataset.trackingStatus;
  drawMotion(null);
}

function renderSample() {
  const scale = width / 256;
  const dx = Math.round(8 * scale * Math.sin(phase * .09)), dy = Math.round(5 * scale * Math.cos(phase * .075));
  phase++;
  const image = sampleCtx.createImageData(width, height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const sx = x - dx + 32, sy = y - dy + 32;
    const hash = (Math.imul(sx, 73856093) ^ Math.imul(sy, 19349663)) >>> 0;
    const noise = (hash ^ (hash >>> 13)) & 31;
    const checker = ((sx >> 4) + (sy >> 4)) % 2;
    const i = (y * width + x) * 4;
    image.data[i] = (checker ? 59 : 21) + noise;
    image.data[i + 1] = (checker ? 153 : 43) + noise;
    image.data[i + 2] = (checker ? 142 : 65) + noise;
    image.data[i + 3] = 255;
  }
  sampleCtx.putImageData(image, 0, 0);
  sampleCtx.fillStyle = "#061a29b0"; sampleCtx.fillRect(50 * scale + dx, 64 * scale + dy, 156 * scale, 57 * scale);
  sampleCtx.fillStyle = "#e9fff8"; sampleCtx.font = `bold ${20 * scale}px system-ui`; sampleCtx.fillText("SpatialRust", 68 * scale + dx, 98 * scale + dy);
  return sample;
}

function setStatus(text) { if ($("status").textContent !== text) $("status").textContent = text; }
function drawMotion(flow) {
  const c = motionCtx; c.clearRect(0, 0, 220, 150);
  c.strokeStyle = "#263e53"; c.lineWidth = 1;
  c.beginPath(); c.arc(110, 72, 48, 0, Math.PI * 2); c.stroke();
  c.beginPath(); c.moveTo(52,72);c.lineTo(168,72);c.moveTo(110,14);c.lineTo(110,130);c.stroke();
  c.fillStyle = "#8ca6bb"; c.font = "10px system-ui"; c.fillText("↑",107,12);c.fillText("→",172,76);
  if (!flow) { $("motion-label").textContent = "背景の動きを待っています"; return; }
  const [dx,dy] = flow;
  const length = Math.hypot(dx,dy), scale = length > 0 ? Math.min(46, length * 10) / length : 0;
  const x = 110 + dx * scale, y = 72 + dy * scale;
  c.strokeStyle = "#57efd0"; c.fillStyle = "#b5fff0";c.lineWidth = 3;
  c.beginPath();c.moveTo(110,72);c.lineTo(x,y);c.stroke();c.beginPath();c.arc(x,y,4,0,Math.PI*2);c.fill();
  $("motion-label").textContent = `背景 Δx ${dx.toFixed(1)} / Δy ${dy.toFixed(1)} px`;
}
