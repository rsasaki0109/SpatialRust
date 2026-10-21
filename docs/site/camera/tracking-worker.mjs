import initRust, { BrowserCameraTracker } from "./pkg/spatialrust_web.js";

import { personMask } from "./person-mask.mjs";

let session = 0;
let initialized = false;
let segmenter = null;
let backend = "OFF";
let tracker = null;
let width = 0;
let height = 0;
let pixels = null;
let context = null;

self.onmessage = async ({ data }) => {
  if (data.kind === "init" && !initialized) {
    initialized = true;
    session = data.session;
    try {
      await initRust();
      if (data.ai) {
        const { FilesetResolver, ImageSegmenter } = await import("./vendor/vision_bundle.mjs");
        const files = await FilesetResolver.forVisionTasks(new URL("./vendor/wasm", import.meta.url).href, true);
        const modelAssetPath = new URL("./vendor/selfie_segmenter.tflite", import.meta.url).href;
        const options = { outputCategoryMask: false, outputConfidenceMasks: true, runningMode: "VIDEO" };
        try {
          segmenter = await ImageSegmenter.createFromOptions(files, {
            ...options, canvas: new OffscreenCanvas(256, 256), baseOptions: { modelAssetPath, delegate: "GPU" },
          });
          backend = "GPU";
        } catch {
          segmenter = await ImageSegmenter.createFromOptions(files, { ...options, baseOptions: { modelAssetPath, delegate: "CPU" } });
          backend = "CPU";
        }
        const labels = segmenter.getLabels().map((label) => label.toLowerCase());
        if (labels.length !== 1 || labels[0] !== "selfie") throw new Error("人物モデルのラベルが想定と異なります。");
      }
      self.postMessage({ kind: "ready", session, backend });
    } catch (error) { failure(error); }
    return;
  }
  if (data.kind !== "frame") return;
  const bitmap = data.bitmap;
  try {
    if (data.session !== session || !initialized) return;
    if (bitmap.width < 64 || bitmap.height < 64 || bitmap.width > 320 || bitmap.height > 320) throw new Error("処理画像のサイズが範囲外です。");
    if (width !== bitmap.width || height !== bitmap.height) {
      tracker?.free();
      width = bitmap.width;
      height = bitmap.height;
      tracker = new BrowserCameraTracker(width, height);
      pixels = new OffscreenCanvas(width, height);
      context = pixels.getContext("2d", { willReadFrequently: true });
      if (!context) throw new Error("画像を処理できませんでした。");
    }
    const started = performance.now();
    context.drawImage(bitmap, 0, 0);
    // Explicit CPU readback of this resized frame; no full-resolution frame is retained.
    const rgba = context.getImageData(0, 0, width, height).data;
    const gray = new Uint8Array(width * height);
    for (let i = 0; i < gray.length; i++) gray[i] = (77 * rgba[4 * i] + 150 * rgba[4 * i + 1] + 29 * rgba[4 * i + 2]) >> 8;
    let mask = new Uint8Array(0);
    if (segmenter) {
      segmenter.segmentForVideo(pixels, data.timestampMs, (result) => {
        const confidence = result.confidenceMasks;
        if (confidence?.length !== 1) throw new Error("人物マスクが返りませんでした。");
        // Explicit float-mask readback before the callback releases the SDK result.
        const foreground = confidence[0];
        mask = personMask(foreground.getAsFloat32Array(), foreground.width, foreground.height, width, height);
      });
    }
    const report = JSON.parse(tracker.track_gray_copy(gray, mask, data.timestampMs));
    self.postMessage({ kind: "result", session, frameId: data.frameId, report, backend, elapsedMs: performance.now() - started, mask: mask.buffer }, [mask.buffer]);
  } catch (error) { failure(error); }
  finally { bitmap?.close(); }
};

function failure(error) {
  self.postMessage({ kind: "error", session, message: error?.message ?? String(error) });
}
