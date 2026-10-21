// The native operation cannot be cancelled; close any bitmap that arrives late.
export function captureBitmap(create, { signal, timeoutMs = 5000 } = {}) {
  if (typeof create !== "function") throw new TypeError("capture function required");
  if (!Number.isInteger(timeoutMs) || timeoutMs <= 0 || timeoutMs > 2147483647) throw new RangeError("invalid capture timeout");
  return new Promise((resolve, reject) => {
    let settled = false, timer;
    const finish = (action, value) => {
      if (settled) { if (action === resolve) value.close();return; }
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      action(value);
    };
    const onAbort = () => finish(reject, new DOMException("capture stopped", "AbortError"));
    if (signal?.aborted) { onAbort();return; }
    signal?.addEventListener("abort", onAbort, { once: true });
    timer = setTimeout(() => {
      const error = new Error(`画像の取り込みが${timeoutMs / 1000}秒間完了しなかったため停止しました。もう一度「カメラ開始」か「サンプル再生」を押してください。`);
      error.name = "FrameCaptureTimeoutError";
      finish(reject, error);
    }, timeoutMs);
    try { Promise.resolve(create()).then((bitmap) => finish(resolve, bitmap), (error) => finish(reject, error)); }
    catch (error) { finish(reject, error); }
  });
}
