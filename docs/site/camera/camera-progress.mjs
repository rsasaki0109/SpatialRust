// Application-observed video time; this does not compare pixels or sensor clocks.
export class CameraProgress {
  #now;
  #updatedAt;
  #videoTime = null;

  constructor(nowMs) {
    if (!Number.isFinite(nowMs) || nowMs < 0) throw new RangeError("映像の監視時刻が想定と異なります。");
    this.#now = this.#updatedAt = nowMs;
  }

  stalled(videoTime, nowMs, ready = true) {
    if (!Number.isFinite(nowMs) || nowMs < this.#now || typeof ready !== "boolean" || (ready && (!Number.isFinite(videoTime) || videoTime < 0))) throw new RangeError("映像の更新時刻が想定と異なります。");
    this.#now = nowMs;
    if (ready && videoTime !== this.#videoTime) { this.#videoTime = videoTime;this.#updatedAt = nowMs; }
    return nowMs - this.#updatedAt >= 5000;
  }
}
