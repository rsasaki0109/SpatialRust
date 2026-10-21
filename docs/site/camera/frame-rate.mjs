// Completed image timestamps, independent of worker message delivery jitter.
export class FrameRate {
  #stamps = [];

  reset() { this.#stamps = []; }

  observe(timestampMs) {
    const last = this.#stamps.at(-1);
    if (!Number.isFinite(timestampMs) || timestampMs < 0 || (last !== undefined && timestampMs <= last)) throw new RangeError("追跡画像の時刻が想定と異なります。");
    this.#stamps = this.#stamps.filter((t) => t >= timestampMs - 2000);
    this.#stamps.push(timestampMs);
    if (this.#stamps.length > 17) this.#stamps.shift();
    if (this.#stamps.length < 2) return null;
    const rate = 1000 * (this.#stamps.length - 1) / (timestampMs - this.#stamps[0]);
    return Number.isFinite(rate) ? rate : null;
  }
}
