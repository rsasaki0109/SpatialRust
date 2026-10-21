// Own one camera/worker epoch. Late async completions never reopen a stopped session.
export class CameraSession {
  constructor({ acquire, createWorker, onReady = async () => {}, onResult = () => {}, onState = () => {}, onError = () => {}, onStop = () => {}, timeoutMs = 45000, resultTimeoutMs = 15000 }) {
    if (!Number.isInteger(resultTimeoutMs) || resultTimeoutMs <= 0 || resultTimeoutMs > 2147483647) throw new RangeError("invalid result timeout");
    Object.assign(this, { acquire, createWorker, onReady, onResult, onState, onError, onStop, timeoutMs, resultTimeoutMs });
    this.epoch = 0;
    this.run = null;
  }

  get token() { return this.run?.token ?? 0; }
  get busy() { return !this.run?.started || !!this.run.inFlight; }

  async start(mode, ai = true) {
    this.stop();
    const run = { token: ++this.epoch, mode, ai, worker: null, stream: null, started: false, inFlight: null, frameId: 0, captureAbort: new AbortController() };
    this.run = run;
    // Release the application wait even if native camera/play work never settles.
    const cancelled = new Promise((resolve) => { run.cancelStart = resolve; });
    this.onState("loading");
    try {
      run.worker = this.createWorker();
      const ready = new Promise((resolve) => {
        // Cleanup resolves readiness; errors follow run cancellation.
        run.resolveReady = resolve;
        run.worker.onmessage = ({ data }) => {
          if (this.run !== run || data.session !== run.token) return;
          if (data.kind === "ready") { run.backend = data.backend; resolve(); }
          else if (data.kind === "result" && run.started && run.inFlight && data.frameId === run.inFlight.frameId) {
            clearTimeout(run.inFlight.timer);
            run.inFlight = null;
            try { this.onResult(data); } catch (error) { this.fail(run, error); }
          } else if (data.kind === "error") {
            const error = new Error(data.message || "処理を開始できませんでした。");
            this.fail(run, error);
          }
        };
        run.worker.onerror = (event) => {
          const error = new Error(event.message || "ワーカーを起動できませんでした。");
          this.fail(run, error);
        };
        run.worker.addEventListener("messageerror", () => {
          const error = new Error("処理との通信に失敗しました。もう一度「カメラ開始」か「サンプル再生」を押してください。");
          error.name = "WorkerMessageError";
          this.fail(run, error);
        }, { signal: run.captureAbort.signal });
      });
      run.timer = setTimeout(() => this.fail(run, new Error("準備に時間がかかっています。通信を確認して、もう一度お試しください。")), this.timeoutMs);
      run.worker.postMessage({ kind: "init", session: run.token, ai });
      const camera = mode === "camera" ? Promise.resolve().then(() => this.run === run ? this.acquire() : null).then((stream) => {
        if (!stream) return null;
        if (this.run !== run) { stopTracks(stream); return null; }
        run.stream = stream;
        run.cameraEvents = new AbortController();
        const onEnd = () => this.checkCamera(run);
        const options = { signal: run.cameraEvents.signal };
        stream.addEventListener?.("inactive", onEnd, options);
        for (const track of stream.getTracks()) if (track.kind === "video") track.addEventListener("ended", onEnd, options);
        if (!this.checkCamera(run)) return null;
        return stream;
      }) : Promise.resolve(null);
      await Promise.race([Promise.all([ready, camera]), cancelled]);
      if (this.run !== run) return false;
      await Promise.race([this.onReady(run), cancelled]);
      if (this.run !== run) return false;
      if (mode === "camera" && !this.checkCamera(run)) return false;
      clearTimeout(run.timer);
      run.started = true;
      this.onState("running", run);
      return true;
    } catch (error) {
      if (this.run === run) this.fail(run, error);
      return false;
    } finally { run.cancelStart = null;run.resolveReady = null; }
  }

  checkCamera(run) {
    if (this.run !== run || !run?.stream) return false;
    if (run.stream.active === false || run.stream.getTracks().some((track) => track.kind === "video" && track.readyState === "ended")) {
      const error = new Error("カメラ映像が終了しました。もう一度「カメラ開始」を押すか、サンプルをお試しください。");
      error.name = "CameraEndedError";
      this.fail(run, error);
      return false;
    }
    return true;
  }

  submit(bitmap, timestampMs) {
    const run = this.run;
    if (!run?.started || run.inFlight) { bitmap.close(); return false; }
    try {
      const pending = { frameId: ++run.frameId, timer: null };
      run.inFlight = pending;
      pending.timer = setTimeout(() => {
        if (this.run !== run || run.inFlight !== pending) return;
        const error = new Error(`処理結果が${this.resultTimeoutMs / 1000}秒間返らなかったため停止しました。もう一度「カメラ開始」か「サンプル再生」を押してください。`);
        error.name = "FrameResultTimeoutError";
        this.fail(run, error);
      }, this.resultTimeoutMs);
      run.worker.postMessage({ kind: "frame", session: run.token, frameId: pending.frameId, bitmap, timestampMs }, [bitmap]);
      return true;
    } catch (error) {
      bitmap.close();
      this.fail(run, error);
      return false;
    }
  }

  fail(run, error) {
    if (this.run !== run) return;
    this.stop();
    this.onState("error");
    this.onError(error);
  }

  stop() {
    const run = this.run;
    this.run = null;
    ++this.epoch;
    if (run) {
      run.cancelStart?.();
      clearTimeout(run.timer);
      clearTimeout(run.inFlight?.timer);
      run.captureAbort.abort();
      run.resolveReady?.();
      run.cameraEvents?.abort();
      stopTracks(run.stream);
      run.worker?.terminate();
      run.worker?.dispose?.();
    }
    this.onStop();
    this.onState("idle");
  }
}

function stopTracks(stream) {
  for (const track of stream?.getTracks() ?? []) track.stop();
}
