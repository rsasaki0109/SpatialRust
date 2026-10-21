// Own one camera/worker epoch. Late async completions never reopen a stopped session.
export class CameraSession {
  constructor({ acquire, createWorker, onReady = async () => {}, onResult = () => {}, onState = () => {}, onError = () => {}, onStop = () => {}, timeoutMs = 45000 }) {
    Object.assign(this, { acquire, createWorker, onReady, onResult, onState, onError, onStop, timeoutMs });
    this.epoch = 0;
    this.run = null;
  }

  get token() { return this.run?.token ?? 0; }
  get busy() { return !this.run?.started || this.run.inFlight; }

  async start(mode, ai = true) {
    this.stop();
    const run = { token: ++this.epoch, mode, ai, worker: null, stream: null, started: false, inFlight: false };
    this.run = run;
    this.onState("loading");
    try {
      run.worker = this.createWorker();
      const ready = new Promise((resolve, reject) => {
        run.rejectReady = reject;
        run.worker.onmessage = ({ data }) => {
          if (this.run !== run || data.session !== run.token) return;
          if (data.kind === "ready") { run.backend = data.backend; resolve(); }
          else if (data.kind === "result" && run.started) {
            run.inFlight = false;
            try { this.onResult(data); } catch (error) { this.fail(run, error); }
          } else if (data.kind === "error") {
            const error = new Error(data.message || "処理を開始できませんでした。");
            reject(error);
            this.fail(run, error);
          }
        };
        run.worker.onerror = (event) => {
          const error = new Error(event.message || "ワーカーを起動できませんでした。");
          reject(error);
          this.fail(run, error);
        };
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
      await Promise.all([ready, camera]);
      if (this.run !== run) return false;
      await this.onReady(run);
      if (this.run !== run) return false;
      if (mode === "camera" && !this.checkCamera(run)) return false;
      clearTimeout(run.timer);
      run.started = true;
      this.onState("running", run);
      return true;
    } catch (error) {
      if (this.run === run) this.fail(run, error);
      return false;
    }
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
      run.inFlight = true;
      run.worker.postMessage({ kind: "frame", session: run.token, bitmap, timestampMs }, [bitmap]);
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
      clearTimeout(run.timer);
      run.rejectReady?.(new Error("stopped"));
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
