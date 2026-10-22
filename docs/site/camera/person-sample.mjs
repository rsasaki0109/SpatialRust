// Load this public, size/SHA-pinned builder asset only after explicit photo play.
const PERSON_PHOTO = new URL("./vendor/person-sample.png", import.meta.url).href;

export function loadPersonSample({ signal, createImage = () => new Image() } = {}) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new DOMException("sample stopped", "AbortError"));return; }
    const image = createImage();
    let settled = false;
    const finish = (error) => {
      if (settled) return;
      settled = true;
      image.removeEventListener("load", onLoad);
      image.removeEventListener("error", onError);
      signal?.removeEventListener("abort", onAbort);
      if (error) { image.src = "";reject(error); }
      else resolve(image);
    };
    const failure = () => {
      const error = new Error("人物サンプルを読み込めませんでした。「サンプル再生」か「カメラ開始」をお試しください。");
      error.name = "SampleImageError";
      return error;
    };
    const onLoad = () => finish(image.naturalWidth === 512 && image.naturalHeight === 512 ? null : failure());
    const onError = () => finish(failure());
    const onAbort = () => finish(new DOMException("sample stopped", "AbortError"));
    image.addEventListener("load", onLoad);
    image.addEventListener("error", onError);
    signal?.addEventListener("abort", onAbort, { once: true });
    image.src = PERSON_PHOTO;
  });
}
