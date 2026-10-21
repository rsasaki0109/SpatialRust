// Keep image-space coordinates bounded without stretching an extreme aspect ratio.
export function cameraFrameSize(sourceWidth, sourceHeight, longEdge = 256) {
  if (longEdge !== 192 && longEdge !== 256) throw new RangeError("処理サイズを選び直してください。");
  if (![sourceWidth, sourceHeight].every((n) => Number.isSafeInteger(n) && n > 0)) throw new RangeError("カメラの映像サイズを読み取れませんでした。");
  const scale = longEdge / Math.max(sourceWidth, sourceHeight);
  const width = Math.round(sourceWidth * scale), height = Math.round(sourceHeight * scale);
  if (width < 64 || height < 64 || width > longEdge || height > longEdge) throw new RangeError("映像の縦横比に対応していません。標準設定や別のカメラでお試しください。");
  return { width, height };
}
