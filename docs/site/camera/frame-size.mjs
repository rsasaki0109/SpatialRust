// Keep image-space coordinates bounded without stretching an extreme aspect ratio.
export function cameraFrameSize(sourceWidth, sourceHeight) {
  if (![sourceWidth, sourceHeight].every((n) => Number.isSafeInteger(n) && n > 0)) throw new RangeError("カメラの映像サイズを読み取れませんでした。");
  const scale = 256 / Math.max(sourceWidth, sourceHeight);
  const width = Math.round(sourceWidth * scale), height = Math.round(sourceHeight * scale);
  if (width < 64 || height < 64 || width > 256 || height > 256) throw new RangeError("映像の縦横比に対応していません。別のカメラでお試しください。");
  return { width, height };
}
