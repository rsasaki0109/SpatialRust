// The pinned model has one "selfie" confidence channel; category IDs vary by SDK.
export function personMask(source, sourceWidth, sourceHeight, width, height) {
  if (!(source instanceof Float32Array) || ![sourceWidth, sourceHeight, width, height].every((n) => Number.isInteger(n) && n > 0 && n <= 320) || source.length !== sourceWidth * sourceHeight) throw new Error("人物マスクのサイズが想定と異なります。");
  for (const value of source) if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error("人物マスクの値が範囲外です。");
  const mask = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const row = Math.floor(y * sourceHeight / height) * sourceWidth;
    for (let x = 0; x < width; x++) mask[y * width + x] = source[row + Math.floor(x * sourceWidth / width)] > 0.5 ? 1 : 0;
  }
  return mask;
}
