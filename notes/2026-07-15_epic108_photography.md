# Epic 108: computational photography and panorama

Epic 108 adds runtime-free gray-world correction, aligned exposure fusion, and
bounded pairwise panorama composition. Source-to-target transforms and output
origins are explicit, image metadata must agree, and the output pixel ceiling
is checked before allocation.

The OpenCV 4.10 receipt produced a `32x88` panorama at origin `(0, 0)`.
SpatialRust's source-only warped region matched `cv2.warpPerspective` exactly
(maximum component error `0`), and gray-world correction reduced the synthetic
channel-mean spread to `0.0`.
