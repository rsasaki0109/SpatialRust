"""Actual Rust ORB and dense flow on a bounded, pinned TUM camera recording.

OpenCV only decodes the AVI. All feature extraction and flow are SpatialRust.
No synthesized images, camera transforms, interpolated frames or fitted arrows.
"""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

URL = "https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz-rgb.avi"
VIDEO_SHA256 = "1820b52939af2a2e7afdf78f6398cfbc7816399c8809b3836a9ba8da9bef8022"
VIDEO_BYTES = 8_059_298


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def render_video(video, output, faces, style):
    import cv2
    import spatialrust as sr

    if video.stat().st_size != VIDEO_BYTES or digest(video.read_bytes()) != VIDEO_SHA256:
        raise ValueError("require the pinned TUM freiburg1_xyz RGB movie")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("cannot decode pinned movie")
    feature_frames, flow_frames, records = [], [], []
    previous = None
    try:
        if capture.get(cv2.CAP_PROP_FPS) != 30:
            raise ValueError("unexpected source movie frame rate")
        for index in range(64):
            ok, bgr = capture.read()
            if not ok or bgr.shape != (480, 640, 3):
                raise ValueError("expected 64 consecutive original camera frames")
            rgb = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
            reduced = rgb.resize((192, 144), Image.Resampling.BOX)
            gray = np.array(reduced.convert("L"))
            flow = None if previous is None else sr.dense_flow_image(
                previous, gray, block_radius=2, search_radius=3)
            if flow is not None:
                if flow.shape != (144, 192, 2):
                    raise ValueError("invalid actual Rust flow geometry")
                valid = np.isfinite(flow).all(axis=2)
                if not valid.any() or np.abs(flow[valid]).max() > 3:
                    raise ValueError("invalid actual Rust flow range")
            if index % 2 == 1:
                keypoints, descriptors = sr.orb_features(gray, max_features=80)
                points = [[float(k.x), float(k.y)] for k in keypoints]
                if not points or descriptors.shape != (len(points), 32):
                    raise ValueError("invalid actual Rust ORB result")
                records.append(dict(source_frame=index, previous_frame=index-1,
                    decoded_rgb_sha256=digest(np.asarray(rgb).tobytes()),
                    gray_sha256=digest(gray.tobytes()),
                    previous_gray_sha256=digest(previous.tobytes()),
                    flow_sha256=digest(flow.astype("<f4").tobytes()),
                    keypoint_count=len(points),
                    keypoints_sha256=digest(np.asarray(points, dtype="<f8").tobytes()),
                    descriptors_sha256=digest(descriptors.tobytes()),
                    valid_flow_pixels=int(np.count_nonzero(valid)),
                    invalid_flow_pixels=int(np.count_nonzero(~valid)),
                    nonzero_flow_pixels=int(np.count_nonzero(valid & np.any(flow != 0, axis=2))),
                    median_flow_px=np.median(flow[valid], axis=0).tolist()))
                for kind, frames in [("features", feature_frames), ("flow", flow_frames)]:
                    image = Image.new("RGB", (464, 360), style.BG)
                    image.paste(reduced.resize((288, 216), Image.Resampling.NEAREST), (88, 64))
                    draw = ImageDraw.Draw(image)
                    style.text(draw, (22, 12), "07 / RUST ORB" if kind == "features" else "08 / RUST DENSE FLOW", faces[12], style.CYAN)
                    style.text(draw, (20, 30), "Features from real video" if kind == "features" else "Motion from real frames", faces[24])
                    if kind == "features":
                        for x, y in points:
                            x, y = 88+x*1.5, 64+y*1.5
                            draw.ellipse((x-3, y-3, x+3, y+3), outline=style.MINT, width=2)
                        detail = f"{len(points)} ORB keypoints / frame {index:02}"
                    else:
                        # Coordinates use the same 1.5x image display scale.
                        # These are raw block matches, not validated object tracks.
                        for y in range(8, 137, 12):
                            for x in range(8, 185, 12):
                                dx, dy = flow[y, x]
                                if not valid[y, x] or (dx == 0 and dy == 0):
                                    continue
                                a = (88+x*1.5, 64+y*1.5)
                                b = (a[0]+float(dx)*1.5, a[1]+float(dy)*1.5)
                                draw.line((a, b), fill=style.ORANGE, width=2)
                                draw.ellipse((b[0]-1, b[1]-1, b[0]+1, b[1]+1), fill=style.MINT)
                        detail = f"Actual pair {index-1:02} > {index:02} / pixels"
                    style.text(draw, (22, 316), detail, faces[14], style.MINT)
                    style.text(draw, (22, 338), "TUM RGB-D / fr1_xyz / CC BY 4.0 / CPU", faces[12], style.MUTED)
                    frames.append(image)
            previous = gray
    finally:
        capture.release()
    style.save_gif(output / "real_orb_features.gif", feature_frames, 125, photographic=True)
    style.save_gif(output / "real_dense_flow.gif", flow_frames, 125, photographic=True)
    return dict(schema="spatialrust.readme-real-video.v1", url=URL,
        bytes=VIDEO_BYTES, sha256=VIDEO_SHA256, license="CC BY 4.0",
        attribution="J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers. A Benchmark for the Evaluation of RGB-D SLAM Systems. IROS 2012.",
        license_source="https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license",
        source_fps=30, decoded_frames=64, processed_adjacent_pairs=63,
        rendered_source_frames=list(range(1, 64, 2)), width=192, height=144,
        frame_duration_ms=125, decoder="OpenCV "+cv2.__version__,
        processing="SpatialRust CPU orb_features and dense_flow_image; no OpenCV vision algorithms",
        flow_controls=dict(block_radius=2, search_radius=3),
        note="Raw integer block matches, no confidence filtering or object identities. Non-finite pixels (including unsupported borders) are counted and never drawn. ORB detections are independent per frame. Playback is slowed; frame numbers identify original movie frames. AVI timing is not a calibrated sensor clock.",
        renderer_sha256=digest(Path(__file__).read_bytes()), frames=records)
