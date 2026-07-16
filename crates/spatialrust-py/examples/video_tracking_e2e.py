"""Frame sequence -> dense optical flow -> deterministic object tracking.

The input is generated deterministically as portable PGM files, then loaded
through SpatialRust's bounded image IO. The same sequence is consumed by the
Rust `video_tracking_e2e` example.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import spatialrust as sr


WIDTH = 96
HEIGHT = 72
FRAME_COUNT = 12


def paint_object(image, x0, y0, width, height, base):
    for local_y in range(height):
        for local_x in range(width):
            image[y0 + local_y, x0 + local_x] = (
                base + (local_x * 7 + local_y * 11 + local_x * local_y * 3) % 25
            )


def generate_frames(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    yy, xx = np.indices((HEIGHT, WIDTH), dtype=np.int32)
    background = (20 + (xx * 3 + yy * 5) % 20).astype(np.uint8)
    for index in range(FRAME_COUNT):
        image = background.copy()
        paint_object(image, 8 + index * 2, 9 + index, 18, 14, 150)
        paint_object(image, 70 - index * 2, 46 - index, 16, 12, 220)
        path = directory / f"frame_{index:02}.pgm"
        path.write_bytes(f"P5\n{WIDTH} {HEIGHT}\n255\n".encode() + image.tobytes())


def load_frames(directory: Path) -> list[np.ndarray]:
    frames = []
    for path in sorted(directory.glob("frame_*.pgm")):
        image, _metadata = sr.read_image(str(path))
        if image.ndim != 2 or image.dtype != np.uint8:
            raise RuntimeError(f"{path} did not decode as Gray8")
        frames.append(image)
    if len(frames) < 2:
        raise RuntimeError("need at least two generated frames")
    return frames


def detect_objects(image: np.ndarray):
    foreground = image >= 100
    visited = np.zeros_like(foreground, dtype=bool)
    boxes = []
    classes = []
    for y0, x0 in np.argwhere(foreground):
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        queue = deque([(int(x0), int(y0))])
        xs = []
        ys = []
        maximum = 0
        while queue:
            x, y = queue.popleft()
            xs.append(x)
            ys.append(y)
            maximum = max(maximum, int(image[y, x]))
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if (
                    0 <= nx < image.shape[1]
                    and 0 <= ny < image.shape[0]
                    and foreground[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        if len(xs) >= 20:
            boxes.append([min(xs), min(ys), max(xs) + 1, max(ys) + 1])
            classes.append(2 if maximum >= 210 else 1)
    order = np.argsort(classes)
    return (
        np.asarray(boxes, dtype=np.float32)[order],
        np.ones(len(boxes), dtype=np.float32)[order],
        np.asarray(classes, dtype=np.int64)[order],
    )


def render_gif(frames, tracks_per_frame, flows, output: Path) -> None:
    from PIL import Image, ImageDraw

    colors = {1: (255, 91, 91), 2: (87, 211, 176)}
    rendered = []
    for index, (frame, tracks) in enumerate(zip(frames, tracks_per_frame)):
        canvas = Image.fromarray(frame, mode="L").convert("RGB").resize((WIDTH * 4, HEIGHT * 4))
        draw = ImageDraw.Draw(canvas)
        for row in tracks:
            track_id = int(row[0])
            box = [int(round(value * 4)) for value in row[1:5]]
            class_id = int(row[5])
            color = colors[class_id]
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0] + 3, box[1] + 3), f"id {track_id}", fill=color)
        if index > 0:
            flow = flows[index - 1]
            for y in range(8, HEIGHT - 8, 12):
                for x in range(8, WIDTH - 8, 12):
                    dx, dy = flow[y, x]
                    if np.isfinite(dx) and np.isfinite(dy) and abs(dx) + abs(dy) >= 1:
                        draw.line(
                            (x * 4, y * 4, (x + float(dx) * 2) * 4, (y + float(dy) * 2) * 4),
                            fill=(255, 207, 112),
                            width=2,
                        )
        draw.text((8, 8), f"frame {index:02}", fill=(240, 245, 255))
        rendered.append(canvas)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered[0].save(
        output,
        save_all=True,
        append_images=rendered[1:],
        duration=140,
        loop=0,
        optimize=False,
    )


def run(frames_dir: Path, gif_path: Path, render: bool = True) -> None:
    generate_frames(frames_dir)
    frames = load_frames(frames_dir)
    tracker = sr.MultiObjectTracker(iou_threshold=0.2, max_missed=1, min_confirmed_hits=2)
    tracks_per_frame = []
    flows = []
    boxes, scores, classes = detect_objects(frames[0])
    tracks_per_frame.append(tracker.update(boxes, scores, classes))
    for previous, current in zip(frames, frames[1:]):
        flow = sr.dense_flow_image(previous, current, block_radius=1, search_radius=3)
        previous_boxes, _previous_scores, previous_classes = detect_objects(previous)
        observed = []
        for box, class_id in zip(previous_boxes, previous_classes):
            x = int((box[0] + box[2]) * 0.5)
            y = int((box[1] + box[3]) * 0.5)
            observed.append((int(class_id), *flow[y, x].tolist()))
        if observed != [(1, 2.0, 1.0), (2, -2.0, -1.0)]:
            raise RuntimeError(f"unexpected object flow: {observed}")
        flows.append(flow)
        boxes, scores, classes = detect_objects(current)
        tracks = tracker.update(boxes, scores, classes)
        if len(tracks) != 2 or [track[0] for track in tracks] != [1, 2]:
            raise RuntimeError(f"unstable tracks: {tracks}")
        tracks_per_frame.append(tracks)
    if render:
        render_gif(frames, tracks_per_frame, flows, gif_path)
    print(
        f"video_tracking_e2e=ok frames={len(frames)} "
        f"flow_pairs={len(flows)} stable_track_ids=1,2 "
        f"gif={gif_path if render else 'skipped'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames-dir", type=Path, default=Path("target/video-tracking-demo/frames")
    )
    parser.add_argument(
        "--gif", type=Path, default=Path("docs/assets/video_tracking_e2e.gif")
    )
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    run(args.frames_dir, args.gif, render=not args.no_gif)


if __name__ == "__main__":
    main()
