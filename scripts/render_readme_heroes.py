"""Render README animations from SpatialRust results; no downloads or GPU claims.

Requires the SpatialRust Python extension, NumPy and Pillow. Supply the public
PCL table_scene_lms400.pcd and reports from readme_tracking_data.mjs explicitly.
Only these four GIFs and a small provenance receipt are written to --output.
Frames are encoded in memory, without a temporary image sequence on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

BG = "#080f1c"
PANEL = "#101d30"
INK = "#ecf5ff"
MUTED = "#94abc3"
CYAN = "#61dfef"
MINT = "#79f2bd"
ORANGE = "#ffbb79"
COLORS = [CYAN, ORANGE, MINT, "#bd9bff", "#ff89b1", "#fae17a"]
NAMES = ("readme_hero.gif", "clusters_rotating.gif", "voxelize_rotating.gif", "camera_tracking.gif")
MAX_GIF_BYTES = 1_500_000
MAX_TOTAL_BYTES = 3_000_000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(path, size):
    return ImageFont.truetype(str(path), size)


def text(draw, xy, value, face, color=INK):
    draw.text(xy, value, font=face, fill=color)


def pill(draw, xy, value, face, color=CYAN):
    x, y = xy
    w = draw.textlength(value, font=face) + 24
    draw.rounded_rectangle((x, y, x + w, y + 27), radius=13, fill=PANEL, outline="#294054")
    text(draw, (x + 12, y + 4), value, face, color)


def cloud_view(points, colors, size, phase, radius=1, voxel_size=None):
    """Orthographic view of original XYZ, using the scan's camera-like Y axis."""
    w, h = size
    center = (points.min(axis=0) + points.max(axis=0)) * .5
    p = points - center
    yaw = .28 + .20 * math.sin(phase)
    depth = -math.sin(yaw) * p[:, 0] + math.cos(yaw) * p[:, 2]
    x = math.cos(yaw) * p[:, 0] + math.sin(yaw) * p[:, 2]
    y = .94 * p[:, 1] - .34 * depth
    # One fixed scale derived from XYZ extents: no per-stage zoom or cropping.
    extent = np.ptp(points, axis=0)
    scale = min((w - 40) / (extent[0] + .5 * extent[2]), (h - 32) / (extent[1] + .4 * extent[2]))
    xy = np.column_stack((w * .5 + x * scale, h * .5 + y * scale)).round().astype(int)
    view = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(view)
    for i in np.argsort(-depth, kind="stable"):
        px, py = xy[i]
        color = colors[i] if not isinstance(colors, str) else colors
        if voxel_size is not None and i < len(points) - 2:
            corners = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                                [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]]) * (voxel_size * .5) + p[i]
            zc = -math.sin(yaw) * corners[:, 0] + math.cos(yaw) * corners[:, 2]
            xc = math.cos(yaw) * corners[:, 0] + math.sin(yaw) * corners[:, 2]
            yc = .94 * corners[:, 1] - .34 * zc
            vertices = np.column_stack((w * .5 + xc * scale, h * .5 + yc * scale))
            for face, fill in [([0,1,2,3], "#2396b4"), ([1,2,6,5], "#28718f"), ([0,1,5,4], CYAN)]:
                d.polygon([tuple(vertices[j]) for j in face], fill=fill, outline="#153b50")
        elif radius == 1:
            d.point((int(px), int(py)), fill=color)
        else:
            d.ellipse((int(px-radius), int(py-radius), int(px+radius), int(py+radius)), fill=color)
    return view


def palette():
    # Fixed palette avoids both temporal color flicker and per-frame dithering.
    values = [Image.new("RGB", (1, 1), c).getpixel((0, 0)) for c in [BG, PANEL, INK, MUTED, CYAN, MINT, ORANGE, *COLORS[3:], "#294054", "#24384b"]]
    values += [(r, g, b) for r in range(0, 256, 51) for g in range(0, 256, 51) for b in range(0, 256, 51)]
    values = (values + [(0, 0, 0)] * 256)[:256]
    p = Image.new("P", (1, 1))
    p.putpalette([c for rgb in values for c in rgb])
    return p


def save_gif(path, frames, duration):
    p = palette()
    indexed = [im.quantize(palette=p, dither=Image.Dither.NONE) for im in frames]
    indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=duration,
                    loop=0, optimize=False, disposal=1)
    if path.stat().st_size > MAX_GIF_BYTES:
        raise ValueError(f"GIF exceeds {MAX_GIF_BYTES} bytes: {path.name}")


def texture(width, height, shift):
    yy, xx = np.indices((height, width), dtype=np.uint32)
    sx = xx + np.uint32(32 - shift[0])
    sy = yy + np.uint32(32 - shift[1])
    h = (sx * np.uint32(73856093)) ^ (sy * np.uint32(19349663))
    return (np.where(((sx >> 4) + (sy >> 4)) % 2, 125, 35) + ((h ^ (h >> 13)) & 31)).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tracking", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True, help="TrueType/OpenType sans font")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import spatialrust as sr

    args.output.mkdir(parents=True, exist_ok=True)
    faces = {n: font(args.font, n) for n in [12, 14, 16, 19, 24, 32, 42]}
    source = sr.read(str(args.input))
    xyz = source.xyz()
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    # Bounded, deterministic public input selection, matching the prior hero cap.
    xyz = np.ascontiguousarray(xyz[::max(1, math.ceil(len(xyz) / 80_000))], dtype=np.float32)
    cloud = sr.PointCloud.from_xyz(xyz)
    result = sr.run_pipeline(cloud, leaf_size=.03, plane_distance=.025,
                             cluster_tolerance=.06, min_cluster_size=8, policy="cpu")
    small = result.downsampled.xyz()
    labeled = result.output.xyz()
    labels = result.labels()
    label_colors = [COLORS[int(v) % len(COLORS)] if v >= 0 else "#294054" for v in labels]
    grid, origin, voxel_size = sr.voxelize(cloud, voxel_size=.08, mode="occupancy")
    indices = np.argwhere(grid > 0)[:, ::-1]
    centers = np.asarray(origin) + (indices + .5) * voxel_size
    policies = getattr(result, "resolved_policies", None)
    # Pipeline output is the labeled plane-outlier cloud. Recover its exact
    # complement in the returned downsampled cloud without fitting another plane.
    row_type = np.dtype((np.void, small.dtype.itemsize * 3))
    outlier_rows = np.ascontiguousarray(labeled).view(row_type).ravel()
    outlier_mask = np.isin(np.ascontiguousarray(small).view(row_type).ravel(), outlier_rows)
    plane, outliers = small[~outlier_mask], small[outlier_mask]
    if len(plane) != result.plane_inliers or len(outliers) != len(labeled):
        raise ValueError("pipeline output cannot identify plane membership exactly")
    stages = [(xyz, CYAN), (small, INK),
              (np.concatenate((plane, outliers)), [CYAN] * len(plane) + [ORANGE] * len(outliers)),
              (labeled, label_colors)]
    stage_names = ["PUBLIC SCAN", "VOXEL FILTER", "PLANE RANSAC", "OBJECT CLUSTERS"]
    counts = [f"{len(xyz):,} sampled points", f"{len(small):,} centroids", f"{len(plane):,} plane inliers", f"{result.cluster_count} clusters / {len(labeled):,} points"]
    header = []
    for i in range(48):
        stage = i // 12
        image = Image.new("RGB", (960, 420), BG)
        d = ImageDraw.Draw(image)
        text(d, (32, 27), "SPATIAL COMPUTING / RUST", faces[12], CYAN)
        text(d, (30, 71), "SpatialRust", faces[42])
        text(d, (32, 141), "From sensor data", faces[24])
        text(d, (32, 174), "to spatial intelligence.", faces[24])
        text(d, (32, 225), "Point clouds. Images. Robotics.", faces[16], MUTED)
        pill(d, (32, 270), "RUST", faces[12])
        pill(d, (112, 270), "wgpu", faces[12], MINT)
        pill(d, (197, 270), "WASM", faces[12], ORANGE)
        text(d, (32, 330), "REAL DATA / ACTUAL RUST OUTPUT", faces[12], MUTED)
        text(d, (32, 352), "PCL table_scene_lms400", faces[14])
        pts, cs = stages[stage]
        # Use shared input bounds for every stage, including sparse object output.
        padding = np.array([xyz.min(axis=0), xyz.max(axis=0)])
        colors = ([cs] * len(pts) if isinstance(cs, str) else cs) + [BG, BG]
        view = cloud_view(np.concatenate((pts, padding)), colors, (570, 302), i / 48 * math.tau, 1 if stage == 0 else 2)
        image.paste(view, (378, 32))
        d = ImageDraw.Draw(image)
        text(d, (404, 26), stage_names[stage], faces[14], CYAN)
        text(d, (404, 324), counts[stage], faces[16])
        text(d, (404, 352), "Rust pipeline / animated viewpoint", faces[12], MUTED)
        for j, name in enumerate(["SCAN", "FILTER", "SEGMENT", "CLUSTER"]):
            x = 32 + j * 232
            d.rounded_rectangle((x, 390, x + 216, 393), radius=1, fill=CYAN if j == stage else PANEL)
            text(d, (x, 372), name, faces[12], INK if j == stage else MUTED)
        header.append(image)
    save_gif(args.output / NAMES[0], header, 125)

    for name, points, colors, title, detail, radius in [
        (NAMES[1], labeled, label_colors, "Separate the scene", f"{result.cluster_count} Euclidean clusters / Rust", 2),
        (NAMES[2], centers, CYAN, "Build spatial structure", f"{len(centers):,} occupied cells / 0.08 m", 3),
    ]:
        frames = []
        for i in range(32):
            image = Image.new("RGB", (464, 340), BG)
            padding = np.array([xyz.min(axis=0), xyz.max(axis=0)])
            cs = ([colors] * len(points) if isinstance(colors, str) else colors) + [BG, BG]
            image.paste(cloud_view(np.concatenate((points, padding)), cs, (448, 235), i / 32 * math.tau,
                                   radius, voxel_size if name == NAMES[2] else None), (8, 58))
            d = ImageDraw.Draw(image)
            text(d, (22, 18), title, faces[24])
            text(d, (22, 305), detail, faces[14], CYAN)
            frames.append(image)
        save_gif(args.output / name, frames, 150)

    tracking = json.loads(args.tracking.read_text(encoding="utf-8"))
    if tracking["schema"] != "spatialrust.readme-tracking.v1" or not tracking["synthetic_only"] or tracking["ai_enabled"]:
        raise ValueError("expected synthetic unmasked actual-WASM reports")
    frames = []
    for f in tracking["frames"]:
        gray = texture(tracking["width"], tracking["height"], f["shift"])
        if hashlib.sha256(gray.tobytes()).hexdigest() != f["gray_sha256"]:
            raise ValueError("rendered image differs from actual WASM input")
        rgb = np.stack((gray // 3, (gray.astype(np.uint16) * 1.1).clip(0, 255).astype(np.uint8), gray), axis=-1)
        scene = Image.fromarray(rgb).resize((384, 288), Image.Resampling.NEAREST)
        image = Image.new("RGB", (464, 400), BG)
        image.paste(scene, (40, 64))
        d = ImageDraw.Draw(image)
        text(d, (22, 16), "Track image motion", faces[24])
        for t in f["report"]["tracks"]:
            a = (40 + t["from"][0] * 1.5, 64 + t["from"][1] * 1.5)
            b = (40 + t["to"][0] * 1.5, 64 + t["to"][1] * 1.5)
            d.line((a, b), fill=MINT, width=2)
            d.ellipse((b[0]-2, b[1]-2, b[0]+2, b[1]+2), fill=INK)
        text(d, (22, 363), f"{len(f['report']['tracks']):02d} tracks / Rust + WASM", faces[14], MINT)
        flow = f["report"]["background_flow_px"]
        if flow is not None:
            text(d, (282, 363), f"dx {flow[0]:+.1f} px", faces[14], MUTED)
        pill(d, (282, 18), "SYNTHETIC / AI OFF", faces[12], ORANGE)
        frames.append(image)
    save_gif(args.output / NAMES[3], frames, tracking["frame_duration_ms"])

    assets = [{"path": name, "bytes": (args.output/name).stat().st_size, "sha256": sha(args.output/name)} for name in NAMES]
    if sum(a["bytes"] for a in assets) > MAX_TOTAL_BYTES:
        raise ValueError("hero group exceeds 3 MB budget")
    extensions = list(Path(sr.__file__).parent.glob("*.pyd")) + list(Path(sr.__file__).parent.glob("*.so"))
    receipt = dict(schema="spatialrust.readme-heroes.v1", assets=assets,
                   public_input=dict(file=args.input.name, sha256=sha(args.input), points=len(source), sampled_points=len(xyz)),
                   execution="SpatialRust Python bindings (voxel policy cpu); CPU Rust/WASM camera tracker",
                   spatialrust_version=sr.__version__, resolved_policies=policies,
                   python_extension_sha256=sha(extensions[0]) if len(extensions) == 1 else None,
                   generator_sha256=sha(__file__),
                   tracking_generator_sha256=sha(Path(__file__).with_name("readme_tracking_data.mjs")),
                   numpy_version=np.__version__, pillow_version=pillow_version,
                   font_sha256=sha(args.font),
                   pipeline=dict(voxel_leaf_m=.03, plane_distance_m=.025, cluster_tolerance_m=.06,
                                 min_cluster_size=8, downsampled_points=len(small),
                                 plane_inliers=len(plane), cluster_count=result.cluster_count,
                                 occupancy_cell_m=.08, occupied_cells=len(centers)),
                   tracking_wasm_sha256=tracking["wasm_sha256"], tracking_frames=len(tracking["frames"]),
                   claims="Illustrations of actual outputs; animation playback is not a throughput benchmark. Synthetic tracking has AI disabled.")
    (args.output / "readme_heroes.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
