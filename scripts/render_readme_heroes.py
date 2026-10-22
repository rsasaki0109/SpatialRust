"""Render README animations from SpatialRust results; no downloads or GPU claims.

Requires the SpatialRust Python extension, NumPy and Pillow. Supply the public
PCL table_scene_lms400.pcd and the pinned real TUM RGB movie explicitly.
Only the nine opening GIFs and a small provenance receipt are written to --output.
Frames are encoded in memory, without a temporary image sequence on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
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
NAMES = ("readme_hero.gif", "clusters_rotating.gif", "voxelize_rotating.gif", "real_orb_features.gif",
         "scan_reveal.gif", "voxel_filter.gif", "plane_segmentation.gif", "plane_removal.gif", "real_dense_flow.gif")
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


def cloud_view(points, colors, size, phase, radius=1, voxel_size=None, bounds=None):
    """Orthographic view of original XYZ, using the scan's camera-like Y axis."""
    w, h = size
    reference = points if bounds is None else bounds
    center = (reference.min(axis=0) + reference.max(axis=0)) * .5
    p = points - center
    yaw = .28 + .20 * math.sin(phase)
    depth = -math.sin(yaw) * p[:, 0] + math.cos(yaw) * p[:, 2]
    x = math.cos(yaw) * p[:, 0] + math.sin(yaw) * p[:, 2]
    y = .94 * p[:, 1] - .34 * depth
    # One fixed scale derived from XYZ extents: no per-stage zoom or cropping.
    extent = np.ptp(reference, axis=0)
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


def card(accent=CYAN):
    image = Image.new("RGB", (464, 340), BG)
    d = ImageDraw.Draw(image)
    # A quiet technical grid frames the data without obscuring it.
    for x in range(20, 464, 28):
        d.line((x, 67, x, 283), fill="#101d30")
    for y in range(67, 284, 28):
        d.line((20, y, 444, y), fill="#101d30")
    d.rounded_rectangle((18, 292, 446, 326), radius=8, fill=PANEL)
    d.rectangle((18, 292, 21, 326), fill=accent)
    return image


def gallery(output, xyz, small, plane, outliers, faces):
    """Visualize actual Rust output membership; orbit is a presentation effect."""
    bounds = np.array([xyz.min(axis=0), xyz.max(axis=0)])
    zmin, zmax = bounds[:, 2]
    depth = ((xyz[:, 2] - zmin) / max(zmax-zmin, 1e-9)).clip(0, 1)
    depth_colors = [COLORS[min(int(v*5), 5)] for v in depth]
    entries = [
        (NAMES[4], "01 / CAPTURE", "A scene, in points", f"{len(xyz):,} public scan samples", CYAN),
        (NAMES[5], "02 / SIMPLIFY", "Keep the structure", f"{len(xyz):,} points  >  {len(small):,} centroids", MINT),
        (NAMES[6], "03 / SEGMENT", "Find the surface", f"{len(plane):,} RANSAC plane inliers", ORANGE),
        (NAMES[7], "06 / REMOVE PLANE", "Keep the objects", f"{len(small):,} centroids > {len(outliers):,} outliers", "#bd9bff"),
    ]
    for name, eyebrow, title, detail, accent in entries:
        frames=[]
        for i in range(32):
            phase=i/32*math.tau
            if name==NAMES[4]:
                pts,colors=xyz,depth_colors
            elif name==NAMES[5]:
                # Alternation compares actual input and actual centroids.
                pts=xyz if i<16 else small
                colors=CYAN if i<16 else MINT
            elif name==NAMES[6]:
                pts=np.concatenate((plane,outliers))
                colors=[CYAN]*len(plane)+[ORANGE]*len(outliers)
            else:
                pts=small if i<16 else outliers
                colors=CYAN if i<16 else ORANGE
            image=card(accent)
            image.paste(cloud_view(pts,colors,(424,211),phase,1 if len(pts)>10000 else 2,bounds=bounds),(20,76))
            d=ImageDraw.Draw(image)
            text(d,(22,12),eyebrow,faces[12],accent)
            text(d,(20,32),title,faces[24])
            text(d,(32,300),detail,faces[14],accent)
            if name==NAMES[5]:
                pill(d,(296,76),"INPUT" if i<16 else "FILTERED",faces[12],colors)
            elif name==NAMES[6]:
                pill(d,(22,76),"PLANE + OBJECTS",faces[12],ORANGE)
            elif name==NAMES[7]:
                pill(d,(296,76),"INPUT" if i<16 else "PLANE OUT",faces[12],colors)
            for j in range(32):
                d.rectangle((22+j*13,333,31+j*13,335),fill=accent if j<=i else PANEL)
            frames.append(image)
        save_gif(output/name,frames,125)


def palette():
    # Fixed palette avoids both temporal color flicker and per-frame dithering.
    values = [Image.new("RGB", (1, 1), c).getpixel((0, 0)) for c in [BG, PANEL, INK, MUTED, CYAN, MINT, ORANGE, *COLORS[3:], "#294054", "#24384b"]]
    values += [(r, g, b) for r in range(0, 256, 51) for g in range(0, 256, 51) for b in range(0, 256, 51)]
    values = (values + [(0, 0, 0)] * 256)[:256]
    p = Image.new("P", (1, 1))
    p.putpalette([c for rgb in values for c in rgb])
    return p


def save_gif(path, frames, duration, photographic=False):
    p = palette()
    if photographic:
        # One shared palette learned from the complete real clip, never one
        # independently chosen palette per frame. Preserve natural image tones.
        sample = Image.new("RGB", (128, 96 * len(frames)))
        for index, frame in enumerate(frames):
            sample.paste(frame.resize((128, 96)), (0, index * 96))
        p = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    indexed = [im.quantize(palette=p, dither=Image.Dither.NONE) for im in frames]
    indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=duration,
                    loop=0, optimize=False, disposal=1)
    if path.stat().st_size > MAX_GIF_BYTES:
        raise ValueError(f"GIF exceeds {MAX_GIF_BYTES} bytes: {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
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
        d.rectangle((0, 0, 959, 3), fill=COLORS[stage])
        text(d, (32, 27), "CAPTURE / UNDERSTAND / BUILD", faces[12], CYAN)
        text(d, (30, 71), "SpatialRust", faces[42])
        text(d, (32, 141), "Make space", faces[32])
        text(d, (32, 183), "computable.", faces[32], COLORS[stage])
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
        text(d, (404, 26), f"0{stage+1} / {stage_names[stage]}", faces[14], COLORS[stage])
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

    gallery(args.output, xyz, small, plane, outliers, faces)
    from readme_real_video import render_video
    video_receipt = render_video(args.video, args.output, faces, sys.modules[__name__])

    assets = [{"path": name, "bytes": (args.output/name).stat().st_size, "sha256": sha(args.output/name)} for name in NAMES]
    if sum(a["bytes"] for a in assets) > MAX_TOTAL_BYTES:
        raise ValueError("hero group exceeds 3 MB budget")
    extensions = list(Path(sr.__file__).parent.glob("*.pyd")) + list(Path(sr.__file__).parent.glob("*.so"))
    receipt = dict(schema="spatialrust.readme-heroes.v2", assets=assets,
                   public_input=dict(file=args.input.name, sha256=sha(args.input), points=len(source), sampled_points=len(xyz)),
                   execution="SpatialRust Python bindings (voxel policy cpu); CPU Rust ORB and dense flow on real recorded frames",
                   spatialrust_version=sr.__version__, resolved_policies=policies,
                   python_extension_sha256=sha(extensions[0]) if len(extensions) == 1 else None,
                   generator_sha256=sha(__file__),
                   video_renderer_sha256=sha(Path(__file__).with_name("readme_real_video.py")),
                   numpy_version=np.__version__, pillow_version=pillow_version,
                   font_sha256=sha(args.font),
                   pipeline=dict(voxel_leaf_m=.03, plane_distance_m=.025, cluster_tolerance_m=.06,
                                 min_cluster_size=8, downsampled_points=len(small),
                                 plane_inliers=len(plane), cluster_count=result.cluster_count,
                                 occupancy_cell_m=.08, occupied_cells=len(centers)),
                   real_video=video_receipt,
                   claims="Actual Rust outputs on real public scan and recorded camera frames. Orbit and stage switching are presentation effects; movie playback is slowed. No synthetic motion, GPU speed, metric pose or SLAM accuracy claim.")
    (args.output / "readme_heroes.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
