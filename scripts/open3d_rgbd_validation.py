"""Bounded producer-side raw warp validation, independent of the full-frame audit."""
import numpy as np

from evaluate_pose_reference import rigid


def raw_statistics(source_counts, target_counts, source_gray, target_gray, pose, camera, settings):
    pose = rigid(pose)
    source = (source_counts.astype(np.float64)*camera['depth_unit_m']).astype(np.float32)
    target = (target_counts.astype(np.float64)*camera['depth_unit_m']).astype(np.float32).astype(np.float64)
    good_target = np.isfinite(target) & (target >= settings['min_depth']) & (target <= settings['max_depth'])
    target[~good_target] = np.nan
    height, width = source.shape
    total, count, photo_square, depth_square = 0, 0, 0., 0.
    for start in range(0, height, 16):
        tile = source[start:start+16].astype(np.float64)
        yy, xx = np.indices(tile.shape)
        yy += start
        valid = np.isfinite(tile) & (tile >= settings['min_depth']) & (tile <= settings['max_depth'])
        total += int(np.count_nonzero(valid))
        z, x, y = tile[valid], xx[valid], yy[valid]
        px, py = (x-camera['cx'])*z/camera['fx'], (y-camera['cy'])*z/camera['fy']
        moved = [pose[a, 0]*px + pose[a, 1]*py + pose[a, 2]*z + pose[a, 3] for a in range(3)]
        positive = np.isfinite(moved[2]) & (moved[2] > 0)
        x, y = x[positive], y[positive]
        moved = [v[positive] for v in moved]
        u = camera['fx']*moved[0]/moved[2] + camera['cx']
        v = camera['fy']*moved[1]/moved[2] + camera['cy']
        inside = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0) & (u < width-1) & (v < height-1)
        u, v, zc, x, y = u[inside], v[inside], moved[2][inside], x[inside], y[inside]
        ix, iy = u.astype(np.int64), v.astype(np.int64)
        du, dv = u-ix, v-iy
        a, b, c, d = target[iy, ix], target[iy, ix+1], target[iy+1, ix], target[iy+1, ix+1]
        depth = (a*(1-du)+b*du)*(1-dv) + (c*(1-du)+d*du)*dv
        residual = depth-zc
        accepted = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d) & (np.abs(residual) <= settings['max_depth_difference'])
        count += int(np.count_nonzero(accepted))
        depth_square += float(np.sum(residual[accepted]**2))
        a, b, c, d = [target_gray[ry, rx].astype(np.float64)/255. for ry, rx in
                       ((iy, ix), (iy, ix+1), (iy+1, ix), (iy+1, ix+1))]
        photo = (a*(1-du)+b*du)*(1-dv) + (c*(1-du)+d*du)*dv - source_gray[y, x].astype(np.float64)/255.
        photo_square += float(np.sum(photo[accepted]**2))
    return dict(valid_source_pixels=total, correspondences=count,
        correspondence_fraction=count/total if total else 0.,
        photometric_rmse=float(np.sqrt(photo_square/count)) if count else None,
        depth_rmse_m=float(np.sqrt(depth_square/count)) if count else None)
