"""Synthetic RGB-D -> colored cloud -> SpatialRust MVP pipeline demo."""

import numpy as np
import spatialrust as sr


def main() -> None:
    height, width = 48, 64
    depth = np.ones((height, width), dtype=np.float32)
    depth[18:30, 26:38] = 0.7
    color = np.zeros((height, width, 3), dtype=np.uint8)
    color[..., 1] = 160
    color[18:30, 26:38] = (230, 80, 40)

    cloud = sr.rgbd_to_point_cloud(
        depth,
        color,
        fx=60.0,
        fy=60.0,
        cx=(width - 1) / 2,
        cy=(height - 1) / 2,
    )
    result = sr.run_pipeline(
        cloud,
        leaf_size=0.025,
        plane_distance=0.02,
        cluster_tolerance=0.08,
        min_cluster_size=4,
    )
    print(f"RGB-D points: {len(cloud)}")
    print(f"plane inliers: {result.plane_inliers}")
    print(f"clusters: {result.cluster_count}")


if __name__ == "__main__":
    main()
