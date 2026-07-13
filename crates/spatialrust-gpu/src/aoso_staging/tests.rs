use super::{
    build_radius_grid_aoso_gpu, compute_voxel_keys_aoso_chunks,
    downsample_voxel_centroid_aoso_chunks, estimate_normals_aoso_gpu,
    estimate_normals_radius_grid_aoso_gpu, reduce_voxel_attributes_aoso_chunks,
    upload_spatial_tensor_attribute_chunks, upload_spatial_tensor_xyz_chunks,
    AoSoAAttributeAggregation, GpuAoSoXyzChunk,
};
use crate::runtime::WgpuRuntime;
use spatialrust_core::{AoSoAAttributeLayout, PointCloudBuilder, SpatialTensor, StandardSchemas};

#[test]
fn upload_matches_interleaved_byte_length() {
    let mut builder = PointCloudBuilder::xyz();
    builder.push_point([1.0, 2.0, 3.0]).unwrap();
    builder.push_point([4.0, 5.0, 6.0]).unwrap();
    let cloud = builder.build().unwrap();
    let packed =
        cloud.spatial_tensor_chunks(4).unwrap().chunks().next().unwrap().pack_xyz(&cloud).unwrap();

    let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
    let gpu_chunk = GpuAoSoXyzChunk::upload(&runtime, "aoso-upload-test", &packed).unwrap();
    assert_eq!(gpu_chunk.point_count(), 2);
    assert_eq!(gpu_chunk.byte_len(), 24);
    gpu_chunk.recycle(&runtime);
}

#[test]
fn uploads_all_tensor_chunks() {
    let mut builder = PointCloudBuilder::xyz();
    for index in 0..5 {
        builder.push_point([index as f32, 0.0, 0.0]).unwrap();
    }
    let cloud = builder.build().unwrap();
    let tensor = SpatialTensor::new(&cloud, 2).unwrap();

    let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
    let gpu_chunks = upload_spatial_tensor_xyz_chunks(&runtime, &tensor).unwrap();
    assert_eq!(gpu_chunks.len(), 3);
    assert_eq!(gpu_chunks[0].point_count(), 2);
    assert_eq!(gpu_chunks[2].point_count(), 1);
    for chunk in gpu_chunks {
        chunk.recycle(&runtime);
    }
}

#[test]
fn chunk_dispatch_matches_global_voxel_keys() {
    let mut builder = PointCloudBuilder::xyz();
    for point in
        [[-0.6, 0.0, 1.2], [-0.1, 0.4, 0.9], [0.0, 0.5, 0.0], [0.6, 1.1, -0.2], [1.4, -0.7, 0.3]]
    {
        builder.push_point(point).unwrap();
    }
    let cloud = builder.build().unwrap();
    let tensor = SpatialTensor::new(&cloud, 2).unwrap();
    let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
    let chunks = upload_spatial_tensor_xyz_chunks(&runtime, &tensor).unwrap();

    let origin = [-1.0, -1.0, -1.0];
    let inv_leaf = 2.0;
    let actual = compute_voxel_keys_aoso_chunks(&runtime, &chunks, origin, inv_leaf)
        .unwrap()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let expected =
        [(-0.6, 0.0, 1.2), (-0.1, 0.4, 0.9), (0.0, 0.5, 0.0), (0.6, 1.1, -0.2), (1.4, -0.7, 0.3)]
            .map(|(x, y, z)| {
                (
                    ((x - origin[0]) * inv_leaf).floor() as i64,
                    ((y - origin[1]) * inv_leaf).floor() as i64,
                    ((z - origin[2]) * inv_leaf).floor() as i64,
                )
            });
    assert_eq!(actual, expected);
    for chunk in chunks {
        chunk.recycle(&runtime);
    }
}

#[test]
fn centroid_pipeline_merges_voxels_across_chunks() {
    let mut builder = PointCloudBuilder::xyz();
    for point in
        [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [1.3, 0.0, 0.0], [2.1, 0.0, 0.0], [2.5, 0.0, 0.0]]
    {
        builder.push_point(point).unwrap();
    }
    let cloud = builder.build().unwrap();
    let tensor = SpatialTensor::new(&cloud, 2).unwrap();
    let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
    let chunks = upload_spatial_tensor_xyz_chunks(&runtime, &tensor).unwrap();

    let result = downsample_voxel_centroid_aoso_chunks(&runtime, &chunks, [0.0; 3], 1.0).unwrap();
    assert_eq!(result.positions.point_count(), 5);
    assert_eq!(result.positions.byte_len(), 5 * 3 * 4);
    let segments = result.segments.to_voxel_segments(&runtime).unwrap();
    assert_eq!(segments.keys, vec![(0, 0, 0), (1, 0, 0), (2, 0, 0)]);
    assert_eq!(result.out_x.len(), 3);
    assert!((result.out_x[0] - 0.1).abs() < 1e-6);
    assert!((result.out_x[1] - 1.2).abs() < 1e-6);
    assert!((result.out_x[2] - 2.3).abs() < 1e-6);
    assert_eq!(result.out_y, vec![0.0; 3]);
    assert_eq!(result.out_z, vec![0.0; 3]);
    result.recycle(&runtime);
    for chunk in chunks {
        chunk.recycle(&runtime);
    }
}

#[test]
fn empty_centroid_pipeline_has_recyclable_positions() {
    let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
    let chunks = Vec::new();

    let result = downsample_voxel_centroid_aoso_chunks(&runtime, &chunks, [0.0; 3], 1.0).unwrap();
    assert!(result.is_empty());
    assert_eq!(result.positions.point_count(), 0);
    result.recycle(&runtime);
}

#[test]
fn uploads_composite_attribute_chunks_with_layout() {
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzinormal());
    for index in 0..5 {
        builder.push_point([index as f32, 1.0, 2.0, 0.5, 0.0, 0.0, 1.0]).unwrap();
    }
    let cloud = builder.build().unwrap();
    let tensor = SpatialTensor::new(&cloud, 2).unwrap();
    let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");

    let chunks = upload_spatial_tensor_attribute_chunks(
        &runtime,
        &tensor,
        AoSoAAttributeLayout::XYZ_INTENSITY_NORMALS,
    )
    .unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].layout().stride_f32(), 7);
    assert_eq!(chunks[0].byte_len(), 2 * 7 * 4);
    assert_eq!(chunks[2].point_count(), 1);
    assert_eq!(chunks[2].byte_len(), 7 * 4);
    for chunk in chunks {
        chunk.recycle(&runtime);
    }
}

#[test]
fn reduces_attributes_across_chunk_boundaries() {
    let mut builder = PointCloudBuilder::new(StandardSchemas::point_xyzinormal());
    for point in [
        [0.1, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0],
        [1.1, 0.0, 0.0, 4.0, 0.0, 1.0, 0.0],
        [1.3, 0.0, 0.0, 8.0, 0.0, 0.0, 1.0],
        [2.1, 0.0, 0.0, 6.0, 1.0, 0.0, 0.0],
    ] {
        builder.push_point(point).unwrap();
    }
    let cloud = builder.build().unwrap();
    let tensor = SpatialTensor::new(&cloud, 2).unwrap();
    let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
    let xyz_chunks = upload_spatial_tensor_xyz_chunks(&runtime, &tensor).unwrap();
    let attribute_chunks = upload_spatial_tensor_attribute_chunks(
        &runtime,
        &tensor,
        AoSoAAttributeLayout::XYZ_INTENSITY_NORMALS,
    )
    .unwrap();
    let voxel =
        downsample_voxel_centroid_aoso_chunks(&runtime, &xyz_chunks, [0.0; 3], 1.0).unwrap();

    let average = reduce_voxel_attributes_aoso_chunks(
        &runtime,
        &attribute_chunks,
        &voxel.segments,
        AoSoAAttributeAggregation::Average,
    )
    .unwrap();
    assert_eq!(average.len(), 3);
    assert_eq!(average.layout(), AoSoAAttributeLayout::XYZ_INTENSITY_NORMALS);
    assert_eq!(
        average.as_slice(),
        &[
            0.1, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.2, 0.0, 0.0, 6.0, 0.0, 0.5, 0.5, 2.1, 0.0, 0.0,
            6.0, 1.0, 0.0, 0.0,
        ]
    );

    let first = reduce_voxel_attributes_aoso_chunks(
        &runtime,
        &attribute_chunks,
        &voxel.segments,
        AoSoAAttributeAggregation::First,
    )
    .unwrap();
    assert_eq!(&first.as_slice()[7..14], &[1.1, 0.0, 0.0, 4.0, 0.0, 1.0, 0.0]);

    voxel.recycle(&runtime);
    for chunk in xyz_chunks {
        chunk.recycle(&runtime);
    }
    for chunk in attribute_chunks {
        chunk.recycle(&runtime);
    }
}

#[test]
fn retained_aoso_positions_match_existing_gpu_normals() {
    let mut builder = PointCloudBuilder::xyz();
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();
    for row in 0..3 {
        for column in 0..3 {
            let point = [column as f32 * 0.1, row as f32 * 0.1, 0.0];
            builder.push_point(point).unwrap();
            x.push(point[0]);
            y.push(point[1]);
            z.push(point[2]);
        }
    }
    let cloud = builder.build().unwrap();
    let tensor = SpatialTensor::new(&cloud, 4).unwrap();
    let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
    let chunks = upload_spatial_tensor_xyz_chunks(&runtime, &tensor).unwrap();
    let voxel = downsample_voxel_centroid_aoso_chunks(&runtime, &chunks, [0.0; 3], 10.0).unwrap();
    let neighbors = (0..9_u32).cycle().take(9 * 9).collect::<Vec<_>>();

    let retained = estimate_normals_aoso_gpu(&runtime, &voxel.positions, &neighbors, 9).unwrap();
    let actual = retained.readback(&runtime).unwrap();
    let expected = crate::estimate_normals_gpu(&runtime, &x, &y, &z, &neighbors, 9).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual.normal[2].abs() - expected.normal[2].abs()).abs() < 1e-6);
        assert!((actual.curvature - expected.curvature).abs() < 1e-6);
    }

    retained.recycle(&runtime);
    voxel.recycle(&runtime);
    for chunk in chunks {
        chunk.recycle(&runtime);
    }
}

#[test]
fn builds_sparse_radius_grid_for_negative_chunked_positions() {
    let mut builder = PointCloudBuilder::xyz();
    for point in
        [[-0.6, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.1, 0.0, 0.0], [0.4, 0.0, 0.0], [1.1, 0.0, 0.0]]
    {
        builder.push_point(point).unwrap();
    }
    let cloud = builder.build().unwrap();
    let tensor = SpatialTensor::new(&cloud, 2).unwrap();
    let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
    let chunks = upload_spatial_tensor_xyz_chunks(&runtime, &tensor).unwrap();
    let voxel = downsample_voxel_centroid_aoso_chunks(&runtime, &chunks, [-1.0; 3], 10.0).unwrap();

    let grid = build_radius_grid_aoso_gpu(&runtime, &voxel.positions, 0.5).unwrap();
    assert_eq!(grid.radius(), 0.5);
    let segments = grid.segments().to_voxel_segments(&runtime).unwrap();
    assert_eq!(segments.keys, vec![(-2, 0, 0), (-1, 0, 0), (0, 0, 0), (2, 0, 0)]);
    assert_eq!(segments.cell_counts, vec![1, 1, 2, 1]);
    assert_eq!(segments.point_indices, vec![0, 1, 2, 3, 4]);

    voxel.recycle(&runtime);
    for chunk in chunks {
        chunk.recycle(&runtime);
    }
}

#[test]
fn sparse_radius_normals_match_existing_grid_path() {
    let mut builder = PointCloudBuilder::xyz();
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();
    for row in 0..12 {
        for column in 0..12 {
            let point = [column as f32 * 0.1 - 0.55, row as f32 * 0.1 - 0.55, 0.0];
            builder.push_point(point).unwrap();
            x.push(point[0]);
            y.push(point[1]);
            z.push(point[2]);
        }
    }
    let cloud = builder.build().unwrap();
    let tensor = SpatialTensor::new(&cloud, 31).unwrap();
    let runtime = WgpuRuntime::new_headless().expect("wgpu runtime");
    let chunks = upload_spatial_tensor_xyz_chunks(&runtime, &tensor).unwrap();
    let voxel = downsample_voxel_centroid_aoso_chunks(&runtime, &chunks, [-1.0; 3], 100.0).unwrap();
    let grid = build_radius_grid_aoso_gpu(&runtime, &voxel.positions, 0.25).unwrap();
    let retained =
        estimate_normals_radius_grid_aoso_gpu(&runtime, &voxel.positions, &grid).unwrap();
    let actual = retained.readback(&runtime).unwrap();
    let expected = crate::estimate_normals_grid_gpu(&runtime, &x, &y, &z, 0.25).unwrap();

    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual.normal[2].abs() - expected.normal[2].abs()).abs() < 1e-5);
        assert!((actual.curvature - expected.curvature).abs() < 1e-5);
    }
    retained.recycle(&runtime);
    voxel.recycle(&runtime);
    for chunk in chunks {
        chunk.recycle(&runtime);
    }
}
