//! Execute fixed public RGB-D image/point checks with explicit device transfers.
//!
//! Input preparation and source provenance are documented with the fixed plan.
//! Usage: gpu_public_rgbd_conformance INPUT_DIRECTORY MANIFEST_SHA256 high|low

use std::{collections::BTreeMap, error::Error, fs, io, path::Path};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use spatialrust_gpu::{
    box_blur_gpu, downsample_voxel_centroid_gpu, histogram_u8_gpu, rgb_to_gray_gpu, GpuImage,
    GpuImageBorder, GpuKdTree, GpuQueryBatch, GpuResourcePoolStats, WgpuPowerPreference,
    WgpuRuntime,
};
use spatialrust_image::Image;
use spatialrust_search::KdTree;

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const PIXELS: usize = WIDTH * HEIGHT;
const QUERIES: usize = 64;
const INV_LEAF: f64 = 32.0;
const RADIUS: f64 = 0.03125;
const CENTROID_TOLERANCE: f64 = 0.00001;
const SOURCE_SHA: &str = "a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98";

fn invalid(message: &str) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidData, message).into()
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn checked_payload(directory: &Path, record: &Value, expected_bytes: usize) -> Result<Vec<u8>> {
    let name = record["path"].as_str().ok_or_else(|| invalid("missing payload path"))?;
    if Path::new(name).components().count() != 1 || !name.is_ascii() {
        return Err(invalid("payload must be a single ASCII filename"));
    }
    let path = directory.join(name).canonicalize()?;
    if path.parent() != Some(directory) || record["bytes"].as_u64() != Some(expected_bytes as u64) {
        return Err(invalid("payload escaped input directory or has invalid length"));
    }
    let bytes = fs::read(path)?;
    if bytes.len() != expected_bytes || record["sha256"].as_str() != Some(&sha(&bytes)) {
        return Err(invalid("payload content/length differs from frozen manifest"));
    }
    Ok(bytes)
}

fn reference_gray(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .map(|pixel| {
            let numerator =
                77 * u32::from(pixel[0]) + 150 * u32::from(pixel[1]) + 29 * u32::from(pixel[2]);
            ((numerator + 128) / 256) as u8
        })
        .collect()
}

// Independent two-dimensional integer integral over replicate-padded pixels.
// The device implementation uses texture dispatch and separable sum passes.
fn reference_box(source: &[u8], width: usize, height: usize, kw: usize, kh: usize) -> Vec<u8> {
    let (rx, ry) = (kw / 2, kh / 2);
    let (pw, ph) = (width + 2 * rx, height + 2 * ry);
    let stride = pw + 1;
    let mut integral = vec![0_u64; stride * (ph + 1)];
    for row in 0..ph {
        let y = row.saturating_sub(ry).min(height - 1);
        let mut sum = 0_u64;
        for col in 0..pw {
            let x = col.saturating_sub(rx).min(width - 1);
            sum += u64::from(source[y * width + x]);
            integral[(row + 1) * stride + col + 1] = integral[row * stride + col + 1] + sum;
        }
    }
    let mut result = Vec::with_capacity(width * height);
    let mass = (kw * kh) as u64;
    for y in 0..height {
        for x in 0..width {
            let top = integral[y * stride + x + kw] - integral[y * stride + x];
            let bottom = integral[(y + kh) * stride + x + kw] - integral[(y + kh) * stride + x];
            result.push(((bottom - top + mass / 2) / mass) as u8);
        }
    }
    result
}

fn mismatches<T: PartialEq>(actual: &[T], expected: &[T]) -> usize {
    actual.iter().zip(expected).filter(|(a, b)| a != b).count()
        + actual.len().abs_diff(expected.len())
}

#[derive(Default)]
struct Cell {
    sum: [f64; 3],
    indices: Vec<u32>,
}

fn reference_cells(positions: &[[f32; 3]]) -> BTreeMap<(i64, i64, i64), Cell> {
    let mut cells = BTreeMap::<_, Cell>::new();
    for (index, position) in positions.iter().enumerate() {
        let key = position.map(|coordinate| (f64::from(coordinate) * INV_LEAF).floor() as i64);
        let cell = cells.entry((key[0], key[1], key[2])).or_default();
        for (sum, coordinate) in cell.sum.iter_mut().zip(position) {
            *sum += f64::from(*coordinate);
        }
        cell.indices.push(index as u32);
    }
    cells
}

fn query_positions(positions: &[[f32; 3]]) -> Vec<[f32; 3]> {
    (0..QUERIES)
        .map(|index| {
            let p = positions[index * positions.len() / QUERIES];
            [p[0] + 0.00031, p[1] - 0.00019, p[2] + 0.00011]
        })
        .collect()
}

fn reference_counts(positions: &[[f32; 3]], queries: &[[f32; 3]]) -> Vec<u32> {
    queries
        .iter()
        .map(|query| {
            positions
                .iter()
                .filter(|point| {
                    point
                        .iter()
                        .zip(query)
                        .map(|(a, b)| {
                            let delta = f64::from(*a) - f64::from(*b);
                            delta * delta
                        })
                        .sum::<f64>()
                        <= RADIUS * RADIUS
                })
                .count() as u32
        })
        .collect()
}

fn pool_receipt(stats: GpuResourcePoolStats) -> Value {
    json!({"budget_bytes":stats.budget_bytes,"cached_bytes":stats.cached_bytes,
        "cached_resources":stats.cached_resources,"allocations":stats.allocation_count,
        "reuses":stats.reuse_count,"discarded":stats.discarded_count})
}

fn run_frame(runtime: &WgpuRuntime, directory: &Path, frame: &Value) -> Result<Value> {
    let count =
        frame["point_count"].as_u64().ok_or_else(|| invalid("missing point count"))? as usize;
    if !(QUERIES..=PIXELS).contains(&count) {
        return Err(invalid("point count outside fixed frame bounds"));
    }
    let rgb = checked_payload(directory, &frame["rgb"], PIXELS * 3)?;
    let raw_positions = checked_payload(directory, &frame["positions"], count * 12)?;
    let positions: Vec<[f32; 3]> = raw_positions
        .chunks_exact(12)
        .map(|record| {
            std::array::from_fn(|axis| {
                f32::from_le_bytes(record[axis * 4..axis * 4 + 4].try_into().unwrap())
            })
        })
        .collect();
    if positions.iter().flatten().any(|value| !value.is_finite()) {
        return Err(invalid("position payload contains nonfinite coordinates"));
    }
    let gray_reference = reference_gray(&rgb);
    let source = Image::<u8, 3>::try_new(WIDTH, HEIGHT, rgb)?;
    let uploaded = GpuImage::upload_u8(runtime, source.view())?;
    let mut gray = rgb_to_gray_gpu(runtime, &uploaded)?;
    let mut small = box_blur_gpu(runtime, &gray, 11, 11, GpuImageBorder::Replicate)?;
    let mut wide = box_blur_gpu(runtime, &gray, 259, 3, GpuImageBorder::Replicate)?;
    let mut histogram = histogram_u8_gpu(runtime, &small, 0)?;
    let upload_bytes = (PIXELS * 4) as u64;
    for receipt in [gray.receipt(), small.receipt(), wide.receipt(), histogram.receipt()] {
        receipt.validate_resident_chain(upload_bytes)?;
    }
    // All device consumers are submitted before the first explicit host readback.
    let gray_host = gray.readback_u8::<1>(runtime)?;
    let small_host = small.readback_u8::<1>(runtime)?;
    let wide_host = wide.readback_u8::<1>(runtime)?;
    let bins = histogram.readback_u32(runtime)?;
    let small_reference = reference_box(&gray_reference, WIDTH, HEIGHT, 11, 11);
    let wide_reference = reference_box(&gray_reference, WIDTH, HEIGHT, 259, 3);
    let mut expected_bins = [0_u32; 256];
    for &pixel in &small_reference {
        expected_bins[pixel as usize] += 1;
    }
    let image_errors = [
        mismatches(gray_host.as_slice(), &gray_reference),
        mismatches(small_host.as_slice(), &small_reference),
        mismatches(wide_host.as_slice(), &wide_reference),
        mismatches(&bins, &expected_bins),
    ];
    let download_bytes = gray.receipt().device_to_host_bytes()
        + small.receipt().device_to_host_bytes()
        + wide.receipt().device_to_host_bytes()
        + histogram.receipt().device_to_host_bytes();
    if download_bytes != 3 * upload_bytes + 1024 {
        return Err(invalid("unexpected explicit image/histogram readback accounting"));
    }
    let image = json!({"gray_mismatches":image_errors[0],"box_11x11_mismatches":image_errors[1],
        "box_259x3_mismatches":image_errors[2],"histogram_bin_mismatches":image_errors[3],
        "host_to_device_bytes":upload_bytes,"explicit_device_to_host_bytes":download_bytes,
        "readbacks_before_device_consumers":0,"histogram_pixels":bins.iter().map(|&v|u64::from(v)).sum::<u64>(),
        "gray_sha256":sha(gray_host.as_slice()),"box_11x11_sha256":sha(small_host.as_slice()),
        "box_259x3_sha256":sha(wide_host.as_slice()),"resident_stages":histogram.receipt().stages()});
    gray.recycle(runtime);
    small.recycle(runtime);
    wide.recycle(runtime);
    histogram.recycle(runtime);
    uploaded.recycle(runtime);

    let x: Vec<_> = positions.iter().map(|p| p[0]).collect();
    let y: Vec<_> = positions.iter().map(|p| p[1]).collect();
    let z: Vec<_> = positions.iter().map(|p| p[2]).collect();
    let reference = reference_cells(&positions);
    let voxel = downsample_voxel_centroid_gpu(runtime, &x, &y, &z, [0.; 3], INV_LEAF as f32)?;
    let metadata = voxel.segments.to_voxel_segments(runtime)?;
    let cell_count = reference.len();
    let mut metadata_errors = usize::from(metadata.len() != cell_count);
    let mut maximum_centroid_error = 0.0_f64;
    for (cell_index, (key, expected)) in reference.iter().enumerate() {
        if metadata.keys.get(cell_index) != Some(key) {
            metadata_errors += 1;
            continue;
        }
        let start = metadata.cell_starts[cell_index] as usize;
        let length = metadata.cell_counts[cell_index] as usize;
        metadata_errors +=
            mismatches(&metadata.point_indices[start..start + length], &expected.indices);
        for (axis, output) in [&voxel.out_x, &voxel.out_y, &voxel.out_z].iter().enumerate() {
            if let Some(&actual) = output.get(cell_index) {
                let error =
                    (f64::from(actual) - expected.sum[axis] / expected.indices.len() as f64).abs();
                if !actual.is_finite() {
                    return Err(invalid("nonfinite GPU centroid"));
                }
                maximum_centroid_error = maximum_centroid_error.max(error);
            } else {
                metadata_errors += 1;
            }
        }
    }
    let construction_bytes = voxel.segments.construction_readback_bytes();
    if construction_bytes != 4 || voxel.segments.point_count() as usize != count {
        return Err(invalid("unexpected voxel construction transfer/count"));
    }
    let voxel_receipt = json!({"point_count":count,"occupied_cells":cell_count,
        "metadata_mismatches":metadata_errors,"maximum_centroid_coordinate_error_m":maximum_centroid_error,
        "centroid_tolerance_m":CENTROID_TOLERANCE,"host_to_device_payload_bytes":count*12,
        "construction_device_to_host_bytes":construction_bytes,
        "centroid_device_to_host_bytes":metadata.len()*12,
        "metadata_device_to_host_bytes":metadata.len()*20+count*4});
    voxel.positions.recycle(runtime);

    // The CPU KD-tree only builds the index. Radius traversal/counting is GPU work.
    let cpu_tree = KdTree::from_slices(&x, &y, &z);
    let tree = GpuKdTree::upload(runtime, &cpu_tree)?;
    let queries = query_positions(&positions);
    let expected_counts = reference_counts(&positions, &queries);
    let query_batch = GpuQueryBatch::upload(runtime, &queries)?;
    let resident = tree.radius_neighbors_resident(runtime, &query_batch, RADIUS as f32, 0)?;
    let execution = resident.execution_transfers();
    if execution.host_to_device_bytes() != 0
        || execution.device_to_host_bytes() != 0
        || tree.upload_transfers().device_to_host_bytes() != 0
        || query_batch.upload_transfers().host_to_device_bytes() != (QUERIES * 16) as u64
    {
        return Err(invalid("unexpected resident radius transfer"));
    }
    let counts = resident.download(runtime)?;
    let radius_errors = mismatches(counts.total_counts(), &expected_counts);
    if counts.transfers().device_to_host_bytes() != (QUERIES * 8) as u64
        || counts.complete_rows().is_ok()
    {
        return Err(invalid(
            "count-only result must retain overflow instead of exposing empty complete rows",
        ));
    }
    let radius = json!({"queries":QUERIES,"radius_m":RADIUS,"count_mismatches":radius_errors,
        "total_neighbors":counts.total_counts().iter().map(|&v|u64::from(v)).sum::<u64>(),
        "tree_host_to_device_bytes":tree.upload_transfers().host_to_device_bytes(),
        "queries_host_to_device_bytes":query_batch.upload_transfers().host_to_device_bytes(),
        "search_host_to_device_bytes":execution.host_to_device_bytes(),
        "search_device_to_host_bytes":execution.device_to_host_bytes(),
        "explicit_device_to_host_bytes":counts.transfers().device_to_host_bytes(),
        "complete_row_access_rejected":true});
    runtime.wait_idle();
    let pools = [
        runtime.image_texture_pool_stats(),
        runtime.scratch_buffer_pool_stats(),
        runtime.buffer_pool().stats(),
    ];
    let resource_ok = pools.iter().all(|stats| stats.cached_bytes <= stats.budget_bytes);
    Ok(json!({"index":frame["index"],"image":image,"voxel":voxel_receipt,"radius":radius,
        "pools":{"image":pool_receipt(pools[0]),"scratch":pool_receipt(pools[1]),"buffer":pool_receipt(pools[2])},
        "accepted":image_errors == [0;4] && metadata_errors == 0
            && maximum_centroid_error <= CENTROID_TOLERANCE && radius_errors == 0 && resource_ok}))
}

fn main() -> Result<()> {
    let arguments: Vec<_> = std::env::args().collect();
    if arguments.len() != 4 {
        return Err(invalid("usage: INPUT_DIRECTORY MANIFEST_SHA256 high|low"));
    }
    let preference = match arguments[3].as_str() {
        "high" => WgpuPowerPreference::HighPerformance,
        "low" => WgpuPowerPreference::LowPower,
        _ => return Err(invalid("power preference must be high or low")),
    };
    let directory = Path::new(&arguments[1]).canonicalize()?;
    let raw_manifest = fs::read(directory.join("manifest.json"))?;
    if sha(&raw_manifest) != arguments[2] {
        return Err(invalid("manifest SHA-256 mismatch"));
    }
    let manifest: Value = serde_json::from_slice(&raw_manifest)?;
    let frames = manifest["frames"].as_array().ok_or_else(|| invalid("missing frames"))?;
    if manifest["schema"] != "spatialrust.public-gpu-inputs.v1"
        || manifest["source_sha256"] != SOURCE_SHA
        || manifest["width"] != WIDTH
        || manifest["height"] != HEIGHT
        || frames.iter().map(|frame| frame["index"].as_u64()).collect::<Vec<_>>()
            != [Some(0), Some(399), Some(797)]
    {
        return Err(invalid("manifest differs from fixed dataset, geometry or frame indices"));
    }
    let runtime = WgpuRuntime::new_headless_with_preference(preference)?;
    let mut outcomes = Vec::new();
    for frame in frames {
        outcomes.push(match run_frame(&runtime, &directory, frame) {
            Ok(result) => result,
            Err(error) => {
                json!({"index":frame["index"],"accepted":false,"error":error.to_string()})
            }
        });
    }
    let accepted = outcomes.iter().all(|result| result["accepted"] == true);
    let info = runtime.adapter_info();
    let limits = runtime.device().limits();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"spatialrust.public-gpu-conformance.v1","manifest_sha256":arguments[2],
            "preference":arguments[3],"backend":info.backend,"device_type":info.device_type,
            "os":std::env::consts::OS,"architecture":std::env::consts::ARCH,
            "limits":{"max_storage_buffers_per_shader_stage":limits.max_storage_buffers_per_shader_stage,
                "max_texture_dimension_2d":limits.max_texture_dimension_2d,
                "max_compute_workgroup_storage_size":limits.max_compute_workgroup_storage_size},
            "frames":outcomes,"accepted":accepted,
            "scope":"Three fixed development frames; no timing/accuracy or all-algorithm portability claim. Transfer payloads exclude uniforms/driver staging; pool counters exclude live/driver/staging allocations."
        }))?
    );
    if !accepted {
        return Err(invalid("one or more fixed GPU conformance cases failed"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integral_box_matches_hand_computed_replicate_borders() {
        assert_eq!(reference_box(&[0, 9, 18, 27], 2, 2, 3, 3), vec![9, 12, 15, 18]);
        assert_eq!(reference_box(&[7], 1, 1, 259, 3), vec![7]);
        assert_eq!(reference_box(&[0, 255], 2, 1, 1, 1), vec![0, 255]);
    }

    #[test]
    fn geometry_reference_keeps_negative_cells_and_every_source_index() {
        let cells = reference_cells(&[[-0.001, 0., 1.], [0., 0., 1.], [0.001, 0., 1.]]);
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[&(-1, 0, 32)].indices, vec![0]);
        assert_eq!(cells[&(0, 0, 32)].indices, vec![1, 2]);
        assert_eq!(
            reference_counts(&[[0., 0., 0.], [0.03125, 0., 0.], [0.0625, 0., 0.]], &[[0., 0., 0.]]),
            vec![2]
        );
    }
}
