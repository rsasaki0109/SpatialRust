use spatialrust_core::{HasPositions3, PointCloud};
use spatialrust_features::{IssKeypointConfig, IssKeypointResult};
use spatialrust_gpu::{GpuIssInput, GpuIssNeighborhoods, WgpuRuntime};
use spatialrust_math::{symmetric_eigen3, Mat3};
use spatialrust_search::{KdTree, RadiusSearchIndex};

pub struct Audit {
    pub indices: Vec<u32>,
    pub differences: usize,
    pub max_absolute_eigen_error: f64,
}

fn cpu_scores(cloud: &PointCloud, config: IssKeypointConfig, tree: &KdTree) -> Vec<[f64; 4]> {
    let (x, y, z) = cloud.positions3().unwrap();
    (0..cloud.len())
        .map(|i| {
            let neighbors = tree.radius_search(x[i], y[i], z[i], config.salient_radius);
            if neighbors.len() < 3 || neighbors.len() < config.min_neighbors {
                return [0.0; 4];
            }
            let count = neighbors.len() as f64;
            let mut mean = [0.0; 3];
            for n in &neighbors {
                for (m, value) in mean.iter_mut().zip([x[n.index], y[n.index], z[n.index]]) {
                    *m += f64::from(value);
                }
            }
            for m in &mut mean {
                *m /= count;
            }
            let mut c = [0.0; 6];
            for n in &neighbors {
                let d = [
                    f64::from(x[n.index]) - mean[0],
                    f64::from(y[n.index]) - mean[1],
                    f64::from(z[n.index]) - mean[2],
                ];
                for (sum, value) in c.iter_mut().zip([
                    d[0] * d[0],
                    d[1] * d[1],
                    d[2] * d[2],
                    d[0] * d[1],
                    d[0] * d[2],
                    d[1] * d[2],
                ]) {
                    *sum += value;
                }
            }
            for value in &mut c {
                *value *= 1.0 / count;
            }
            let e = symmetric_eigen3(Mat3::from_rows(
                [c[0], c[3], c[4]],
                [c[3], c[1], c[5]],
                [c[4], c[5], c[2]],
            ))
            .eigenvalues;
            let eligible = e[2] > 0.0
                && e[1] > 0.0
                && e[1] / e[2] < f64::from(config.gamma_21)
                && e[0] / e[1] < f64::from(config.gamma_32);
            [e[0], e[1], e[2], if eligible { 1.0 } else { 0.0 }]
        })
        .collect()
}

// Bound eigenvalue errors against f64 math, independently reproduce device NMS
// from its downloaded scores and a CPU radius graph, and explain every CPU/GPU
// selection difference by a threshold or competing-saliency uncertainty band.
pub fn audit(
    runtime: &WgpuRuntime,
    cloud: &PointCloud,
    config: IssKeypointConfig,
    input: &GpuIssInput,
    rows: &GpuIssNeighborhoods,
    cpu: &IssKeypointResult,
) -> Audit {
    let output = input
        .keypoints_resident(
            runtime,
            rows,
            config.gamma_21,
            config.gamma_32,
            u32::try_from(config.min_neighbors).unwrap_or(u32::MAX),
        )
        .unwrap();
    let (scores, receipt) = output.download_scores(runtime).unwrap();
    assert_eq!(
        receipt.device_to_host_bytes(),
        if cloud.is_empty() { 0 } else { (cloud.len() * 16 + 32) as u64 }
    );
    let indices = output.download(runtime).unwrap().0;
    let tree = KdTree::from_point_cloud(cloud).unwrap();
    let reference = cpu_scores(cloud, config, &tree);
    let tolerances: Vec<_> =
        reference.iter().map(|row| 1e-12 + 5e-6 * (row[0] + row[1] + row[2]).abs()).collect();
    let mut max_absolute_eigen_error = 0.0_f64;
    let mut ambiguous = vec![false; cloud.len()];
    for (i, (a, b)) in scores.iter().zip(&reference).enumerate() {
        if b.iter().all(|&v| v == 0.0) {
            assert_eq!(a[3], 0.0, "unsupported/zero covariance must be ineligible at row {i}");
        }
        for bin in 0..3 {
            let error = (f64::from(a[bin]) - b[bin]).abs();
            max_absolute_eigen_error = max_absolute_eigen_error.max(error);
            assert!(
                error <= tolerances[i],
                "eigen row={i} bin={bin} gpu={} cpu={} error={error} bound={}",
                a[bin],
                b[bin],
                tolerances[i]
            );
        }
        let boundary = b[2].abs() <= tolerances[i]
            || b[1].abs() <= tolerances[i]
            || (b[1] - f64::from(config.gamma_21) * b[2]).abs()
                <= (1.0 + f64::from(config.gamma_21)) * tolerances[i]
            || (b[0] - f64::from(config.gamma_32) * b[1]).abs()
                <= (1.0 + f64::from(config.gamma_32)) * tolerances[i];
        ambiguous[i] = boundary;
        assert!(
            f64::from(a[3]) == b[3] || boundary,
            "unexplained eligibility difference at row {i}"
        );
    }
    let (x, y, z) = cloud.positions3().unwrap();
    let mut from_scores = vec![];
    let mut differences = 0;
    for i in 0..cloud.len() {
        let neighbors = tree.radius_search(x[i], y[i], z[i], config.non_max_radius);
        let selected = scores[i][3] != 0.0
            && neighbors.iter().all(|n| {
                n.index == i || scores[n.index][3] == 0.0 || scores[i][0] >= scores[n.index][0]
            });
        if selected {
            from_scores.push(i as u32);
        }
        if selected != cpu.indices.binary_search(&i).is_ok() {
            differences += 1;
            assert!(
                ambiguous[i]
                    || neighbors.iter().any(|n| n.index != i
                        && (reference[n.index][3] != 0.0 || scores[n.index][3] != 0.0)
                        && (ambiguous[n.index]
                            || (f64::from(reference[i][0] as f32)
                                - f64::from(reference[n.index][0] as f32))
                            .abs()
                                <= tolerances[i] + tolerances[n.index])),
                "unexplained NMS difference at row {i}"
            );
        }
    }
    assert_eq!(
        indices, from_scores,
        "device NMS must match independent selection from its own scores"
    );
    Audit { indices, differences, max_absolute_eigen_error }
}
