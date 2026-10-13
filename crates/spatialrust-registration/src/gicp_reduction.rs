use super::{covariances, delta_from_solution, update_magnitude, GicpRegistration};
use crate::RegistrationResult;
use spatialrust_core::{
    HasPositions3, PointCloud, SpatialError, SpatialResult, TransferDirection, TransferStats,
};
use spatialrust_gpu::{GpuGicpEquations, GpuGicpInput, WgpuRuntime};
use spatialrust_math::{solve_linear_system, Isometry3, LeastSquaresResult};
use spatialrust_search::KdTree;

/// One completed update in the explicit resident GICP precision path.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuGicpIteration {
    /// One-based completed update number.
    pub iteration: usize,
    /// Gated pairs used by the normal-equation solve.
    pub correspondences: u32,
    /// Gated pairs used for post-update Euclidean fitness.
    pub evaluated_correspondences: u32,
    /// Post-update Euclidean mean squared distance, f64::MAX when unmatched.
    pub fitness: f64,
    /// Norm of the world rotation/translation linearized solution, as in CPU GICP.
    pub update_magnitude: f64,
}

/// Explicit GICP resident reduction result, trace and full transfer receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuGicpReductionDiagnostics {
    /// Estimated pose, Euclidean fitness and stopping result.
    pub result: RegistrationResult,
    /// Completed update rows from this device precision path.
    pub history: Vec<GpuGicpIteration>,
    /// Position/tree/covariance upload, optional radius covariance transfers and equations readback.
    pub transfers: TransferStats,
    /// Completed resident evaluations (updates plus one).
    pub evaluation_batches: usize,
    /// Explicit GPU covariance estimates, zero or two (radius or resident k-NN).
    pub covariance_batches: usize,
}

fn invalid() -> SpatialError {
    SpatialError::InvalidArgument("GPU GICP reduction requires at least one iteration, minimum six correspondences, finite positive covariance epsilon/distance square, finite nonnegative stopping thresholds and a finite unit initial pose".to_owned())
}
fn valid_pose(pose: Isometry3<f32>) -> bool {
    let q = pose.rotation();
    let t = pose.translation();
    let norm = f64::from(q.x).hypot(f64::from(q.y)).hypot(f64::from(q.z)).hypot(f64::from(q.w));
    t.x.is_finite()
        && t.y.is_finite()
        && t.z.is_finite()
        && norm.is_finite()
        && (norm - 1.0).abs() <= 1e-4
}
fn record(total: &mut TransferStats, part: TransferStats) {
    total.record(TransferDirection::HostToDevice, part.host_to_device_bytes());
    total.record(TransferDirection::DeviceToHost, part.device_to_host_bytes());
}
fn fitness(equations: &GpuGicpEquations) -> f64 {
    if equations.fitness_correspondences == 0 {
        f64::MAX
    } else {
        equations.squared_error_sum / f64::from(equations.fitness_correspondences)
    }
}

#[derive(Clone, Copy)]
enum CovariancePreparation {
    Legacy,
    ResidentKnn,
    ResidentRadius(u32),
    ResidentGrid(u32),
}

impl GicpRegistration {
    /// Retains source/target/covariances on GPU for transform, nearest matching,
    /// rotated covariance inverse, Hessian/gradient and Euclidean fitness passes.
    /// CPU solves normalized 6x6 equations and converts the linearized step to
    /// the existing world update/stopping convention. Each evaluation explicitly
    /// downloads 256 bytes; post-update equations feed the next solve/final fitness.
    /// Covariance preparation remains CPU k-NN or explicit radius-GPU readback
    /// and re-upload with full receipts. No automatic CPU fallback occurs.
    /// Configuration/conditioning restrictions are stricter than legacy GICP;
    /// f32 weights, transforms and sums can change pose/matching/stopping.
    pub fn align_with_gpu_reduction(
        &self,
        runtime: &WgpuRuntime,
        source: &PointCloud,
        target: &PointCloud,
    ) -> SpatialResult<GpuGicpReductionDiagnostics> {
        self.align_reduction_impl(runtime, source, target, CovariancePreparation::Legacy)
    }

    /// Computes k-NN neighborhoods and plane covariances on GPU, retaining the
    /// covariance buffers for the existing resident equation/update loop. No
    /// neighborhood/covariance download or re-upload occurs. CPU builds both
    /// trees, solves normalized 6x6 equations and composes/stops the pose.
    /// Requires covariance_radius=None, k_neighbors=3..=64, epsilon (as f32)
    /// in 1e-4..=1, and the same valid configuration as align_with_gpu_reduction.
    /// Centered/scaled f32 covariance/eigenvectors may differ from CPU raw
    /// products, particularly at repeated eigenvalues and search boundaries.
    /// Invalid device preparation rejects equations without CPU fallback.
    pub fn align_with_gpu_resident_covariances(
        &self,
        runtime: &WgpuRuntime,
        source: &PointCloud,
        target: &PointCloud,
    ) -> SpatialResult<GpuGicpReductionDiagnostics> {
        self.align_reduction_impl(runtime, source, target, CovariancePreparation::ResidentKnn)
    }

    /// Computes complete inclusive radius neighborhoods and covariances on GPU
    /// and retains them for the resident equation/update loop. Requires configured
    /// covariance_radius=Some(finite positive radius) with a normal f32 square,
    /// epsilon (as f32) in 1e-4..=1 and positive `neighbor_capacity` per point.
    /// Capacity includes self/duplicates; overflow fails without truncation or
    /// fallback. Fewer than three neighbors use epsilon*I as in the grid route.
    /// Centered/scaled compensated f32 covariance and max-pivot Jacobi can differ
    /// from grid accumulation/sweeps, especially at repeated eigenvalues/gates.
    /// CPU builds both trees, solves 6x6 equations and composes/stops the pose.
    /// Upload is 32*(S+T+source_nodes+target_nodes) bytes; downloads are only
    /// 256 bytes per evaluation. No neighborhood/covariance payload crosses CPU.
    pub fn align_with_gpu_resident_radius_covariances(
        &self,
        runtime: &WgpuRuntime,
        source: &PointCloud,
        target: &PointCloud,
        neighbor_capacity: u32,
    ) -> SpatialResult<GpuGicpReductionDiagnostics> {
        self.align_reduction_impl(
            runtime,
            source,
            target,
            CovariancePreparation::ResidentRadius(neighbor_capacity),
        )
    }

    /// Prepares radius covariances with CPU counting-sort grids and GPU gather,
    /// retaining the canonical grid's raw f32 means/central sums/Jacobi sweeps.
    /// No source KD-tree, per-point neighborhood buffer or covariance readback
    /// is needed. Positive capacity includes self/duplicates; overflow rejects
    /// all equations. Configured radius/epsilon and normal-equation constraints
    /// match the resident-radius API. Dense grids must fit existing cell/device
    /// limits; no automatic sparse-path/CPU fallback occurs. CPU prepares grids
    /// and the target tree and solves 6x6 equations. Downloads remain 256 bytes
    /// per evaluation; all grid-index uploads are included in the receipt.
    pub fn align_with_gpu_resident_grid_covariances(
        &self,
        runtime: &WgpuRuntime,
        source: &PointCloud,
        target: &PointCloud,
        neighbor_capacity: u32,
    ) -> SpatialResult<GpuGicpReductionDiagnostics> {
        self.align_reduction_impl(
            runtime,
            source,
            target,
            CovariancePreparation::ResidentGrid(neighbor_capacity),
        )
    }

    fn align_reduction_impl(
        &self,
        runtime: &WgpuRuntime,
        source: &PointCloud,
        target: &PointCloud,
        preparation: CovariancePreparation,
    ) -> SpatialResult<GpuGicpReductionDiagnostics> {
        self.validate_inputs(source, target)?;
        let c = self.config;
        let distance_squared = c.max_correspondence_distance * c.max_correspondence_distance;
        let epsilon = c.epsilon as f32;
        if matches!(preparation, CovariancePreparation::ResidentKnn)
            && (c.covariance_radius.is_some()
                || c.k_neighbors > spatialrust_gpu::MAX_GPU_K_NEIGHBORS as usize
                || !(1e-4..=1.0).contains(&epsilon))
        {
            return Err(SpatialError::InvalidArgument("resident GPU GICP covariances require covariance_radius=None, k_neighbors=3..=64 and epsilon (as f32) in 1e-4..=1".to_owned()));
        }
        if let CovariancePreparation::ResidentRadius(capacity)
        | CovariancePreparation::ResidentGrid(capacity) = preparation
        {
            if capacity == 0
                || !(1e-4..=1.0).contains(&epsilon)
                || c.covariance_radius
                    .map_or(true, |r| !r.is_finite() || r <= 0.0 || !(r * r).is_normal())
            {
                return Err(SpatialError::InvalidArgument("resident GPU GICP radius covariances require a positive capacity including self, covariance_radius=Some(finite positive radius) with a normal f32 square and epsilon (as f32) in 1e-4..=1".to_owned()));
            }
        }
        if c.max_iterations == 0
            || c.min_correspondences < 6
            || !c.epsilon.is_finite()
            || epsilon <= 0.0
            || !epsilon.is_finite()
            || !c.max_correspondence_distance.is_finite()
            || c.max_correspondence_distance <= 0.0
            || !distance_squared.is_finite()
            || distance_squared == 0.0
            || !c.transformation_epsilon.is_finite()
            || c.transformation_epsilon < 0.0
            || !c.fitness_epsilon.is_finite()
            || c.fitness_epsilon < 0.0
            || !valid_pose(c.initial_guess)
        {
            return Err(invalid());
        }
        let (sx, sy, sz) = source.positions3()?;
        let (tx, ty, tz) = target.positions3()?;
        if sx.iter().chain(sy).chain(sz).chain(tx).chain(ty).chain(tz).any(|v| !v.is_finite()) {
            return Err(SpatialError::InvalidArgument("GPU GICP requires finite XYZ".to_owned()));
        }
        let tree = KdTree::from_slices(tx, ty, tz);
        let mut transfers = TransferStats::default();
        let mut covariance_batches = 0;
        let input = if let CovariancePreparation::ResidentGrid(capacity) = preparation {
            covariance_batches = 2;
            GpuGicpInput::upload_with_gpu_grid_covariances(
                runtime,
                (sx, sy, sz),
                &tree,
                c.covariance_radius.unwrap(),
                capacity,
                epsilon,
            )?
        } else if !matches!(preparation, CovariancePreparation::Legacy) {
            let source_tree = KdTree::from_slices(sx, sy, sz);
            covariance_batches = 2;
            match preparation {
                CovariancePreparation::ResidentKnn => {
                    let k = u32::try_from(c.k_neighbors).map_err(|_| invalid())?;
                    GpuGicpInput::upload_with_gpu_covariances(
                        runtime,
                        &source_tree,
                        &tree,
                        k,
                        epsilon,
                    )?
                }
                CovariancePreparation::ResidentRadius(capacity) => {
                    GpuGicpInput::upload_with_gpu_radius_covariances(
                        runtime,
                        &source_tree,
                        &tree,
                        c.covariance_radius.unwrap(),
                        capacity,
                        epsilon,
                    )?
                }
                CovariancePreparation::Legacy | CovariancePreparation::ResidentGrid(_) => {
                    unreachable!()
                }
            }
        } else {
            let mut prepare = |x: &[f32],
                               y: &[f32],
                               z: &[f32],
                               tree: Option<&KdTree>|
             -> SpatialResult<Vec<[f32; 6]>> {
                if let Some(radius) = c.covariance_radius {
                    let (covariance, receipt) =
                        spatialrust_gpu::estimate_plane_covariances_grid_gpu_with_receipt(
                            runtime, x, y, z, radius, epsilon,
                        )?;
                    record(&mut transfers, receipt);
                    covariance_batches += 1;
                    Ok(covariance)
                } else {
                    let built;
                    let tree = if let Some(tree) = tree {
                        tree
                    } else {
                        built = KdTree::from_slices(x, y, z);
                        &built
                    };
                    Ok(covariances(x, y, z, tree, c.k_neighbors, c.epsilon)
                        .into_iter()
                        .map(|m| {
                            [
                                m[0][0] as f32,
                                m[1][1] as f32,
                                m[2][2] as f32,
                                m[0][1] as f32,
                                m[0][2] as f32,
                                m[1][2] as f32,
                            ]
                        })
                        .collect())
                }
            };
            let source_cov = prepare(sx, sy, sz, None)?;
            let target_cov = prepare(tx, ty, tz, Some(&tree))?;
            let positions: Vec<_> = (0..source.len()).map(|i| [sx[i], sy[i], sz[i]]).collect();
            GpuGicpInput::upload(runtime, &positions, &source_cov, &tree, &target_cov)?
        };
        record(&mut transfers, input.upload_transfers());
        let mut evaluation_batches = 0;
        let mut evaluate = |pose| -> SpatialResult<GpuGicpEquations> {
            let equations = input
                .equations_resident(runtime, pose, c.max_correspondence_distance)?
                .download(runtime)?;
            record(&mut transfers, equations.transfers);
            evaluation_batches += 1;
            Ok(equations)
        };
        let mut transform = c.initial_guess;
        let mut equations = evaluate(transform)?;
        let mut converged = false;
        let mut history = Vec::new();
        for iteration in 1..=c.max_iterations {
            let count = equations.correspondences;
            if (count as usize) < c.min_correspondences {
                return Err(SpatialError::InvalidArgument(format!(
                    "GICP found only {count} correspondences, minimum is {}",
                    c.min_correspondences
                )));
            }
            let rows = equations.hessian.iter().map(|row| row.to_vec()).collect();
            let gradient = equations.gradient.iter().map(|g| -g).collect();
            let LeastSquaresResult::Solved(solution) = solve_linear_system(rows, gradient) else {
                return Err(SpatialError::InvalidArgument(
                    "GICP normal equations were singular".to_owned(),
                ));
            };
            let solution: [f64; 6] = solution.try_into().map_err(|_| invalid())?;
            let world = equations.world_step(solution)?;
            transform = delta_from_solution(&world).compose(transform);
            if !valid_pose(transform) {
                return Err(invalid());
            }
            equations = evaluate(transform)?;
            let fitness = fitness(&equations);
            let update_magnitude = update_magnitude(&world);
            history.push(GpuGicpIteration {
                iteration,
                correspondences: count,
                evaluated_correspondences: equations.fitness_correspondences,
                fitness,
                update_magnitude,
            });
            if update_magnitude < c.transformation_epsilon || fitness < c.fitness_epsilon {
                converged = true;
                break;
            }
        }
        Ok(GpuGicpReductionDiagnostics {
            result: RegistrationResult {
                transform,
                fitness: fitness(&equations),
                iterations: history.len(),
                converged,
            },
            history,
            transfers,
            evaluation_batches,
            covariance_batches,
        })
    }
}
