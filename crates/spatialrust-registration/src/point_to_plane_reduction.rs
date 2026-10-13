use super::{delta_from_solution, update_magnitude, PointToPlaneIcp};
use crate::RegistrationResult;
use spatialrust_core::{
    HasNormals3, HasPositions3, PointCloud, SpatialError, SpatialResult, TransferDirection,
    TransferStats,
};
use spatialrust_gpu::{GpuPointToPlaneEquations, GpuPointToPlaneInput, WgpuRuntime};
use spatialrust_math::{solve_linear_system, Isometry3, LeastSquaresResult};
use spatialrust_search::KdTree;

/// One completed update in the explicit resident point-to-plane ICP precision path.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuPointToPlaneIteration {
    /// One-based completed update number.
    pub iteration: usize,
    /// Gated pairs used by the normal-equation solve.
    pub correspondences: u32,
    /// Gated pairs used for post-update raw-normal plane fitness.
    pub evaluated_correspondences: u32,
    /// Post-update raw-normal plane mean squared distance, f64::MAX when unmatched.
    pub fitness: f64,
    /// Norm of the world rotation/translation linearized solution, as in CPU point-to-plane ICP.
    pub update_magnitude: f64,
}

/// Explicit point-to-plane ICP resident reduction result, trace and full transfer receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuPointToPlaneReductionDiagnostics {
    /// Estimated pose, raw-normal plane fitness and stopping result.
    pub result: RegistrationResult,
    /// Completed update rows from this device precision path.
    pub history: Vec<GpuPointToPlaneIteration>,
    /// Position/tree/normal upload and fixed-size equations readback.
    pub transfers: TransferStats,
    /// Completed resident evaluations (updates plus one).
    pub evaluation_batches: usize,
}

fn invalid() -> SpatialError {
    SpatialError::InvalidArgument("GPU point-to-plane ICP reduction requires nonempty clouds, at least one iteration, minimum six correspondences, a finite positive distance square, finite nonnegative stopping thresholds and a finite unit initial pose".to_owned())
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
fn fitness(equations: &GpuPointToPlaneEquations) -> f64 {
    if equations.fitness_correspondences == 0 {
        f64::MAX
    } else {
        equations.squared_error_sum / f64::from(equations.fitness_correspondences)
    }
}

impl PointToPlaneIcp {
    /// Retains source/target/tree/raw normals on GPU for transforms, nearest
    /// matching, plane Hessian/gradient and raw-normal plane fitness reductions.
    /// CPU solves normalized 6x6 equations and converts the linearized step to
    /// the existing world update/stopping convention. Each evaluation explicitly
    /// downloads 256 bytes; post-update equations feed the next solve/final fitness.
    /// No CPU fallback occurs. Raw normals keep relative length weights and need
    /// not be unit vectors. f32 arithmetic may change matches, pose and stopping.
    pub fn align_with_gpu_reduction(
        &self,
        runtime: &WgpuRuntime,
        source: &PointCloud,
        target: &PointCloud,
    ) -> SpatialResult<GpuPointToPlaneReductionDiagnostics> {
        if source.is_empty() || target.is_empty() {
            return Err(invalid());
        }
        let c = self.config;
        let distance_squared = c.max_correspondence_distance * c.max_correspondence_distance;
        if c.max_iterations == 0
            || c.min_correspondences < 6
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
            return Err(SpatialError::InvalidArgument(
                "GPU point-to-plane ICP requires finite XYZ".to_owned(),
            ));
        }
        let tree = KdTree::from_slices(tx, ty, tz);
        let mut transfers = TransferStats::default();
        let (nx, ny, nz) = target.normals3()?;
        let positions: Vec<_> = (0..source.len()).map(|i| [sx[i], sy[i], sz[i]]).collect();
        let normals: Vec<_> = (0..target.len()).map(|i| [nx[i], ny[i], nz[i]]).collect();
        let input = GpuPointToPlaneInput::upload(runtime, &positions, &tree, &normals)?;
        record(&mut transfers, input.upload_transfers());
        let mut evaluation_batches = 0;
        let mut evaluate = |pose| -> SpatialResult<GpuPointToPlaneEquations> {
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
                    "point-to-plane ICP found only {count} correspondences, minimum is {}",
                    c.min_correspondences
                )));
            }
            let rows = equations.hessian.iter().map(|row| row.to_vec()).collect();
            let gradient = equations.gradient.iter().map(|g| -g).collect();
            let LeastSquaresResult::Solved(solution) = solve_linear_system(rows, gradient) else {
                return Err(SpatialError::InvalidArgument(
                    "point-to-plane ICP normal equations were singular".to_owned(),
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
            history.push(GpuPointToPlaneIteration {
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
        Ok(GpuPointToPlaneReductionDiagnostics {
            result: RegistrationResult {
                transform,
                fitness: fitness(&equations),
                iterations: history.len(),
                converged,
            },
            history,
            transfers,
            evaluation_batches,
        })
    }
}
