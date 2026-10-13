use super::{
    transform_delta_magnitudes, validate_icp_transform, IcpDiagnostics, IcpIteration,
    IcpRegistration, IcpStopReason,
};
use crate::{kabsch::estimate_rigid_transform_from_moments, RegistrationResult};
use spatialrust_core::{
    HasPositions3, PointCloud, SpatialError, SpatialResult, TransferDirection, TransferStats,
};
use spatialrust_gpu::{GpuIcpInput, GpuIcpTrimPrecision, GpuIcpTrimmedStatistics, WgpuRuntime};
use spatialrust_search::KdTree;

/// Explicit resident-transform/reduction ICP result and fixed-size transfer receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuIcpReductionDiagnostics {
    /// Pose, convergence and per-update measurements from this GPU precision path.
    pub diagnostics: IcpDiagnostics,
    /// Source/target/tree upload plus fixed-size moments readbacks.
    pub transfers: TransferStats,
    /// Completed transformation/search/reduction evaluations (updates plus one).
    pub evaluation_batches: usize,
}

impl IcpRegistration {
    /// Aligns with explicitly uploaded source/target retained across iterations.
    /// GPU transforms original source points, finds bounded correspondences and
    /// reduces centered cross-covariance/centroids/fitness. Only 176 moment/status
    /// bytes are downloaded per evaluation; pose solving and convergence stay CPU.
    /// Reuses post-update moments for the next update and final fitness. No point
    /// or correspondence readback, repeated payload upload or CPU fallback occurs.
    /// Requires trim_fraction=1; `align_with_gpu_trimmed_reduction` supports
    /// resident trimming. f32 transforms/reductions can change matches, pose, fitness and
    /// stopping near numerical/geometry degeneracy compared with CPU f64 sums.
    #[cfg(feature = "register-icp-gpu")]
    pub fn align_with_gpu_reduction(
        &self,
        runtime: &WgpuRuntime,
        source: &PointCloud,
        target: &PointCloud,
    ) -> SpatialResult<GpuIcpReductionDiagnostics> {
        self.align_reduction_impl(runtime, source, target, false, GpuIcpTrimPrecision::DoubleSingle)
    }

    /// Aligns using GPU stable distance/source-index trimming and resident
    /// moments. floor(trim_fraction * gated_count) pairs contribute to each
    /// pose solve. Post-update fitness/stopping use ALL gated pairs. Only 176
    /// bytes cross to CPU per evaluation; source/target/search/selection remain
    /// resident. CPU target-tree construction, Horn solve and convergence remain.
    /// A fraction of one uses the canonical untrimmed GPU evaluation. f32 device
    /// matching/moments can differ from CPU near gates, ties or degeneracy.
    #[cfg(feature = "register-icp-gpu")]
    pub fn align_with_gpu_trimmed_reduction(
        &self,
        runtime: &WgpuRuntime,
        source: &PointCloud,
        target: &PointCloud,
    ) -> SpatialResult<GpuIcpReductionDiagnostics> {
        self.align_with_gpu_trimmed_reduction_with_precision(
            runtime,
            source,
            target,
            GpuIcpTrimPrecision::DoubleSingle,
        )
    }

    /// Aligns with an explicit retained-moment precision choice. DoubleSingle
    /// (the default resident trimmed route) carries normalized quotient/product
    /// residuals in two f32 words. CompensatedF32 retains the earlier lower-cost
    /// arithmetic. Transform/search precision, all-gated fitness, CPU solving
    /// and explicit fixed-size transfer receipts are the same for both choices.
    #[cfg(feature = "register-icp-gpu")]
    pub fn align_with_gpu_trimmed_reduction_with_precision(
        &self,
        runtime: &WgpuRuntime,
        source: &PointCloud,
        target: &PointCloud,
        precision: GpuIcpTrimPrecision,
    ) -> SpatialResult<GpuIcpReductionDiagnostics> {
        self.align_reduction_impl(runtime, source, target, true, precision)
    }

    fn align_reduction_impl(
        &self,
        runtime: &WgpuRuntime,
        source: &PointCloud,
        target: &PointCloud,
        trimmed: bool,
        precision: GpuIcpTrimPrecision,
    ) -> SpatialResult<GpuIcpReductionDiagnostics> {
        self.validate_inputs(source, target)?;
        if !trimmed && self.trim_fraction != 1.0 {
            return Err(SpatialError::InvalidArgument("GPU ICP reduction requires trim_fraction=1; use the explicit correspondence API for trimming".to_owned()));
        }
        let (sx, sy, sz) = source.positions3()?;
        let (tx, ty, tz) = target.positions3()?;
        let source_positions: Vec<_> = (0..source.len()).map(|i| [sx[i], sy[i], sz[i]]).collect();
        let tree = KdTree::from_slices(tx, ty, tz);
        let input = GpuIcpInput::upload(runtime, &source_positions, &tree)?;
        let mut transfers = input.upload_transfers();
        let mut evaluation_batches = 0;
        let mut evaluate = |pose| -> SpatialResult<GpuIcpTrimmedStatistics> {
            let statistics = input
                .trimmed_moments_resident_with_precision(
                    runtime,
                    pose,
                    self.config.max_correspondence_distance,
                    self.trim_fraction,
                    precision,
                )?
                .download(runtime)?;
            transfers.record(
                TransferDirection::DeviceToHost,
                statistics.moments.transfers.device_to_host_bytes(),
            );
            evaluation_batches += 1;
            Ok(statistics)
        };
        let mut transform = self.config.initial_guess;
        let mut moments = evaluate(transform)?;
        let mut history = Vec::new();
        let mut previous_fitness = None;
        let mut converged = false;
        let mut stop_reason = IcpStopReason::IterationLimit;
        for iteration in 1..=self.config.max_iterations {
            let evaluated = moments.evaluated_correspondences as usize;
            let count = moments.moments.correspondences as usize;
            if evaluated < self.config.min_correspondences {
                return Err(SpatialError::InvalidArgument(format!(
                    "ICP found only {evaluated} correspondences, minimum is {}",
                    self.config.min_correspondences
                )));
            }
            if count < self.config.min_correspondences {
                return Err(SpatialError::InvalidArgument(format!(
                    "trim_fraction retains only {count} of {evaluated} correspondences, minimum is {}",
                    self.config.min_correspondences
                )));
            }
            let delta = estimate_rigid_transform_from_moments(
                count,
                moments.moments.source_centroid,
                moments.moments.target_centroid,
                moments.moments.cross_covariance,
            )
            .ok_or_else(|| {
                SpatialError::InvalidArgument("ICP failed to estimate a rigid transform".to_owned())
            })?;
            validate_icp_transform(delta)?;
            transform = delta.compose(transform);
            validate_icp_transform(transform)?;
            moments = evaluate(transform)?;
            let fitness = fitness(&moments);
            let (translation_delta, rotation_delta_radians) = transform_delta_magnitudes(delta);
            history.push(IcpIteration {
                iteration,
                correspondences: count,
                evaluated_correspondences: moments.evaluated_correspondences as usize,
                fitness,
                fitness_change: previous_fitness.map(|previous| previous - fitness),
                translation_delta,
                rotation_delta_radians,
            });
            previous_fitness = Some(fitness);
            if fitness < self.criteria.fitness_epsilon {
                converged = true;
                stop_reason = IcpStopReason::FitnessThreshold;
                break;
            }
            if translation_delta < self.criteria.translation_epsilon
                && rotation_delta_radians < self.criteria.rotation_epsilon
            {
                converged = true;
                stop_reason = IcpStopReason::TransformThreshold;
                break;
            }
        }
        Ok(GpuIcpReductionDiagnostics {
            diagnostics: IcpDiagnostics {
                result: RegistrationResult {
                    transform,
                    fitness: fitness(&moments),
                    iterations: history.len(),
                    converged,
                },
                history,
                stop_reason,
            },
            transfers,
            evaluation_batches,
        })
    }
}

fn fitness(statistics: &GpuIcpTrimmedStatistics) -> f64 {
    if statistics.evaluated_correspondences == 0 {
        f64::MAX
    } else {
        statistics.moments.squared_error_sum / f64::from(statistics.evaluated_correspondences)
    }
}
