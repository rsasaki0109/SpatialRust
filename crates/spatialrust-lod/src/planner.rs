use std::collections::BTreeSet;

use spatialrust_math::Vec3;
use spatialrust_viz::{Camera, Projection};

use crate::{LodBudgets, LodError, LodIndex, LodResult, NodeId};

/// Screen-space refinement and hysteresis settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodPlannerOptions {
    /// Enter refinement above this projected error in pixels.
    pub refine_enter_pixels: f32,
    /// Leave refinement below this lower projected error in pixels.
    pub refine_exit_pixels: f32,
    /// Logical viewport height in pixels.
    pub viewport_height: u32,
}

impl LodPlannerOptions {
    /// Validates positive thresholds with exit below enter.
    pub fn validate(self) -> LodResult<()> {
        if !self.refine_enter_pixels.is_finite()
            || !self.refine_exit_pixels.is_finite()
            || self.refine_exit_pixels <= 0.0
            || self.refine_enter_pixels <= self.refine_exit_pixels
            || self.viewport_height == 0
        {
            return Err(LodError::InvalidPlanner(
                "refinement thresholds require 0 < exit < enter and a non-zero viewport".into(),
            ));
        }
        Ok(())
    }
}

/// One deterministic camera-driven LOD decision.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LodPlan {
    /// Camera-visible nodes admitted by point/request budgets.
    pub desired: Vec<NodeId>,
    /// Resident desired nodes or resident ancestors retained for continuity.
    pub display: Vec<NodeId>,
    /// Desired nodes that must be requested.
    pub request: Vec<NodeId>,
    /// Obsolete in-flight nodes to cancel.
    pub cancel: Vec<NodeId>,
    /// Visible candidates denied by hard plan budgets.
    pub denied: Vec<NodeId>,
    /// Monotonic plan generation.
    pub generation: u64,
}

/// Stateful planner retaining only hysteresis/refinement state.
#[derive(Clone, Debug)]
pub struct LodPlanner {
    options: LodPlannerOptions,
    refined: BTreeSet<NodeId>,
    generation: u64,
}

impl LodPlanner {
    /// Creates a validated planner.
    pub fn try_new(options: LodPlannerOptions) -> LodResult<Self> {
        options.validate()?;
        Ok(Self { options, refined: BTreeSet::new(), generation: 0 })
    }

    /// Plans selection, requests, cancellation, and progressive display.
    ///
    /// `resident` and `in_flight` are snapshots owned by the caller's explicit
    /// residency/source layer. Traversal order in the source index cannot affect
    /// the result.
    pub fn plan(
        &mut self,
        index: &LodIndex,
        camera: &Camera,
        aspect: f32,
        budgets: LodBudgets,
        resident: &BTreeSet<NodeId>,
        in_flight: &BTreeSet<NodeId>,
    ) -> LodResult<LodPlan> {
        self.options.validate()?;
        budgets.validate()?;
        Camera::try_new(camera.eye, camera.target, camera.up, camera.projection)
            .map_err(|error| LodError::InvalidPlanner(error.to_string()))?;
        if !aspect.is_finite() || aspect <= 0.0 {
            return Err(LodError::InvalidPlanner(
                "camera aspect must be finite and positive".into(),
            ));
        }
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| LodError::InvalidPlanner("LOD plan generation overflow".into()))?;

        let view = CameraView::try_new(*camera, aspect, self.options.viewport_height)?;
        let mut candidates = Vec::new();
        let mut next_refined = BTreeSet::new();
        for root in index.roots() {
            self.collect(index, *root, &view, &mut candidates, &mut next_refined)?;
        }
        self.refined = next_refined;

        candidates.sort_by(|a, b| {
            b.screen_error.total_cmp(&a.screen_error).then_with(|| a.id.cmp(&b.id))
        });
        let mut desired = Vec::new();
        let mut denied = Vec::new();
        let mut points = 0_u64;
        let mut request_bytes = 0_u64;
        let mut new_requests = 0_usize;
        for candidate in candidates {
            let node = index.node(candidate.id).expect("candidate comes from index");
            let next_points = points.checked_add(node.point_count);
            let needs_request = !resident.contains(&node.id) && !in_flight.contains(&node.id);
            let next_request_bytes =
                request_bytes.checked_add(if needs_request { node.upload_bytes } else { 0 });
            let next_requests = new_requests + usize::from(needs_request);
            if next_points.map_or(true, |value| value > budgets.max_points)
                || next_request_bytes
                    .map_or(true, |value| value > budgets.max_upload_bytes_per_frame)
                || next_requests > budgets.max_in_flight
            {
                denied.push(node.id);
                continue;
            }
            points = next_points.expect("checked above");
            request_bytes = next_request_bytes.expect("checked above");
            new_requests = next_requests;
            desired.push(node.id);
        }
        desired.sort_unstable();
        denied.sort_unstable();

        let desired_set: BTreeSet<_> = desired.iter().copied().collect();
        let request = desired
            .iter()
            .copied()
            .filter(|id| !resident.contains(id) && !in_flight.contains(id))
            .collect();
        let cancel = in_flight.iter().copied().filter(|id| !desired_set.contains(id)).collect();
        let mut display = BTreeSet::new();
        for id in &desired {
            if let Some(display_id) =
                index.nearest_ancestor(*id, |candidate| resident.contains(&candidate))
            {
                display.insert(display_id);
            }
        }
        Ok(LodPlan {
            desired,
            display: display.into_iter().collect(),
            request,
            cancel,
            denied,
            generation: self.generation,
        })
    }

    fn collect(
        &self,
        index: &LodIndex,
        id: NodeId,
        view: &CameraView,
        candidates: &mut Vec<Candidate>,
        refined: &mut BTreeSet<NodeId>,
    ) -> LodResult<()> {
        let node = index.node(id).ok_or(LodError::UnknownNode(id.0))?;
        let Some(screen_error) = view.screen_error(node.bounds, node.geometric_error) else {
            return Ok(());
        };
        let threshold = if self.refined.contains(&id) {
            self.options.refine_exit_pixels
        } else {
            self.options.refine_enter_pixels
        };
        if !node.children.is_empty() && screen_error > threshold {
            refined.insert(id);
            for child in &node.children {
                self.collect(index, *child, view, candidates, refined)?;
            }
        } else {
            candidates.push(Candidate { id, screen_error });
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    id: NodeId,
    screen_error: f32,
}

struct CameraView {
    eye: Vec3<f32>,
    forward: Vec3<f32>,
    right: Vec3<f32>,
    up: Vec3<f32>,
    tan_half_vertical: f32,
    aspect: f32,
    near: f32,
    far: f32,
    viewport_height: f32,
    orthographic_span: Option<f32>,
}

impl CameraView {
    fn try_new(camera: Camera, aspect: f32, viewport_height: u32) -> LodResult<Self> {
        let forward = subtract(camera.target, camera.eye).normalize();
        let right = forward.cross(camera.up).normalize();
        let up = right.cross(forward).normalize();
        let (tan_half_vertical, near, far, orthographic_span) = match camera.projection {
            Projection::Perspective { vertical_fov_radians, near, far } => {
                ((vertical_fov_radians * 0.5).tan(), near, far, None)
            }
            Projection::Orthographic { vertical_span, near, far } => {
                (0.0, near, far, Some(vertical_span))
            }
        };
        Ok(Self {
            eye: camera.eye,
            forward,
            right,
            up,
            tan_half_vertical,
            aspect,
            near,
            far,
            viewport_height: viewport_height as f32,
            orthographic_span,
        })
    }

    fn screen_error(&self, bounds: crate::LodBounds, geometric_error: f32) -> Option<f32> {
        let center = bounds.center();
        let radius = bounds.radius();
        let relative = subtract(center, self.eye);
        let depth = relative.dot(self.forward);
        if depth + radius < self.near || depth - radius > self.far {
            return None;
        }
        let horizontal = relative.dot(self.right).abs();
        let vertical = relative.dot(self.up).abs();
        if let Some(span) = self.orthographic_span {
            if vertical - radius > span * 0.5 || horizontal - radius > span * self.aspect * 0.5 {
                return None;
            }
            return Some(geometric_error * self.viewport_height / span);
        }
        if depth + radius <= 0.0
            || vertical - radius > depth.max(0.0) * self.tan_half_vertical
            || horizontal - radius > depth.max(0.0) * self.tan_half_vertical * self.aspect
        {
            return None;
        }
        Some(
            geometric_error * self.viewport_height
                / (2.0 * depth.max(self.near) * self.tan_half_vertical),
        )
    }
}

fn subtract(lhs: Vec3<f32>, rhs: Vec3<f32>) -> Vec3<f32> {
    Vec3::new(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use spatialrust_math::Vec3;
    use spatialrust_viz::{Camera, Projection};

    use crate::{LodBounds, LodBudgets, LodIndex, LodNode, LodPlanner, LodPlannerOptions, NodeId};

    fn index(reverse_children: bool) -> LodIndex {
        let bounds =
            LodBounds::try_new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)).unwrap();
        let mut children = vec![NodeId(2), NodeId(1)];
        if !reverse_children {
            children.reverse();
        }
        LodIndex::try_new([
            LodNode {
                id: NodeId(0),
                parent: None,
                children,
                bounds,
                geometric_error: 1.0,
                point_count: 100,
                host_bytes: 400,
                upload_bytes: 400,
                gpu_bytes: 400,
            },
            LodNode {
                id: NodeId(1),
                parent: Some(NodeId(0)),
                children: Vec::new(),
                bounds: LodBounds::try_new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(0.0, 1.0, 1.0))
                    .unwrap(),
                geometric_error: 0.05,
                point_count: 40,
                host_bytes: 160,
                upload_bytes: 160,
                gpu_bytes: 160,
            },
            LodNode {
                id: NodeId(2),
                parent: Some(NodeId(0)),
                children: Vec::new(),
                bounds: LodBounds::try_new(Vec3::new(0.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0))
                    .unwrap(),
                geometric_error: 0.05,
                point_count: 60,
                host_bytes: 240,
                upload_bytes: 240,
                gpu_bytes: 240,
            },
        ])
        .unwrap()
    }

    fn camera(z: f32) -> Camera {
        Camera::try_new(
            Vec3::new(0.0, 0.0, z),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Projection::Perspective { vertical_fov_radians: 1.0, near: 0.1, far: 100.0 },
        )
        .unwrap()
    }

    fn budgets() -> LodBudgets {
        LodBudgets {
            max_points: 1_000,
            max_host_bytes: 1_000,
            max_gpu_bytes: 1_000,
            max_upload_bytes_per_frame: 1_000,
            max_in_flight: 8,
        }
    }

    #[test]
    fn selection_is_traversal_order_independent_and_progressive() {
        let options = LodPlannerOptions {
            refine_enter_pixels: 20.0,
            refine_exit_pixels: 15.0,
            viewport_height: 600,
        };
        let resident = BTreeSet::from([NodeId(0)]);
        let mut a = LodPlanner::try_new(options).unwrap();
        let mut b = LodPlanner::try_new(options).unwrap();
        let first = a
            .plan(&index(false), &camera(5.0), 1.0, budgets(), &resident, &BTreeSet::new())
            .unwrap();
        let second = b
            .plan(&index(true), &camera(5.0), 1.0, budgets(), &resident, &BTreeSet::new())
            .unwrap();
        assert_eq!(first.desired, second.desired);
        assert_eq!(first.request, vec![NodeId(1), NodeId(2)]);
        assert_eq!(first.display, vec![NodeId(0)]);
    }

    #[test]
    fn hysteresis_resists_small_camera_jitter_and_cancels_obsolete_requests() {
        let options = LodPlannerOptions {
            refine_enter_pixels: 60.0,
            refine_exit_pixels: 45.0,
            viewport_height: 600,
        };
        let mut planner = LodPlanner::try_new(options).unwrap();
        let index = index(false);
        let first = planner
            .plan(&index, &camera(8.0), 1.0, budgets(), &BTreeSet::new(), &BTreeSet::new())
            .unwrap();
        let in_flight: BTreeSet<_> = first.request.iter().copied().collect();
        let jittered = planner
            .plan(&index, &camera(8.1), 1.0, budgets(), &BTreeSet::new(), &in_flight)
            .unwrap();
        assert_eq!(first.desired, jittered.desired);
        let far = planner
            .plan(&index, &camera(40.0), 1.0, budgets(), &BTreeSet::new(), &in_flight)
            .unwrap();
        assert_eq!(far.desired, vec![NodeId(0)]);
        assert_eq!(far.cancel, vec![NodeId(1), NodeId(2)]);
    }

    #[test]
    fn point_upload_and_inflight_budgets_fail_before_admission() {
        let mut planner = LodPlanner::try_new(LodPlannerOptions {
            refine_enter_pixels: 20.0,
            refine_exit_pixels: 10.0,
            viewport_height: 600,
        })
        .unwrap();
        let mut limited = budgets();
        limited.max_points = 50;
        limited.max_upload_bytes_per_frame = 200;
        limited.max_in_flight = 1;
        let plan = planner
            .plan(&index(false), &camera(5.0), 1.0, limited, &BTreeSet::new(), &BTreeSet::new())
            .unwrap();
        assert_eq!(plan.desired, vec![NodeId(1)]);
        assert_eq!(plan.denied, vec![NodeId(2)]);
    }
}
