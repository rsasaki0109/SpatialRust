//! Pose graph nodes, relative edges, and loop-closure candidates.

use std::collections::{HashMap, VecDeque};

use spatialrust_math::Isometry3;

use crate::{MappingError, MappingResult, StampedPose};

/// Stable pose-graph node identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PoseNodeId(pub String);

impl PoseNodeId {
    /// Creates a node id.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

impl From<&str> for PoseNodeId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// Relative pose constraint between two nodes: `to_T_from`.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseGraphEdge {
    /// Source node.
    pub from: PoseNodeId,
    /// Target node.
    pub to: PoseNodeId,
    /// Transform mapping `from` coordinates into `to` coordinates.
    pub to_t_from: Isometry3<f32>,
    /// Whether this edge is a loop closure.
    pub loop_closure: bool,
}

/// Pose graph with absolute node estimates and relative constraints.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PoseGraph {
    nodes: HashMap<String, StampedPose>,
    edges: Vec<PoseGraphEdge>,
}

impl PoseGraph {
    /// Creates an empty pose graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces a node estimate.
    pub fn upsert_node(&mut self, id: impl Into<PoseNodeId>, pose: StampedPose) {
        self.nodes.insert(id.into().0, pose);
    }

    /// Adds a relative constraint.
    pub fn add_edge(&mut self, edge: PoseGraphEdge) -> MappingResult<()> {
        if !self.nodes.contains_key(&edge.from.0) || !self.nodes.contains_key(&edge.to.0) {
            return Err(MappingError::Missing("pose graph endpoint".into()));
        }
        self.edges.push(edge);
        Ok(())
    }

    /// Returns node estimates.
    #[must_use]
    pub fn nodes(&self) -> &HashMap<String, StampedPose> {
        &self.nodes
    }

    /// Returns relative edges.
    #[must_use]
    pub fn edges(&self) -> &[PoseGraphEdge] {
        &self.edges
    }

    /// Propagates complete poses through the directed non-loop tree rooted at `root`.
    ///
    /// Stored isometries use `node_T_anchor`, with
    /// `to_T_anchor = to_T_from * from_T_anchor`. The original root and all stamps
    /// remain fixed; disconnected components and loop-closure edges are untouched.
    /// Callers needing `anchor_T_node` explicitly invert the resulting isometry.
    ///
    /// The reachable non-loop component must be a tree: cycles, duplicate edges
    /// and multiple incoming paths return [`MappingError::Graph`]. Non-finite
    /// reachable transforms or composed results also fail. Every failure leaves
    /// the graph unchanged. Work and temporary storage are O(nodes + edges).
    pub fn localize_from_root(&mut self, root: &PoseNodeId) -> MappingResult<()> {
        let root_pose = self
            .nodes
            .get(&root.0)
            .cloned()
            .ok_or_else(|| MappingError::Missing(format!("root node `{}`", root.0)))?;
        if !finite_isometry(root_pose.pose.isometry) {
            return Err(MappingError::Graph("non-finite root pose".into()));
        }

        let mut outgoing: HashMap<&str, Vec<&PoseGraphEdge>> = HashMap::new();
        for edge in self.edges.iter().filter(|edge| !edge.loop_closure) {
            outgoing.entry(edge.from.0.as_str()).or_default().push(edge);
        }
        let mut localized = HashMap::new();
        localized.insert(root.0.as_str(), root_pose.pose.isometry);
        let mut pending = VecDeque::from([root.0.as_str()]);
        while let Some(from) = pending.pop_front() {
            let Some(edges) = outgoing.get(from) else {
                continue;
            };
            let from_pose = localized[from];
            for edge in edges {
                let to = edge.to.0.as_str();
                if localized.contains_key(to) {
                    return Err(MappingError::Graph(format!(
                        "reachable non-loop constraints must form a tree; repeated node `{to}`"
                    )));
                }
                if !finite_isometry(edge.to_t_from) {
                    return Err(MappingError::Graph(format!("non-finite constraint to `{to}`")));
                }
                let predicted = edge.to_t_from.compose(from_pose);
                if !finite_isometry(predicted) {
                    return Err(MappingError::Graph(format!("non-finite localized pose `{to}`")));
                }
                localized.insert(to, predicted);
                pending.push_back(to);
            }
        }

        // Validation completes before the first mutation. add_edge guarantees
        // every endpoint exists, and this traversal never removes a node.
        for (id, isometry) in localized {
            if id != root.0 {
                let target = self.nodes.get_mut(id).expect("validated pose graph endpoint");
                target.pose = spatialrust_math::Pose3::new(isometry);
            }
        }
        Ok(())
    }

    /// Suggests loop closures for nodes within `max_distance` translation of each other.
    pub fn loop_closure_candidates(&self, max_distance: f32) -> Vec<(PoseNodeId, PoseNodeId)> {
        let mut ids: Vec<_> = self.nodes.keys().cloned().collect();
        ids.sort();
        let mut out = Vec::new();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let a = &self.nodes[&ids[i]];
                let b = &self.nodes[&ids[j]];
                let delta =
                    (a.pose.isometry.translation() - b.pose.isometry.translation()).length();
                if delta <= max_distance {
                    out.push((PoseNodeId(ids[i].clone()), PoseNodeId(ids[j].clone())));
                }
            }
        }
        out
    }
}

fn finite_isometry(pose: Isometry3<f32>) -> bool {
    let rotation = pose.rotation();
    let translation = pose.translation();
    [rotation.x, rotation.y, rotation.z, rotation.w, translation.x, translation.y, translation.z]
        .iter()
        .all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::{PoseGraph, PoseGraphEdge, PoseNodeId};
    use crate::StampedPose;
    use spatialrust_core::Timestamp;
    use spatialrust_math::{Isometry3, Pose3, Quat, Vec3};
    use spatialrust_sync::{ClockDomain, StampedTime};

    fn stamped(x: f32, nanos: u64) -> StampedPose {
        StampedPose::new(
            StampedTime::exact("host", ClockDomain::HostSteady, Timestamp::from_nanos(nanos)),
            Pose3::new(Isometry3::new(Quat::new(0.0, 0.0, 0.0, 1.0), Vec3::new(x, 0.0, 0.0))),
        )
    }

    #[test]
    fn localizes_child_from_relative_edge() {
        let mut graph = PoseGraph::new();
        graph.upsert_node("a", stamped(0.0, 0));
        graph.upsert_node("b", stamped(0.0, 1));
        graph
            .add_edge(PoseGraphEdge {
                from: PoseNodeId::new("a"),
                to: PoseNodeId::new("b"),
                to_t_from: Isometry3::new(Quat::new(0.0, 0.0, 0.0, 1.0), Vec3::new(2.0, 0.0, 0.0)),
                loop_closure: false,
            })
            .unwrap();
        graph.localize_from_root(&PoseNodeId::new("a")).unwrap();
        assert!((graph.nodes()["b"].pose.isometry.translation().x - 2.0).abs() < 1e-4);
        let loops = graph.loop_closure_candidates(0.1);
        assert!(loops.is_empty());
    }
}
