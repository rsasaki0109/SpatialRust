//! Calibrated frame graph with parent→child isometries.

use std::collections::{HashMap, HashSet, VecDeque};

use spatialrust_core::FrameId;
use spatialrust_math::Isometry3;

use crate::{SyncError, SyncResult};

/// One directed edge: `child_T_parent` transform.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameEdge {
    /// Parent frame.
    pub parent: FrameId,
    /// Child frame.
    pub child: FrameId,
    /// Transform that maps parent coordinates into child coordinates.
    pub child_t_parent: Isometry3<f32>,
}

/// Directed calibrated frame graph.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameGraph {
    /// Adjacency: parent → (child, child_T_parent).
    edges: HashMap<String, Vec<(String, Isometry3<f32>)>>,
    /// Reverse adjacency for lookups towards parents.
    reverse: HashMap<String, Vec<(String, Isometry3<f32>)>>,
}

impl FrameGraph {
    /// Creates an empty frame graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces a parent→child edge.
    pub fn insert_edge(&mut self, edge: FrameEdge) -> SyncResult<()> {
        if edge.parent.0 == edge.child.0 {
            return Err(SyncError::InvalidConfiguration(
                "frame edge parent and child must differ".into(),
            ));
        }
        self.edges
            .entry(edge.parent.0.clone())
            .or_default()
            .retain(|(child, _)| child != &edge.child.0);
        self.edges
            .entry(edge.parent.0.clone())
            .or_default()
            .push((edge.child.0.clone(), edge.child_t_parent));

        let parent_t_child = edge.child_t_parent.inverse();
        self.reverse
            .entry(edge.child.0.clone())
            .or_default()
            .retain(|(parent, _)| parent != &edge.parent.0);
        self.reverse.entry(edge.child.0.clone()).or_default().push((edge.parent.0, parent_t_child));
        Ok(())
    }

    /// Looks up a transform that maps `from` coordinates into `to` coordinates.
    pub fn lookup(&self, from: &FrameId, to: &FrameId) -> SyncResult<Isometry3<f32>> {
        if from == to {
            return Ok(Isometry3::identity());
        }
        let mut queue = VecDeque::from([(from.0.clone(), Isometry3::identity())]);
        let mut visited = HashSet::from([from.0.clone()]);
        while let Some((node, acc)) = queue.pop_front() {
            for (next, edge) in self.neighbors(&node) {
                if !visited.insert(next.clone()) {
                    continue;
                }
                let composed = edge.compose(acc);
                if next == to.0 {
                    return Ok(composed);
                }
                queue.push_back((next, composed));
            }
        }
        Err(SyncError::NoTransformPath { from: from.0.clone(), to: to.0.clone() })
    }

    fn neighbors(&self, node: &str) -> Vec<(String, Isometry3<f32>)> {
        let mut out = Vec::new();
        if let Some(forward) = self.edges.get(node) {
            out.extend(forward.iter().cloned());
        }
        if let Some(back) = self.reverse.get(node) {
            out.extend(back.iter().cloned());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameEdge, FrameGraph};
    use spatialrust_core::FrameId;
    use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};

    #[test]
    fn composes_chain_base_to_lidar() {
        let mut graph = FrameGraph::new();
        graph
            .insert_edge(FrameEdge {
                parent: FrameId::new("base"),
                child: FrameId::new("sensor"),
                child_t_parent: Isometry3::new(
                    Quat::new(0.0, 0.0, 0.0, 1.0),
                    Vec3::new(1.0, 0.0, 0.0),
                ),
            })
            .unwrap();
        graph
            .insert_edge(FrameEdge {
                parent: FrameId::new("sensor"),
                child: FrameId::new("lidar"),
                child_t_parent: Isometry3::new(
                    Quat::new(0.0, 0.0, 0.0, 1.0),
                    Vec3::new(0.0, 2.0, 0.0),
                ),
            })
            .unwrap();
        let t = graph.lookup(&FrameId::new("base"), &FrameId::new("lidar")).unwrap();
        let p = t.transform_point(Vec3::new(0.0, 0.0, 0.0));
        assert!((p.x - 1.0).abs() < 1e-5);
        assert!((p.y - 2.0).abs() < 1e-5);
    }
}
