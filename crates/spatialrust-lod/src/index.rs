use std::collections::{BTreeMap, BTreeSet};

use spatialrust_math::Vec3;

use crate::{LodError, LodResult};

/// Stable LOD/index node identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u64);

/// Finite axis-aligned node bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodBounds {
    /// Inclusive minimum corner.
    pub min: Vec3<f32>,
    /// Inclusive maximum corner.
    pub max: Vec3<f32>,
}

impl LodBounds {
    /// Creates ordered finite bounds.
    pub fn try_new(min: Vec3<f32>, max: Vec3<f32>) -> LodResult<Self> {
        if [min.x, min.y, min.z, max.x, max.y, max.z].iter().any(|value| !value.is_finite())
            || min.x > max.x
            || min.y > max.y
            || min.z > max.z
        {
            return Err(LodError::InvalidIndex("node bounds must be finite and ordered".into()));
        }
        Ok(Self { min, max })
    }

    /// Bounds center.
    #[must_use]
    pub fn center(self) -> Vec3<f32> {
        Vec3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    /// Radius of a sphere containing these bounds.
    #[must_use]
    pub fn radius(self) -> f32 {
        let half = Vec3::new(
            (self.max.x - self.min.x) * 0.5,
            (self.max.y - self.min.y) * 0.5,
            (self.max.z - self.min.z) * 0.5,
        );
        half.length()
    }

    /// Union of two bounds.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: Vec3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vec3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }
}

/// One node from a persisted spatial/COPC hierarchy.
#[derive(Clone, Debug, PartialEq)]
pub struct LodNode {
    /// Stable identity.
    pub id: NodeId,
    /// Parent identity; root nodes have none.
    pub parent: Option<NodeId>,
    /// Child identities.
    pub children: Vec<NodeId>,
    /// World-space bounds.
    pub bounds: LodBounds,
    /// Conservative geometric error in world units.
    pub geometric_error: f32,
    /// Points materialized by this node.
    pub point_count: u64,
    /// Estimated host bytes needed to decode this node.
    pub host_bytes: u64,
    /// Estimated bytes uploaded for this node.
    pub upload_bytes: u64,
    /// GPU bytes retained after upload.
    pub gpu_bytes: u64,
}

impl LodNode {
    fn validate(&self) -> LodResult<()> {
        if !self.geometric_error.is_finite()
            || self.geometric_error < 0.0
            || self.point_count == 0
            || self.host_bytes == 0
            || self.upload_bytes == 0
            || self.gpu_bytes == 0
        {
            return Err(LodError::InvalidIndex(format!(
                "node {} requires finite non-negative error and positive counts/bytes",
                self.id.0
            )));
        }
        let mut children = self.children.clone();
        children.sort_unstable();
        children.dedup();
        if children.len() != self.children.len() || children.contains(&self.id) {
            return Err(LodError::InvalidIndex(format!(
                "node {} has duplicate or self child",
                self.id.0
            )));
        }
        Ok(())
    }
}

/// Validated deterministic LOD hierarchy.
#[derive(Clone, Debug, PartialEq)]
pub struct LodIndex {
    nodes: BTreeMap<NodeId, LodNode>,
    roots: Vec<NodeId>,
}

impl LodIndex {
    /// Validates nodes, reciprocal parent links, roots, reachability, and cycles.
    pub fn try_new(nodes: impl IntoIterator<Item = LodNode>) -> LodResult<Self> {
        let mut map = BTreeMap::new();
        for mut node in nodes {
            node.validate()?;
            node.children.sort_unstable();
            if map.insert(node.id, node).is_some() {
                return Err(LodError::InvalidIndex("duplicate node ID".into()));
            }
        }
        if map.is_empty() {
            return Err(LodError::InvalidIndex("LOD index must not be empty".into()));
        }
        for node in map.values() {
            if let Some(parent) = node.parent {
                let parent_node = map.get(&parent).ok_or_else(|| {
                    LodError::InvalidIndex(format!("missing parent {}", parent.0))
                })?;
                if !parent_node.children.contains(&node.id) {
                    return Err(LodError::InvalidIndex(format!(
                        "parent {} does not reference child {}",
                        parent.0, node.id.0
                    )));
                }
            }
            for child in &node.children {
                let child_node = map
                    .get(child)
                    .ok_or_else(|| LodError::InvalidIndex(format!("missing child {}", child.0)))?;
                if child_node.parent != Some(node.id) {
                    return Err(LodError::InvalidIndex(format!(
                        "child {} has inconsistent parent",
                        child.0
                    )));
                }
            }
        }
        let roots: Vec<_> =
            map.values().filter(|node| node.parent.is_none()).map(|node| node.id).collect();
        if roots.is_empty() {
            return Err(LodError::InvalidIndex("LOD index has no root".into()));
        }
        let mut visited = BTreeSet::new();
        let mut active = BTreeSet::new();
        for root in &roots {
            visit(*root, &map, &mut visited, &mut active)?;
        }
        if visited.len() != map.len() {
            return Err(LodError::InvalidIndex("LOD index contains unreachable nodes".into()));
        }
        Ok(Self { nodes: map, roots })
    }

    /// Finds a node.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<&LodNode> {
        self.nodes.get(&id)
    }

    /// Root IDs in deterministic order.
    #[must_use]
    pub fn roots(&self) -> &[NodeId] {
        &self.roots
    }

    /// All nodes ordered by ID.
    pub fn nodes(&self) -> impl ExactSizeIterator<Item = &LodNode> {
        self.nodes.values()
    }

    /// Returns the nearest ancestor, including `id`, satisfying `predicate`.
    pub(crate) fn nearest_ancestor(
        &self,
        id: NodeId,
        mut predicate: impl FnMut(NodeId) -> bool,
    ) -> Option<NodeId> {
        let mut current = Some(id);
        while let Some(node_id) = current {
            if predicate(node_id) {
                return Some(node_id);
            }
            current = self.nodes.get(&node_id).and_then(|node| node.parent);
        }
        None
    }
}

fn visit(
    id: NodeId,
    nodes: &BTreeMap<NodeId, LodNode>,
    visited: &mut BTreeSet<NodeId>,
    active: &mut BTreeSet<NodeId>,
) -> LodResult<()> {
    if active.contains(&id) {
        return Err(LodError::InvalidIndex("LOD hierarchy contains a cycle".into()));
    }
    if !visited.insert(id) {
        return Ok(());
    }
    active.insert(id);
    for child in &nodes[&id].children {
        visit(*child, nodes, visited, active)?;
    }
    active.remove(&id);
    Ok(())
}
