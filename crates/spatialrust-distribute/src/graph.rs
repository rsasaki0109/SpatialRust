//! Partition graphs for edge/distributed placement.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::{DistributeError, DistributeResult};

/// One execution node on a device/host.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExecutionNode {
    /// Node id.
    pub id: String,
    /// Device or host label.
    pub device: String,
}

/// Named partition of execution nodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionPartition {
    /// Partition id.
    pub id: String,
    /// Member node ids.
    pub nodes: Vec<String>,
}

impl ExecutionPartition {
    /// Creates a validated non-empty partition.
    pub fn try_new(id: impl Into<String>, nodes: Vec<String>) -> DistributeResult<Self> {
        let id = id.into();
        if id.is_empty() {
            return Err(DistributeError::InvalidConfiguration(
                "partition id must be non-empty".into(),
            ));
        }
        if nodes.is_empty() {
            return Err(DistributeError::InvalidConfiguration(
                "partition must contain at least one node".into(),
            ));
        }
        if nodes.iter().any(|n| n.is_empty()) {
            return Err(DistributeError::InvalidConfiguration(
                "node ids must be non-empty".into(),
            ));
        }
        Ok(Self { id, nodes })
    }

    /// Returns true when `node_id` is a member.
    #[must_use]
    pub fn contains(&self, node_id: &str) -> bool {
        self.nodes.iter().any(|n| n == node_id)
    }
}

/// Directed partition graph with adjacency between partitions.
#[derive(Clone, Debug, Default)]
pub struct PartitionGraph {
    partitions: HashMap<String, ExecutionPartition>,
    edges: Vec<(String, String)>,
}

impl PartitionGraph {
    /// Creates an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a partition.
    pub fn insert_partition(&mut self, partition: ExecutionPartition) -> DistributeResult<()> {
        if partition.id.is_empty() {
            return Err(DistributeError::InvalidConfiguration(
                "partition id must be non-empty".into(),
            ));
        }
        if partition.nodes.is_empty() {
            return Err(DistributeError::InvalidConfiguration(
                "partition must contain at least one node".into(),
            ));
        }
        self.partitions.insert(partition.id.clone(), partition);
        Ok(())
    }

    /// Connects two partitions with a directed edge.
    pub fn connect(&mut self, from: impl Into<String>, to: impl Into<String>) -> DistributeResult<()> {
        let from = from.into();
        let to = to.into();
        if !self.partitions.contains_key(&from) || !self.partitions.contains_key(&to) {
            return Err(DistributeError::Missing("partition endpoint".into()));
        }
        if from == to {
            return Err(DistributeError::InvalidConfiguration(
                "self-edges are not allowed".into(),
            ));
        }
        if self.edges.iter().any(|(a, b)| a == &from && b == &to) {
            return Ok(());
        }
        self.edges.push((from, to));
        Ok(())
    }

    /// Returns partitions.
    #[must_use]
    pub fn partitions(&self) -> &HashMap<String, ExecutionPartition> {
        &self.partitions
    }

    /// Returns edges.
    #[must_use]
    pub fn edges(&self) -> &[(String, String)] {
        &self.edges
    }

    /// Looks up a partition by id.
    #[must_use]
    pub fn partition(&self, id: &str) -> Option<&ExecutionPartition> {
        self.partitions.get(id)
    }

    /// Finds which partition owns a node id.
    #[must_use]
    pub fn partition_of_node(&self, node_id: &str) -> Option<&ExecutionPartition> {
        self.partitions
            .values()
            .find(|partition| partition.contains(node_id))
    }

    /// Returns outgoing neighbors of a partition.
    #[must_use]
    pub fn successors(&self, id: &str) -> Vec<&str> {
        self.edges
            .iter()
            .filter_map(|(from, to)| (from == id).then_some(to.as_str()))
            .collect()
    }

    /// Returns a topological order of partitions, or errors on cycles / missing nodes.
    pub fn topological_order(&self) -> DistributeResult<Vec<String>> {
        let mut indegree: HashMap<&str, usize> =
            self.partitions.keys().map(|id| (id.as_str(), 0usize)).collect();
        for (from, to) in &self.edges {
            if !indegree.contains_key(from.as_str()) || !indegree.contains_key(to.as_str()) {
                return Err(DistributeError::Missing("partition endpoint".into()));
            }
            *indegree.get_mut(to.as_str()).unwrap() += 1;
        }

        let mut queue: VecDeque<String> = indegree
            .iter()
            .filter_map(|(id, deg)| (*deg == 0).then(|| (*id).to_string()))
            .collect();
        // Stable order for deterministic schedules.
        let mut queued: HashSet<String> = queue.iter().cloned().collect();
        let mut order = Vec::with_capacity(self.partitions.len());

        while let Some(id) = {
            // Pop lexicographically smallest ready id for determinism.
            let next = queue.iter().min().cloned();
            if let Some(ref n) = next {
                queue.retain(|x| x != n);
                queued.remove(n);
            }
            next
        } {
            order.push(id.clone());
            for succ in self.successors(&id) {
                let deg = indegree.get_mut(succ).unwrap();
                *deg = deg.saturating_sub(1);
                if *deg == 0 && !queued.contains(succ) {
                    queued.insert(succ.to_string());
                    queue.push_back(succ.to_string());
                }
            }
        }

        if order.len() != self.partitions.len() {
            return Err(DistributeError::CycleDetected);
        }
        Ok(order)
    }
}

#[cfg(test)]
mod tests {
    use super::{ExecutionPartition, PartitionGraph};

    #[test]
    fn connects_partitions_and_orders() {
        let mut graph = PartitionGraph::new();
        graph
            .insert_partition(ExecutionPartition::try_new("edge", vec!["cam".into()]).unwrap())
            .unwrap();
        graph
            .insert_partition(ExecutionPartition::try_new("cloud", vec!["infer".into()]).unwrap())
            .unwrap();
        graph.connect("edge", "cloud").unwrap();
        assert_eq!(graph.edges().len(), 1);
        assert_eq!(graph.topological_order().unwrap(), vec!["edge", "cloud"]);
        assert_eq!(graph.partition_of_node("cam").unwrap().id, "edge");
    }

    #[test]
    fn detects_cycle() {
        let mut graph = PartitionGraph::new();
        graph
            .insert_partition(ExecutionPartition::try_new("a", vec!["n0".into()]).unwrap())
            .unwrap();
        graph
            .insert_partition(ExecutionPartition::try_new("b", vec!["n1".into()]).unwrap())
            .unwrap();
        graph.connect("a", "b").unwrap();
        graph.connect("b", "a").unwrap();
        assert!(matches!(
            graph.topological_order(),
            Err(crate::DistributeError::CycleDetected)
        ));
    }
}
