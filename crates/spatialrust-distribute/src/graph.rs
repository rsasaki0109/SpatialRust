//! Partition graphs for edge/distributed placement.

use std::collections::HashMap;

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
}

#[cfg(test)]
mod tests {
    use super::{ExecutionPartition, PartitionGraph};

    #[test]
    fn connects_partitions() {
        let mut graph = PartitionGraph::new();
        graph
            .insert_partition(ExecutionPartition {
                id: "edge".into(),
                nodes: vec!["cam".into()],
            })
            .unwrap();
        graph
            .insert_partition(ExecutionPartition {
                id: "cloud".into(),
                nodes: vec!["infer".into()],
            })
            .unwrap();
        graph.connect("edge", "cloud").unwrap();
        assert_eq!(graph.edges().len(), 1);
    }
}
