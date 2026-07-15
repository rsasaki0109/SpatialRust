//! Bounded spatial operator graph with explicit fusion and transfer receipts.

use std::collections::{HashMap, HashSet, VecDeque};

use spatialrust_distribute::{
    BackpressurePolicy, BackpressureSignal, NamedTransfer, TransferLedger,
};

use crate::{RuntimeError, RuntimeResult};

/// Static placement and fusion properties for one operator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphNodeSpec {
    /// Stable node identifier.
    pub id: String,
    /// Host/device placement label.
    pub device: String,
    /// Whether this operator permits queue-eliding fusion with neighbors.
    pub fusable: bool,
}

impl GraphNodeSpec {
    /// Creates a checked node specification.
    pub fn try_new(
        id: impl Into<String>,
        device: impl Into<String>,
        fusable: bool,
    ) -> RuntimeResult<Self> {
        let (id, device) = (id.into(), device.into());
        if id.is_empty() || device.is_empty() {
            return Err(RuntimeError::InvalidConfiguration(
                "graph node id and device must be non-empty".into(),
            ));
        }
        Ok(Self { id, device, fusable })
    }
}

/// Stateful transformation executed at one graph node.
pub trait GraphOperator<T>: Send {
    /// Processes one owned value without an implicit device transfer.
    fn process(&mut self, value: T) -> RuntimeResult<T>;
}

/// Closure-backed graph operator.
pub struct FnOperator<F>(F);

impl<F> FnOperator<F> {
    /// Wraps a stateful transformation closure.
    pub fn new(function: F) -> Self {
        Self(function)
    }
}

impl<T, F> GraphOperator<T> for FnOperator<F>
where
    F: FnMut(T) -> RuntimeResult<T> + Send,
{
    fn process(&mut self, value: T) -> RuntimeResult<T> {
        (self.0)(value)
    }
}

struct Node<T> {
    spec: GraphNodeSpec,
    operator: Box<dyn GraphOperator<T>>,
}

/// Mutable graph builder. Compilation freezes topology and validates transfers.
pub struct SpatialExecutionGraph<T> {
    nodes: HashMap<String, Node<T>>,
    edges: Vec<(String, String)>,
    transfers: Vec<NamedTransfer>,
    pressure: BackpressurePolicy,
}

impl<T> SpatialExecutionGraph<T> {
    /// Creates an empty graph with explicit input watermarks.
    pub fn new(pressure: BackpressurePolicy) -> Self {
        Self { nodes: HashMap::new(), edges: Vec::new(), transfers: Vec::new(), pressure }
    }

    /// Adds a uniquely named operator.
    pub fn add_node(
        &mut self,
        spec: GraphNodeSpec,
        operator: impl GraphOperator<T> + 'static,
    ) -> RuntimeResult<()> {
        if self.nodes.contains_key(&spec.id) {
            return Err(RuntimeError::InvalidConfiguration(format!(
                "duplicate graph node `{}`",
                spec.id
            )));
        }
        self.nodes.insert(spec.id.clone(), Node { spec, operator: Box::new(operator) });
        Ok(())
    }

    /// Adds a directed data edge.
    pub fn connect(&mut self, from: impl Into<String>, to: impl Into<String>) -> RuntimeResult<()> {
        let (from, to) = (from.into(), to.into());
        if from == to || !self.nodes.contains_key(&from) || !self.nodes.contains_key(&to) {
            return Err(RuntimeError::InvalidConfiguration(
                "graph edge requires distinct existing endpoints".into(),
            ));
        }
        if !self.edges.iter().any(|edge| edge == &(from.clone(), to.clone())) {
            self.edges.push((from, to));
        }
        Ok(())
    }

    /// Declares a named cross-placement transfer for one edge.
    pub fn declare_transfer(&mut self, transfer: NamedTransfer) {
        self.transfers.push(transfer);
    }

    /// Validates topology/placements and builds deterministic fusion groups.
    pub fn compile(self) -> RuntimeResult<CompiledSpatialGraph<T>> {
        if self.nodes.is_empty() {
            return Err(RuntimeError::InvalidConfiguration("execution graph is empty".into()));
        }
        let order = topological_order(&self.nodes, &self.edges)?;
        let sources = order.iter().filter(|id| indegree(id, &self.edges) == 0).count();
        if sources != 1 {
            return Err(RuntimeError::InvalidConfiguration(
                "execution graph requires exactly one source".into(),
            ));
        }
        let transfer_edges = self
            .transfers
            .iter()
            .map(|transfer| (transfer.from.as_str(), transfer.to.as_str()))
            .collect::<HashSet<_>>();
        for (from, to) in &self.edges {
            let source = &self.nodes[from].spec;
            let target = &self.nodes[to].spec;
            if source.device != target.device && !transfer_edges.contains(&(from, to)) {
                return Err(RuntimeError::InvalidConfiguration(format!(
                    "cross-device edge `{from}` -> `{to}` requires a named transfer"
                )));
            }
        }
        for transfer in &self.transfers {
            if !self.edges.iter().any(|edge| edge == &(transfer.from.clone(), transfer.to.clone()))
            {
                return Err(RuntimeError::InvalidConfiguration(format!(
                    "transfer `{}` does not name a graph edge",
                    transfer.name
                )));
            }
        }
        let groups = fusion_groups(&order, &self.nodes, &self.edges);
        Ok(CompiledSpatialGraph {
            nodes: self.nodes,
            edges: self.edges,
            transfers: self.transfers,
            order,
            groups,
            pressure: self.pressure,
            inputs: VecDeque::new(),
            soft_trips: 0,
            hard_rejects: 0,
        })
    }
}

/// Per-run schedule, transfer, and backpressure evidence.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionReceipt {
    /// Nodes executed in deterministic topological order.
    pub stages: Vec<String>,
    /// Queue-eliding same-device linear groups (singletons included).
    pub fusion_groups: Vec<Vec<String>>,
    /// Named transfers completed during the run.
    pub transfers: TransferLedger,
    /// Highest observed input queue depth.
    pub max_input_depth: usize,
    /// Cumulative soft-watermark admissions.
    pub soft_trips: u64,
    /// Cumulative hard-watermark rejections.
    pub hard_rejects: u64,
}

/// Validated graph with a bounded input queue.
pub struct CompiledSpatialGraph<T> {
    nodes: HashMap<String, Node<T>>,
    edges: Vec<(String, String)>,
    transfers: Vec<NamedTransfer>,
    order: Vec<String>,
    groups: Vec<Vec<String>>,
    pressure: BackpressurePolicy,
    inputs: VecDeque<T>,
    soft_trips: u64,
    hard_rejects: u64,
}

impl<T: Clone> CompiledSpatialGraph<T> {
    /// Admits one source value or rejects at the hard watermark.
    pub fn try_submit(&mut self, value: T) -> RuntimeResult<BackpressureSignal> {
        let signal = self.pressure.evaluate(self.inputs.len());
        if signal == BackpressureSignal::HardLimit {
            self.hard_rejects += 1;
            return Err(RuntimeError::CapacityExceeded("spatial-graph-input".into()));
        }
        if signal == BackpressureSignal::SoftLimit {
            self.soft_trips += 1;
        }
        self.inputs.push_back(value);
        Ok(self.pressure.evaluate(self.inputs.len()))
    }

    /// Executes the oldest input and returns one value per sink plus a receipt.
    pub fn run_next(&mut self) -> RuntimeResult<Option<(Vec<T>, ExecutionReceipt)>> {
        let Some(value) = self.inputs.pop_front() else {
            return Ok(None);
        };
        let max_input_depth = self.inputs.len() + 1;
        let source = self
            .order
            .iter()
            .find(|id| indegree(id, &self.edges) == 0)
            .expect("compiled graph has one source")
            .clone();
        let mut pending = HashMap::<String, Vec<T>>::new();
        pending.insert(source, vec![value]);
        let mut sinks = Vec::new();
        let mut ledger = TransferLedger::new();
        for id in &self.order {
            let values = pending.remove(id).unwrap_or_default();
            for value in values {
                let output =
                    self.nodes.get_mut(id).expect("compiled node").operator.process(value)?;
                let successors = successors(id, &self.edges);
                if successors.is_empty() {
                    sinks.push(output);
                    continue;
                }
                for successor in &successors {
                    if let Some(transfer) = self
                        .transfers
                        .iter()
                        .find(|item| item.from == *id && item.to == **successor)
                    {
                        ledger.record(transfer.clone());
                    }
                    pending.entry((*successor).to_string()).or_default().push(output.clone());
                }
            }
        }
        Ok(Some((
            sinks,
            ExecutionReceipt {
                stages: self.order.clone(),
                fusion_groups: self.groups.clone(),
                transfers: ledger,
                max_input_depth,
                soft_trips: self.soft_trips,
                hard_rejects: self.hard_rejects,
            },
        )))
    }

    /// Returns the compiled queue-elision schedule.
    pub fn fusion_groups(&self) -> &[Vec<String>] {
        &self.groups
    }
}

fn indegree(id: &str, edges: &[(String, String)]) -> usize {
    edges.iter().filter(|(_, to)| to == id).count()
}

fn successors<'a>(id: &str, edges: &'a [(String, String)]) -> Vec<&'a str> {
    edges.iter().filter_map(|(from, to)| (from == id).then_some(to.as_str())).collect()
}

fn topological_order<T>(
    nodes: &HashMap<String, Node<T>>,
    edges: &[(String, String)],
) -> RuntimeResult<Vec<String>> {
    let mut degree =
        nodes.keys().map(|id| (id.clone(), indegree(id, edges))).collect::<HashMap<_, _>>();
    let mut order: Vec<String> = Vec::with_capacity(nodes.len());
    while order.len() < nodes.len() {
        let next = degree
            .iter()
            .filter(|(_, &value)| value == 0)
            .map(|(id, _)| id)
            .filter(|id| !order.contains(id))
            .min()
            .cloned();
        let Some(next) = next else {
            return Err(RuntimeError::InvalidConfiguration(
                "execution graph contains a cycle".into(),
            ));
        };
        order.push(next.clone());
        for successor in successors(&next, edges) {
            *degree.get_mut(successor).expect("validated endpoint") -= 1;
        }
    }
    Ok(order)
}

fn fusion_groups<T>(
    order: &[String],
    nodes: &HashMap<String, Node<T>>,
    edges: &[(String, String)],
) -> Vec<Vec<String>> {
    let mut groups: Vec<Vec<String>> = Vec::new();
    for id in order {
        let append = groups.last().and_then(|group| group.last()).is_some_and(|previous| {
            edges.iter().any(|edge| edge == &(previous.clone(), id.clone()))
                && successors(previous, edges).len() == 1
                && indegree(id, edges) == 1
                && nodes[previous].spec.fusable
                && nodes[id].spec.fusable
                && nodes[previous].spec.device == nodes[id].spec.device
        });
        if append {
            groups.last_mut().unwrap().push(id.clone());
        } else {
            groups.push(vec![id.clone()]);
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::{FnOperator, GraphNodeSpec, SpatialExecutionGraph};
    use spatialrust_distribute::{
        BackpressurePolicy, NamedTransfer, TransferDirection, TransferKind,
    };

    #[test]
    fn fuses_local_chain_and_receipts_named_device_copy() {
        let mut graph = SpatialExecutionGraph::new(BackpressurePolicy::try_new(2, 3).unwrap());
        for (id, device, add) in [("decode", "cpu", 1), ("gray", "cpu", 2), ("infer", "gpu", 4)] {
            graph
                .add_node(
                    GraphNodeSpec::try_new(id, device, true).unwrap(),
                    FnOperator::new(move |value: i32| Ok(value + add)),
                )
                .unwrap();
        }
        graph.connect("decode", "gray").unwrap();
        graph.connect("gray", "infer").unwrap();
        graph.declare_transfer(
            NamedTransfer::try_new(
                "gray-upload",
                TransferDirection::HostToDevice,
                TransferKind::ExplicitCopy,
                "gray",
                "infer",
                1024,
            )
            .unwrap(),
        );
        let mut compiled = graph.compile().unwrap();
        assert_eq!(
            compiled.fusion_groups(),
            &[vec!["decode".to_string(), "gray".to_string()], vec!["infer".to_string()]]
        );
        compiled.try_submit(10).unwrap();
        let (outputs, receipt) = compiled.run_next().unwrap().unwrap();
        assert_eq!(outputs, vec![17]);
        assert_eq!(receipt.transfers.counted_copy_bytes(), 1024);
        assert_eq!(receipt.transfers.completed()[0].name, "gray-upload");
    }

    #[test]
    fn rejects_missing_transfer_cycle_and_full_input() {
        let mut graph =
            SpatialExecutionGraph::<i32>::new(BackpressurePolicy::try_new(1, 1).unwrap());
        graph
            .add_node(
                GraphNodeSpec::try_new("a", "cpu", true).unwrap(),
                FnOperator::new(Ok::<_, crate::RuntimeError>),
            )
            .unwrap();
        graph
            .add_node(
                GraphNodeSpec::try_new("b", "gpu", true).unwrap(),
                FnOperator::new(Ok::<_, crate::RuntimeError>),
            )
            .unwrap();
        graph.connect("a", "b").unwrap();
        assert!(graph.compile().is_err());

        let mut graph =
            SpatialExecutionGraph::<i32>::new(BackpressurePolicy::try_new(1, 1).unwrap());
        graph
            .add_node(
                GraphNodeSpec::try_new("a", "cpu", true).unwrap(),
                FnOperator::new(Ok::<_, crate::RuntimeError>),
            )
            .unwrap();
        let mut compiled = graph.compile().unwrap();
        compiled.try_submit(1).unwrap();
        assert!(compiled.try_submit(2).is_err());
    }
}
