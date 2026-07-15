//! Backend-independent model, session, named-I/O, and explicit binding contracts.
//!
//! The default build contains no inference runtime. ONNX Runtime and hardware
//! execution providers are additive features and must preserve the copy/device
//! choices represented by these types.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use std::{path::PathBuf, sync::Arc};

use spatialrust_tensor::{DataType, Device, TensorBuffer, TensorDescriptor};

mod mock;
pub use mock::{MockInferenceBackend, MockProfile};

#[cfg(feature = "onnxruntime")]
mod onnxruntime;
#[cfg(feature = "onnxruntime")]
pub use onnxruntime::{OnnxRuntimeBackend, OnnxRuntimeSession};

/// Result type for inference operations.
pub type AiResult<T> = Result<T, AiError>;

/// Errors shared by inference contracts and backend adapters.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AiError {
    /// A named input or output appears more than once.
    #[error("duplicate tensor name `{0}`")]
    DuplicateName(String),
    /// A required model input was not supplied.
    #[error("missing required input `{0}`")]
    MissingInput(String),
    /// A tensor name is not declared by the model.
    #[error("unexpected tensor `{0}`")]
    UnexpectedTensor(String),
    /// Actual dtype differs from the model contract.
    #[error("tensor `{name}` requires {expected:?}, found {actual:?}")]
    DataTypeMismatch {
        /// Tensor name.
        name: String,
        /// Model dtype.
        expected: DataType,
        /// Supplied dtype.
        actual: DataType,
    },
    /// Actual rank or dimension differs from the model contract.
    #[error("tensor `{name}` shape {actual:?} does not match {expected:?}")]
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// Model dimensions.
        expected: Vec<Dimension>,
        /// Supplied dimensions.
        actual: Vec<usize>,
    },
    /// A preallocated output does not have enough bytes.
    #[error("preallocated output `{name}` needs {required} bytes, found {found}")]
    OutputBufferTooSmall {
        /// Output name.
        name: String,
        /// Required allocation bytes.
        required: usize,
        /// Available allocation bytes.
        found: usize,
    },
    /// The selected backend cannot honor an operation or option.
    #[error("backend `{backend}` does not support {operation}")]
    Unsupported {
        /// Backend identifier.
        backend: String,
        /// Unsupported operation.
        operation: String,
    },
    /// Backend-specific failure with stable outer context.
    #[error("backend `{backend}` failed: {message}")]
    Backend {
        /// Backend identifier.
        backend: String,
        /// Backend error message.
        message: String,
    },
    /// Invalid public configuration.
    #[error("invalid inference configuration: {0}")]
    InvalidConfiguration(String),
    /// A backend would need a copy that the caller did not authorize.
    #[error(
        "{direction} copy is required for tensor `{name}`; opt in explicitly or use I/O binding"
    )]
    CopyRequired {
        /// Input or output transfer direction.
        direction: &'static str,
        /// Model-visible tensor name.
        name: String,
    },
}

/// One model dimension, fixed, unconstrained, or symbolically dynamic.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dimension {
    /// Exact dimension size.
    Fixed(usize),
    /// Dynamic dimension without a model-provided symbol.
    Dynamic,
    /// Dynamic dimension sharing a model-provided symbolic name.
    Symbol(String),
}

impl Dimension {
    fn accepts(&self, actual: usize) -> bool {
        match self {
            Self::Fixed(expected) => *expected == actual,
            Self::Dynamic | Self::Symbol(_) => true,
        }
    }
}

/// One named model input or output contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    /// ONNX/model-visible name.
    pub name: String,
    /// Required scalar/vector dtype.
    pub dtype: DataType,
    /// Ordered fixed or dynamic dimensions.
    pub shape: Vec<Dimension>,
}

impl TensorSpec {
    /// Creates a named tensor specification.
    pub fn new(name: impl Into<String>, dtype: DataType, shape: Vec<Dimension>) -> Self {
        Self { name: name.into(), dtype, shape }
    }

    /// Validates a concrete tensor descriptor against this specification.
    pub fn validate(&self, descriptor: &TensorDescriptor) -> AiResult<()> {
        if descriptor.dtype() != self.dtype {
            return Err(AiError::DataTypeMismatch {
                name: self.name.clone(),
                expected: self.dtype,
                actual: descriptor.dtype(),
            });
        }
        if descriptor.shape().len() != self.shape.len()
            || !self
                .shape
                .iter()
                .zip(descriptor.shape())
                .all(|(expected, &actual)| expected.accepts(actual))
        {
            return Err(AiError::ShapeMismatch {
                name: self.name.clone(),
                expected: self.shape.clone(),
                actual: descriptor.shape().to_vec(),
            });
        }
        Ok(())
    }
}

/// Model-visible named input and output metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelInfo {
    /// Optional producer/model identifier.
    pub name: Option<String>,
    /// Ordered model inputs.
    pub inputs: Vec<TensorSpec>,
    /// Ordered model outputs.
    pub outputs: Vec<TensorSpec>,
}

impl ModelInfo {
    /// Validates uniqueness of all input and output names.
    pub fn validate(&self) -> AiResult<()> {
        ensure_unique(self.inputs.iter().map(|spec| spec.name.as_str()))?;
        ensure_unique(self.outputs.iter().map(|spec| spec.name.as_str()))?;
        Ok(())
    }

    /// Validates required names, dtype, and dynamic/fixed shapes for one request.
    pub fn validate_inputs(&self, inputs: &NamedTensors) -> AiResult<()> {
        for spec in &self.inputs {
            let tensor =
                inputs.get(&spec.name).ok_or_else(|| AiError::MissingInput(spec.name.clone()))?;
            spec.validate(tensor.descriptor())?;
        }
        for (name, _) in inputs.iter() {
            if !self.inputs.iter().any(|spec| spec.name == name) {
                return Err(AiError::UnexpectedTensor(name.to_owned()));
            }
        }
        Ok(())
    }
}

fn ensure_unique<'a>(names: impl IntoIterator<Item = &'a str>) -> AiResult<()> {
    let mut seen = Vec::<&str>::new();
    for name in names {
        if seen.contains(&name) {
            return Err(AiError::DuplicateName(name.to_owned()));
        }
        seen.push(name);
    }
    Ok(())
}

/// Ordered, uniquely named tensor collection.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NamedTensors {
    values: Vec<(String, TensorBuffer)>,
}

impl NamedTensors {
    /// Creates an empty collection.
    pub const fn new() -> Self {
        Self { values: Vec::new() }
    }

    /// Inserts a unique named tensor while retaining insertion order.
    pub fn insert(&mut self, name: impl Into<String>, tensor: TensorBuffer) -> AiResult<()> {
        let name = name.into();
        if self.values.iter().any(|(existing, _)| existing == &name) {
            return Err(AiError::DuplicateName(name));
        }
        self.values.push((name, tensor));
        Ok(())
    }

    /// Returns a tensor by model-visible name.
    pub fn get(&self, name: &str) -> Option<&TensorBuffer> {
        self.values.iter().find(|(candidate, _)| candidate == name).map(|(_, value)| value)
    }

    /// Returns ordered `(name, tensor)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TensorBuffer)> {
        self.values.iter().map(|(name, tensor)| (name.as_str(), tensor))
    }

    /// Returns the tensor count.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether no tensors are present.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Consumes the collection into ordered pairs.
    pub fn into_values(self) -> Vec<(String, TensorBuffer)> {
        self.values
    }
}

/// Model bytes, filesystem path, or built-in mock profile for session creation.
#[derive(Clone, Debug)]
pub enum ModelSource {
    /// Read model bytes from this path during session creation.
    Path(PathBuf),
    /// Immutable in-memory model bytes.
    Bytes(Arc<[u8]>),
    /// Deterministic in-process mock profile (no ONNX / no weights).
    Mock(MockProfile),
}

/// Graph optimization level requested from a backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GraphOptimization {
    /// Disable graph rewrites.
    Disabled,
    /// Apply safe basic rewrites.
    Basic,
    /// Apply extended rewrites.
    Extended,
    /// Apply all backend-supported rewrites.
    #[default]
    All,
}

/// Runtime-independent session configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionOptions {
    /// Intra-operator worker count; `None` delegates to the backend.
    pub intra_threads: Option<usize>,
    /// Inter-operator worker count; `None` delegates to the backend.
    pub inter_threads: Option<usize>,
    /// Graph rewrite level.
    pub graph_optimization: GraphOptimization,
    /// Request deterministic kernels where supported.
    pub deterministic: bool,
}

/// Whether a run may allocate and copy host tensor data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CopyPolicy {
    /// Fail instead of silently copying.
    #[default]
    Forbid,
    /// Permit a documented host-to-host copy for this run.
    Allow,
}

/// Per-run host copy permissions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RunOptions {
    /// Permission to repack/copy named inputs into backend values.
    pub input_copy: CopyPolicy,
    /// Permission to copy backend-owned outputs into `TensorBuffer`.
    pub output_copy: CopyPolicy,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            intra_threads: None,
            inter_threads: None,
            graph_optimization: GraphOptimization::All,
            deterministic: false,
        }
    }
}

impl SessionOptions {
    /// Rejects zero thread counts.
    pub fn validate(&self) -> AiResult<()> {
        if self.intra_threads == Some(0) || self.inter_threads == Some(0) {
            return Err(AiError::InvalidConfiguration("thread counts must be positive".into()));
        }
        Ok(())
    }
}

/// Explicit output destination for an I/O-bound run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputBinding {
    /// Ask the backend to allocate this output on the named device.
    Allocate {
        /// Model output name.
        name: String,
        /// Required allocation device.
        device: Device,
    },
    /// Write into this caller-owned CPU allocation.
    PreallocatedCpu(PreallocatedOutput),
}

/// Caller-owned mutable bytes for an explicitly bound CPU output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreallocatedOutput {
    name: String,
    descriptor: TensorDescriptor,
    storage: Option<PreallocatedStorage>,
}

#[derive(Clone, Debug)]
pub(crate) enum PreallocatedStorage {
    Bytes(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

impl PreallocatedStorage {
    fn allocation_bytes(&self) -> &[u8] {
        match self {
            Self::Bytes(values) => values,
            Self::U16(values) => bytemuck::cast_slice(values),
            Self::F32(values) => bytemuck::cast_slice(values),
        }
    }

    fn allocation_bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Self::Bytes(values) => values,
            Self::U16(values) => bytemuck::cast_slice_mut(values),
            Self::F32(values) => bytemuck::cast_slice_mut(values),
        }
    }
}

impl PartialEq for PreallocatedStorage {
    fn eq(&self, other: &Self) -> bool {
        self.allocation_bytes() == other.allocation_bytes()
    }
}

impl Eq for PreallocatedStorage {}

impl PreallocatedOutput {
    /// Creates a checked preallocated CPU output.
    pub fn try_new(
        name: impl Into<String>,
        descriptor: TensorDescriptor,
        bytes: Vec<u8>,
    ) -> AiResult<Self> {
        let name = name.into();
        let required = descriptor
            .required_byte_range()
            .map_err(|error| AiError::InvalidConfiguration(error.to_string()))?
            .end;
        if !descriptor.device().is_host_accessible() {
            return Err(AiError::InvalidConfiguration(
                "PreallocatedCpu requires host-accessible storage".into(),
            ));
        }
        if bytes.len() < required {
            return Err(AiError::OutputBufferTooSmall { name, required, found: bytes.len() });
        }
        Ok(Self { name, descriptor, storage: Some(PreallocatedStorage::Bytes(bytes)) })
    }

    /// Allocates aligned storage for a compact `u8`, `u16`, or `f32` CPU output.
    pub fn allocate(name: impl Into<String>, descriptor: TensorDescriptor) -> AiResult<Self> {
        if !descriptor.is_c_contiguous() || descriptor.byte_offset() != 0 {
            return Err(AiError::InvalidConfiguration(
                "preallocated output must be compact with byte_offset=0".into(),
            ));
        }
        if descriptor.device() != Device::CPU {
            return Err(AiError::InvalidConfiguration(
                "preallocated output allocation currently requires Device::CPU".into(),
            ));
        }
        let elements = descriptor
            .element_count()
            .map_err(|error| AiError::InvalidConfiguration(error.to_string()))?;
        let storage = match descriptor.dtype() {
            DataType::U8 => PreallocatedStorage::Bytes(vec![0; elements]),
            DataType::U16 => PreallocatedStorage::U16(vec![0; elements]),
            DataType::F32 => PreallocatedStorage::F32(vec![0.0; elements]),
            dtype => {
                return Err(AiError::InvalidConfiguration(format!(
                    "preallocated output dtype {dtype:?} is not supported; use u8, u16, or f32"
                )))
            }
        };
        Ok(Self { name: name.into(), descriptor, storage: Some(storage) })
    }

    /// Returns the model output name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns output metadata.
    pub const fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    /// Returns mutable caller-owned output bytes for a backend binding.
    pub fn allocation_bytes_mut(&mut self) -> AiResult<&mut [u8]> {
        self.storage.as_mut().map(PreallocatedStorage::allocation_bytes_mut).ok_or_else(|| {
            AiError::InvalidConfiguration(format!(
                "preallocated output `{}` has already been consumed",
                self.name
            ))
        })
    }

    /// Returns whether a bound run has taken ownership of this allocation.
    pub fn is_consumed(&self) -> bool {
        self.storage.is_none()
    }

    /// Converts a completed binding into generic owned tensor storage.
    pub fn into_tensor(mut self) -> AiResult<TensorBuffer> {
        let storage = self.take_storage()?;
        tensor_from_preallocated(storage, self.descriptor)
    }

    pub(crate) fn take_storage(&mut self) -> AiResult<PreallocatedStorage> {
        self.storage.take().ok_or_else(|| {
            AiError::InvalidConfiguration(format!(
                "preallocated output `{}` has already been consumed",
                self.name
            ))
        })
    }
}

fn tensor_from_preallocated(
    storage: PreallocatedStorage,
    descriptor: TensorDescriptor,
) -> AiResult<TensorBuffer> {
    let result = match storage {
        PreallocatedStorage::Bytes(values) => TensorBuffer::try_new(values, descriptor),
        PreallocatedStorage::U16(values) => TensorBuffer::try_from_u16(values, descriptor),
        PreallocatedStorage::F32(values) => TensorBuffer::try_from_f32(values, descriptor),
    };
    result.map_err(|error| AiError::InvalidConfiguration(error.to_string()))
}

/// Inputs, requested output destinations, and completed named results.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IoBinding {
    inputs: NamedTensors,
    outputs: Vec<OutputBinding>,
    results: Option<NamedTensors>,
}

impl IoBinding {
    /// Creates an explicit binding request.
    pub fn try_new(inputs: NamedTensors, outputs: Vec<OutputBinding>) -> AiResult<Self> {
        ensure_unique(outputs.iter().map(|output| match output {
            OutputBinding::Allocate { name, .. } => name.as_str(),
            OutputBinding::PreallocatedCpu(output) => output.name(),
        }))?;
        Ok(Self { inputs, outputs, results: None })
    }

    /// Returns bound named inputs.
    pub const fn inputs(&self) -> &NamedTensors {
        &self.inputs
    }

    /// Returns requested output destinations.
    pub fn outputs(&self) -> &[OutputBinding] {
        &self.outputs
    }

    /// Returns mutable requested output destinations.
    pub fn outputs_mut(&mut self) -> &mut [OutputBinding] {
        &mut self.outputs
    }

    /// Stores completed named results. Intended for backend implementers.
    pub fn set_results(&mut self, results: NamedTensors) {
        self.results = Some(results);
    }

    /// Clears results before a backend starts another run.
    pub fn clear_results(&mut self) {
        self.results = None;
    }

    /// Returns completed results after a successful bound run.
    pub fn results(&self) -> Option<&NamedTensors> {
        self.results.as_ref()
    }

    /// Consumes completed results, if present.
    pub fn into_results(self) -> Option<NamedTensors> {
        self.results
    }
}

/// Stable interface implemented by inference engines.
pub trait InferenceBackend: Send + Sync {
    /// Stable backend identifier such as `onnxruntime-cpu`.
    fn name(&self) -> &str;

    /// Loads one model and returns an independent mutable session.
    fn create_session(
        &self,
        source: &ModelSource,
        options: &SessionOptions,
    ) -> AiResult<Box<dyn ModelSession>>;
}

/// Loaded model session with named dynamic I/O.
pub trait ModelSession: Send {
    /// Stable backend identifier for diagnostics and capability checks.
    fn backend_name(&self) -> &str;

    /// Returns model input/output metadata captured at load time.
    fn model_info(&self) -> &ModelInfo;

    /// Runs without authorizing hidden host copies.
    fn run(&mut self, inputs: NamedTensors) -> AiResult<NamedTensors> {
        self.run_with_options(inputs, RunOptions::default())
    }

    /// Runs named I/O with explicit host copy permissions.
    fn run_with_options(
        &mut self,
        inputs: NamedTensors,
        options: RunOptions,
    ) -> AiResult<NamedTensors>;

    /// Runs with explicit input/output device or allocation bindings.
    fn run_with_binding(&mut self, _binding: &mut IoBinding) -> AiResult<()> {
        Err(AiError::Unsupported {
            backend: self.backend_name().into(),
            operation: "explicit I/O binding".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AiError, Dimension, IoBinding, ModelInfo, NamedTensors, OutputBinding, PreallocatedOutput,
        TensorSpec,
    };
    use spatialrust_tensor::{DataType, Device, TensorBuffer, TensorDescriptor};

    fn tensor(shape: Vec<usize>) -> TensorBuffer {
        let len = shape.iter().product::<usize>() * 4;
        TensorBuffer::try_new(
            vec![0; len],
            TensorDescriptor::contiguous(DataType::F32, shape, Device::CPU),
        )
        .unwrap()
    }

    #[test]
    fn validates_named_dynamic_inputs() {
        let info = ModelInfo {
            name: Some("dynamic-batch".into()),
            inputs: vec![TensorSpec::new(
                "images",
                DataType::F32,
                vec![Dimension::Symbol("batch".into()), Dimension::Fixed(3), Dimension::Dynamic],
            )],
            outputs: vec![],
        };
        let mut inputs = NamedTensors::new();
        inputs.insert("images", tensor(vec![4, 3, 224])).unwrap();
        info.validate_inputs(&inputs).unwrap();
        let wrong = TensorDescriptor::contiguous(DataType::F32, vec![4, 1, 224], Device::CPU);
        assert!(matches!(info.inputs[0].validate(&wrong), Err(AiError::ShapeMismatch { .. })));
    }

    #[test]
    fn named_tensors_reject_duplicates_and_missing_inputs() {
        let mut inputs = NamedTensors::new();
        inputs.insert("x", tensor(vec![2])).unwrap();
        assert!(matches!(inputs.insert("x", tensor(vec![2])), Err(AiError::DuplicateName(_))));
        let info = ModelInfo {
            name: None,
            inputs: vec![TensorSpec::new("y", DataType::F32, vec![Dimension::Fixed(2)])],
            outputs: vec![],
        };
        assert!(matches!(info.validate_inputs(&inputs), Err(AiError::MissingInput(_))));
    }

    #[test]
    fn io_binding_requires_explicit_unique_outputs() {
        let descriptor = TensorDescriptor::contiguous(DataType::F32, vec![2], Device::CPU);
        let output = PreallocatedOutput::allocate("scores", descriptor).unwrap();
        let binding =
            IoBinding::try_new(NamedTensors::new(), vec![OutputBinding::PreallocatedCpu(output)])
                .unwrap();
        assert_eq!(binding.outputs().len(), 1);
        assert!(IoBinding::try_new(
            NamedTensors::new(),
            vec![
                OutputBinding::Allocate { name: "y".into(), device: Device::CPU },
                OutputBinding::Allocate { name: "y".into(), device: Device::CPU },
            ],
        )
        .is_err());
    }
}
