//! ONNX Runtime CPU execution provider adapter.

use std::sync::Arc;

use ort::{
    ep::CPU,
    memory::MemoryInfo,
    session::{builder::GraphOptimizationLevel, Session},
    value::{DynValue, Tensor, TensorElementType, TensorRef, TensorValueType, ValueType},
};
use spatialrust_tensor::{DataType, Device, HostTensorStorage, TensorBuffer, TensorDescriptor};

use crate::{
    AiError, AiResult, CopyPolicy, Dimension, GraphOptimization, InferenceBackend,
    IoBinding as SpatialIoBinding, ModelInfo, ModelSession, ModelSource, NamedTensors,
    OutputBinding, PreallocatedStorage, RunOptions, SessionOptions, TensorSpec,
};

const BACKEND_NAME: &str = "onnxruntime-cpu";

/// ONNX Runtime backend configured explicitly for the CPU execution provider.
#[derive(Clone, Copy, Debug, Default)]
pub struct OnnxRuntimeBackend;

impl InferenceBackend for OnnxRuntimeBackend {
    fn name(&self) -> &str {
        BACKEND_NAME
    }

    fn create_session(
        &self,
        source: &ModelSource,
        options: &SessionOptions,
    ) -> AiResult<Box<dyn ModelSession>> {
        options.validate()?;
        let mut builder = Session::builder().map_err(ort_error)?;
        builder = builder.with_execution_providers([CPU::default().build()]).map_err(ort_error)?;
        if let Some(threads) = options.intra_threads {
            builder = builder.with_intra_threads(threads).map_err(ort_error)?;
        }
        if let Some(threads) = options.inter_threads {
            builder = builder.with_inter_threads(threads).map_err(ort_error)?;
        }
        builder = builder
            .with_optimization_level(match options.graph_optimization {
                GraphOptimization::Disabled => GraphOptimizationLevel::Disable,
                GraphOptimization::Basic => GraphOptimizationLevel::Level1,
                GraphOptimization::Extended => GraphOptimizationLevel::Level2,
                GraphOptimization::All => GraphOptimizationLevel::Level3,
            })
            .map_err(ort_error)?;
        builder = builder.with_deterministic_compute(options.deterministic).map_err(ort_error)?;
        let session = match source {
            ModelSource::Path(path) => builder.commit_from_file(path).map_err(ort_error)?,
            ModelSource::Bytes(bytes) => builder.commit_from_memory(bytes).map_err(ort_error)?,
        };
        let info = model_info(&session)?;
        Ok(Box::new(OnnxRuntimeSession { session, info }))
    }
}

/// Loaded ONNX Runtime CPU session.
#[derive(Debug)]
pub struct OnnxRuntimeSession {
    session: Session,
    info: ModelInfo,
}

impl ModelSession for OnnxRuntimeSession {
    fn backend_name(&self) -> &str {
        BACKEND_NAME
    }

    fn model_info(&self) -> &ModelInfo {
        &self.info
    }

    fn run_with_options(
        &mut self,
        inputs: NamedTensors,
        options: RunOptions,
    ) -> AiResult<NamedTensors> {
        self.info.validate_inputs(&inputs)?;
        if options.input_copy == CopyPolicy::Forbid {
            return Err(AiError::CopyRequired {
                direction: "input host",
                name: self.info.inputs.first().map_or_else(String::new, |spec| spec.name.clone()),
            });
        }
        if options.output_copy == CopyPolicy::Forbid {
            return Err(AiError::CopyRequired {
                direction: "output host",
                name: self.info.outputs.first().map_or_else(String::new, |spec| spec.name.clone()),
            });
        }
        let values = inputs
            .iter()
            .map(|(name, tensor)| Ok((name.to_owned(), to_ort_tensor(name, tensor)?)))
            .collect::<AiResult<Vec<_>>>()?;
        let outputs = self.session.run(values).map_err(ort_error)?;
        let mut named = NamedTensors::new();
        for spec in &self.info.outputs {
            let value = outputs.get(&spec.name).ok_or_else(|| AiError::Backend {
                backend: BACKEND_NAME.into(),
                message: format!("runtime omitted output `{}`", spec.name),
            })?;
            named.insert(spec.name.clone(), copy_ort_output(spec, value)?)?;
        }
        Ok(named)
    }

    fn run_with_binding(&mut self, binding: &mut SpatialIoBinding) -> AiResult<()> {
        binding.clear_results();
        self.info.validate_inputs(binding.inputs())?;
        let mut ort_binding = self.session.create_binding().map_err(ort_error)?;
        for (name, tensor) in binding.inputs().iter() {
            bind_zero_copy_input(&mut ort_binding, name, tensor)?;
        }

        let requested = binding
            .outputs()
            .iter()
            .map(|output| match output {
                OutputBinding::Allocate { name, .. } => name.clone(),
                OutputBinding::PreallocatedCpu(output) => output.name().to_owned(),
            })
            .collect::<Vec<_>>();
        for output in binding.outputs_mut() {
            match output {
                OutputBinding::Allocate { name, device } => {
                    output_spec(&self.info, name)?;
                    if *device != Device::CPU {
                        return Err(AiError::Unsupported {
                            backend: BACKEND_NAME.into(),
                            operation: format!("bound output device {device:?}"),
                        });
                    }
                    ort_binding
                        .bind_output_to_device(name.clone(), &MemoryInfo::default())
                        .map_err(ort_error)?;
                }
                OutputBinding::PreallocatedCpu(output) => {
                    let spec = output_spec(&self.info, output.name())?;
                    spec.validate(output.descriptor())?;
                    if !output.descriptor().is_c_contiguous()
                        || output.descriptor().byte_offset() != 0
                    {
                        return Err(AiError::Unsupported {
                            backend: BACKEND_NAME.into(),
                            operation: format!(
                                "strided preallocated output `{}`; allocate a compact buffer",
                                output.name()
                            ),
                        });
                    }
                    let name = output.name().to_owned();
                    let shape = output.descriptor().shape().to_vec();
                    bind_preallocated_output(
                        &mut ort_binding,
                        &name,
                        shape,
                        output.take_storage()?,
                        spec.dtype,
                    )?;
                }
            }
        }

        let mut outputs = self.session.run_binding(&ort_binding).map_err(ort_error)?;
        let mut results = NamedTensors::new();
        for name in requested {
            let spec = output_spec(&self.info, &name)?;
            let value = outputs.remove(&name).ok_or_else(|| AiError::Backend {
                backend: BACKEND_NAME.into(),
                message: format!("runtime omitted bound output `{name}`"),
            })?;
            results.insert(name, retain_ort_output(spec, value)?)?;
        }
        drop(outputs);
        binding.set_results(results);
        Ok(())
    }
}

fn output_spec<'a>(info: &'a ModelInfo, name: &str) -> AiResult<&'a TensorSpec> {
    info.outputs
        .iter()
        .find(|spec| spec.name == name)
        .ok_or_else(|| AiError::UnexpectedTensor(name.to_owned()))
}

fn bind_zero_copy_input(
    binding: &mut ort::session::IoBinding,
    name: &str,
    tensor: &TensorBuffer,
) -> AiResult<()> {
    let descriptor = tensor.descriptor();
    if !descriptor.is_c_contiguous() || descriptor.byte_offset() != 0 {
        return Err(AiError::Unsupported {
            backend: BACKEND_NAME.into(),
            operation: format!("strided bound input `{name}`; explicitly pack it first"),
        });
    }
    if let Some(storage) = tensor.host_storage() {
        if let Some(storage) = storage.as_any().downcast_ref::<OrtHostStorage>() {
            return storage.bind_input(binding, name);
        }
    }
    let expected = descriptor
        .element_count()
        .map_err(|error| AiError::InvalidConfiguration(error.to_string()))?;
    let shape = descriptor.shape().to_vec();
    macro_rules! bind_arc {
        ($getter:ident, $type:ty) => {{
            let values = tensor.$getter().ok_or_else(|| AiError::CopyRequired {
                direction: "input alignment",
                name: name.to_owned(),
            })?;
            if values.len() != expected {
                return Err(AiError::InvalidConfiguration(format!(
                    "bound input `{name}` allocation has {} elements, expected {expected}",
                    values.len()
                )));
            }
            let value = TensorRef::<$type>::from_array_view((shape, values)).map_err(ort_error)?;
            binding.bind_input(name, &value).map_err(ort_error)
        }};
    }
    match descriptor.dtype() {
        DataType::U8 => bind_arc!(shared_bytes, u8),
        DataType::U16 => bind_arc!(shared_u16, u16),
        DataType::U32 => bind_arc!(shared_u32, u32),
        DataType::I16 => bind_arc!(shared_i16, i16),
        DataType::I32 => bind_arc!(shared_i32, i32),
        DataType::I64 => bind_arc!(shared_i64, i64),
        DataType::F32 => bind_arc!(shared_f32, f32),
        DataType::F64 => bind_arc!(shared_f64, f64),
        dtype => Err(AiError::Unsupported {
            backend: BACKEND_NAME.into(),
            operation: format!("zero-copy bound input dtype {dtype:?}"),
        }),
    }
}

fn bind_preallocated_output(
    binding: &mut ort::session::IoBinding,
    name: &str,
    shape: Vec<usize>,
    storage: PreallocatedStorage,
    dtype: DataType,
) -> AiResult<()> {
    match (storage, dtype) {
        (PreallocatedStorage::Bytes(values), DataType::U8) => binding
            .bind_output(name, Tensor::<u8>::from_array((shape, values)).map_err(ort_error)?)
            .map_err(ort_error),
        (PreallocatedStorage::U16(values), DataType::U16) => binding
            .bind_output(name, Tensor::<u16>::from_array((shape, values)).map_err(ort_error)?)
            .map_err(ort_error),
        (PreallocatedStorage::F32(values), DataType::F32) => binding
            .bind_output(name, Tensor::<f32>::from_array((shape, values)).map_err(ort_error)?)
            .map_err(ort_error),
        _ => Err(AiError::CopyRequired { direction: "output alignment", name: name.to_owned() }),
    }
}

#[derive(Debug)]
enum OrtHostStorage {
    U8(Tensor<u8>),
    U16(Tensor<u16>),
    U32(Tensor<u32>),
    I8(Tensor<i8>),
    I16(Tensor<i16>),
    I32(Tensor<i32>),
    I64(Tensor<i64>),
    F32(Tensor<f32>),
    F64(Tensor<f64>),
}

impl OrtHostStorage {
    fn bind_input(&self, binding: &mut ort::session::IoBinding, name: &str) -> AiResult<()> {
        macro_rules! bind {
            ($value:expr) => {
                binding.bind_input(name, $value).map_err(ort_error)
            };
        }
        match self {
            Self::U8(value) => bind!(value),
            Self::U16(value) => bind!(value),
            Self::U32(value) => bind!(value),
            Self::I8(value) => bind!(value),
            Self::I16(value) => bind!(value),
            Self::I32(value) => bind!(value),
            Self::I64(value) => bind!(value),
            Self::F32(value) => bind!(value),
            Self::F64(value) => bind!(value),
        }
    }
}

impl HostTensorStorage for OrtHostStorage {
    fn dtype(&self) -> DataType {
        match self {
            Self::U8(_) => DataType::U8,
            Self::U16(_) => DataType::U16,
            Self::U32(_) => DataType::U32,
            Self::I8(_) => DataType::I8,
            Self::I16(_) => DataType::I16,
            Self::I32(_) => DataType::I32,
            Self::I64(_) => DataType::I64,
            Self::F32(_) => DataType::F32,
            Self::F64(_) => DataType::F64,
        }
    }

    fn allocation_bytes(&self) -> &[u8] {
        macro_rules! bytes {
            ($value:expr) => {
                bytemuck::cast_slice($value.extract_tensor().1)
            };
        }
        match self {
            Self::U8(value) => value.extract_tensor().1,
            Self::U16(value) => bytes!(value),
            Self::U32(value) => bytes!(value),
            Self::I8(value) => bytes!(value),
            Self::I16(value) => bytes!(value),
            Self::I32(value) => bytes!(value),
            Self::I64(value) => bytes!(value),
            Self::F32(value) => bytes!(value),
            Self::F64(value) => bytes!(value),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn retain_ort_output(spec: &TensorSpec, value: DynValue) -> AiResult<TensorBuffer> {
    macro_rules! retain {
        ($type:ty, $variant:ident) => {{
            let tensor = value.downcast::<TensorValueType<$type>>().map_err(ort_error)?;
            let shape = tensor
                .extract_tensor()
                .0
                .iter()
                .map(|&dimension| usize::try_from(dimension).expect("runtime shape is concrete"))
                .collect::<Vec<_>>();
            let storage: Arc<dyn HostTensorStorage> = Arc::new(OrtHostStorage::$variant(tensor));
            (shape, storage)
        }};
    }
    let (shape, storage) = match spec.dtype {
        DataType::U8 => retain!(u8, U8),
        DataType::U16 => retain!(u16, U16),
        DataType::U32 => retain!(u32, U32),
        DataType::I8 => retain!(i8, I8),
        DataType::I16 => retain!(i16, I16),
        DataType::I32 => retain!(i32, I32),
        DataType::I64 => retain!(i64, I64),
        DataType::F32 => retain!(f32, F32),
        DataType::F64 => retain!(f64, F64),
        dtype => {
            return Err(AiError::Unsupported {
                backend: BACKEND_NAME.into(),
                operation: format!("retaining bound output dtype {dtype:?}"),
            })
        }
    };
    TensorBuffer::try_from_host_storage(
        storage,
        TensorDescriptor::contiguous(spec.dtype, shape, Device::CPU),
    )
    .map_err(|error| AiError::Backend { backend: BACKEND_NAME.into(), message: error.to_string() })
}

fn model_info(session: &Session) -> AiResult<ModelInfo> {
    let inputs = session.inputs().iter().map(outlet_spec).collect::<AiResult<Vec<_>>>()?;
    let outputs = session.outputs().iter().map(outlet_spec).collect::<AiResult<Vec<_>>>()?;
    let info = ModelInfo { name: None, inputs, outputs };
    info.validate()?;
    Ok(info)
}

fn outlet_spec(outlet: &ort::value::Outlet) -> AiResult<TensorSpec> {
    let ValueType::Tensor { ty, shape, dimension_symbols } = outlet.dtype() else {
        return Err(AiError::Unsupported {
            backend: BACKEND_NAME.into(),
            operation: format!("non-tensor ONNX value `{}`", outlet.name()),
        });
    };
    let dtype = decode_ort_dtype(*ty).ok_or_else(|| AiError::Unsupported {
        backend: BACKEND_NAME.into(),
        operation: format!("ONNX dtype {ty} for `{}`", outlet.name()),
    })?;
    let dimensions = shape
        .iter()
        .enumerate()
        .map(|(index, &dimension)| {
            if dimension >= 0 {
                Dimension::Fixed(dimension as usize)
            } else {
                let symbol = &dimension_symbols[index];
                if symbol.is_empty() {
                    Dimension::Dynamic
                } else {
                    Dimension::Symbol(symbol.clone())
                }
            }
        })
        .collect();
    Ok(TensorSpec::new(outlet.name(), dtype, dimensions))
}

fn decode_ort_dtype(dtype: TensorElementType) -> Option<DataType> {
    Some(match dtype {
        TensorElementType::Float32 => DataType::F32,
        TensorElementType::Float64 => DataType::F64,
        TensorElementType::Uint8 => DataType::U8,
        TensorElementType::Uint16 => DataType::U16,
        TensorElementType::Uint32 => DataType::U32,
        TensorElementType::Int8 => DataType::I8,
        TensorElementType::Int16 => DataType::I16,
        TensorElementType::Int32 => DataType::I32,
        TensorElementType::Int64 => DataType::I64,
        TensorElementType::Float16 => DataType::F16,
        TensorElementType::Bfloat16 => DataType::BF16,
        TensorElementType::Bool => DataType::BOOL,
        _ => return None,
    })
}

fn to_ort_tensor(name: &str, tensor: &TensorBuffer) -> AiResult<DynValue> {
    let descriptor = tensor.descriptor();
    if !descriptor.is_c_contiguous() {
        return Err(AiError::Unsupported {
            backend: BACKEND_NAME.into(),
            operation: format!("strided input `{name}`; explicitly pack it first"),
        });
    }
    let range = descriptor.required_byte_range().map_err(|error| AiError::Backend {
        backend: BACKEND_NAME.into(),
        message: error.to_string(),
    })?;
    let bytes = &tensor.allocation_bytes()[range];
    let shape = descriptor.shape().to_vec();
    macro_rules! typed {
        ($type:ty, $width:expr) => {{
            let values = decode_ne::<$type>(bytes, $width, |chunk| {
                <$type>::from_ne_bytes(chunk.try_into().expect("fixed chunk width"))
            });
            Tensor::<$type>::from_array((shape, values)).map(|value| value.into_dyn())
        }};
    }
    let value = match descriptor.dtype() {
        DataType::U8 => Tensor::<u8>::from_array((shape, bytes.to_vec())).map(|v| v.into_dyn()),
        DataType::U16 => typed!(u16, 2),
        DataType::U32 => typed!(u32, 4),
        DataType::I8 => Tensor::<i8>::from_array((
            shape,
            bytes.iter().map(|&value| value as i8).collect::<Vec<_>>(),
        ))
        .map(|v| v.into_dyn()),
        DataType::I16 => typed!(i16, 2),
        DataType::I32 => typed!(i32, 4),
        DataType::I64 => typed!(i64, 8),
        DataType::F32 => typed!(f32, 4),
        DataType::F64 => typed!(f64, 8),
        dtype => {
            return Err(AiError::Unsupported {
                backend: BACKEND_NAME.into(),
                operation: format!("input dtype {dtype:?}"),
            })
        }
    };
    value.map_err(ort_error)
}

fn decode_ne<T>(bytes: &[u8], width: usize, decode: impl Fn(&[u8]) -> T) -> Vec<T> {
    debug_assert_eq!(bytes.len() % width, 0);
    bytes.chunks_exact(width).map(decode).collect()
}

fn copy_ort_output(spec: &TensorSpec, value: &DynValue) -> AiResult<TensorBuffer> {
    macro_rules! extract {
        ($type:ty, $constructor:ident) => {{
            let (shape, values) = value.try_extract_tensor::<$type>().map_err(ort_error)?;
            let descriptor = TensorDescriptor::contiguous(
                spec.dtype,
                shape.iter().map(|&dimension| dimension as usize).collect(),
                Device::CPU,
            );
            TensorBuffer::$constructor(values.to_vec(), descriptor)
        }};
    }
    let result = match spec.dtype {
        DataType::U8 => extract!(u8, try_new),
        DataType::U16 => extract!(u16, try_from_u16),
        DataType::U32 => extract!(u32, try_from_u32),
        DataType::I8 => {
            let (shape, values) = value.try_extract_tensor::<i8>().map_err(ort_error)?;
            let descriptor = TensorDescriptor::contiguous(
                DataType::I8,
                shape.iter().map(|&dimension| dimension as usize).collect(),
                Device::CPU,
            );
            TensorBuffer::try_new(values.iter().map(|&item| item as u8).collect(), descriptor)
        }
        DataType::I16 => extract!(i16, try_from_i16),
        DataType::I32 => extract!(i32, try_from_i32),
        DataType::I64 => extract!(i64, try_from_i64),
        DataType::F32 => extract!(f32, try_from_f32),
        DataType::F64 => extract!(f64, try_from_f64),
        dtype => {
            return Err(AiError::Unsupported {
                backend: BACKEND_NAME.into(),
                operation: format!("output dtype {dtype:?}"),
            })
        }
    };
    result.map_err(|error| AiError::Backend {
        backend: BACKEND_NAME.into(),
        message: error.to_string(),
    })
}

fn ort_error(error: impl std::fmt::Display) -> AiError {
    AiError::Backend { backend: BACKEND_NAME.into(), message: error.to_string() }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::OnnxRuntimeBackend;
    use crate::{
        AiError, CopyPolicy, Dimension, InferenceBackend, IoBinding, ModelSource, NamedTensors,
        OutputBinding, PreallocatedOutput, RunOptions, SessionOptions,
    };
    use spatialrust_tensor::{DataType, Device, TensorBuffer, TensorDescriptor};

    const DOUBLE_DYNAMIC: &[u8] = &[
        8, 8, 18, 16, 115, 112, 97, 116, 105, 97, 108, 114, 117, 115, 116, 45, 116, 101, 115, 116,
        58, 106, 10, 27, 10, 5, 105, 110, 112, 117, 116, 10, 5, 105, 110, 112, 117, 116, 18, 6,
        111, 117, 116, 112, 117, 116, 34, 3, 65, 100, 100, 18, 14, 100, 111, 117, 98, 108, 101, 95,
        100, 121, 110, 97, 109, 105, 99, 90, 28, 10, 5, 105, 110, 112, 117, 116, 18, 19, 10, 17, 8,
        1, 18, 13, 10, 7, 18, 5, 98, 97, 116, 99, 104, 10, 2, 8, 3, 98, 29, 10, 6, 111, 117, 116,
        112, 117, 116, 18, 19, 10, 17, 8, 1, 18, 13, 10, 7, 18, 5, 98, 97, 116, 99, 104, 10, 2, 8,
        3, 66, 4, 10, 0, 16, 13,
    ];

    fn f32_tensor(shape: Vec<usize>, values: &[f32]) -> TensorBuffer {
        TensorBuffer::try_from_f32(
            values.to_vec(),
            TensorDescriptor::contiguous(DataType::F32, shape, Device::CPU),
        )
        .unwrap()
    }

    fn f32_values(tensor: &TensorBuffer) -> Vec<f32> {
        tensor
            .allocation_bytes()
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn dynamic_identity_model_with_onnx_dtype(dtype: u8) -> Arc<[u8]> {
        let mut model = DOUBLE_DYNAMIC.to_vec();
        let graph = model
            .windows(4)
            .position(|window| window == [58, 106, 10, 27])
            .expect("embedded graph and node lengths");
        model[graph + 1] = 104;
        let node = graph + 2;
        let identity_node = [
            10, 25, 10, 5, 105, 110, 112, 117, 116, 18, 6, 111, 117, 116, 112, 117, 116, 34, 8, 73,
            100, 101, 110, 116, 105, 116, 121,
        ];
        model.splice(node..node + 29, identity_node);
        let mut replacements = 0;
        for index in 0..model.len().saturating_sub(3) {
            if model[index..index + 4] == [8, 1, 18, 13] {
                model[index + 1] = dtype;
                replacements += 1;
            }
        }
        assert_eq!(replacements, 2, "input and output tensor types must both be replaced");
        Arc::from(model)
    }

    #[test]
    fn cpu_session_preserves_named_dynamic_contract_and_requires_copy_opt_in() {
        let backend = OnnxRuntimeBackend;
        let mut session = backend
            .create_session(
                &ModelSource::Bytes(Arc::from(DOUBLE_DYNAMIC)),
                &SessionOptions::default(),
            )
            .unwrap();
        assert_eq!(session.model_info().inputs[0].name, "input");
        assert_eq!(
            session.model_info().inputs[0].shape,
            vec![Dimension::Symbol("batch".into()), Dimension::Fixed(3)]
        );

        let mut inputs = NamedTensors::new();
        inputs.insert("input", f32_tensor(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
        assert!(matches!(session.run(inputs.clone()), Err(AiError::CopyRequired { .. })));
        let outputs = session
            .run_with_options(
                inputs,
                RunOptions { input_copy: CopyPolicy::Allow, output_copy: CopyPolicy::Allow },
            )
            .unwrap();
        let output = outputs.get("output").unwrap();
        assert_eq!(output.descriptor().shape(), &[2, 3]);
        assert_eq!(f32_values(output), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn io_binding_allocates_dynamic_output_and_reuses_it_as_zero_copy_input() {
        let backend = OnnxRuntimeBackend;
        let mut session = backend
            .create_session(
                &ModelSource::Bytes(Arc::from(DOUBLE_DYNAMIC)),
                &SessionOptions::default(),
            )
            .unwrap();
        let mut inputs = NamedTensors::new();
        inputs.insert("input", f32_tensor(vec![1, 3], &[1.0, 2.0, 3.0])).unwrap();
        let mut binding = IoBinding::try_new(
            inputs,
            vec![OutputBinding::Allocate { name: "output".into(), device: Device::CPU }],
        )
        .unwrap();
        session.run_with_binding(&mut binding).unwrap();
        let first = binding.results().unwrap().get("output").unwrap();
        assert!(first.host_storage().is_some());
        assert_eq!(f32_values(first), &[2.0, 4.0, 6.0]);

        let mut inputs = NamedTensors::new();
        inputs.insert("input", first.clone()).unwrap();
        let mut chained = IoBinding::try_new(
            inputs,
            vec![OutputBinding::Allocate { name: "output".into(), device: Device::CPU }],
        )
        .unwrap();
        session.run_with_binding(&mut chained).unwrap();
        assert_eq!(
            f32_values(chained.results().unwrap().get("output").unwrap()),
            &[4.0, 8.0, 12.0]
        );
    }

    #[test]
    fn io_binding_writes_directly_into_caller_preallocated_f32_storage() {
        let backend = OnnxRuntimeBackend;
        let mut session = backend
            .create_session(
                &ModelSource::Bytes(Arc::from(DOUBLE_DYNAMIC)),
                &SessionOptions::default(),
            )
            .unwrap();
        let mut inputs = NamedTensors::new();
        inputs.insert("input", f32_tensor(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
        let descriptor = TensorDescriptor::contiguous(DataType::F32, vec![2, 3], Device::CPU);
        let mut output = PreallocatedOutput::allocate("output", descriptor).unwrap();
        let allocation = output.allocation_bytes_mut().unwrap().as_ptr();
        let mut binding =
            IoBinding::try_new(inputs, vec![OutputBinding::PreallocatedCpu(output)]).unwrap();
        session.run_with_binding(&mut binding).unwrap();
        let result = binding.results().unwrap().get("output").unwrap();
        assert_eq!(result.allocation_bytes().as_ptr(), allocation);
        assert_eq!(f32_values(result), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn io_binding_rejects_unaligned_raw_f32_storage_instead_of_copying() {
        let backend = OnnxRuntimeBackend;
        let mut session = backend
            .create_session(
                &ModelSource::Bytes(Arc::from(DOUBLE_DYNAMIC)),
                &SessionOptions::default(),
            )
            .unwrap();
        let descriptor = TensorDescriptor::contiguous(DataType::F32, vec![1, 3], Device::CPU);
        let input = TensorBuffer::try_new(vec![0; 12], descriptor).unwrap();
        let mut inputs = NamedTensors::new();
        inputs.insert("input", input).unwrap();
        let mut binding = IoBinding::try_new(
            inputs,
            vec![OutputBinding::Allocate { name: "output".into(), device: Device::CPU }],
        )
        .unwrap();
        assert!(matches!(
            session.run_with_binding(&mut binding),
            Err(AiError::CopyRequired { direction: "input alignment", .. })
        ));
    }

    #[test]
    fn io_binding_supports_aligned_u8_and_u16_storage() {
        let backend = OnnxRuntimeBackend;

        let mut u8_session = backend
            .create_session(
                &ModelSource::Bytes(dynamic_identity_model_with_onnx_dtype(2)),
                &SessionOptions::default(),
            )
            .unwrap();
        let mut u8_inputs = NamedTensors::new();
        u8_inputs
            .insert(
                "input",
                TensorBuffer::try_new(
                    vec![1, 2, 3],
                    TensorDescriptor::contiguous(DataType::U8, vec![1, 3], Device::CPU),
                )
                .unwrap(),
            )
            .unwrap();
        let mut u8_binding = IoBinding::try_new(
            u8_inputs,
            vec![OutputBinding::Allocate { name: "output".into(), device: Device::CPU }],
        )
        .unwrap();
        u8_session.run_with_binding(&mut u8_binding).unwrap();
        assert_eq!(
            u8_binding.results().unwrap().get("output").unwrap().allocation_bytes(),
            &[1, 2, 3]
        );

        let mut u16_session = backend
            .create_session(
                &ModelSource::Bytes(dynamic_identity_model_with_onnx_dtype(4)),
                &SessionOptions::default(),
            )
            .unwrap();
        let descriptor = TensorDescriptor::contiguous(DataType::U16, vec![1, 3], Device::CPU);
        let mut u16_inputs = NamedTensors::new();
        u16_inputs
            .insert(
                "input",
                TensorBuffer::try_from_u16(vec![10, 20, 30], descriptor.clone()).unwrap(),
            )
            .unwrap();
        let output = PreallocatedOutput::allocate("output", descriptor).unwrap();
        let mut u16_binding =
            IoBinding::try_new(u16_inputs, vec![OutputBinding::PreallocatedCpu(output)]).unwrap();
        u16_session.run_with_binding(&mut u16_binding).unwrap();
        assert_eq!(
            bytemuck::cast_slice::<u8, u16>(
                u16_binding.results().unwrap().get("output").unwrap().allocation_bytes()
            ),
            &[10, 20, 30]
        );
    }
}
