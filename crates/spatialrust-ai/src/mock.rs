//! Deterministic in-process mock inference backends.

use spatialrust_tensor::{DataType, Device, TensorBuffer, TensorDescriptor};

use crate::{
    AiError, AiResult, CopyPolicy, Dimension, InferenceBackend, ModelInfo, ModelSession,
    ModelSource, NamedTensors, RunOptions, SessionOptions, TensorSpec,
};

/// Built-in mock model profiles selected through [`ModelSource::Mock`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MockProfile {
    /// Emits metric depth `[1,1,H,W]` from RGB NCHW `[1,3,H,W]` using BT.601 luma.
    ///
    /// Depth meters are `1.0 + 0.5 * (1.0 - luminance)` so brighter pixels map
    /// nearer, giving a deterministic textured plane for image → AI → XYZ demos.
    SyntheticDepth,
    /// Emits deterministic class IDs `[1,N]` and confidences `[1,N]` from
    /// normalized point features `[1,4,N]`.
    ///
    /// The four channels are `x`, `y`, `z`, and horizontal radial distance.
    /// This profile is a visual/demo fixture, not a learned ontology or a
    /// substitute for a production model receipt.
    SemanticClasses,
}

/// Backend that never loads ONNX bytes and always uses [`MockProfile`].
#[derive(Clone, Debug, Default)]
pub struct MockInferenceBackend;

impl InferenceBackend for MockInferenceBackend {
    fn name(&self) -> &str {
        "mock"
    }

    fn create_session(
        &self,
        source: &ModelSource,
        options: &SessionOptions,
    ) -> AiResult<Box<dyn ModelSession>> {
        options.validate()?;
        let profile = match source {
            ModelSource::Mock(profile) => *profile,
            ModelSource::Path(_) | ModelSource::Bytes(_) => {
                return Err(AiError::Unsupported {
                    backend: "mock".into(),
                    operation: "filesystem or byte-backed model loading".into(),
                });
            }
        };
        Ok(Box::new(MockSession { profile, info: profile.model_info() }))
    }
}

struct MockSession {
    profile: MockProfile,
    info: ModelInfo,
}

impl ModelSession for MockSession {
    fn backend_name(&self) -> &str {
        "mock"
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
        // Output is always a newly allocated host `TensorBuffer`.
        if options.output_copy == CopyPolicy::Forbid {
            let name = match self.profile {
                MockProfile::SyntheticDepth => "depth",
                MockProfile::SemanticClasses => "class_ids",
            };
            return Err(AiError::CopyRequired { direction: "output host", name: name.into() });
        }
        match self.profile {
            MockProfile::SyntheticDepth => run_synthetic_depth(inputs, options.input_copy),
            MockProfile::SemanticClasses => run_semantic_classes(inputs, options.input_copy),
        }
    }
}

impl MockProfile {
    fn model_info(self) -> ModelInfo {
        match self {
            Self::SyntheticDepth => ModelInfo {
                name: Some("mock-synthetic-depth".into()),
                inputs: vec![TensorSpec::new(
                    "images",
                    DataType::F32,
                    vec![
                        Dimension::Fixed(1),
                        Dimension::Fixed(3),
                        Dimension::Dynamic,
                        Dimension::Dynamic,
                    ],
                )],
                outputs: vec![TensorSpec::new(
                    "depth",
                    DataType::F32,
                    vec![
                        Dimension::Fixed(1),
                        Dimension::Fixed(1),
                        Dimension::Dynamic,
                        Dimension::Dynamic,
                    ],
                )],
            },
            Self::SemanticClasses => ModelInfo {
                name: Some("mock-semantic-classes".into()),
                inputs: vec![TensorSpec::new(
                    "features",
                    DataType::F32,
                    vec![Dimension::Fixed(1), Dimension::Fixed(4), Dimension::Dynamic],
                )],
                outputs: vec![
                    TensorSpec::new(
                        "class_ids",
                        DataType::U32,
                        vec![Dimension::Fixed(1), Dimension::Dynamic],
                    ),
                    TensorSpec::new(
                        "confidence",
                        DataType::F32,
                        vec![Dimension::Fixed(1), Dimension::Dynamic],
                    ),
                ],
            },
        }
    }
}

fn run_synthetic_depth(inputs: NamedTensors, input_copy: CopyPolicy) -> AiResult<NamedTensors> {
    let input = inputs.get("images").ok_or_else(|| AiError::MissingInput("images".into()))?;
    let descriptor = input.descriptor();
    let shape = descriptor.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
        return Err(AiError::ShapeMismatch {
            name: "images".into(),
            expected: vec![
                Dimension::Fixed(1),
                Dimension::Fixed(3),
                Dimension::Dynamic,
                Dimension::Dynamic,
            ],
            actual: shape.to_vec(),
        });
    }
    if !descriptor.is_c_contiguous() || descriptor.byte_offset() != 0 {
        if input_copy == CopyPolicy::Forbid {
            return Err(AiError::CopyRequired { direction: "input host", name: "images".into() });
        }
        return Err(AiError::Unsupported {
            backend: "mock".into(),
            operation: "non-contiguous input packing for SyntheticDepth".into(),
        });
    }
    let height = shape[2];
    let width = shape[3];
    let values = f32_values(input)?;
    let plane = height.saturating_mul(width);
    if values.len() != plane.saturating_mul(3) {
        return Err(AiError::InvalidConfiguration(
            "images storage length does not match NCHW shape".into(),
        ));
    }
    let mut depth = Vec::with_capacity(plane);
    for index in 0..plane {
        let r = values[index];
        let g = values[plane + index];
        let b = values[2 * plane + index];
        let luma = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0);
        depth.push(1.0 + 0.5 * (1.0 - luma));
    }
    let output = TensorBuffer::try_from_f32(
        depth,
        TensorDescriptor::contiguous(DataType::F32, vec![1, 1, height, width], Device::CPU),
    )
    .map_err(|error| AiError::InvalidConfiguration(error.to_string()))?;
    let mut outputs = NamedTensors::new();
    outputs.insert("depth", output)?;
    Ok(outputs)
}

fn run_semantic_classes(inputs: NamedTensors, input_copy: CopyPolicy) -> AiResult<NamedTensors> {
    let input = inputs.get("features").ok_or_else(|| AiError::MissingInput("features".into()))?;
    let descriptor = input.descriptor();
    let shape = descriptor.shape();
    if shape.len() != 3 || shape[0] != 1 || shape[1] != 4 {
        return Err(AiError::ShapeMismatch {
            name: "features".into(),
            expected: vec![Dimension::Fixed(1), Dimension::Fixed(4), Dimension::Dynamic],
            actual: shape.to_vec(),
        });
    }
    if !descriptor.is_c_contiguous() || descriptor.byte_offset() != 0 {
        if input_copy == CopyPolicy::Forbid {
            return Err(AiError::CopyRequired { direction: "input host", name: "features".into() });
        }
        return Err(AiError::Unsupported {
            backend: "mock".into(),
            operation: "non-contiguous input packing for SemanticClasses".into(),
        });
    }
    let count = shape[2];
    let values = f32_values(input)?;
    let plane = count;
    if values.len() != plane.saturating_mul(4) {
        return Err(AiError::InvalidConfiguration(
            "features storage length does not match [1,4,N] shape".into(),
        ));
    }
    let mut class_ids = Vec::with_capacity(count);
    let mut confidence = Vec::with_capacity(count);
    for index in 0..count {
        let x = values[index];
        let y = values[plane + index];
        let z = values[2 * plane + index];
        let radial = values[3 * plane + index];
        if [x, y, z, radial].iter().any(|value| !value.is_finite()) {
            return Err(AiError::InvalidConfiguration(
                "SemanticClasses features must be finite".into(),
            ));
        }
        let class_id = if z < 0.28 {
            0
        } else if radial > 0.55 {
            1
        } else {
            2
        };
        let margin = match class_id {
            0 => (0.28 - z).abs(),
            1 => (radial - 0.55).abs(),
            _ => (z - 0.28).abs().min((radial - 0.55).abs()),
        };
        class_ids.push(class_id);
        confidence.push((0.60 + 0.40 * (margin * 2.0).clamp(0.0, 1.0)).clamp(0.0, 1.0));
    }
    let class_output = TensorBuffer::try_from_u32(
        class_ids,
        TensorDescriptor::contiguous(DataType::U32, vec![1, count], Device::CPU),
    )
    .map_err(|error| AiError::InvalidConfiguration(error.to_string()))?;
    let confidence_output = TensorBuffer::try_from_f32(
        confidence,
        TensorDescriptor::contiguous(DataType::F32, vec![1, count], Device::CPU),
    )
    .map_err(|error| AiError::InvalidConfiguration(error.to_string()))?;
    let mut outputs = NamedTensors::new();
    outputs.insert("class_ids", class_output)?;
    outputs.insert("confidence", confidence_output)?;
    Ok(outputs)
}

fn f32_values(tensor: &TensorBuffer) -> AiResult<Vec<f32>> {
    if tensor.descriptor().dtype() != DataType::F32 {
        return Err(AiError::DataTypeMismatch {
            name: "images".into(),
            expected: DataType::F32,
            actual: tensor.descriptor().dtype(),
        });
    }
    if let Some(values) = tensor.shared_f32() {
        return Ok(values.to_vec());
    }
    let bytes = tensor.allocation_bytes();
    if bytes.len() % 4 != 0 {
        return Err(AiError::InvalidConfiguration(
            "f32 tensor allocation is not a multiple of 4 bytes".into(),
        ));
    }
    Ok(bytemuck::cast_slice(bytes).to_vec())
}

#[cfg(test)]
mod tests {
    use super::{MockInferenceBackend, MockProfile};
    use crate::{
        AiError, CopyPolicy, InferenceBackend, ModelSource, NamedTensors, RunOptions,
        SessionOptions,
    };
    use spatialrust_tensor::{DataType, Device, TensorBuffer, TensorDescriptor};

    #[test]
    fn synthetic_depth_emits_metric_plane() {
        let backend = MockInferenceBackend;
        let mut session = backend
            .create_session(
                &ModelSource::Mock(MockProfile::SyntheticDepth),
                &SessionOptions::default(),
            )
            .unwrap();
        // planar RRRRGGGG BBBB for 2x2
        let values = vec![
            0.0, 1.0, 0.5, 0.25, // R
            0.0, 1.0, 0.5, 0.25, // G
            0.0, 1.0, 0.5, 0.25, // B
        ];
        let input = TensorBuffer::try_from_f32(
            values,
            TensorDescriptor::contiguous(DataType::F32, vec![1, 3, 2, 2], Device::CPU),
        )
        .unwrap();
        let mut inputs = NamedTensors::new();
        inputs.insert("images", input).unwrap();
        let outputs = session
            .run_with_options(
                inputs,
                RunOptions { input_copy: CopyPolicy::Forbid, output_copy: CopyPolicy::Allow },
            )
            .unwrap();
        let depth = outputs.get("depth").unwrap();
        assert_eq!(depth.descriptor().shape(), &[1, 1, 2, 2]);
        assert_eq!(session.model_info().name.as_deref(), Some("mock-synthetic-depth"));
        let depth_values = depth.shared_f32().unwrap();
        let near = depth_values[1]; // bright pixel -> nearer
        let far = depth_values[0];
        assert!(near < far);
    }

    #[test]
    fn synthetic_depth_requires_output_copy_permission() {
        let backend = MockInferenceBackend;
        let mut session = backend
            .create_session(
                &ModelSource::Mock(MockProfile::SyntheticDepth),
                &SessionOptions::default(),
            )
            .unwrap();
        let input = TensorBuffer::try_from_f32(
            vec![0.0; 12],
            TensorDescriptor::contiguous(DataType::F32, vec![1, 3, 2, 2], Device::CPU),
        )
        .unwrap();
        let mut inputs = NamedTensors::new();
        inputs.insert("images", input).unwrap();
        assert!(matches!(
            session.run(inputs),
            Err(AiError::CopyRequired { direction: "output host", .. })
        ));
    }

    #[test]
    fn semantic_classes_are_deterministic_and_explicitly_host_bound() {
        let backend = MockInferenceBackend;
        let mut session = backend
            .create_session(
                &ModelSource::Mock(MockProfile::SemanticClasses),
                &SessionOptions { deterministic: true, ..SessionOptions::default() },
            )
            .unwrap();
        let values = vec![
            0.1, 0.2, 0.3, // x
            0.1, 0.2, 0.3, // y
            0.1, 0.5, 0.8, // z
            0.1, 0.7, 0.2, // radial
        ];
        let input = TensorBuffer::try_from_f32(
            values,
            TensorDescriptor::contiguous(DataType::F32, vec![1, 4, 3], Device::CPU),
        )
        .unwrap();
        let mut inputs = NamedTensors::new();
        inputs.insert("features", input).unwrap();
        let outputs = session
            .run_with_options(
                inputs,
                RunOptions { input_copy: CopyPolicy::Forbid, output_copy: CopyPolicy::Allow },
            )
            .unwrap();
        assert_eq!(outputs.get("class_ids").unwrap().shared_u32().unwrap().as_ref(), &[0, 1, 2]);
        assert_eq!(outputs.get("confidence").unwrap().descriptor().shape(), &[1, 3]);
        assert_eq!(session.model_info().name.as_deref(), Some("mock-semantic-classes"));
    }
}
