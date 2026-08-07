//! Real-model entity embedding through an explicit model session.

use spatialrust_ai::{CopyPolicy, ModelSession, NamedTensors, RunOptions};
use spatialrust_tensor::{Device, TensorBuffer, TensorDescriptor};

use crate::{Embedding, SemanticError, SemanticResult};

/// Runs entity features through an already-open model session to produce an
/// [`Embedding`].
///
/// The embedder never loads a model, chooses a backend, or moves data across a
/// device boundary: the caller supplies an open session, names the input and
/// output tensors, and selects the copy policy. This keeps backend identity and
/// transfer semantics explicit and auditable.
#[derive(Clone, Debug)]
pub struct OnnxEntityEmbedder {
    input_name: String,
    output_name: String,
    /// Input tensor descriptor (feature shape).
    input_descriptor: TensorDescriptor,
    /// Output tensor descriptor (embedding shape).
    output_descriptor: TensorDescriptor,
    copy_policy: CopyPolicy,
}

impl OnnxEntityEmbedder {
    /// Creates an embedder for one model input/output pair.
    pub fn try_new(
        input_name: impl Into<String>,
        output_name: impl Into<String>,
        input_descriptor: TensorDescriptor,
        output_descriptor: TensorDescriptor,
        copy_policy: CopyPolicy,
    ) -> SemanticResult<Self> {
        if input_descriptor.device() != Device::CPU || output_descriptor.device() != Device::CPU {
            return Err(SemanticError::InvalidConfiguration(
                "entity embedder requires CPU-hosted input and output tensors".into(),
            ));
        }
        if output_descriptor.shape().is_empty() || output_descriptor.shape().last() == Some(&0) {
            return Err(SemanticError::InvalidConfiguration(
                "embedding output must have a non-empty trailing dimension".into(),
            ));
        }
        Ok(Self {
            input_name: input_name.into(),
            output_name: output_name.into(),
            input_descriptor,
            output_descriptor,
            copy_policy,
        })
    }

    /// Embeds one feature vector (flattened `f32` values) into an embedding.
    ///
    /// `features` must contain exactly the element count implied by
    /// `input_descriptor`. The session's `output_name` tensor must match
    /// `output_descriptor`.
    pub fn embed_one(
        &self,
        session: &mut dyn ModelSession,
        features: &[f32],
    ) -> SemanticResult<Embedding> {
        let expected = self.input_descriptor.element_count().map_err(|error| {
            SemanticError::InvalidConfiguration(format!("input descriptor: {error}"))
        })?;
        if features.len() != expected {
            return Err(SemanticError::InvalidConfiguration(format!(
                "entity features have {} elements; expected {}",
                features.len(),
                expected
            )));
        }
        let bytes = features_to_bytes(features)?;
        let tensor = TensorBuffer::try_new(bytes, self.input_descriptor.clone())
            .map_err(|error| SemanticError::InvalidConfiguration(error.to_string()))?;
        let mut inputs = NamedTensors::new();
        inputs
            .insert(self.input_name.clone(), tensor)
            .map_err(|error| SemanticError::InvalidConfiguration(error.to_string()))?;

        let options = RunOptions { input_copy: self.copy_policy, output_copy: self.copy_policy };
        let outputs = session
            .run_with_options(inputs, options)
            .map_err(|error| SemanticError::InvalidConfiguration(error.to_string()))?;
        let output = outputs.get(&self.output_name).ok_or_else(|| {
            SemanticError::InvalidConfiguration(format!(
                "model output `{}` not found",
                self.output_name
            ))
        })?;

        let expected_output = self.output_descriptor.element_count().map_err(|error| {
            SemanticError::InvalidConfiguration(format!("output descriptor: {error}"))
        })?;
        let output_shape = output.descriptor().shape();
        if output_shape.iter().product::<usize>() != expected_output {
            return Err(SemanticError::InvalidConfiguration(format!(
                "model output `{}` has shape {output_shape:?}; expected {expected_output} elements",
                self.output_name
            )));
        }
        let bytes = output.allocation_bytes();
        if bytes.len() != expected_output * 4 {
            return Err(SemanticError::InvalidConfiguration(format!(
                "model output `{}` has {} bytes; expected {}",
                self.output_name,
                bytes.len(),
                expected_output * 4
            )));
        }
        let mut values = Vec::with_capacity(expected_output);
        for chunk in bytes.chunks_exact(4) {
            values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Embedding::try_new(values)
    }

    /// Returns the configured input descriptor.
    #[must_use]
    pub fn input_descriptor(&self) -> &TensorDescriptor {
        &self.input_descriptor
    }

    /// Returns the configured output descriptor.
    #[must_use]
    pub fn output_descriptor(&self) -> &TensorDescriptor {
        &self.output_descriptor
    }
}

fn features_to_bytes(features: &[f32]) -> SemanticResult<Vec<u8>> {
    let mut bytes = Vec::with_capacity(features.len() * 4);
    for value in features {
        if !value.is_finite() {
            return Err(SemanticError::InvalidConfiguration(
                "entity features must contain finite values".into(),
            ));
        }
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::OnnxEntityEmbedder;
    use crate::Embedding;
    use spatialrust_ai::{CopyPolicy, ModelInfo, ModelSession, NamedTensors, RunOptions};
    use spatialrust_tensor::{DataType, Device, TensorDescriptor};

    #[derive(Clone, Debug)]
    struct IdentitySession {
        info: ModelInfo,
    }

    impl Default for IdentitySession {
        fn default() -> Self {
            Self { info: ModelInfo { name: None, inputs: Vec::new(), outputs: Vec::new() } }
        }
    }

    impl ModelSession for IdentitySession {
        fn backend_name(&self) -> &str {
            "test-identity"
        }

        fn model_info(&self) -> &ModelInfo {
            &self.info
        }

        fn run_with_options(
            &mut self,
            inputs: NamedTensors,
            _options: RunOptions,
        ) -> spatialrust_ai::AiResult<NamedTensors> {
            let mut outputs = NamedTensors::new();
            for (name, tensor) in inputs.into_values() {
                let output_name = if name == "input" { "output".to_owned() } else { name };
                outputs.insert(output_name, tensor)?;
            }
            Ok(outputs)
        }
    }

    fn descriptor(shape: &[usize]) -> TensorDescriptor {
        TensorDescriptor::contiguous(DataType::F32, shape.to_vec(), Device::CPU)
    }

    #[test]
    fn embeds_identity_features() {
        let embedder = OnnxEntityEmbedder::try_new(
            "input",
            "output",
            descriptor(&[1, 4]),
            descriptor(&[1, 4]),
            CopyPolicy::Allow,
        )
        .unwrap();
        let mut session = IdentitySession::default();
        let embedding = embedder.embed_one(&mut session, &[0.1, 0.2, 0.3, 0.4]).unwrap();
        assert_eq!(embedding, Embedding::try_new(vec![0.1, 0.2, 0.3, 0.4]).unwrap());
    }

    #[test]
    fn rejects_feature_count_mismatch() {
        let embedder = OnnxEntityEmbedder::try_new(
            "input",
            "output",
            descriptor(&[1, 4]),
            descriptor(&[1, 4]),
            CopyPolicy::Allow,
        )
        .unwrap();
        let mut session = IdentitySession::default();
        assert!(embedder.embed_one(&mut session, &[0.1, 0.2]).is_err());
    }

    #[test]
    fn rejects_non_finite_features() {
        let embedder = OnnxEntityEmbedder::try_new(
            "input",
            "output",
            descriptor(&[1, 3]),
            descriptor(&[1, 3]),
            CopyPolicy::Allow,
        )
        .unwrap();
        let mut session = IdentitySession::default();
        assert!(embedder.embed_one(&mut session, &[f32::NAN, 0.0, 0.0]).is_err());
    }

    #[test]
    fn rejects_device_mismatch() {
        let gpu = TensorDescriptor::contiguous(
            DataType::F32,
            vec![1, 3],
            Device { kind: spatialrust_tensor::DeviceKind::Cuda, id: 0 },
        );
        assert!(OnnxEntityEmbedder::try_new(
            "input",
            "output",
            gpu,
            descriptor(&[1, 3]),
            CopyPolicy::Allow,
        )
        .is_err());
    }

    #[test]
    fn rejects_zero_dim_output() {
        assert!(OnnxEntityEmbedder::try_new(
            "input",
            "output",
            descriptor(&[1, 3]),
            descriptor(&[0]),
            CopyPolicy::Allow,
        )
        .is_err());
    }
}
