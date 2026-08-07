//! Real ONNX model embedding through `OnnxEntityEmbedder` (Epic 150B).
//!
//! The committed `double_dynamic.onnx` fixture maps `input` [1,3] → `output`
//! [1,3] by adding the input to itself (doubling). This proves the full
//! path: feature vector → ONNX session → embedding, without external weights.

#![cfg(all(feature = "ai-onnxruntime", feature = "semantic-model"))]

use std::sync::Arc;

use spatialrust::ai::{
    CopyPolicy, InferenceBackend, ModelSource, OnnxRuntimeBackend, SessionOptions,
};
use spatialrust::semantic::OnnxEntityEmbedder;
use spatialrust::tensor::{DataType, Device, TensorDescriptor};

#[test]
fn onnx_embedder_runs_real_model_and_round_trips_embedding() {
    let model_bytes: &[u8] = include_bytes!("fixtures/double_dynamic.onnx");
    let backend = OnnxRuntimeBackend;
    let mut session = backend
        .create_session(&ModelSource::Bytes(Arc::from(model_bytes)), &SessionOptions::default())
        .expect("open ONNX fixture");

    let descriptor =
        |shape: &[usize]| TensorDescriptor::contiguous(DataType::F32, shape.to_vec(), Device::CPU);
    let embedder = OnnxEntityEmbedder::try_new(
        "input",
        "output",
        descriptor(&[1, 3]),
        descriptor(&[1, 3]),
        CopyPolicy::Allow,
    )
    .expect("embedder");

    // The fixture doubles its input, so [1,2,3] → [2,4,6].
    let embedding = embedder.embed_one(session.as_mut(), &[1.0, 2.0, 3.0]).expect("embed");
    assert_eq!(embedding.dim(), 3);
    assert_eq!(embedding.as_slice(), &[2.0, 4.0, 6.0]);

    // Search integration: embedding feeds the semantic search index.
    let mut index = spatialrust::semantic::SemanticSearchIndex::new();
    index.insert(spatialrust::semantic::SemanticEntity {
        id: spatialrust::semantic::EntityId::new("entity-a"),
        centroid: None,
        labels: vec![spatialrust::semantic::OpenVocabLabel {
            text: "doubler".into(),
            confidence: 1.0,
        }],
        embedding: Some(embedding),
    });

    // Query with an embedding close to [2,4,6] returns the indexed entity.
    let query = spatialrust::semantic::Embedding::try_new(vec![2.0, 4.0, 6.0]).unwrap();
    let results =
        index.search(&query, spatialrust::semantic::MultimodalFusion::default(), 1).unwrap();
    assert_eq!(results[0].0, spatialrust::semantic::EntityId::new("entity-a"));
}

#[test]
fn onnx_embedder_rejects_feature_shape_mismatch() {
    let model_bytes: &[u8] = include_bytes!("fixtures/double_dynamic.onnx");
    let backend = OnnxRuntimeBackend;
    let mut session = backend
        .create_session(&ModelSource::Bytes(Arc::from(model_bytes)), &SessionOptions::default())
        .expect("open ONNX fixture");

    let descriptor =
        |shape: &[usize]| TensorDescriptor::contiguous(DataType::F32, shape.to_vec(), Device::CPU);
    let embedder = OnnxEntityEmbedder::try_new(
        "input",
        "output",
        descriptor(&[1, 3]),
        descriptor(&[1, 3]),
        CopyPolicy::Allow,
    )
    .expect("embedder");

    // Only two features supplied but the model expects three.
    assert!(embedder.embed_one(session.as_mut(), &[1.0, 2.0]).is_err());
}
