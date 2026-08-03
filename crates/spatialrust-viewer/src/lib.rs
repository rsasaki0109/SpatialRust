//! Viewer state, interaction controls, native window shell, and debug overlays.
//!
//! The default feature set is renderer- and window-system-independent. Native
//! window creation is opt-in through `native`; geometry uploads remain explicit
//! operations in `spatialrust-render-wgpu`.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod adapters;
mod controls;
mod dataset_health;
mod digital_twin;
mod error;
mod map_diff;
#[cfg(feature = "native")]
mod native;
mod observatory;
mod overlay;
mod replay;
mod semantic_overlay;
mod state;
mod studio;
mod timeline;

#[cfg(feature = "camera")]
pub use adapters::camera_frustum_visual;
#[cfg(feature = "scene-gaussian")]
pub use adapters::gaussian_visual;
#[cfg(feature = "scene")]
pub use adapters::{mesh_visual, surfel_visual};
#[cfg(feature = "mapping")]
pub use adapters::{pose_graph_visual, trajectory_visual};
#[cfg(feature = "semantic")]
pub use adapters::{semantic_overlay_visual, semantic_visual, spatial_record_entity_visual};
pub use adapters::{AdaptedGeometry, AdaptedVisual, AdapterReceipt};
pub use controls::{InputAction, ViewerController};
pub use dataset_health::{
    DatasetHealthCheck, DatasetHealthStage, DatasetHealthState, DatasetHealthSummary,
    DatasetHealthTopic, DATASET_HEALTH_STATE_VERSION,
};
pub use digital_twin::{
    DigitalTwinAsset, DigitalTwinState, DigitalTwinSummary, DIGITAL_TWIN_STATE_VERSION,
};
pub use error::{ViewerError, ViewerResult};
pub use map_diff::{
    MapDiffBounds, MapDiffCell, MapDiffMap, MapDiffState, MapDiffSummary, MAP_DIFF_STATE_VERSION,
};
#[cfg(feature = "native")]
pub use native::{NativeViewer, NativeViewerOptions};
pub use observatory::{
    CalibrationArtifact, CalibrationObservatoryState, ClockCalibration, FrameTransform,
    CALIBRATION_OBSERVATORY_STATE_VERSION,
};
pub use overlay::{DebugOverlay, OverlayGeometry, OverlayKind};
pub use replay::{
    ReplayArtifact, ReplayDemoState, ReplaySample, ReplaySummary, ReplayTopic,
    REPLAY_DEMO_STATE_VERSION,
};
pub use semantic_overlay::{
    SemanticOverlayClass, SemanticOverlayEntity, SemanticOverlayModel, SemanticOverlayState,
    SemanticOverlaySummary, SEMANTIC_CONFIDENCE_SCALE, SEMANTIC_OVERLAY_STATE_VERSION,
};
pub use state::{
    AttributeSummary, InspectorSelection, LayerPresentation, ViewerState, ViewportSize,
    VIEWER_STATE_VERSION,
};
pub use studio::{
    StudioCalibration, StudioFrameGraph, StudioLayer, StudioPerformance, StudioSource,
    StudioStageMetric, StudioState, StudioTimeline, STUDIO_STATE_VERSION,
};
pub use timeline::{FrameTimestamps, RgbdFrameView, RgbdPixelSample, RgbdTimeline};
