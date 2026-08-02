//! Bounded execution pipelines with explicit capacity.

use crate::{RuntimeError, RuntimeResult, TraceEvent, TraceLevel, TraceLog};

/// Named pipeline stage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineStage(pub String);

impl PipelineStage {
    /// Creates a stage name.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

/// Pipeline capacity / timeout configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PipelineConfig {
    /// Maximum in-flight messages.
    pub max_inflight: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self { max_inflight: 8 }
    }
}

/// Bounded FIFO pipeline with tracing.
#[derive(Clone, Debug)]
pub struct BoundedPipeline<T> {
    config: PipelineConfig,
    queue: Vec<(PipelineStage, T)>,
    trace: TraceLog,
}

impl<T> BoundedPipeline<T> {
    /// Creates an empty pipeline.
    #[must_use]
    pub fn new(config: PipelineConfig) -> Self {
        Self { config, queue: Vec::new(), trace: TraceLog::new() }
    }

    /// Enqueues a message or rejects when at capacity.
    pub fn push(&mut self, stage: PipelineStage, value: T) -> RuntimeResult<()> {
        if self.queue.len() >= self.config.max_inflight {
            self.trace.push(TraceEvent {
                level: TraceLevel::Error,
                stage: stage.0.clone(),
                message: "capacity exceeded".into(),
            });
            return Err(RuntimeError::CapacityExceeded(stage.0));
        }
        self.trace.push(TraceEvent {
            level: TraceLevel::Info,
            stage: stage.0.clone(),
            message: "enqueued".into(),
        });
        self.queue.push((stage, value));
        Ok(())
    }

    /// Pops the next message.
    pub fn pop(&mut self) -> Option<(PipelineStage, T)> {
        if self.queue.is_empty() {
            None
        } else {
            Some(self.queue.remove(0))
        }
    }

    /// Returns the trace log.
    #[must_use]
    pub fn trace(&self) -> &TraceLog {
        &self.trace
    }

    /// Returns in-flight count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns whether the pipeline is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{BoundedPipeline, PipelineConfig, PipelineStage};

    #[test]
    fn rejects_when_full() {
        let mut pipe = BoundedPipeline::new(PipelineConfig { max_inflight: 1 });
        pipe.push(PipelineStage::new("a"), 1).unwrap();
        assert!(pipe.push(PipelineStage::new("b"), 2).is_err());
    }
}
