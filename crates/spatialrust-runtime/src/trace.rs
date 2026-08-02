//! Lightweight structured tracing for robotics pipelines.

/// Trace severity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TraceLevel {
    /// Informational.
    Info,
    /// Warning.
    Warn,
    /// Error.
    Error,
}

/// One pipeline trace event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceEvent {
    /// Severity.
    pub level: TraceLevel,
    /// Stage name.
    pub stage: String,
    /// Message.
    pub message: String,
}

/// Append-only trace log.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TraceLog {
    events: Vec<TraceEvent>,
}

impl TraceLog {
    /// Creates an empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pushes an event.
    pub fn push(&mut self, event: TraceEvent) {
        self.events.push(event);
    }

    /// Returns events.
    #[must_use]
    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }
}
