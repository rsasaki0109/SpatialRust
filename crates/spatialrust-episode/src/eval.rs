//! Lightweight evaluation metrics for episodes.

/// Named scalar evaluation metric.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalMetric {
    /// Metric name.
    pub name: String,
    /// Metric value.
    pub value: f64,
}

/// Bundle of evaluation metrics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EvalReport {
    /// Metrics.
    pub metrics: Vec<EvalMetric>,
}

impl EvalReport {
    /// Creates an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pushes a metric.
    pub fn push(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.push(EvalMetric { name: name.into(), value });
    }
}
