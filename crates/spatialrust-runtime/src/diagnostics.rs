//! Failure diagnostics codes.

/// Machine-readable diagnostic code.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DiagnosticCode(pub String);

/// Structured failure diagnostic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FailureDiagnostic {
    /// Diagnostic code.
    pub code: DiagnosticCode,
    /// Human-readable summary.
    pub summary: String,
    /// Optional remediation hint.
    pub remediation: Option<String>,
}

impl FailureDiagnostic {
    /// Creates a diagnostic.
    #[must_use]
    pub fn new(
        code: impl Into<String>,
        summary: impl Into<String>,
        remediation: Option<String>,
    ) -> Self {
        Self {
            code: DiagnosticCode(code.into()),
            summary: summary.into(),
            remediation,
        }
    }
}
