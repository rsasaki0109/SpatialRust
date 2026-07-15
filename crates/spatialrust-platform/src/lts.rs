//! Long-term support policy and calendar helpers.

/// Support window for one major line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SupportWindow {
    /// Major version tag, e.g. `1.x`.
    pub major_line: String,
    /// Months of active support.
    pub active_months: u32,
    /// Months of security-only support after active ends.
    pub security_months: u32,
}

impl SupportWindow {
    /// Total supported months (active + security-only).
    #[must_use]
    pub fn total_months(&self) -> u32 {
        self.active_months.saturating_add(self.security_months)
    }
}

/// Long-term support policy.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LtsPolicy {
    windows: Vec<SupportWindow>,
}

impl LtsPolicy {
    /// Creates an empty policy.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a support window.
    pub fn declare(&mut self, window: SupportWindow) {
        self.windows.push(window);
    }

    /// Returns declared windows.
    #[must_use]
    pub fn windows(&self) -> &[SupportWindow] {
        &self.windows
    }

    /// Looks up a major line window.
    #[must_use]
    pub fn window_for(&self, major_line: &str) -> Option<&SupportWindow> {
        self.windows
            .iter()
            .find(|window| window.major_line == major_line)
    }

    /// Default SpatialRust 1.x policy used by Epic 100.
    #[must_use]
    pub fn spatialrust_v1() -> Self {
        let mut policy = Self::new();
        policy.declare(SupportWindow {
            major_line: "1.x".into(),
            active_months: 18,
            security_months: 6,
        });
        policy
    }
}

#[cfg(test)]
mod tests {
    use super::LtsPolicy;

    #[test]
    fn v1_policy_declares_window() {
        let policy = LtsPolicy::spatialrust_v1();
        assert_eq!(policy.windows().len(), 1);
        assert_eq!(policy.window_for("1.x").unwrap().total_months(), 24);
    }
}
