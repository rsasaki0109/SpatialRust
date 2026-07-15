//! Security audit checklist items.

/// One security audit checklist item.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SecurityAuditItem {
    /// Item id.
    pub id: String,
    /// Description.
    pub description: String,
    /// Whether the item is currently satisfied.
    pub satisfied: bool,
}

/// Security checklist for release gates.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SecurityChecklist {
    items: Vec<SecurityAuditItem>,
}

impl SecurityChecklist {
    /// Creates an empty checklist.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an item.
    pub fn push(&mut self, id: impl Into<String>, description: impl Into<String>, satisfied: bool) {
        self.items.push(SecurityAuditItem {
            id: id.into(),
            description: description.into(),
            satisfied,
        });
    }

    /// Marks an existing item satisfied by id.
    pub fn mark_satisfied(&mut self, id: &str) -> bool {
        if let Some(item) = self.items.iter_mut().find(|item| item.id == id) {
            item.satisfied = true;
            true
        } else {
            false
        }
    }

    /// Returns whether every item is satisfied.
    #[must_use]
    pub fn all_satisfied(&self) -> bool {
        !self.items.is_empty() && self.items.iter().all(|item| item.satisfied)
    }

    /// Returns unsatisfied item ids.
    #[must_use]
    pub fn unsatisfied_ids(&self) -> Vec<&str> {
        self.items
            .iter()
            .filter(|item| !item.satisfied)
            .map(|item| item.id.as_str())
            .collect()
    }

    /// Returns items.
    #[must_use]
    pub fn items(&self) -> &[SecurityAuditItem] {
        &self.items
    }

    /// Baseline checklist for north-star release gates (initially unsatisfied).
    #[must_use]
    pub fn north_star_baseline() -> Self {
        let mut checklist = Self::new();
        checklist.push(
            "no-silent-device-copies",
            "Production APIs never perform implicit host/device copies",
            false,
        );
        checklist.push(
            "no-secrets-in-fixtures",
            "Repository fixtures contain no private keys or customer sensor dumps",
            false,
        );
        checklist.push(
            "feature-gated-heavy-runtimes",
            "ONNX/ROS2/CUDA/OpenUSD native deps remain opt-in features",
            false,
        );
        checklist.push(
            "deny-unsafe-public-surface",
            "Public crates keep #![deny(unsafe_code)] outside audited FFI/GPU boundaries",
            false,
        );
        checklist
    }

    /// Satisfied copy of [`Self::north_star_baseline`] for integration proofs.
    #[must_use]
    pub fn north_star_baseline_satisfied() -> Self {
        let mut checklist = Self::north_star_baseline();
        for item in &mut checklist.items {
            item.satisfied = true;
        }
        checklist
    }
}

#[cfg(test)]
mod tests {
    use super::SecurityChecklist;

    #[test]
    fn baseline_starts_unsatisfied() {
        let checklist = SecurityChecklist::north_star_baseline();
        assert!(!checklist.all_satisfied());
        assert_eq!(checklist.unsatisfied_ids().len(), 4);
        assert!(SecurityChecklist::north_star_baseline_satisfied().all_satisfied());
    }
}
