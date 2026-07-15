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

    /// Returns whether every item is satisfied.
    #[must_use]
    pub fn all_satisfied(&self) -> bool {
        !self.items.is_empty() && self.items.iter().all(|item| item.satisfied)
    }

    /// Returns items.
    #[must_use]
    pub fn items(&self) -> &[SecurityAuditItem] {
        &self.items
    }
}
