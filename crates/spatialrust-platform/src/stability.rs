//! API surface stability registry.

/// Stability class for a public API item.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ApiStabilityClass {
    /// Guaranteed within a major version.
    Stable,
    /// May change with notice inside a major version.
    Provisional,
    /// Explicitly experimental.
    Experimental,
}

/// One registered public API surface item.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiSurfaceItem {
    /// Crate or feature path.
    pub path: String,
    /// Stability class.
    pub class: ApiStabilityClass,
}

/// Registry of API surface items used by freeze checklists.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StabilityRegistry {
    items: Vec<ApiSurfaceItem>,
}

impl StabilityRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers an item.
    pub fn register(&mut self, path: impl Into<String>, class: ApiStabilityClass) {
        self.items.push(ApiSurfaceItem { path: path.into(), class });
    }

    /// Returns items.
    #[must_use]
    pub fn items(&self) -> &[ApiSurfaceItem] {
        &self.items
    }

    /// Counts provisional APIs.
    #[must_use]
    pub fn provisional_count(&self) -> usize {
        self.items.iter().filter(|item| item.class == ApiStabilityClass::Provisional).count()
    }
}
