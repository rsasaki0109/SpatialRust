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

    /// Registers an item (replaces an existing path if present).
    pub fn register(&mut self, path: impl Into<String>, class: ApiStabilityClass) {
        let path = path.into();
        if let Some(existing) = self.items.iter_mut().find(|item| item.path == path) {
            existing.class = class;
            return;
        }
        self.items.push(ApiSurfaceItem { path, class });
    }

    /// Looks up one path.
    #[must_use]
    pub fn lookup(&self, path: &str) -> Option<&ApiSurfaceItem> {
        self.items.iter().find(|item| item.path == path)
    }

    /// Returns items.
    #[must_use]
    pub fn items(&self) -> &[ApiSurfaceItem] {
        &self.items
    }

    /// Counts provisional APIs.
    #[must_use]
    pub fn provisional_count(&self) -> usize {
        self.items
            .iter()
            .filter(|item| item.class == ApiStabilityClass::Provisional)
            .count()
    }

    /// Counts experimental APIs.
    #[must_use]
    pub fn experimental_count(&self) -> usize {
        self.items
            .iter()
            .filter(|item| item.class == ApiStabilityClass::Experimental)
            .count()
    }

    /// Seeds the north-star crate surface used by Epic 100 gates.
    #[must_use]
    pub fn north_star_surface() -> Self {
        let mut registry = Self::new();
        let provisional = [
            "spatialrust-records",
            "spatialrust-arrow",
            "spatialrust-sync",
            "spatialrust-mapping",
            "spatialrust-scene",
            "spatialrust-semantic",
            "spatialrust-episode",
            "spatialrust-runtime",
            "spatialrust-interchange",
            "spatialrust-distribute",
            "spatialrust-platform",
        ];
        for path in provisional {
            registry.register(path, ApiStabilityClass::Provisional);
        }
        registry.register("spatialrust-core", ApiStabilityClass::Stable);
        registry.register("spatialrust-math", ApiStabilityClass::Stable);
        registry.register("spatialrust-platform::LtsPolicy", ApiStabilityClass::Stable);
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::{ApiStabilityClass, StabilityRegistry};

    #[test]
    fn north_star_surface_has_core_stable() {
        let registry = StabilityRegistry::north_star_surface();
        assert_eq!(
            registry.lookup("spatialrust-core").unwrap().class,
            ApiStabilityClass::Stable
        );
        assert!(registry.provisional_count() >= 10);
        assert_eq!(registry.experimental_count(), 0);
    }
}
