//! TSDF volumes, surfaces, triangle meshes, and optional Gaussian scenes.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod marching_cubes;
mod mesh;
mod surfel;
mod tsdf;

#[cfg(feature = "gaussian")]
mod gaussian;

pub use error::{SceneError, SceneResult};
pub use mesh::TriangleMesh;
pub use surfel::{Surfel, SurfelCloud};
pub use tsdf::TsdfVolume;

#[cfg(feature = "gaussian")]
pub use gaussian::{GaussianPrimitive, GaussianScene};
