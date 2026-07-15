//! TSDF volumes, surfaces, triangle meshes, and optional Gaussian scenes.
//!
//! Enable `gaussian` for anisotropic primitives and a CPU soft-splat renderer.

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
pub use gaussian::{
    render_gaussians_cpu, GaussianCamera, GaussianFramebuffer, GaussianPrimitive, GaussianScene,
};
