//! AI-ready CPU image processing and vision algorithms.
//!
//! Algorithms are feature-gated by area. GPU implementations belong in an
//! explicit backend and must not introduce hidden device transfers.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod pixel;

#[cfg(feature = "dense")]
mod dense;
#[cfg(feature = "detection")]
mod detection;
#[cfg(feature = "preprocess")]
mod preprocess;
#[cfg(feature = "resize")]
mod resize;
#[cfg(feature = "spatial")]
mod spatial;
#[cfg(feature = "warp")]
mod warp;

pub use error::{VisionError, VisionResult};
pub use pixel::PixelComponent;

#[cfg(feature = "dense")]
pub use dense::*;
#[cfg(feature = "detection")]
pub use detection::*;
#[cfg(feature = "preprocess")]
pub use preprocess::*;
#[cfg(feature = "resize")]
pub use resize::*;
#[cfg(feature = "spatial")]
pub use spatial::*;
#[cfg(feature = "warp")]
pub use warp::*;
