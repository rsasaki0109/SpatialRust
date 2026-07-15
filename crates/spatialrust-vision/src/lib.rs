//! AI-ready CPU image processing and vision algorithms.
//!
//! Algorithms are feature-gated by area. GPU implementations belong in an
//! explicit backend and must not introduce hidden device transfers.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod border;
mod error;
mod pixel;

#[cfg(feature = "imgproc-filter")]
mod advanced_filter;
#[cfg(feature = "imgproc-analysis")]
mod analysis;
#[cfg(feature = "imgproc-canny")]
mod canny;
#[cfg(feature = "feature2d")]
mod corners;

#[cfg(feature = "dense")]
mod dense;
#[cfg(feature = "detection")]
mod detection;
#[cfg(feature = "feature2d")]
mod feature2d;
#[cfg(feature = "imgproc-filter")]
mod filter;
#[cfg(feature = "geometry")]
mod geometry;
#[cfg(feature = "feature2d")]
mod matcher;
#[cfg(feature = "imgproc-morphology")]
mod morphology;
#[cfg(feature = "geometry")]
mod multiview;
#[cfg(feature = "geometry")]
mod optical_flow;
#[cfg(feature = "geometry")]
mod pnp;
#[cfg(feature = "ai-adapters")]
mod adapters;
#[cfg(feature = "feature2d")]
mod orb;
#[cfg(feature = "preprocess")]
mod preprocess;
#[cfg(feature = "resize")]
mod resize;
#[cfg(feature = "spatial")]
mod spatial;
#[cfg(feature = "geometry")]
mod stereo;
#[cfg(feature = "warp")]
mod warp;
#[cfg(feature = "video")]
mod video;

pub use border::BorderMode;
pub use error::{VisionError, VisionResult};
pub use pixel::PixelComponent;

#[cfg(feature = "imgproc-filter")]
pub use advanced_filter::*;
#[cfg(feature = "imgproc-analysis")]
pub use analysis::*;
#[cfg(feature = "imgproc-canny")]
pub use canny::*;
#[cfg(feature = "feature2d")]
pub use corners::*;

#[cfg(feature = "dense")]
pub use dense::*;
#[cfg(feature = "detection")]
pub use detection::*;
#[cfg(feature = "feature2d")]
pub use feature2d::*;
#[cfg(feature = "imgproc-filter")]
pub use filter::*;
#[cfg(feature = "geometry")]
pub use geometry::*;
#[cfg(feature = "feature2d")]
pub use matcher::*;
#[cfg(feature = "imgproc-morphology")]
pub use morphology::*;
#[cfg(feature = "geometry")]
pub use multiview::*;
#[cfg(feature = "geometry")]
pub use optical_flow::*;
#[cfg(feature = "geometry")]
pub use pnp::*;
#[cfg(feature = "feature2d")]
pub use orb::*;
#[cfg(feature = "ai-adapters")]
pub use adapters::*;
#[cfg(feature = "preprocess")]
pub use preprocess::*;
#[cfg(feature = "resize")]
pub use resize::*;
#[cfg(feature = "spatial")]
pub use spatial::*;
#[cfg(feature = "geometry")]
pub use stereo::*;
#[cfg(feature = "warp")]
pub use warp::*;
#[cfg(feature = "video")]
pub use video::*;
