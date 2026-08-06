//! Python bindings for the OGC 3D Tiles 1.1 tileset export surface.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::PyRef;

use spatialrust::interchange::{
    build_point_tileset, export_copc_tileset, write_point_tileset, CopcTilesetOptions,
    TilesetBuilderOptions, TilesetWriteReceipt,
};
use spatialrust::HasPositions3;

/// Exports a point cloud into a 3D Tiles 1.1 tileset directory.
///
/// Args:
///     cloud: spatialrust.PointCloud
///     out_dir: str — directory that receives `tileset.json` plus `.pnts` files
///     max_points_per_tile: int (default 100000) — octree split budget
///     max_depth: int (default 12) — maximum octree depth
///
/// Returns a dict with `tileset_json_bytes`, `tile_count`, `point_count`,
/// and `pnts_bytes`.
#[pyfunction]
#[pyo3(signature = (cloud, out_dir, max_points_per_tile = 100_000, max_depth = 12))]
pub fn export_tiles3d<'py>(
    py: Python<'py>,
    cloud: &Bound<'_, PyAny>,
    out_dir: &str,
    max_points_per_tile: usize,
    max_depth: u32,
) -> PyResult<Bound<'py, PyDict>> {
    let cloud = cloud.extract::<PyRef<'_, crate::PyPointCloud>>()?;
    let (x, y, z) = cloud
        .inner
        .positions3()
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let mut positions = Vec::with_capacity(cloud.inner.len() * 3);
    for index in 0..cloud.inner.len() {
        positions.push(x[index]);
        positions.push(y[index]);
        positions.push(z[index]);
    }
    let built = build_point_tileset(
        &positions,
        None,
        &TilesetBuilderOptions {
            max_points_per_tile,
            max_depth,
            ..Default::default()
        },
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let receipt = write_point_tileset(out_dir, &built)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(receipt_to_dict(py, &receipt))
}

/// Exports a COPC file into a 3D Tiles 1.1 tileset directory without
/// materializing the whole cloud.
///
/// Args:
///     copc_path: str — path to a `.copc.laz` file
///     out_dir: str — directory that receives `tileset.json` plus `.pnts` files
///     max_level: int | None (default None) — maximum octree level to export
///
/// Returns a dict with `tileset_json_bytes`, `tile_count`, `point_count`,
/// and `pnts_bytes`.
#[pyfunction]
#[pyo3(signature = (copc_path, out_dir, max_level = None))]
pub fn export_copc_tiles3d<'py>(
    py: Python<'py>,
    copc_path: &str,
    out_dir: &str,
    max_level: Option<i32>,
) -> PyResult<Bound<'py, PyDict>> {
    let receipt = export_copc_tileset(
        copc_path,
        out_dir,
        &CopcTilesetOptions { max_level, ..Default::default() },
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(receipt_to_dict(py, &receipt))
}

fn receipt_to_dict<'py>(
    py: Python<'py>,
    receipt: &TilesetWriteReceipt,
) -> Bound<'py, PyDict> {
    let dict = PyDict::new_bound(py);
    dict.set_item("tileset_json_bytes", receipt.tileset_json_bytes).unwrap();
    dict.set_item("tile_count", receipt.tile_count).unwrap();
    dict.set_item("point_count", receipt.point_count).unwrap();
    dict.set_item("pnts_bytes", receipt.pnts_bytes).unwrap();
    dict
}
