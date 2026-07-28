use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use spatialrust::math::Vec3;
use spatialrust::viewer::{NativeViewer, NativeViewerOptions, ViewerState, ViewportSize};
use spatialrust::viz::{Camera, PositionColumns3, Projection};
use spatialrust::web::{BrowserInput, WebViewerState};

use crate::to_py_err;

#[pyclass(name = "ViewerState")]
#[derive(Clone)]
pub(crate) struct PyViewerState {
    inner: WebViewerState,
}

#[pymethods]
impl PyViewerState {
    #[new]
    #[pyo3(signature = (width=1280, height=720))]
    fn new(width: u32, height: u32) -> PyResult<Self> {
        let camera = Camera::try_new(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Projection::Perspective { vertical_fov_radians: 1.0, near: 0.1, far: 100.0 },
        )
        .map_err(to_py_err)?;
        let viewer =
            ViewerState::try_new(camera, ViewportSize::try_new(width, height).map_err(to_py_err)?)
                .map_err(to_py_err)?;
        Ok(Self { inner: WebViewerState::try_new(viewer).map_err(to_py_err)? })
    }

    #[staticmethod]
    fn from_json(state_json: &str) -> PyResult<Self> {
        Ok(Self { inner: WebViewerState::from_json(state_json).map_err(to_py_err)? })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(to_py_err)
    }

    fn apply_input_json(&mut self, input_json: &str) -> PyResult<()> {
        let input: BrowserInput = serde_json::from_str(input_json).map_err(to_py_err)?;
        self.inner.apply(input).map_err(to_py_err)
    }

    #[getter]
    fn version(&self) -> u32 {
        self.inner.version
    }

    #[getter]
    fn revision(&self) -> u64 {
        self.inner.revision
    }

    fn native_launch_receipt(&self, title: &str) -> PyResult<String> {
        let options = NativeViewerOptions {
            title: title.to_owned(),
            width: self.inner.viewer.viewport.width,
            height: self.inner.viewer.viewport.height,
        };
        NativeViewer::try_new(self.inner.viewer.clone(), options).map_err(to_py_err)?;
        serde_json::to_string(&serde_json::json!({
            "state_version": self.inner.version,
            "state_revision": self.inner.revision,
            "width": self.inner.viewer.viewport.width,
            "height": self.inner.viewer.viewport.height,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
        }))
        .map_err(to_py_err)
    }

    #[pyo3(signature = (title="SpatialRust Viewer"))]
    fn launch_native(&self, py: Python<'_>, title: &str) -> PyResult<()> {
        let state = self.inner.viewer.clone();
        let options = NativeViewerOptions {
            title: title.to_owned(),
            width: state.viewport.width,
            height: state.viewport.height,
        };
        py.allow_threads(move || NativeViewer::try_new(state, options).and_then(NativeViewer::run))
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "ViewerState(version={}, revision={}, viewport={}x{})",
            self.inner.version,
            self.inner.revision,
            self.inner.viewer.viewport.width,
            self.inner.viewer.viewport.height
        )
    }
}

enum PointStorage {
    Borrowed { x: Py<PyAny>, y: Py<PyAny>, z: Py<PyAny>, pointers: [usize; 3], len: usize },
    Owned { x: Vec<f32>, y: Vec<f32>, z: Vec<f32>, source_bytes: u64 },
}

#[pyclass(name = "ViewerPointSource", unsendable)]
pub(crate) struct PyViewerPointSource {
    storage: PointStorage,
}

#[pymethods]
impl PyViewerPointSource {
    #[staticmethod]
    fn borrow_numpy(
        x: PyReadonlyArray1<'_, f32>,
        y: PyReadonlyArray1<'_, f32>,
        z: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Self> {
        let x_slice = x
            .as_slice()
            .map_err(|_| PyValueError::new_err("x must be a contiguous float32 array"))?;
        let y_slice = y
            .as_slice()
            .map_err(|_| PyValueError::new_err("y must be a contiguous float32 array"))?;
        let z_slice = z
            .as_slice()
            .map_err(|_| PyValueError::new_err("z must be a contiguous float32 array"))?;
        PositionColumns3::try_new(x_slice, y_slice, z_slice).map_err(to_py_err)?;
        let pointers =
            [x_slice.as_ptr() as usize, y_slice.as_ptr() as usize, z_slice.as_ptr() as usize];
        let len = x_slice.len();
        let x = x.as_untyped().clone().into_any().unbind();
        let y = y.as_untyped().clone().into_any().unbind();
        let z = z.as_untyped().clone().into_any().unbind();
        Ok(Self { storage: PointStorage::Borrowed { x, y, z, pointers, len } })
    }

    #[staticmethod]
    fn copy_from_numpy(positions: PyReadonlyArray2<'_, f32>) -> PyResult<Self> {
        let positions = positions.as_array();
        if positions.shape().len() != 2 || positions.shape()[1] != 3 {
            return Err(PyValueError::new_err("positions must have shape (N, 3)"));
        }
        let len = positions.shape()[0];
        let source_bytes = u64::try_from(len)
            .ok()
            .and_then(|count| count.checked_mul(12))
            .ok_or_else(|| PyValueError::new_err("point byte count overflow"))?;
        let mut x = Vec::with_capacity(len);
        let mut y = Vec::with_capacity(len);
        let mut z = Vec::with_capacity(len);
        for row in positions.rows() {
            x.push(row[0]);
            y.push(row[1]);
            z.push(row[2]);
        }
        PositionColumns3::try_new(&x, &y, &z).map_err(to_py_err)?;
        Ok(Self { storage: PointStorage::Owned { x, y, z, source_bytes } })
    }

    #[getter]
    fn ownership(&self) -> &'static str {
        match self.storage {
            PointStorage::Borrowed { .. } => "borrowed_numpy",
            PointStorage::Owned { .. } => "owned_rust",
        }
    }

    fn __len__(&self) -> usize {
        match &self.storage {
            PointStorage::Borrowed { len, .. } => *len,
            PointStorage::Owned { x, .. } => x.len(),
        }
    }

    #[getter]
    fn source_pointers(&self) -> (usize, usize, usize) {
        let pointers = match &self.storage {
            PointStorage::Borrowed { pointers, .. } => *pointers,
            PointStorage::Owned { x, y, z, .. } => {
                [x.as_ptr() as usize, y.as_ptr() as usize, z.as_ptr() as usize]
            }
        };
        (pointers[0], pointers[1], pointers[2])
    }

    fn transfer_receipt_json(&self) -> PyResult<String> {
        let point_count = self.__len__();
        let source_bytes = u64::try_from(point_count)
            .ok()
            .and_then(|count| count.checked_mul(12))
            .ok_or_else(|| PyValueError::new_err("point byte count overflow"))?;
        let host_to_host_bytes = match &self.storage {
            PointStorage::Borrowed { .. } => 0,
            PointStorage::Owned { source_bytes, .. } => *source_bytes,
        };
        serde_json::to_string(&serde_json::json!({
            "ownership": self.ownership(),
            "point_count": point_count,
            "source_bytes": source_bytes,
            "host_to_host_bytes": host_to_host_bytes,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
        }))
        .map_err(to_py_err)
    }

    fn copy_to_numpy<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray2<f32>>, String)> {
        let mut values = Vec::with_capacity(self.__len__().saturating_mul(3));
        match &self.storage {
            PointStorage::Borrowed { x, y, z, .. } => {
                let x = x.bind(py).downcast::<PyArray1<f32>>()?.readonly();
                let y = y.bind(py).downcast::<PyArray1<f32>>()?.readonly();
                let z = z.bind(py).downcast::<PyArray1<f32>>()?.readonly();
                let x = x.as_slice().map_err(|_| {
                    PyValueError::new_err("retained x array is no longer contiguous")
                })?;
                let y = y.as_slice().map_err(|_| {
                    PyValueError::new_err("retained y array is no longer contiguous")
                })?;
                let z = z.as_slice().map_err(|_| {
                    PyValueError::new_err("retained z array is no longer contiguous")
                })?;
                PositionColumns3::try_new(x, y, z).map_err(to_py_err)?;
                append_interleaved(&mut values, x, y, z);
            }
            PointStorage::Owned { x, y, z, .. } => {
                append_interleaved(&mut values, x, y, z);
            }
        }
        let byte_count = u64::try_from(values.len())
            .ok()
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| PyValueError::new_err("snapshot byte count overflow"))?;
        let array = Array2::from_shape_vec((self.__len__(), 3), values)
            .map_err(to_py_err)?
            .into_pyarray_bound(py);
        let receipt = serde_json::to_string(&serde_json::json!({
            "direction": "rust_to_numpy",
            "copied_bytes": byte_count,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
        }))
        .map_err(to_py_err)?;
        Ok((array, receipt))
    }

    fn __repr__(&self) -> String {
        format!("ViewerPointSource(ownership='{}', len={})", self.ownership(), self.__len__())
    }
}

fn append_interleaved(output: &mut Vec<f32>, x: &[f32], y: &[f32], z: &[f32]) {
    for ((x, y), z) in x.iter().zip(y).zip(z) {
        output.extend_from_slice(&[*x, *y, *z]);
    }
}
