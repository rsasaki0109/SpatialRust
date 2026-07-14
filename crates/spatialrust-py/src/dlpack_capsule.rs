//! Audited CPython capsule boundary for DLPack ownership transfer.

use pyo3::{
    exceptions::PyBufferError,
    prelude::*,
    types::{PyAny, PyDict},
};
use spatialrust::tensor::{release_dlpack_raw, DlpackExport, DlpackImport, TensorBuffer};

const VERSIONED_NAME: &[u8] = b"dltensor_versioned\0";
const USED_VERSIONED_NAME: &[u8] = b"used_dltensor_versioned\0";

unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let name = VERSIONED_NAME.as_ptr().cast();
    // SAFETY: CPython calls the destructor with the capsule object. A consumer
    // renames consumed capsules, so only the original name retains ownership.
    if unsafe { pyo3::ffi::PyCapsule_IsValid(capsule, name) } == 1 {
        // SAFETY: validity above proves the name and capsule pointer contract.
        let raw = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, name) };
        if !raw.is_null() {
            // SAFETY: the unconsumed capsule uniquely owns deleter responsibility.
            unsafe { release_dlpack_raw(raw) };
        }
    }
}

pub(crate) fn export_tensor(py: Python<'_>, tensor: &TensorBuffer) -> PyResult<Py<PyAny>> {
    let export = DlpackExport::from_tensor(tensor)
        .map_err(|error| PyBufferError::new_err(error.to_string()))?;
    let raw = export.into_raw();
    // SAFETY: `raw` is live and the static nul-terminated name outlives the capsule.
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(raw, VERSIONED_NAME.as_ptr().cast(), Some(capsule_destructor))
    };
    if capsule.is_null() {
        // SAFETY: capsule construction failed, so ownership was not transferred.
        unsafe { release_dlpack_raw(raw) };
        return Err(PyErr::fetch(py));
    }
    // SAFETY: PyCapsule_New returned one new owned Python reference.
    Ok(unsafe { Py::from_owned_ptr(py, capsule) })
}

pub(crate) fn import_tensor(producer: &Bound<'_, PyAny>) -> PyResult<DlpackImport> {
    let kwargs = PyDict::new_bound(producer.py());
    kwargs.set_item("max_version", (1, 0))?;
    kwargs.set_item("copy", false)?;
    let capsule = producer.call_method("__dlpack__", (), Some(&kwargs))?;
    let name = VERSIONED_NAME.as_ptr().cast();
    // SAFETY: this only asks CPython to validate the exact capsule/name pair.
    if unsafe { pyo3::ffi::PyCapsule_IsValid(capsule.as_ptr(), name) } != 1 {
        return Err(PyBufferError::new_err("producer did not return a dltensor_versioned capsule"));
    }
    // SAFETY: capsule validity proves that the payload is non-null for this name.
    let raw = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule.as_ptr(), name) };
    // SAFETY: both names are static nul-terminated strings and capsule is valid.
    if unsafe {
        pyo3::ffi::PyCapsule_SetName(capsule.as_ptr(), USED_VERSIONED_NAME.as_ptr().cast())
    } != 0
    {
        return Err(PyErr::fetch(producer.py()));
    }
    // SAFETY: renaming transferred exclusive deleter ownership from the capsule.
    unsafe { DlpackImport::from_raw(raw) }
        .map_err(|error| PyBufferError::new_err(error.to_string()))
}
