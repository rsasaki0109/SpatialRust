//! Audited CPython capsule boundary for DLPack ownership transfer.

use pyo3::{
    exceptions::{PyBufferError, PyTypeError},
    prelude::*,
    types::{PyAny, PyDict},
};
use spatialrust::tensor::{
    release_dlpack_legacy_raw, release_dlpack_raw, DlpackExport, DlpackImport,
    DlpackLegacyExport, TensorBuffer,
};

const LEGACY_NAME: &[u8] = b"dltensor\0";
const USED_LEGACY_NAME: &[u8] = b"used_dltensor\0";
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
        return;
    }
    let legacy_name = LEGACY_NAME.as_ptr().cast();
    // SAFETY: this only validates the legacy capsule/name pair.
    if unsafe { pyo3::ffi::PyCapsule_IsValid(capsule, legacy_name) } == 1 {
        // SAFETY: validity above proves the name and capsule pointer contract.
        let raw = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, legacy_name) };
        if !raw.is_null() {
            // SAFETY: the unconsumed capsule uniquely owns deleter responsibility.
            unsafe { release_dlpack_legacy_raw(raw) };
        }
    }
}

pub(crate) fn export_tensor(
    py: Python<'_>,
    tensor: &TensorBuffer,
    versioned: bool,
) -> PyResult<Py<PyAny>> {
    let (raw, name) = if versioned {
        let export = DlpackExport::from_tensor(tensor)
            .map_err(|error| PyBufferError::new_err(error.to_string()))?;
        (export.into_raw(), VERSIONED_NAME)
    } else {
        let export = DlpackLegacyExport::from_tensor(tensor)
            .map_err(|error| PyBufferError::new_err(error.to_string()))?;
        (export.into_raw(), LEGACY_NAME)
    };
    // SAFETY: `raw` is live and the static nul-terminated name outlives the capsule.
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(raw, name.as_ptr().cast(), Some(capsule_destructor))
    };
    if capsule.is_null() {
        // SAFETY: capsule construction failed, so ownership was not transferred.
        unsafe {
            if versioned {
                release_dlpack_raw(raw);
            } else {
                release_dlpack_legacy_raw(raw);
            }
        };
        return Err(PyErr::fetch(py));
    }
    // SAFETY: PyCapsule_New returned one new owned Python reference.
    Ok(unsafe { Py::from_owned_ptr(py, capsule) })
}

pub(crate) fn import_tensor(producer: &Bound<'_, PyAny>) -> PyResult<DlpackImport> {
    let kwargs = PyDict::new_bound(producer.py());
    kwargs.set_item("max_version", (1, 0))?;
    kwargs.set_item("copy", false)?;
    let capsule = match producer.call_method("__dlpack__", (), Some(&kwargs)) {
        Ok(capsule) => capsule,
        Err(error) if error.is_instance_of::<PyTypeError>(producer.py()) => {
            producer.call_method0("__dlpack__")?
        }
        Err(error) => return Err(error),
    };
    let name = VERSIONED_NAME.as_ptr().cast();
    // SAFETY: this only asks CPython to validate the exact capsule/name pair.
    if unsafe { pyo3::ffi::PyCapsule_IsValid(capsule.as_ptr(), name) } == 1 {
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
        return unsafe { DlpackImport::from_raw(raw) }
            .map_err(|error| PyBufferError::new_err(error.to_string()));
    }
    let legacy_name = LEGACY_NAME.as_ptr().cast();
    if unsafe { pyo3::ffi::PyCapsule_IsValid(capsule.as_ptr(), legacy_name) } != 1 {
        return Err(PyBufferError::new_err(
            "producer did not return a dltensor or dltensor_versioned capsule",
        ));
    }
    // SAFETY: capsule validity proves that the payload is non-null for this name.
    let raw = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule.as_ptr(), legacy_name) };
    // SAFETY: both names are static nul-terminated strings and capsule is valid.
    if unsafe { pyo3::ffi::PyCapsule_SetName(capsule.as_ptr(), USED_LEGACY_NAME.as_ptr().cast()) }
        != 0
    {
        return Err(PyErr::fetch(producer.py()));
    }
    // SAFETY: renaming transferred exclusive deleter ownership from the capsule.
    unsafe { DlpackImport::from_legacy_raw(raw) }
        .map_err(|error| PyBufferError::new_err(error.to_string()))
}
