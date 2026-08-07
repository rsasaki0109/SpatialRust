//! CPython capsule boundary for the Arrow C Data Interface.

use std::ffi::{c_char, c_void, CString};
use std::ptr;

use pyo3::exceptions::PyBufferError;
use pyo3::prelude::*;

use spatialrust::arrow::{export_point_cloud_c_data, ArrowArray, ArrowArrayStream, ArrowSchema};

const ARRAY_CAPSULE_NAME: &[u8] = b"arrow_array\0";
const SCHEMA_CAPSULE_NAME: &[u8] = b"arrow_schema\0";
const STREAM_CAPSULE_NAME: &[u8] = b"arrow_array_stream\0";

unsafe extern "C" fn array_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let name = ARRAY_CAPSULE_NAME.as_ptr().cast();
    // SAFETY: CPython calls the destructor with the capsule object. The
    // unconsumed capsule uniquely owns the Arrow array export.
    if unsafe { pyo3::ffi::PyCapsule_IsValid(capsule, name) } == 1 {
        let raw = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, name) }.cast::<ArrowArray>();
        if !raw.is_null() {
            // SAFETY: the unconsumed capsule owns the ArrowArray; call its
            // release callback and free the Box allocation.
            if let Some(release) = (*raw).release {
                unsafe { release(raw) };
            }
            unsafe { drop(Box::from_raw(raw)) };
        }
    }
}

unsafe extern "C" fn schema_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let name = SCHEMA_CAPSULE_NAME.as_ptr().cast();
    // SAFETY: CPython calls the destructor with the capsule object. The
    // unconsumed capsule uniquely owns the Arrow schema export.
    if unsafe { pyo3::ffi::PyCapsule_IsValid(capsule, name) } == 1 {
        let raw = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, name) }.cast::<ArrowSchema>();
        if !raw.is_null() {
            // SAFETY: the unconsumed capsule owns the ArrowSchema; call its
            // release callback and free the Box allocation.
            if let Some(release) = (*raw).release {
                unsafe { release(raw) };
            }
            unsafe { drop(Box::from_raw(raw)) };
        }
    }
}

/// Exports a point cloud as an `(array, schema)` capsule tuple for
/// `__arrow_c_array__`. The capsules hold owned exports whose `Drop` releases
/// the Arrow C Data resources exactly once.
pub(crate) fn export_point_cloud(
    py: Python<'_>,
    cloud: &spatialrust::core::PointCloud,
) -> PyResult<Py<PyAny>> {
    let (mut schema, mut array) = export_point_cloud_c_data(cloud)
        .map_err(|error| PyBufferError::new_err(error.to_string()))?;

    // Forget the Rust owners; the capsules take over. On failure we rebuild
    // the owners so their Drop still releases the Arrow resources.
    let array_ptr = array.as_mut_ptr();
    let schema_ptr = schema.as_mut_ptr();
    std::mem::forget(array);
    std::mem::forget(schema);

    // SAFETY: as_mut_ptr hands ownership of the Box allocations to the
    // capsules; the destructors rebuild the Box and drop it, releasing the
    // Arrow release callbacks.
    let array_capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            array_ptr.cast(),
            ARRAY_CAPSULE_NAME.as_ptr().cast(),
            Some(array_capsule_destructor),
        )
    };
    let schema_capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            schema_ptr.cast(),
            SCHEMA_CAPSULE_NAME.as_ptr().cast(),
            Some(schema_capsule_destructor),
        )
    };

    let fail = array_capsule.is_null() || schema_capsule.is_null();
    if fail {
        // SAFETY: PyCapsule_New returned one new owned Python reference when the
        // pointer is non-null; rebuild the owned Box for the failed one.
        if !array_capsule.is_null() {
            unsafe { pyo3::ffi::Py_DECREF(array_capsule) };
        } else {
            unsafe { drop(Box::from_raw(array_ptr)) };
        }
        if !schema_capsule.is_null() {
            unsafe { pyo3::ffi::Py_DECREF(schema_capsule) };
        } else {
            unsafe { drop(Box::from_raw(schema_ptr)) };
        }
        return Err(PyBufferError::new_err("failed to create Arrow C Data capsules"));
    }

    // SAFETY: PyTuple_New returns a new owned reference. PyTuple_SetItem steals
    // the capsule references (they remain alive inside the tuple). The Arrow
    // protocol expects `(schema, array)` capsule order.
    let tuple = unsafe { pyo3::ffi::PyTuple_New(2) };
    if tuple.is_null() {
        // SAFETY: the capsules still own their Box allocations; drop them.
        unsafe { pyo3::ffi::Py_DECREF(array_capsule) };
        unsafe { pyo3::ffi::Py_DECREF(schema_capsule) };
        return Err(PyErr::fetch(py));
    }
    // SAFETY: PyTuple_SetItem steals the reference, so no decref here.
    unsafe { pyo3::ffi::PyTuple_SetItem(tuple, 0, schema_capsule) };
    unsafe { pyo3::ffi::PyTuple_SetItem(tuple, 1, array_capsule) };
    // SAFETY: PyTuple_New returned one new owned reference.
    Ok(unsafe { Py::<PyAny>::from_owned_ptr(py, tuple) })
}

/// Adapter from a chunk iterator to a pullable source for the Arrow C Stream.
struct StreamPrivate {
    iter: spatialrust::pipeline::StreamingPipelineIter,
    last_error: Option<CString>,
}

/// Exports a chunk iterator as an Arrow C Stream capsule for
/// `__arrow_c_stream__`.
pub(crate) fn export_stream(
    py: Python<'_>,
    iter: spatialrust::pipeline::StreamingPipelineIter,
) -> PyResult<Py<PyAny>> {
    let private = Box::new(StreamPrivate { iter, last_error: None });
    let raw = Box::new(ArrowArrayStream {
        get_schema: Some(stream_get_schema),
        get_next: Some(stream_get_next),
        get_last_error: Some(stream_get_last_error),
        release: Some(stream_release),
        private_data: Box::into_raw(private) as *mut c_void,
    });
    let stream_ptr = Box::into_raw(raw);

    // SAFETY: as_mut_ptr hands ownership of the Box allocation to the capsule;
    // the destructor calls the stream release callback and frees the Box.
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            stream_ptr.cast(),
            STREAM_CAPSULE_NAME.as_ptr().cast(),
            Some(stream_capsule_destructor),
        )
    };
    if capsule.is_null() {
        // SAFETY: capsule construction failed, so ownership was not transferred.
        unsafe {
            if let Some(release) = (*stream_ptr).release {
                release(stream_ptr);
            }
            drop(Box::from_raw(stream_ptr));
        };
        return Err(PyErr::fetch(py));
    }
    // SAFETY: PyCapsule_New returned one new owned Python reference.
    Ok(unsafe { Py::<PyAny>::from_owned_ptr(py, capsule) })
}

unsafe extern "C" fn stream_get_schema(
    stream: *mut ArrowArrayStream,
    out: *mut ArrowSchema,
) -> i32 {
    if stream.is_null() || out.is_null() {
        return 22; // EINVAL
    }
    let private = &mut *stream;
    if private.private_data.is_null() {
        return 22;
    }
    let private = &mut *(private.private_data as *mut StreamPrivate);
    let schema = private.iter.schema().point_schema().clone();
    // Build an empty cloud to drive the C Data schema export.
    let cloud = match empty_cloud(&schema) {
        Ok(cloud) => cloud,
        Err(message) => {
            set_error(private, message);
            return 5;
        }
    };
    match export_point_cloud_c_data(&cloud) {
        Ok((mut exported_schema, exported_array)) => {
            ptr::write(out, ptr::read(exported_schema.as_mut_ptr()));
            unsafe {
                (*exported_schema.as_mut_ptr()).release = None;
                (*exported_schema.as_mut_ptr()).private_data = ptr::null_mut();
                (*exported_schema.as_mut_ptr()).children = ptr::null_mut();
            }
            drop(exported_array);
            0
        }
        Err(error) => {
            set_error(private, error.to_string());
            5
        }
    }
}

unsafe extern "C" fn stream_get_next(stream: *mut ArrowArrayStream, out: *mut ArrowArray) -> i32 {
    if stream.is_null() || out.is_null() {
        return 22;
    }
    let private = &mut *stream;
    if private.private_data.is_null() {
        return 22;
    }
    let private = &mut *(private.private_data as *mut StreamPrivate);
    match private.iter.next() {
        None => {
            ptr::write(out, null_array());
            0
        }
        Some(Ok(chunk)) => {
            let cloud = chunk.record().cloud().clone();
            match export_point_cloud_c_data(&cloud) {
                Ok((_schema, mut array)) => {
                    ptr::write(out, ptr::read(array.as_mut_ptr()));
                    unsafe {
                        (*array.as_mut_ptr()).release = None;
                        (*array.as_mut_ptr()).private_data = ptr::null_mut();
                        (*array.as_mut_ptr()).buffers = ptr::null_mut();
                        (*array.as_mut_ptr()).children = ptr::null_mut();
                    }
                    0
                }
                Err(error) => {
                    set_error(private, error.to_string());
                    5
                }
            }
        }
        Some(Err(error)) => {
            set_error(private, error.to_string());
            5
        }
    }
}

unsafe extern "C" fn stream_get_last_error(stream: *mut ArrowArrayStream) -> *const c_char {
    if stream.is_null() {
        return ptr::null();
    }
    let private = &mut *stream;
    if private.private_data.is_null() {
        return ptr::null();
    }
    let private = &mut *(private.private_data as *mut StreamPrivate);
    private.last_error.as_ref().map(|value| value.as_ptr()).unwrap_or(ptr::null())
}

unsafe extern "C" fn stream_release(stream: *mut ArrowArrayStream) {
    if stream.is_null() {
        return;
    }
    let stream = &mut *stream;
    if stream.release.is_none() {
        return;
    }
    if !stream.private_data.is_null() {
        drop(Box::from_raw(stream.private_data as *mut StreamPrivate));
    }
    stream.get_schema = None;
    stream.get_next = None;
    stream.get_last_error = None;
    stream.release = None;
    stream.private_data = ptr::null_mut();
}

unsafe extern "C" fn stream_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let name = STREAM_CAPSULE_NAME.as_ptr().cast();
    // SAFETY: CPython calls the destructor with the capsule object. The
    // unconsumed capsule uniquely owns the Arrow stream.
    if unsafe { pyo3::ffi::PyCapsule_IsValid(capsule, name) } == 1 {
        let raw = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, name) }
            .cast::<ArrowArrayStream>();
        if !raw.is_null() {
            // SAFETY: the unconsumed capsule owns the stream; call its release
            // callback and free the Box allocation.
            if let Some(release) = (*raw).release {
                unsafe { release(raw) };
            }
            unsafe { drop(Box::from_raw(raw)) };
        }
    }
}

fn set_error(private: &mut StreamPrivate, message: String) {
    private.last_error = CString::new(message).ok();
}

fn null_array() -> ArrowArray {
    ArrowArray {
        length: 0,
        null_count: 0,
        offset: 0,
        n_buffers: 0,
        n_children: 0,
        buffers: ptr::null_mut(),
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: None,
        private_data: ptr::null_mut(),
    }
}

fn empty_cloud(schema: &spatialrust::core::PointSchema) -> Result<spatialrust::core::PointCloud, String> {
    use spatialrust::core::{PointBuffer, PointBufferSet, PointCloud, SpatialMetadata};
    let mut buffers = PointBufferSet::new();
    for field in schema.fields() {
        buffers.insert(field.name.clone(), PointBuffer::with_capacity(field.dtype, 0));
    }
    PointCloud::try_from_parts(schema.clone(), buffers, SpatialMetadata::default())
        .map_err(|error| error.to_string())
}
