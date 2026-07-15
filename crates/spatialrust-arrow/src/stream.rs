//! Arrow C Stream Interface over SpatialRust record sources.

use std::{
    ffi::{c_char, c_void, CString},
    ptr,
};

use spatialrust_records::SpatialRecordSource;

use crate::{cdata::{export_point_cloud_c_data, ArrowArray, ArrowSchema}, ArrowBridgeResult};

/// Arrow C Stream Interface object.
#[repr(C)]
pub struct ArrowArrayStream {
    /// Fills `out` with the stream schema.
    pub get_schema: Option<unsafe extern "C" fn(stream: *mut ArrowArrayStream, out: *mut ArrowSchema) -> i32>,
    /// Fills `out` with the next array (`release=null` when exhausted).
    pub get_next: Option<unsafe extern "C" fn(stream: *mut ArrowArrayStream, out: *mut ArrowArray) -> i32>,
    /// Optional last-error message.
    pub get_last_error: Option<unsafe extern "C" fn(stream: *mut ArrowArrayStream) -> *const c_char>,
    /// Release callback.
    pub release: Option<unsafe extern "C" fn(stream: *mut ArrowArrayStream)>,
    /// Implementation private data.
    pub private_data: *mut c_void,
}

/// Owned Arrow C Stream export.
pub struct ExportedArrowArrayStream {
    raw: Box<ArrowArrayStream>,
}

impl ExportedArrowArrayStream {
    /// Returns a mutable raw stream pointer for FFI handoff.
    pub fn as_mut_ptr(&mut self) -> *mut ArrowArrayStream {
        self.raw.as_mut()
    }
}

impl Drop for ExportedArrowArrayStream {
    fn drop(&mut self) {
        if let Some(release) = self.raw.release {
            unsafe { release(self.raw.as_mut()) };
        }
    }
}

struct StreamPrivate {
    source: Box<dyn SpatialRecordSource + Send>,
    last_error: Option<CString>,
}

/// Exports a [`SpatialRecordSource`] as an Arrow C Stream of point-cloud structs.
pub fn export_record_source_c_stream(
    source: Box<dyn SpatialRecordSource + Send>,
) -> ArrowBridgeResult<ExportedArrowArrayStream> {
    let private = Box::new(StreamPrivate { source, last_error: None });
    let raw = Box::new(ArrowArrayStream {
        get_schema: Some(stream_get_schema),
        get_next: Some(stream_get_next),
        get_last_error: Some(stream_get_last_error),
        release: Some(stream_release),
        private_data: Box::into_raw(private) as *mut c_void,
    });
    Ok(ExportedArrowArrayStream { raw })
}

unsafe extern "C" fn stream_get_schema(
    stream: *mut ArrowArrayStream,
    out: *mut ArrowSchema,
) -> i32 {
    if stream.is_null() || out.is_null() {
        return EINVAL;
    }
    let private = match private_mut(stream) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let cloud = match empty_cloud(private.source.schema().point_schema()) {
        Ok(cloud) => cloud,
        Err(error) => {
            set_error(private, error.to_string());
            return EIO;
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
            EIO
        }
    }
}

unsafe extern "C" fn stream_get_next(stream: *mut ArrowArrayStream, out: *mut ArrowArray) -> i32 {
    if stream.is_null() || out.is_null() {
        return EINVAL;
    }
    let private = match private_mut(stream) {
        Ok(value) => value,
        Err(code) => return code,
    };
    match private.source.next_record() {
        None => {
            ptr::write(out, null_array());
            0
        }
        Some(Ok(record)) => match export_point_cloud_c_data(record.cloud()) {
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
                EIO
            }
        },
        Some(Err(error)) => {
            set_error(private, error.to_string());
            EIO
        }
    }
}

unsafe extern "C" fn stream_get_last_error(stream: *mut ArrowArrayStream) -> *const c_char {
    if stream.is_null() {
        return ptr::null();
    }
    match private_mut(stream) {
        Ok(private) => private
            .last_error
            .as_ref()
            .map(|value| value.as_ptr())
            .unwrap_or(ptr::null()),
        Err(_) => ptr::null(),
    }
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

unsafe fn private_mut(stream: *mut ArrowArrayStream) -> Result<&'static mut StreamPrivate, i32> {
    let stream = &mut *stream;
    if stream.private_data.is_null() {
        return Err(EINVAL);
    }
    Ok(&mut *(stream.private_data as *mut StreamPrivate))
}

fn empty_cloud(schema: &spatialrust_core::PointSchema) -> ArrowBridgeResult<spatialrust_core::PointCloud> {
    use spatialrust_core::{PointBuffer, PointBufferSet, PointCloud, SpatialMetadata};
    let mut buffers = PointBufferSet::new();
    for field in schema.fields() {
        buffers.insert(field.name.clone(), PointBuffer::with_capacity(field.dtype, 0));
    }
    Ok(PointCloud::try_from_parts(schema.clone(), buffers, SpatialMetadata::default())?)
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

const EINVAL: i32 = 22;
const EIO: i32 = 5;

#[cfg(test)]
mod tests {
    use super::export_record_source_c_stream;
    use crate::cdata::import_point_cloud_c_data;
    use spatialrust_core::{
        PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas,
    };
    use spatialrust_records::{
        MemoryChunkSource, SchemaDescriptor, SchemaVersion,
    };

    #[test]
    fn stream_yields_chunked_clouds() {
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![0.0, 1.0, 2.0]));
        buffers.insert("y", PointBuffer::from_f32(vec![0.0; 3]));
        buffers.insert("z", PointBuffer::from_f32(vec![1.0; 3]));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::default(),
        )
        .unwrap();
        let schema =
            SchemaDescriptor::try_new("point", SchemaVersion::new(1, 0), cloud.schema().clone())
                .unwrap();
        let source = MemoryChunkSource::try_new(schema, cloud, 2).unwrap();
        let mut stream = export_record_source_c_stream(Box::new(source)).unwrap();
        let stream_ptr = stream.as_mut_ptr();
        unsafe {
            let get_next = (*stream_ptr).get_next.expect("get_next");
            let mut first = std::mem::zeroed();
            assert_eq!(get_next(stream_ptr, &mut first), 0);
            assert!(first.release.is_some());
            assert_eq!(first.length, 2);
            let mut second = std::mem::zeroed();
            assert_eq!(get_next(stream_ptr, &mut second), 0);
            assert_eq!(second.length, 1);
            let mut done = std::mem::zeroed();
            assert_eq!(get_next(stream_ptr, &mut done), 0);
            assert!(done.release.is_none());
            if let Some(release) = first.release {
                release(&mut first);
            }
            if let Some(release) = second.release {
                release(&mut second);
            }
        }
        let _ = import_point_cloud_c_data;
    }
}
