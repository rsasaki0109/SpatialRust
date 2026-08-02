//! Arrow C Device Data Interface (CPU-only v1).

use std::ptr;

use spatialrust_core::{PointCloud, SpatialMetadata};

use crate::{
    cdata::{
        export_point_cloud_c_data, import_point_cloud_c_data, ArrowArray, ExportedArrowSchema,
    },
    ArrowBridgeError, ArrowBridgeResult,
};

/// Arrow device type codes used by the C Device Data Interface.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArrowDeviceType {
    /// Host CPU memory.
    Cpu = 1,
    /// CUDA device memory (recognized but not exported by this crate yet).
    Cuda = 2,
}

/// Arrow C Device Array wrapper around an [`ArrowArray`].
#[repr(C)]
pub struct ArrowDeviceArray {
    /// Device-resident or host array payload.
    pub array: ArrowArray,
    /// Device type.
    pub device_id: i64,
    /// Device type code.
    pub device_type: i32,
    /// Sync event handle; unused for CPU.
    pub sync_event: *mut std::ffi::c_void,
    /// Reserved for ABI growth.
    pub reserved: [i64; 3],
}

/// Owned CPU device-array export.
pub struct ExportedArrowDeviceArray {
    schema: ExportedArrowSchema,
    device: Box<ArrowDeviceArray>,
}

impl ExportedArrowDeviceArray {
    /// Returns the exported schema.
    #[must_use]
    pub fn schema(&mut self) -> &mut ExportedArrowSchema {
        &mut self.schema
    }

    /// Returns a mutable device-array pointer for FFI handoff.
    pub fn as_mut_ptr(&mut self) -> *mut ArrowDeviceArray {
        self.device.as_mut()
    }
}

impl Drop for ExportedArrowDeviceArray {
    fn drop(&mut self) {
        if let Some(release) = self.device.array.release {
            unsafe { release(&mut self.device.array) };
            self.device.array.release = None;
        }
    }
}

/// Exports a point cloud as an Arrow device array on CPU.
pub fn export_point_cloud_device_array(
    cloud: &PointCloud,
) -> ArrowBridgeResult<ExportedArrowDeviceArray> {
    let (schema, mut array) = export_point_cloud_c_data(cloud)?;
    let moved = unsafe { ptr::read(array.as_mut_ptr()) };
    unsafe {
        (*array.as_mut_ptr()).release = None;
        (*array.as_mut_ptr()).private_data = ptr::null_mut();
        (*array.as_mut_ptr()).buffers = ptr::null_mut();
        (*array.as_mut_ptr()).children = ptr::null_mut();
    }
    drop(array);
    let device = Box::new(ArrowDeviceArray {
        array: moved,
        device_id: -1,
        device_type: ArrowDeviceType::Cpu as i32,
        sync_event: ptr::null_mut(),
        reserved: [0; 3],
    });
    Ok(ExportedArrowDeviceArray { schema, device })
}

/// Imports a CPU Arrow device array into a point cloud.
///
/// Non-CPU device types are rejected until an explicit device copy path exists.
///
/// # Safety
///
/// `device_array` and `schema` must form a valid exported pair.
pub unsafe fn import_point_cloud_device_array(
    schema: *const crate::cdata::ArrowSchema,
    device_array: *const ArrowDeviceArray,
    metadata: SpatialMetadata,
) -> ArrowBridgeResult<PointCloud> {
    if device_array.is_null() {
        return Err(ArrowBridgeError::NullPointer("device array".into()));
    }
    let device_array = &*device_array;
    if device_array.device_type != ArrowDeviceType::Cpu as i32 {
        return Err(ArrowBridgeError::InvalidConfiguration(format!(
            "Arrow device type {} is not supported without an explicit device copy",
            device_array.device_type
        )));
    }
    import_point_cloud_c_data(schema, &device_array.array, metadata)
}

#[cfg(test)]
mod tests {
    use super::{
        export_point_cloud_device_array, import_point_cloud_device_array, ArrowDeviceType,
    };
    use spatialrust_core::{
        PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas,
    };

    #[test]
    fn cpu_device_array_roundtrip() {
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![1.0]));
        buffers.insert("y", PointBuffer::from_f32(vec![2.0]));
        buffers.insert("z", PointBuffer::from_f32(vec![3.0]));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::default(),
        )
        .unwrap();
        let mut exported = export_point_cloud_device_array(&cloud).unwrap();
        let schema_ptr = exported.schema().as_mut_ptr();
        let device_ptr = exported.as_mut_ptr();
        assert_eq!(unsafe { (*device_ptr).device_type }, ArrowDeviceType::Cpu as i32);
        let imported = unsafe {
            import_point_cloud_device_array(schema_ptr, device_ptr, SpatialMetadata::default())
        }
        .unwrap();
        assert_eq!(imported.field("x").unwrap().as_f32().unwrap(), &[1.0]);
    }
}
