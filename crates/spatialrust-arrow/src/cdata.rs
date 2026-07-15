//! Arrow C Data Interface export/import for `PointCloud` columns.

use std::{
    ffi::{c_char, c_void, CString},
    ptr,
};

use spatialrust_core::{
    DType, PointBuffer, PointBufferSet, PointCloud, PointField, PointSchema, SpatialMetadata,
};

use crate::{ArrowBridgeError, ArrowBridgeResult};

/// Arrow C Data Interface schema object.
#[repr(C)]
#[derive(Debug)]
pub struct ArrowSchema {
    /// Format string (`f`, `g`, `i`, `+s`, ...).
    pub format: *const c_char,
    /// Field name.
    pub name: *const c_char,
    /// Optional metadata key/value blob.
    pub metadata: *const c_char,
    /// Dictionary / nullable flags.
    pub flags: i64,
    /// Child count for nested types.
    pub n_children: i64,
    /// Child schemas.
    pub children: *mut *mut ArrowSchema,
    /// Dictionary schema.
    pub dictionary: *mut ArrowSchema,
    /// Release callback.
    pub release: Option<unsafe extern "C" fn(schema: *mut ArrowSchema)>,
    /// Implementation private data.
    pub private_data: *mut c_void,
}

/// Arrow C Data Interface array object.
#[repr(C)]
#[derive(Debug)]
pub struct ArrowArray {
    /// Logical length.
    pub length: i64,
    /// Null count (`-1` if unknown).
    pub null_count: i64,
    /// Buffer offset.
    pub offset: i64,
    /// Number of buffers.
    pub n_buffers: i64,
    /// Nested child count.
    pub n_children: i64,
    /// Buffer pointer table.
    pub buffers: *mut *const c_void,
    /// Child arrays.
    pub children: *mut *mut ArrowArray,
    /// Dictionary array.
    pub dictionary: *mut ArrowArray,
    /// Release callback.
    pub release: Option<unsafe extern "C" fn(array: *mut ArrowArray)>,
    /// Implementation private data.
    pub private_data: *mut c_void,
}

/// Owned export that keeps `ArrowSchema` alive until dropped or released.
pub struct ExportedArrowSchema {
    raw: Box<ArrowSchema>,
}

impl ExportedArrowSchema {
    /// Returns a mutable raw pointer suitable for FFI handoff.
    pub fn as_mut_ptr(&mut self) -> *mut ArrowSchema {
        self.raw.as_mut()
    }

    /// Borrows the schema.
    #[must_use]
    pub fn schema(&self) -> &ArrowSchema {
        &self.raw
    }
}

impl Drop for ExportedArrowSchema {
    fn drop(&mut self) {
        if let Some(release) = self.raw.release {
            unsafe { release(self.raw.as_mut()) };
        }
    }
}

/// Owned export that keeps `ArrowArray` alive until dropped or released.
pub struct ExportedArrowArray {
    raw: Box<ArrowArray>,
}

impl ExportedArrowArray {
    /// Returns a mutable raw pointer suitable for FFI handoff.
    pub fn as_mut_ptr(&mut self) -> *mut ArrowArray {
        self.raw.as_mut()
    }

    /// Borrows the array.
    #[must_use]
    pub fn array(&self) -> &ArrowArray {
        &self.raw
    }
}

impl Drop for ExportedArrowArray {
    fn drop(&mut self) {
        if let Some(release) = self.raw.release {
            unsafe { release(self.raw.as_mut()) };
        }
    }
}

/// Exports a point cloud as an Arrow struct array plus matching schema.
pub fn export_point_cloud_c_data(
    cloud: &PointCloud,
) -> ArrowBridgeResult<(ExportedArrowSchema, ExportedArrowArray)> {
    cloud.validate()?;
    let schema = export_schema(cloud.schema())?;
    let array = export_array(cloud)?;
    Ok((schema, array))
}

/// Imports a SpatialRust point cloud from Arrow C Data struct columns.
///
/// # Safety
///
/// `schema` and `array` must form a valid exported Arrow C Data pair produced
/// for a SpatialRust point cloud (or an equivalent struct of primitive columns).
pub unsafe fn import_point_cloud_c_data(
    schema: *const ArrowSchema,
    array: *const ArrowArray,
    metadata: SpatialMetadata,
) -> ArrowBridgeResult<PointCloud> {
    if schema.is_null() || array.is_null() {
        return Err(ArrowBridgeError::NullPointer("schema/array".into()));
    }
    let schema = &*schema;
    let array = &*array;
    if schema.format.is_null() {
        return Err(ArrowBridgeError::NullPointer("schema.format".into()));
    }
    let format = std::ffi::CStr::from_ptr(schema.format).to_string_lossy();
    if format.as_ref() != "+s" {
        return Err(ArrowBridgeError::SchemaMismatch(format!(
            "expected Arrow struct (+s), found {format}"
        )));
    }
    if schema.n_children < 0 || array.n_children < 0 || schema.n_children != array.n_children {
        return Err(ArrowBridgeError::SchemaMismatch(
            "schema/array child counts disagree".into(),
        ));
    }
    let n = schema.n_children as usize;
    let mut point_schema = PointSchema::new();
    let mut buffers = PointBufferSet::new();
    let length = usize::try_from(array.length).map_err(|_| {
        ArrowBridgeError::InvalidConfiguration("array length does not fit usize".into())
    })?;
    for index in 0..n {
        let child_schema = *schema.children.add(index);
        let child_array = *array.children.add(index);
        if child_schema.is_null() || child_array.is_null() {
            return Err(ArrowBridgeError::NullPointer("child schema/array".into()));
        }
        let (field, buffer) = import_child(&*child_schema, &*child_array, length)?;
        buffers.insert(field.name.clone(), buffer);
        point_schema = point_schema.with_field(field);
    }
    Ok(PointCloud::try_from_parts(point_schema, buffers, metadata)?)
}

fn export_schema(point_schema: &PointSchema) -> ArrowBridgeResult<ExportedArrowSchema> {
    let children = point_schema
        .fields()
        .iter()
        .map(export_field_schema)
        .collect::<ArrowBridgeResult<Vec<_>>>()?;
    let mut child_ptrs =
        children.into_iter().map(Box::into_raw).collect::<Vec<*mut ArrowSchema>>();
    let child_table = child_ptrs.as_mut_ptr();
    let n_children = child_ptrs.len() as i64;
    // Leak the vec table into private data; release rebuilds and frees it.
    std::mem::forget(child_ptrs);

    let format = CString::new("+s").expect("static");
    let name = CString::new("PointCloud").expect("static");
    let private = Box::new(SchemaPrivate {
        format,
        name,
        children_table: child_table,
        n_children: n_children as usize,
    });
    let raw = Box::new(ArrowSchema {
        format: private.format.as_ptr(),
        name: private.name.as_ptr(),
        metadata: ptr::null(),
        flags: 0,
        n_children,
        children: child_table,
        dictionary: ptr::null_mut(),
        release: Some(release_schema),
        private_data: Box::into_raw(private) as *mut c_void,
    });
    Ok(ExportedArrowSchema { raw })
}

fn export_field_schema(field: &PointField) -> ArrowBridgeResult<Box<ArrowSchema>> {
    if field.components != 1 {
        return Err(ArrowBridgeError::InvalidConfiguration(
            "Arrow C Data export currently supports scalar fields only".into(),
        ));
    }
    let format = CString::new(dtype_format(field.dtype)?).map_err(|_| {
        ArrowBridgeError::InvalidConfiguration("field format contained NUL".into())
    })?;
    let name = CString::new(field.name.as_str()).map_err(|_| {
        ArrowBridgeError::InvalidConfiguration("field name contained NUL".into())
    })?;
    let private = Box::new(SchemaPrivate {
        format,
        name,
        children_table: ptr::null_mut(),
        n_children: 0,
    });
    Ok(Box::new(ArrowSchema {
        format: private.format.as_ptr(),
        name: private.name.as_ptr(),
        metadata: ptr::null(),
        flags: 0,
        n_children: 0,
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_schema),
        private_data: Box::into_raw(private) as *mut c_void,
    }))
}

fn export_array(cloud: &PointCloud) -> ArrowBridgeResult<ExportedArrowArray> {
    let children = cloud
        .schema()
        .fields()
        .iter()
        .map(|field| {
            let buffer = cloud.field(&field.name)?;
            export_primitive_array(buffer, cloud.len())
        })
        .collect::<ArrowBridgeResult<Vec<_>>>()?;
    let mut child_ptrs = children.into_iter().map(Box::into_raw).collect::<Vec<*mut ArrowArray>>();
    let child_table = child_ptrs.as_mut_ptr();
    let n_children = child_ptrs.len() as i64;
    std::mem::forget(child_ptrs);

    let private = Box::new(ArrayPrivate {
        buffers_table: ptr::null_mut(),
        n_buffers: 0,
        children_table: child_table,
        n_children: n_children as usize,
        owned_buffers: Vec::new(),
    });
    let raw = Box::new(ArrowArray {
        length: cloud.len() as i64,
        null_count: 0,
        offset: 0,
        n_buffers: 1, // struct arrays expose a single validity/null bitmap buffer slot
        n_children,
        buffers: {
            // Struct: one null bitmap pointer (null means non-nullable / no bitmap).
            let mut buffers = vec![ptr::null()];
            let table = buffers.as_mut_ptr();
            // Store buffers table in private for release.
            // Safety: overwrite after private box created - reassign below via raw.
            std::mem::forget(buffers);
            table
        },
        children: child_table,
        dictionary: ptr::null_mut(),
        release: Some(release_array),
        private_data: ptr::null_mut(),
    });
    // Move buffers_table into private and attach private_data.
    let mut raw = raw;
    let buffers_table = raw.buffers;
    let mut private = private;
    private.buffers_table = buffers_table;
    private.n_buffers = 1;
    raw.private_data = Box::into_raw(private) as *mut c_void;
    Ok(ExportedArrowArray { raw })
}

fn export_primitive_array(buffer: &PointBuffer, len: usize) -> ArrowBridgeResult<Box<ArrowArray>> {
    if buffer.len() != len {
        return Err(ArrowBridgeError::SchemaMismatch(
            "column length does not match point count".into(),
        ));
    }
    let owned = clone_bytes(buffer);
    let data_ptr = owned.as_ptr() as *const c_void;
    let mut buffers = vec![ptr::null(), data_ptr];
    let buffers_table = buffers.as_mut_ptr();
    std::mem::forget(buffers);
    let private = Box::new(ArrayPrivate {
        buffers_table,
        n_buffers: 2,
        children_table: ptr::null_mut(),
        n_children: 0,
        owned_buffers: vec![owned],
    });
    Ok(Box::new(ArrowArray {
        length: len as i64,
        null_count: 0,
        offset: 0,
        n_buffers: 2,
        n_children: 0,
        buffers: buffers_table,
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_array),
        private_data: Box::into_raw(private) as *mut c_void,
    }))
}

unsafe fn import_child(
    schema: &ArrowSchema,
    array: &ArrowArray,
    expected_len: usize,
) -> ArrowBridgeResult<(PointField, PointBuffer)> {
    if schema.format.is_null() || schema.name.is_null() {
        return Err(ArrowBridgeError::NullPointer("child format/name".into()));
    }
    let format = std::ffi::CStr::from_ptr(schema.format).to_string_lossy();
    let name = std::ffi::CStr::from_ptr(schema.name)
        .to_str()
        .map_err(|_| ArrowBridgeError::InvalidConfiguration("field name is not UTF-8".into()))?
        .to_owned();
    let dtype = format_dtype(format.as_ref())?;
    let length = usize::try_from(array.length).map_err(|_| {
        ArrowBridgeError::InvalidConfiguration("child length does not fit usize".into())
    })?;
    if length != expected_len {
        return Err(ArrowBridgeError::SchemaMismatch(format!(
            "child `{name}` length {length} != parent {expected_len}"
        )));
    }
    if array.n_buffers < 2 || array.buffers.is_null() {
        return Err(ArrowBridgeError::InvalidConfiguration(
            "primitive arrays require validity + data buffers".into(),
        ));
    }
    let data_ptr = *array.buffers.add(1);
    if length > 0 && data_ptr.is_null() {
        return Err(ArrowBridgeError::NullPointer(format!("data for `{name}`")));
    }
    let buffer = copy_buffer(dtype, data_ptr, length)?;
    let semantic = semantic_for_name(&name);
    Ok((PointField::scalar(name, semantic, dtype), buffer))
}

fn clone_bytes(buffer: &PointBuffer) -> Vec<u8> {
    match buffer {
        PointBuffer::F32(values) => bytemuck_bytes(values),
        PointBuffer::F64(values) => bytemuck_bytes(values),
        PointBuffer::U8(values) => values.clone(),
        PointBuffer::U16(values) => bytemuck_bytes(values),
        PointBuffer::U32(values) => bytemuck_bytes(values),
        PointBuffer::I32(values) => bytemuck_bytes(values),
    }
}

fn bytemuck_bytes<T: Copy>(values: &[T]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        let ptr = value as *const T as *const u8;
        let slice = unsafe { std::slice::from_raw_parts(ptr, std::mem::size_of::<T>()) };
        bytes.extend_from_slice(slice);
    }
    bytes
}

unsafe fn copy_buffer(
    dtype: DType,
    data_ptr: *const c_void,
    length: usize,
) -> ArrowBridgeResult<PointBuffer> {
    if length == 0 {
        return Ok(PointBuffer::with_capacity(dtype, 0));
    }
    Ok(match dtype {
        DType::F32 => PointBuffer::F32(copy_typed(data_ptr, length)),
        DType::F64 => PointBuffer::F64(copy_typed(data_ptr, length)),
        DType::U8 => PointBuffer::U8(copy_typed(data_ptr, length)),
        DType::U16 => PointBuffer::U16(copy_typed(data_ptr, length)),
        DType::U32 => PointBuffer::U32(copy_typed(data_ptr, length)),
        DType::I32 => PointBuffer::I32(copy_typed(data_ptr, length)),
        DType::F16 => {
            return Err(ArrowBridgeError::InvalidConfiguration(
                "F16 Arrow import is not implemented".into(),
            ))
        }
    })
}

unsafe fn copy_typed<T: Copy>(data_ptr: *const c_void, length: usize) -> Vec<T> {
    let slice = std::slice::from_raw_parts(data_ptr as *const T, length);
    slice.to_vec()
}

fn dtype_format(dtype: DType) -> ArrowBridgeResult<&'static str> {
    match dtype {
        DType::F32 => Ok("f"),
        DType::F64 => Ok("g"),
        DType::U8 => Ok("C"),
        DType::U16 => Ok("S"),
        DType::U32 => Ok("I"),
        DType::I32 => Ok("i"),
        DType::F16 => Err(ArrowBridgeError::InvalidConfiguration(
            "F16 Arrow export is not implemented".into(),
        )),
    }
}

fn format_dtype(format: &str) -> ArrowBridgeResult<DType> {
    match format {
        "f" => Ok(DType::F32),
        "g" => Ok(DType::F64),
        "C" => Ok(DType::U8),
        "S" => Ok(DType::U16),
        "I" => Ok(DType::U32),
        "i" => Ok(DType::I32),
        other => Err(ArrowBridgeError::SchemaMismatch(format!(
            "unsupported Arrow format `{other}`"
        ))),
    }
}

fn semantic_for_name(name: &str) -> spatialrust_core::FieldSemantic {
    use spatialrust_core::FieldSemantic;
    match name {
        "x" => FieldSemantic::PositionX,
        "y" => FieldSemantic::PositionY,
        "z" => FieldSemantic::PositionZ,
        "intensity" => FieldSemantic::Intensity,
        "nx" | "normal_x" => FieldSemantic::NormalX,
        "ny" | "normal_y" => FieldSemantic::NormalY,
        "nz" | "normal_z" => FieldSemantic::NormalZ,
        "r" | "red" => FieldSemantic::ColorR,
        "g" | "green" => FieldSemantic::ColorG,
        "b" | "blue" => FieldSemantic::ColorB,
        _ => FieldSemantic::Unknown,
    }
}

struct SchemaPrivate {
    format: CString,
    name: CString,
    children_table: *mut *mut ArrowSchema,
    n_children: usize,
}

struct ArrayPrivate {
    buffers_table: *mut *const c_void,
    n_buffers: usize,
    children_table: *mut *mut ArrowArray,
    n_children: usize,
    /// Keeps exported column bytes alive for the Arrow buffer pointers.
    #[allow(dead_code)]
    owned_buffers: Vec<Vec<u8>>,
}

unsafe extern "C" fn release_schema(schema: *mut ArrowSchema) {
    if schema.is_null() {
        return;
    }
    let schema = &mut *schema;
    if schema.release.is_none() {
        return;
    }
    if !schema.private_data.is_null() {
        let private = Box::from_raw(schema.private_data as *mut SchemaPrivate);
        if !private.children_table.is_null() && private.n_children > 0 {
            let children =
                Vec::from_raw_parts(private.children_table, private.n_children, private.n_children);
            for child in children {
                if !child.is_null() {
                    if let Some(release) = (*child).release {
                        release(child);
                    }
                    drop(Box::from_raw(child));
                }
            }
        }
        drop(private);
    }
    schema.release = None;
    schema.private_data = ptr::null_mut();
    schema.children = ptr::null_mut();
    schema.format = ptr::null();
    schema.name = ptr::null();
}

unsafe extern "C" fn release_array(array: *mut ArrowArray) {
    if array.is_null() {
        return;
    }
    let array = &mut *array;
    if array.release.is_none() {
        return;
    }
    if !array.private_data.is_null() {
        let private = Box::from_raw(array.private_data as *mut ArrayPrivate);
        if !private.children_table.is_null() && private.n_children > 0 {
            let children = Vec::from_raw_parts(
                private.children_table,
                private.n_children,
                private.n_children,
            );
            for child in children {
                if !child.is_null() {
                    if let Some(release) = (*child).release {
                        release(child);
                    }
                    drop(Box::from_raw(child));
                }
            }
        }
        if !private.buffers_table.is_null() && private.n_buffers > 0 {
            let _ = Vec::from_raw_parts(
                private.buffers_table as *mut *const c_void,
                private.n_buffers,
                private.n_buffers,
            );
        }
        drop(private);
    }
    array.release = None;
    array.private_data = ptr::null_mut();
    array.buffers = ptr::null_mut();
    array.children = ptr::null_mut();
}

#[cfg(test)]
mod tests {
    use super::{export_point_cloud_c_data, import_point_cloud_c_data};
    use spatialrust_core::{
        PointBuffer, PointBufferSet, PointCloud, SpatialMetadata, StandardSchemas,
    };

    #[test]
    fn roundtrip_xyz_cloud() {
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![1.0, 2.0]));
        buffers.insert("y", PointBuffer::from_f32(vec![3.0, 4.0]));
        buffers.insert("z", PointBuffer::from_f32(vec![5.0, 6.0]));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::default(),
        )
        .unwrap();
        let (mut schema, mut array) = export_point_cloud_c_data(&cloud).unwrap();
        let imported = unsafe {
            import_point_cloud_c_data(schema.as_mut_ptr(), array.as_mut_ptr(), SpatialMetadata::default())
        }
        .unwrap();
        assert_eq!(imported.len(), 2);
        assert_eq!(imported.field("x").unwrap().as_f32().unwrap(), &[1.0, 2.0]);
        assert_eq!(imported.field("z").unwrap().as_f32().unwrap(), &[5.0, 6.0]);
    }
}
