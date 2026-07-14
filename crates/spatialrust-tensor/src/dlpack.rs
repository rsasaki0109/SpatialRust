//! Audited DLPack major-version 1 CPU ownership boundary.

use std::{
    ffi::c_void,
    mem::ManuallyDrop,
    ptr::{self, NonNull},
    slice,
};

use crate::{
    DataType, DataTypeCode, Device, DeviceKind, TensorBuffer, TensorDescriptor, TensorError,
    TensorStorage, TensorView,
};

/// DLPack ABI major version supported by this boundary.
pub const DLPACK_MAJOR: u32 = 1;
/// Baseline DLPack minor version emitted by this boundary.
pub const DLPACK_MINOR: u32 = 0;

const FLAG_READ_ONLY: u64 = 1;
const MAX_RANK: usize = 64;

/// Errors raised while validating a DLPack managed tensor.
#[derive(Debug, thiserror::Error)]
pub enum DlpackError {
    /// A null managed-tensor pointer was supplied.
    #[error("DLPack managed tensor pointer is null")]
    NullManagedTensor,
    /// The ABI major version is incompatible.
    #[error("unsupported DLPack ABI version {major}.{minor}; expected major {DLPACK_MAJOR}")]
    IncompatibleVersion {
        /// Producer major version.
        major: u32,
        /// Producer minor version.
        minor: u32,
    },
    /// Rank is negative or exceeds the defensive limit.
    #[error("invalid DLPack tensor rank {0}")]
    InvalidRank(i32),
    /// A ranked tensor has no shape pointer.
    #[error("DLPack tensor shape pointer is null for non-zero rank")]
    NullShape,
    /// A dimension is negative or cannot be represented by `usize`.
    #[error("invalid DLPack shape dimension {0}")]
    InvalidDimension(i64),
    /// The data type code is unsupported or not byte-addressable.
    #[error("unsupported DLPack dtype code={code}, bits={bits}, lanes={lanes}")]
    UnsupportedDataType {
        /// Raw DLPack code.
        code: u8,
        /// Bits per lane.
        bits: u8,
        /// Vector lanes.
        lanes: u16,
    },
    /// The device code is unknown to this DLPack minor implementation.
    #[error("unsupported DLPack device type {0}")]
    UnsupportedDevice(i32),
    /// A non-empty host tensor has no data pointer.
    #[error("non-empty DLPack host tensor has a null data pointer")]
    NullData,
    /// A DLPack integer does not fit the host representation.
    #[error("DLPack metadata does not fit the host integer representation")]
    IntegerConversion,
    /// The decoded tensor layout is invalid.
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawVersion {
    major: u32,
    minor: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct RawTensor {
    data: *mut c_void,
    device: RawDevice,
    ndim: i32,
    dtype: RawDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct RawManagedTensorVersioned {
    version: RawVersion,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut RawManagedTensorVersioned)>,
    flags: u64,
    dl_tensor: RawTensor,
}

struct ExportContext {
    _allocation: TensorStorage,
    _shape: Box<[i64]>,
    _strides: Box<[i64]>,
}

/// Owner for a versioned DLPack managed tensor exported without copying CPU data.
///
/// Dropping this value calls the DLPack deleter. [`Self::into_raw`] transfers
/// that responsibility to a capsule or another consumer.
pub struct DlpackExport {
    raw: NonNull<RawManagedTensorVersioned>,
}

/// Calls the producer deleter for a raw pointer previously returned by
/// [`DlpackExport::into_raw`].
///
/// # Safety
///
/// `raw` must still carry exclusive deleter responsibility for a live
/// `DLManagedTensorVersioned*`. It must not be used after this call.
pub unsafe fn release_dlpack_raw(raw: *mut c_void) {
    // SAFETY: forwarded from the public ownership contract above.
    unsafe { call_deleter(raw.cast()) };
}

impl DlpackExport {
    /// Shares an owned CPU tensor allocation with a DLPack consumer without copying.
    pub fn from_tensor(tensor: &TensorBuffer) -> Result<Self, DlpackError> {
        let descriptor = tensor.descriptor();
        if !descriptor.device().is_host_accessible() {
            return Err(TensorError::DeviceNotHostAccessible(descriptor.device()).into());
        }
        let shape = descriptor
            .shape()
            .iter()
            .map(|&dimension| i64::try_from(dimension).map_err(|_| DlpackError::IntegerConversion))
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        let strides = match descriptor.strides() {
            Some(values) => values
                .iter()
                .map(|&stride| i64::try_from(stride).map_err(|_| DlpackError::IntegerConversion))
                .collect::<Result<Vec<_>, _>>()?,
            None => compact_strides(descriptor.shape())?,
        }
        .into_boxed_slice();
        let ndim =
            i32::try_from(descriptor.shape().len()).map_err(|_| DlpackError::IntegerConversion)?;
        let byte_offset =
            u64::try_from(descriptor.byte_offset()).map_err(|_| DlpackError::IntegerConversion)?;
        let allocation = tensor.shared_allocation();
        let data = if allocation.is_empty() {
            ptr::null_mut()
        } else {
            allocation.as_ptr().cast_mut().cast()
        };
        let shape_ptr = if shape.is_empty() { ptr::null_mut() } else { shape.as_ptr().cast_mut() };
        let strides_ptr =
            if strides.is_empty() { ptr::null_mut() } else { strides.as_ptr().cast_mut() };
        let context =
            Box::new(ExportContext { _allocation: allocation, _shape: shape, _strides: strides });
        let manager_ctx = Box::into_raw(context).cast();
        let raw = Box::new(RawManagedTensorVersioned {
            version: RawVersion { major: DLPACK_MAJOR, minor: DLPACK_MINOR },
            manager_ctx,
            deleter: Some(delete_export),
            flags: FLAG_READ_ONLY,
            dl_tensor: RawTensor {
                data,
                device: encode_device(descriptor.device()),
                ndim,
                dtype: encode_dtype(descriptor.dtype()),
                shape: shape_ptr,
                strides: strides_ptr,
                byte_offset,
            },
        });
        Ok(Self { raw: NonNull::from(Box::leak(raw)) })
    }

    /// Returns the opaque managed-tensor pointer without transferring ownership.
    pub fn as_raw(&self) -> *mut c_void {
        self.raw.as_ptr().cast()
    }

    /// Transfers deleter responsibility to an external DLPack consumer.
    pub fn into_raw(self) -> *mut c_void {
        let this = ManuallyDrop::new(self);
        this.raw.as_ptr().cast()
    }
}

impl Drop for DlpackExport {
    fn drop(&mut self) {
        // SAFETY: `DlpackExport` uniquely owns deleter responsibility until `into_raw`.
        unsafe { call_deleter(self.raw.as_ptr()) };
    }
}

/// Validated owner of a DLPack producer's managed tensor.
#[derive(Debug)]
pub struct DlpackImport {
    raw: NonNull<RawManagedTensorVersioned>,
    descriptor: TensorDescriptor,
    allocation_len: usize,
    data: *const u8,
    version: (u32, u32),
    flags: u64,
}

impl DlpackImport {
    /// Takes deleter ownership of a DLPack managed tensor and validates its host view.
    ///
    /// # Safety
    ///
    /// `raw` must be a live, exclusively transferred `DLManagedTensorVersioned*`
    /// produced according to DLPack. The caller must not use or delete it after
    /// this call, whether validation succeeds or fails.
    pub unsafe fn from_raw(raw: *mut c_void) -> Result<Self, DlpackError> {
        let raw = NonNull::new(raw.cast::<RawManagedTensorVersioned>())
            .ok_or(DlpackError::NullManagedTensor)?;
        let guard = IncomingGuard { raw: Some(raw) };

        // SAFETY: the caller promises a live versioned header. DLPack guarantees
        // version and deleter positions remain accessible across major mismatch.
        let managed = unsafe { raw.as_ref() };
        let version = (managed.version.major, managed.version.minor);
        if version.0 != DLPACK_MAJOR {
            return Err(DlpackError::IncompatibleVersion { major: version.0, minor: version.1 });
        }
        let tensor = &managed.dl_tensor;
        if tensor.ndim < 0 || tensor.ndim as usize > MAX_RANK {
            return Err(DlpackError::InvalidRank(tensor.ndim));
        }
        let rank = tensor.ndim as usize;
        if rank != 0 && tensor.shape.is_null() {
            return Err(DlpackError::NullShape);
        }
        let shape_values = if rank == 0 {
            &[][..]
        } else {
            // SAFETY: producer contract supplies `ndim` readable shape entries.
            unsafe { slice::from_raw_parts(tensor.shape, rank) }
        };
        let shape = shape_values
            .iter()
            .map(|&dimension| {
                usize::try_from(dimension).map_err(|_| DlpackError::InvalidDimension(dimension))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let strides = if tensor.strides.is_null() {
            None
        } else {
            // SAFETY: producer contract supplies `ndim` readable stride entries.
            let values = unsafe { slice::from_raw_parts(tensor.strides, rank) };
            Some(
                values
                    .iter()
                    .map(|&stride| {
                        isize::try_from(stride).map_err(|_| DlpackError::IntegerConversion)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        };
        let dtype = decode_dtype(tensor.dtype)?;
        let device = decode_device(tensor.device)?;
        let byte_offset =
            usize::try_from(tensor.byte_offset).map_err(|_| DlpackError::IntegerConversion)?;
        let descriptor = match strides {
            Some(strides) => {
                TensorDescriptor::try_strided(dtype, shape, strides, byte_offset, device)?
            }
            None => {
                let mut descriptor = TensorDescriptor::contiguous(dtype, shape, device);
                if byte_offset != 0 {
                    descriptor = TensorDescriptor::try_strided(
                        dtype,
                        descriptor.shape().to_vec(),
                        compact_strides_isize(descriptor.shape())?,
                        byte_offset,
                        device,
                    )?;
                }
                descriptor
            }
        };
        let range = descriptor.required_byte_range()?;
        if range.end != 0 && tensor.data.is_null() {
            return Err(DlpackError::NullData);
        }
        let raw = guard.disarm();
        Ok(Self {
            raw,
            descriptor,
            allocation_len: range.end,
            data: tensor.data.cast(),
            version,
            flags: managed.flags,
        })
    }

    /// Returns producer ABI major and minor versions.
    pub const fn version(&self) -> (u32, u32) {
        self.version
    }

    /// Returns raw DLPack flags.
    pub const fn flags(&self) -> u64 {
        self.flags
    }

    /// Returns validated tensor metadata.
    pub const fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    /// Borrows the imported host allocation without copying it.
    pub fn view(&self) -> Result<TensorView<'_>, DlpackError> {
        let bytes = if self.allocation_len == 0 {
            &[][..]
        } else {
            // SAFETY: `from_raw` validated the DLPack producer contract and the
            // managed tensor remains alive until this owner is dropped.
            unsafe { slice::from_raw_parts(self.data, self.allocation_len) }
        };
        Ok(TensorView::try_new(bytes, self.descriptor.clone())?)
    }
}

impl Drop for DlpackImport {
    fn drop(&mut self) {
        // SAFETY: this owner received exclusive deleter responsibility in `from_raw`.
        unsafe { call_deleter(self.raw.as_ptr()) };
    }
}

struct IncomingGuard {
    raw: Option<NonNull<RawManagedTensorVersioned>>,
}

impl IncomingGuard {
    fn disarm(mut self) -> NonNull<RawManagedTensorVersioned> {
        self.raw.take().expect("incoming pointer is present")
    }
}

impl Drop for IncomingGuard {
    fn drop(&mut self) {
        if let Some(raw) = self.raw {
            // SAFETY: the guard owns the transferred pointer on all error paths.
            unsafe { call_deleter(raw.as_ptr()) };
        }
    }
}

unsafe extern "C" fn delete_export(raw: *mut RawManagedTensorVersioned) {
    if raw.is_null() {
        return;
    }
    // SAFETY: this function is installed only for allocations built by
    // `DlpackExport::from_tensor` and is called exactly once by ownership contract.
    let managed = unsafe { Box::from_raw(raw) };
    if !managed.manager_ctx.is_null() {
        // SAFETY: manager_ctx was created with Box::into_raw for ExportContext.
        drop(unsafe { Box::from_raw(managed.manager_ctx.cast::<ExportContext>()) });
    }
}

unsafe fn call_deleter(raw: *mut RawManagedTensorVersioned) {
    if raw.is_null() {
        return;
    }
    // SAFETY: caller owns a live managed-tensor pointer.
    if let Some(deleter) = unsafe { (*raw).deleter } {
        // SAFETY: deleter belongs to this exact managed tensor.
        unsafe { deleter(raw) };
    }
}

fn compact_strides(shape: &[usize]) -> Result<Vec<i64>, DlpackError> {
    let mut output = vec![0; shape.len()];
    let mut stride = 1_i64;
    for (index, &dimension) in shape.iter().enumerate().rev() {
        output[index] = stride;
        stride = stride
            .checked_mul(
                i64::try_from(dimension.max(1)).map_err(|_| DlpackError::IntegerConversion)?,
            )
            .ok_or(DlpackError::IntegerConversion)?;
    }
    Ok(output)
}

fn compact_strides_isize(shape: &[usize]) -> Result<Vec<isize>, DlpackError> {
    compact_strides(shape)?
        .into_iter()
        .map(|stride| isize::try_from(stride).map_err(|_| DlpackError::IntegerConversion))
        .collect()
}

fn encode_dtype(dtype: DataType) -> RawDataType {
    RawDataType { code: dtype.code() as u8, bits: dtype.bits(), lanes: dtype.lanes() }
}

fn decode_dtype(raw: RawDataType) -> Result<DataType, DlpackError> {
    let code = match raw.code {
        0 => DataTypeCode::Int,
        1 => DataTypeCode::UInt,
        2 => DataTypeCode::Float,
        4 => DataTypeCode::BFloat,
        5 => DataTypeCode::Complex,
        6 => DataTypeCode::Bool,
        _ => {
            return Err(DlpackError::UnsupportedDataType {
                code: raw.code,
                bits: raw.bits,
                lanes: raw.lanes,
            })
        }
    };
    DataType::try_new(code, raw.bits, raw.lanes).map_err(|_| DlpackError::UnsupportedDataType {
        code: raw.code,
        bits: raw.bits,
        lanes: raw.lanes,
    })
}

fn encode_device(device: Device) -> RawDevice {
    let device_type = match device.kind {
        DeviceKind::Cpu => 1,
        DeviceKind::Cuda => 2,
        DeviceKind::CudaHost => 3,
        DeviceKind::OpenCl => 4,
        DeviceKind::Vulkan => 7,
        DeviceKind::Metal => 8,
        DeviceKind::Vpi => 9,
        DeviceKind::Rocm => 10,
        DeviceKind::RocmHost => 11,
        DeviceKind::External => 12,
        DeviceKind::CudaManaged => 13,
        DeviceKind::OneApi => 14,
        DeviceKind::WebGpu => 15,
        DeviceKind::Hexagon => 16,
        DeviceKind::Maia => 17,
        DeviceKind::Trainium => 18,
        DeviceKind::Tpu => 19,
        DeviceKind::TpuHost => 20,
    };
    RawDevice { device_type, device_id: device.id }
}

fn decode_device(raw: RawDevice) -> Result<Device, DlpackError> {
    let kind = match raw.device_type {
        1 => DeviceKind::Cpu,
        2 => DeviceKind::Cuda,
        3 => DeviceKind::CudaHost,
        4 => DeviceKind::OpenCl,
        7 => DeviceKind::Vulkan,
        8 => DeviceKind::Metal,
        9 => DeviceKind::Vpi,
        10 => DeviceKind::Rocm,
        11 => DeviceKind::RocmHost,
        12 => DeviceKind::External,
        13 => DeviceKind::CudaManaged,
        14 => DeviceKind::OneApi,
        15 => DeviceKind::WebGpu,
        16 => DeviceKind::Hexagon,
        17 => DeviceKind::Maia,
        18 => DeviceKind::Trainium,
        19 => DeviceKind::Tpu,
        20 => DeviceKind::TpuHost,
        other => return Err(DlpackError::UnsupportedDevice(other)),
    };
    Ok(Device { kind, id: raw.device_id })
}

#[cfg(test)]
mod tests {
    use super::{DlpackError, DlpackExport, DlpackImport, DLPACK_MAJOR, DLPACK_MINOR};
    use crate::{DataType, Device, TensorBuffer, TensorDescriptor};

    #[test]
    fn owned_cpu_roundtrip_is_zero_copy_and_versioned() {
        let tensor = TensorBuffer::try_new(
            (0_u8..24).collect(),
            TensorDescriptor::contiguous(DataType::F32, vec![2, 3], Device::CPU),
        )
        .unwrap();
        let original = tensor.allocation_bytes().as_ptr();
        let allocation = tensor.shared_allocation();
        let export = DlpackExport::from_tensor(&tensor).unwrap();
        assert_eq!(allocation.strong_count(), 3);
        // SAFETY: into_raw transfers the live export exactly once.
        let imported = unsafe { DlpackImport::from_raw(export.into_raw()) }.unwrap();
        assert_eq!(imported.version(), (DLPACK_MAJOR, DLPACK_MINOR));
        let view = imported.view().unwrap();
        assert_eq!(view.descriptor().shape(), &[2, 3]);
        assert_eq!(view.allocation_bytes().as_ptr(), original);
        drop(imported);
        assert_eq!(allocation.strong_count(), 2);
    }

    #[test]
    fn negative_stride_and_byte_offset_roundtrip() {
        let descriptor =
            TensorDescriptor::try_strided(DataType::U8, vec![4], vec![-1], 3, Device::CPU).unwrap();
        let tensor = TensorBuffer::try_new(vec![10, 20, 30, 40], descriptor).unwrap();
        let export = DlpackExport::from_tensor(&tensor).unwrap();
        // SAFETY: the export pointer is transferred exactly once.
        let imported = unsafe { DlpackImport::from_raw(export.into_raw()) }.unwrap();
        let view = imported.view().unwrap();
        assert_eq!(view.descriptor().strides(), Some(&[-1][..]));
        assert_eq!(view.descriptor().byte_offset(), 3);
        assert_eq!(view.allocation_bytes(), &[10, 20, 30, 40]);
    }

    #[test]
    fn major_mismatch_is_rejected_and_deleted() {
        let tensor = TensorBuffer::try_new(
            vec![1],
            TensorDescriptor::contiguous(DataType::U8, vec![1], Device::CPU),
        )
        .unwrap();
        let allocation = tensor.shared_allocation();
        let export = DlpackExport::from_tensor(&tensor).unwrap();
        let raw = export.into_raw().cast::<super::RawManagedTensorVersioned>();
        // SAFETY: test exclusively owns this live export and only changes its version header.
        unsafe { (*raw).version.major = 99 };
        // SAFETY: the pointer is still live and transferred exactly once.
        let error = unsafe { DlpackImport::from_raw(raw.cast()) }.unwrap_err();
        assert!(matches!(error, DlpackError::IncompatibleVersion { major: 99, .. }));
        assert_eq!(allocation.strong_count(), 2);
    }

    #[test]
    fn malformed_dtype_is_rejected_and_deleted() {
        let tensor = TensorBuffer::try_new(
            vec![1],
            TensorDescriptor::contiguous(DataType::U8, vec![1], Device::CPU),
        )
        .unwrap();
        let allocation = tensor.shared_allocation();
        let raw = DlpackExport::from_tensor(&tensor)
            .unwrap()
            .into_raw()
            .cast::<super::RawManagedTensorVersioned>();
        // SAFETY: test owns the live export and corrupts only metadata under test.
        unsafe { (*raw).dl_tensor.dtype.code = 255 };
        // SAFETY: the corrupted but live pointer is transferred exactly once.
        let error = unsafe { DlpackImport::from_raw(raw.cast()) }.unwrap_err();
        assert!(matches!(error, DlpackError::UnsupportedDataType { code: 255, .. }));
        assert_eq!(allocation.strong_count(), 2);
    }

    #[test]
    fn null_nonempty_data_is_rejected_and_deleted() {
        let tensor = TensorBuffer::try_new(
            vec![1, 2],
            TensorDescriptor::contiguous(DataType::U8, vec![2], Device::CPU),
        )
        .unwrap();
        let allocation = tensor.shared_allocation();
        let raw = DlpackExport::from_tensor(&tensor)
            .unwrap()
            .into_raw()
            .cast::<super::RawManagedTensorVersioned>();
        // SAFETY: test owns the live export and corrupts only metadata under test.
        unsafe { (*raw).dl_tensor.data = std::ptr::null_mut() };
        // SAFETY: the corrupted but live pointer is transferred exactly once.
        let error = unsafe { DlpackImport::from_raw(raw.cast()) }.unwrap_err();
        assert!(matches!(error, DlpackError::NullData));
        assert_eq!(allocation.strong_count(), 2);
    }

    #[test]
    fn negative_shape_is_rejected_and_deleted() {
        let tensor = TensorBuffer::try_new(
            vec![1],
            TensorDescriptor::contiguous(DataType::U8, vec![1], Device::CPU),
        )
        .unwrap();
        let allocation = tensor.shared_allocation();
        let raw = DlpackExport::from_tensor(&tensor)
            .unwrap()
            .into_raw()
            .cast::<super::RawManagedTensorVersioned>();
        // SAFETY: exported rank is one and shape points to one writable context entry.
        unsafe { *(*raw).dl_tensor.shape = -1 };
        // SAFETY: the corrupted but live pointer is transferred exactly once.
        let error = unsafe { DlpackImport::from_raw(raw.cast()) }.unwrap_err();
        assert!(matches!(error, DlpackError::InvalidDimension(-1)));
        assert_eq!(allocation.strong_count(), 2);
    }
}
