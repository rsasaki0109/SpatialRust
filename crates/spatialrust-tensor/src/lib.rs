//! Small, runtime-independent tensor descriptors and CPU storage views.
//!
//! Shape, element strides, byte offset, device, and ownership are explicit.
//! This crate never uploads, downloads, or otherwise migrates storage. DLPack
//! FFI and image bridges are additive features built on this data model.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use std::{any::Any, fmt::Debug, ops::Range, sync::Arc};

#[cfg(feature = "image")]
mod image;
#[cfg(feature = "image")]
pub use image::{
    interleaved_image_view, pack_interleaved_image, pack_planar_image, planar_image_view,
    TensorElement,
};
#[cfg(feature = "spatial")]
mod spatial;
#[cfg(feature = "spatial")]
pub use spatial::{spatial_f32_field_view, SpatialTensorBridgeError};
#[cfg(feature = "dlpack")]
#[allow(unsafe_code)]
mod dlpack;
#[cfg(feature = "dlpack")]
pub use dlpack::{
    release_dlpack_legacy_raw, release_dlpack_raw, DlpackError, DlpackExport, DlpackImport,
    DlpackLegacyExport, DLPACK_MAJOR, DLPACK_MINOR,
};

/// Tensor construction and layout errors.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TensorError {
    /// The data type has no bits or lanes, or cannot address whole bytes.
    #[error("invalid data type: bits and lanes must be non-zero and form whole bytes")]
    InvalidDataType,
    /// The stride rank differs from the shape rank.
    #[error("stride rank {strides} differs from shape rank {shape}")]
    RankMismatch {
        /// Number of shape dimensions.
        shape: usize,
        /// Number of strides.
        strides: usize,
    },
    /// Shape, stride, or byte-range arithmetic overflowed.
    #[error("tensor layout overflows addressable memory")]
    LayoutOverflow,
    /// The byte offset or reachable tensor span lies outside storage.
    #[error("tensor byte range {start}..{end} lies outside {available} bytes")]
    StorageOutOfBounds {
        /// First reachable byte.
        start: i128,
        /// Exclusive final reachable byte.
        end: i128,
        /// Available storage size.
        available: usize,
    },
    /// A host slice was paired with a device that is not host-accessible.
    #[error("device {0:?} is not directly host-accessible")]
    DeviceNotHostAccessible(Device),
    /// Typed storage does not match its descriptor dtype.
    #[error("typed storage requires {expected:?}, descriptor declares {actual:?}")]
    DataTypeMismatch {
        /// Storage dtype.
        expected: DataType,
        /// Descriptor dtype.
        actual: DataType,
    },
}

/// DLPack-compatible scalar category without depending on a runtime header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DataTypeCode {
    /// Signed integer.
    Int = 0,
    /// Unsigned integer.
    UInt = 1,
    /// IEEE floating point.
    Float = 2,
    /// Brain floating point.
    BFloat = 4,
    /// Complex floating point.
    Complex = 5,
    /// Boolean value.
    Bool = 6,
}

/// Scalar category, bit width, and vector lane count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DataType {
    code: DataTypeCode,
    bits: u8,
    lanes: u16,
}

impl DataType {
    /// Unsigned 8-bit scalar.
    pub const U8: Self = Self::new_unchecked(DataTypeCode::UInt, 8, 1);
    /// Unsigned 16-bit scalar.
    pub const U16: Self = Self::new_unchecked(DataTypeCode::UInt, 16, 1);
    /// Unsigned 32-bit scalar.
    pub const U32: Self = Self::new_unchecked(DataTypeCode::UInt, 32, 1);
    /// Signed 8-bit scalar.
    pub const I8: Self = Self::new_unchecked(DataTypeCode::Int, 8, 1);
    /// Signed 16-bit scalar.
    pub const I16: Self = Self::new_unchecked(DataTypeCode::Int, 16, 1);
    /// Signed 32-bit scalar.
    pub const I32: Self = Self::new_unchecked(DataTypeCode::Int, 32, 1);
    /// Signed 64-bit scalar.
    pub const I64: Self = Self::new_unchecked(DataTypeCode::Int, 64, 1);
    /// IEEE binary16 scalar.
    pub const F16: Self = Self::new_unchecked(DataTypeCode::Float, 16, 1);
    /// Brain floating-point 16-bit scalar.
    pub const BF16: Self = Self::new_unchecked(DataTypeCode::BFloat, 16, 1);
    /// IEEE binary32 scalar.
    pub const F32: Self = Self::new_unchecked(DataTypeCode::Float, 32, 1);
    /// IEEE binary64 scalar.
    pub const F64: Self = Self::new_unchecked(DataTypeCode::Float, 64, 1);
    /// Eight-bit boolean scalar, matching common DLPack producers.
    pub const BOOL: Self = Self::new_unchecked(DataTypeCode::Bool, 8, 1);

    const fn new_unchecked(code: DataTypeCode, bits: u8, lanes: u16) -> Self {
        Self { code, bits, lanes }
    }

    /// Creates a byte-addressable scalar or vector data type.
    pub fn try_new(code: DataTypeCode, bits: u8, lanes: u16) -> Result<Self, TensorError> {
        let total_bits = usize::from(bits)
            .checked_mul(usize::from(lanes))
            .ok_or(TensorError::InvalidDataType)?;
        if bits == 0 || lanes == 0 || total_bits % 8 != 0 {
            return Err(TensorError::InvalidDataType);
        }
        Ok(Self { code, bits, lanes })
    }

    /// Returns the scalar category.
    pub const fn code(self) -> DataTypeCode {
        self.code
    }

    /// Returns bits per lane.
    pub const fn bits(self) -> u8 {
        self.bits
    }

    /// Returns the vector lane count.
    pub const fn lanes(self) -> u16 {
        self.lanes
    }

    /// Returns bytes occupied by one tensor element.
    pub fn element_size(self) -> usize {
        usize::from(self.bits) * usize::from(self.lanes) / 8
    }
}

/// Device category represented independently of any execution backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    /// Ordinary CPU memory.
    Cpu,
    /// CUDA device memory.
    Cuda,
    /// CUDA-pinned host memory.
    CudaHost,
    /// OpenCL device memory.
    OpenCl,
    /// Vulkan device memory.
    Vulkan,
    /// Metal device memory.
    Metal,
    /// Verilog simulator buffer.
    Vpi,
    /// ROCm device memory.
    Rocm,
    /// ROCm-pinned host memory.
    RocmHost,
    /// Backend-specific external memory.
    External,
    /// CUDA managed memory.
    CudaManaged,
    /// oneAPI device memory.
    OneApi,
    /// WebGPU device memory.
    WebGpu,
    /// Hexagon device memory.
    Hexagon,
    /// Microsoft MAIA device memory.
    Maia,
    /// AWS Trainium device memory.
    Trainium,
    /// Google TPU device memory.
    Tpu,
    /// Google TPU pinned host memory.
    TpuHost,
}

/// Device category and backend-local ordinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Device {
    /// Device category.
    pub kind: DeviceKind,
    /// Backend-local device ordinal.
    pub id: i32,
}

impl Device {
    /// Main CPU device.
    pub const CPU: Self = Self { kind: DeviceKind::Cpu, id: 0 };

    /// Returns whether Rust may safely expose this memory as a host byte slice.
    pub const fn is_host_accessible(self) -> bool {
        matches!(
            self.kind,
            DeviceKind::Cpu | DeviceKind::CudaHost | DeviceKind::RocmHost | DeviceKind::TpuHost
        )
    }
}

/// Whether a tensor object owns or borrows its backing allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ownership {
    /// The object owns and drops its storage.
    Owned,
    /// The object cannot outlive borrowed storage.
    Borrowed,
}

/// Shape, element strides, type, offset, and device for one tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDescriptor {
    dtype: DataType,
    shape: Vec<usize>,
    strides: Option<Vec<isize>>,
    byte_offset: usize,
    device: Device,
}

impl TensorDescriptor {
    /// Creates a compact C-order tensor on a device.
    pub fn contiguous(dtype: DataType, shape: Vec<usize>, device: Device) -> Self {
        Self { dtype, shape, strides: None, byte_offset: 0, device }
    }

    /// Creates an explicitly strided tensor. Strides are measured in elements.
    pub fn try_strided(
        dtype: DataType,
        shape: Vec<usize>,
        strides: Vec<isize>,
        byte_offset: usize,
        device: Device,
    ) -> Result<Self, TensorError> {
        if shape.len() != strides.len() {
            return Err(TensorError::RankMismatch { shape: shape.len(), strides: strides.len() });
        }
        let descriptor = Self { dtype, shape, strides: Some(strides), byte_offset, device };
        descriptor.required_byte_range()?;
        Ok(descriptor)
    }

    /// Returns the data type.
    pub const fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Returns dimensions in logical axis order.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns element strides, or `None` for compact C order.
    pub fn strides(&self) -> Option<&[isize]> {
        self.strides.as_deref()
    }

    /// Returns the byte offset from the allocation start to logical element zero.
    pub const fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// Returns the storage device.
    pub const fn device(&self) -> Device {
        self.device
    }

    /// Returns the number of logical elements, including one for a scalar.
    pub fn element_count(&self) -> Result<usize, TensorError> {
        self.shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
            .ok_or(TensorError::LayoutOverflow)
    }

    /// Returns whether the tensor is compact in C order.
    pub fn is_c_contiguous(&self) -> bool {
        let Some(strides) = &self.strides else { return true };
        let mut expected = 1usize;
        for (&dimension, &stride) in self.shape.iter().zip(strides).rev() {
            if dimension > 1 && usize::try_from(stride).ok() != Some(expected) {
                return false;
            }
            let Some(next) = expected.checked_mul(dimension.max(1)) else { return false };
            expected = next;
        }
        true
    }

    /// Computes the smallest allocation byte range reachable by the layout.
    pub fn required_byte_range(&self) -> Result<Range<usize>, TensorError> {
        if self.element_count()? == 0 {
            return Ok(self.byte_offset..self.byte_offset);
        }
        let item_size =
            i128::try_from(self.dtype.element_size()).map_err(|_| TensorError::LayoutOverflow)?;
        let origin = i128::try_from(self.byte_offset).map_err(|_| TensorError::LayoutOverflow)?;
        let mut minimum = origin;
        let mut maximum = origin;
        if let Some(strides) = &self.strides {
            for (&dimension, &stride) in self.shape.iter().zip(strides) {
                let steps = i128::try_from(dimension.saturating_sub(1))
                    .map_err(|_| TensorError::LayoutOverflow)?;
                let stride_bytes =
                    (stride as i128).checked_mul(item_size).ok_or(TensorError::LayoutOverflow)?;
                let delta = steps.checked_mul(stride_bytes).ok_or(TensorError::LayoutOverflow)?;
                if delta < 0 {
                    minimum = minimum.checked_add(delta).ok_or(TensorError::LayoutOverflow)?;
                } else {
                    maximum = maximum.checked_add(delta).ok_or(TensorError::LayoutOverflow)?;
                }
            }
        } else {
            let count =
                i128::try_from(self.element_count()?).map_err(|_| TensorError::LayoutOverflow)?;
            maximum = maximum
                .checked_add((count - 1).checked_mul(item_size).ok_or(TensorError::LayoutOverflow)?)
                .ok_or(TensorError::LayoutOverflow)?;
        }
        let end = maximum.checked_add(item_size).ok_or(TensorError::LayoutOverflow)?;
        let start = usize::try_from(minimum).map_err(|_| TensorError::StorageOutOfBounds {
            start: minimum,
            end,
            available: 0,
        })?;
        let end = usize::try_from(end).map_err(|_| TensorError::LayoutOverflow)?;
        Ok(start..end)
    }

    fn validate_storage(&self, available: usize) -> Result<(), TensorError> {
        if !self.device.is_host_accessible() {
            return Err(TensorError::DeviceNotHostAccessible(self.device));
        }
        let range = self.required_byte_range()?;
        if range.start > available || range.end > available {
            return Err(TensorError::StorageOutOfBounds {
                start: range.start as i128,
                end: range.end as i128,
                available,
            });
        }
        Ok(())
    }
}

/// Lifetime-bound, zero-copy view of host-accessible tensor storage.
#[derive(Clone, Debug)]
pub struct TensorView<'a> {
    bytes: &'a [u8],
    descriptor: TensorDescriptor,
}

impl<'a> TensorView<'a> {
    /// Validates and borrows a host allocation without copying it.
    pub fn try_new(bytes: &'a [u8], descriptor: TensorDescriptor) -> Result<Self, TensorError> {
        descriptor.validate_storage(bytes.len())?;
        Ok(Self { bytes, descriptor })
    }

    /// Returns the complete borrowed allocation slice.
    pub const fn allocation_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Returns tensor metadata.
    pub const fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    /// Reports borrowed ownership.
    pub const fn ownership(&self) -> Ownership {
        Ownership::Borrowed
    }

    /// Performs an explicit host-to-host copy retaining the same layout.
    pub fn to_owned_copy(&self) -> TensorBuffer {
        TensorBuffer {
            storage: copy_storage(self.bytes, self.descriptor.dtype()),
            descriptor: self.descriptor.clone(),
        }
    }
}

/// Owned host allocation paired with a validated tensor descriptor.
#[derive(Clone, Debug)]
pub struct TensorBuffer {
    storage: TensorStorage,
    descriptor: TensorDescriptor,
}

#[derive(Clone, Debug)]
pub(crate) enum TensorStorage {
    Bytes(Arc<[u8]>),
    U16(Arc<[u16]>),
    I16(Arc<[i16]>),
    U32(Arc<[u32]>),
    I32(Arc<[i32]>),
    I64(Arc<[i64]>),
    F32(Arc<[f32]>),
    F64(Arc<[f64]>),
    External(Arc<dyn HostTensorStorage>),
}

/// Runtime-owned, host-accessible storage retained without a runtime dependency.
///
/// Backend crates use this boundary to keep an allocator or runtime value alive
/// while exposing its stable CPU allocation. Implementations must keep the
/// returned allocation address and length unchanged for their entire lifetime.
pub trait HostTensorStorage: Any + Debug + Send + Sync {
    /// Returns the exact tensor element type stored by this allocation.
    fn dtype(&self) -> DataType;

    /// Returns the complete host-accessible allocation.
    fn allocation_bytes(&self) -> &[u8];

    /// Supports backend-specific zero-copy reuse through checked downcasting.
    fn as_any(&self) -> &dyn Any;
}

impl TensorStorage {
    pub(crate) fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Bytes(values) => values,
            Self::U16(values) => bytemuck::cast_slice(values),
            Self::I16(values) => bytemuck::cast_slice(values),
            Self::U32(values) => bytemuck::cast_slice(values),
            Self::I32(values) => bytemuck::cast_slice(values),
            Self::I64(values) => bytemuck::cast_slice(values),
            Self::F32(values) => bytemuck::cast_slice(values),
            Self::F64(values) => bytemuck::cast_slice(values),
            Self::External(storage) => storage.allocation_bytes(),
        }
    }

    #[cfg(feature = "dlpack")]
    pub(crate) fn is_empty(&self) -> bool {
        self.as_bytes().is_empty()
    }

    #[cfg(feature = "dlpack")]
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.as_bytes().as_ptr()
    }

    #[cfg(all(feature = "dlpack", test))]
    pub(crate) fn strong_count(&self) -> usize {
        match self {
            Self::Bytes(values) => Arc::strong_count(values),
            Self::U16(values) => Arc::strong_count(values),
            Self::I16(values) => Arc::strong_count(values),
            Self::U32(values) => Arc::strong_count(values),
            Self::I32(values) => Arc::strong_count(values),
            Self::I64(values) => Arc::strong_count(values),
            Self::F32(values) => Arc::strong_count(values),
            Self::F64(values) => Arc::strong_count(values),
            Self::External(storage) => Arc::strong_count(storage),
        }
    }
}

fn copy_storage(bytes: &[u8], dtype: DataType) -> TensorStorage {
    macro_rules! aligned_copy {
        ($type:ty, $variant:ident) => {
            bytemuck::try_cast_slice::<u8, $type>(bytes)
                .map(|values| TensorStorage::$variant(Arc::from(values)))
                .unwrap_or_else(|_| TensorStorage::Bytes(Arc::from(bytes)))
        };
    }
    match dtype {
        DataType::U16 | DataType::F16 | DataType::BF16 => aligned_copy!(u16, U16),
        DataType::I16 => aligned_copy!(i16, I16),
        DataType::U32 => aligned_copy!(u32, U32),
        DataType::I32 => aligned_copy!(i32, I32),
        DataType::I64 => aligned_copy!(i64, I64),
        DataType::F32 => aligned_copy!(f32, F32),
        DataType::F64 => aligned_copy!(f64, F64),
        _ => TensorStorage::Bytes(Arc::from(bytes)),
    }
}

impl PartialEq for TensorBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor && self.allocation_bytes() == other.allocation_bytes()
    }
}

impl Eq for TensorBuffer {}

impl TensorBuffer {
    /// Validates and takes ownership of a host allocation.
    pub fn try_new(bytes: Vec<u8>, descriptor: TensorDescriptor) -> Result<Self, TensorError> {
        descriptor.validate_storage(bytes.len())?;
        Ok(Self { storage: TensorStorage::Bytes(Arc::from(bytes)), descriptor })
    }

    /// Validates and owns aligned u16 storage without byte repacking.
    pub fn try_from_u16(
        values: Vec<u16>,
        descriptor: TensorDescriptor,
    ) -> Result<Self, TensorError> {
        Self::try_from_storage(TensorStorage::U16(Arc::from(values)), DataType::U16, descriptor)
    }

    /// Validates and owns aligned i16 storage without byte repacking.
    pub fn try_from_i16(
        values: Vec<i16>,
        descriptor: TensorDescriptor,
    ) -> Result<Self, TensorError> {
        Self::try_from_storage(TensorStorage::I16(Arc::from(values)), DataType::I16, descriptor)
    }

    /// Validates and owns aligned u32 storage without byte repacking.
    pub fn try_from_u32(
        values: Vec<u32>,
        descriptor: TensorDescriptor,
    ) -> Result<Self, TensorError> {
        Self::try_from_storage(TensorStorage::U32(Arc::from(values)), DataType::U32, descriptor)
    }

    /// Validates and owns aligned i32 storage without byte repacking.
    pub fn try_from_i32(
        values: Vec<i32>,
        descriptor: TensorDescriptor,
    ) -> Result<Self, TensorError> {
        Self::try_from_storage(TensorStorage::I32(Arc::from(values)), DataType::I32, descriptor)
    }

    /// Validates and owns aligned i64 storage without byte repacking.
    pub fn try_from_i64(
        values: Vec<i64>,
        descriptor: TensorDescriptor,
    ) -> Result<Self, TensorError> {
        Self::try_from_storage(TensorStorage::I64(Arc::from(values)), DataType::I64, descriptor)
    }

    /// Validates and owns aligned f32 storage without byte repacking.
    pub fn try_from_f32(
        values: Vec<f32>,
        descriptor: TensorDescriptor,
    ) -> Result<Self, TensorError> {
        Self::try_from_storage(TensorStorage::F32(Arc::from(values)), DataType::F32, descriptor)
    }

    /// Validates and owns aligned f64 storage without byte repacking.
    pub fn try_from_f64(
        values: Vec<f64>,
        descriptor: TensorDescriptor,
    ) -> Result<Self, TensorError> {
        Self::try_from_storage(TensorStorage::F64(Arc::from(values)), DataType::F64, descriptor)
    }

    /// Validates and retains a runtime-owned host allocation without copying it.
    pub fn try_from_host_storage(
        storage: Arc<dyn HostTensorStorage>,
        descriptor: TensorDescriptor,
    ) -> Result<Self, TensorError> {
        let dtype = storage.dtype();
        Self::try_from_storage(TensorStorage::External(storage), dtype, descriptor)
    }

    fn try_from_storage(
        storage: TensorStorage,
        expected: DataType,
        descriptor: TensorDescriptor,
    ) -> Result<Self, TensorError> {
        if descriptor.dtype() != expected {
            return Err(TensorError::DataTypeMismatch { expected, actual: descriptor.dtype() });
        }
        descriptor.validate_storage(storage.as_bytes().len())?;
        Ok(Self { storage, descriptor })
    }

    /// Returns a zero-copy borrowed view.
    pub fn view(&self) -> TensorView<'_> {
        TensorView { bytes: self.storage.as_bytes(), descriptor: self.descriptor.clone() }
    }

    /// Returns tensor metadata.
    pub const fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    /// Returns the complete owned allocation bytes.
    pub fn allocation_bytes(&self) -> &[u8] {
        self.storage.as_bytes()
    }

    /// Performs an explicit host-to-host copy while preserving typed alignment.
    pub fn to_owned_copy(&self) -> Self {
        Self {
            storage: copy_storage(self.storage.as_bytes(), self.descriptor.dtype()),
            descriptor: self.descriptor.clone(),
        }
    }

    #[cfg(feature = "dlpack")]
    pub(crate) fn shared_allocation(&self) -> TensorStorage {
        self.storage.clone()
    }

    /// Returns shared aligned f32 storage when constructed with [`Self::try_from_f32`].
    pub fn shared_f32(&self) -> Option<Arc<[f32]>> {
        match &self.storage {
            TensorStorage::F32(values) => Some(Arc::clone(values)),
            _ => None,
        }
    }

    /// Returns shared byte storage when constructed with [`Self::try_new`].
    ///
    /// This is suitable for zero-copy `u8` and `i8` tensor adapters. Multi-byte
    /// element types must use their matching typed constructor and accessor.
    pub fn shared_bytes(&self) -> Option<Arc<[u8]>> {
        match &self.storage {
            TensorStorage::Bytes(values) => Some(Arc::clone(values)),
            _ => None,
        }
    }

    /// Returns shared aligned f64 storage when constructed with [`Self::try_from_f64`].
    pub fn shared_f64(&self) -> Option<Arc<[f64]>> {
        match &self.storage {
            TensorStorage::F64(values) => Some(Arc::clone(values)),
            _ => None,
        }
    }

    /// Returns shared aligned i16 storage when constructed with [`Self::try_from_i16`].
    pub fn shared_i16(&self) -> Option<Arc<[i16]>> {
        match &self.storage {
            TensorStorage::I16(values) => Some(Arc::clone(values)),
            _ => None,
        }
    }

    /// Returns shared aligned i32 storage when constructed with [`Self::try_from_i32`].
    pub fn shared_i32(&self) -> Option<Arc<[i32]>> {
        match &self.storage {
            TensorStorage::I32(values) => Some(Arc::clone(values)),
            _ => None,
        }
    }

    /// Returns shared aligned i64 storage when constructed with [`Self::try_from_i64`].
    pub fn shared_i64(&self) -> Option<Arc<[i64]>> {
        match &self.storage {
            TensorStorage::I64(values) => Some(Arc::clone(values)),
            _ => None,
        }
    }

    /// Returns shared aligned u16 storage when constructed with [`Self::try_from_u16`].
    pub fn shared_u16(&self) -> Option<Arc<[u16]>> {
        match &self.storage {
            TensorStorage::U16(values) => Some(Arc::clone(values)),
            _ => None,
        }
    }

    /// Returns shared aligned u32 storage when constructed with [`Self::try_from_u32`].
    pub fn shared_u32(&self) -> Option<Arc<[u32]>> {
        match &self.storage {
            TensorStorage::U32(values) => Some(Arc::clone(values)),
            _ => None,
        }
    }

    /// Returns runtime-owned host storage, when this tensor wraps one.
    pub fn host_storage(&self) -> Option<&Arc<dyn HostTensorStorage>> {
        match &self.storage {
            TensorStorage::External(storage) => Some(storage),
            _ => None,
        }
    }

    /// Reports owned storage.
    pub const fn ownership(&self) -> Ownership {
        Ownership::Owned
    }

    /// Explicitly copies the allocation into bytes and returns its metadata.
    pub fn into_allocation_bytes_copy(self) -> (Vec<u8>, TensorDescriptor) {
        (self.storage.as_bytes().to_vec(), self.descriptor)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DataType, Device, DeviceKind, TensorBuffer, TensorDescriptor, TensorError, TensorView,
    };

    #[test]
    fn compact_tensor_validates_exact_storage() {
        let descriptor = TensorDescriptor::contiguous(DataType::F32, vec![2, 3], Device::CPU);
        let tensor = TensorBuffer::try_new(vec![0; 24], descriptor).unwrap();
        assert_eq!(tensor.descriptor().element_count().unwrap(), 6);
        assert!(tensor.descriptor().is_c_contiguous());
        assert_eq!(tensor.descriptor().required_byte_range().unwrap(), 0..24);
    }

    #[test]
    fn strided_roi_and_negative_stride_have_checked_spans() {
        let roi =
            TensorDescriptor::try_strided(DataType::U8, vec![2, 3], vec![5, 1], 6, Device::CPU)
                .unwrap();
        assert_eq!(roi.required_byte_range().unwrap(), 6..14);
        TensorView::try_new(&[0; 14], roi).unwrap();

        let reversed =
            TensorDescriptor::try_strided(DataType::U8, vec![4], vec![-1], 3, Device::CPU).unwrap();
        assert_eq!(reversed.required_byte_range().unwrap(), 0..4);
        TensorView::try_new(&[0; 4], reversed).unwrap();
    }

    #[test]
    fn rejects_short_storage_and_device_memory_as_host_slice() {
        let descriptor = TensorDescriptor::contiguous(DataType::U16, vec![4], Device::CPU);
        assert!(matches!(
            TensorView::try_new(&[0; 7], descriptor),
            Err(TensorError::StorageOutOfBounds { .. })
        ));
        let cuda = TensorDescriptor::contiguous(
            DataType::U8,
            vec![1],
            Device { kind: DeviceKind::Cuda, id: 0 },
        );
        assert!(matches!(
            TensorView::try_new(&[0], cuda),
            Err(TensorError::DeviceNotHostAccessible(_))
        ));
    }

    #[test]
    fn zero_sized_and_scalar_shapes_are_distinct() {
        let empty = TensorDescriptor::contiguous(DataType::U8, vec![2, 0, 3], Device::CPU);
        assert_eq!(empty.element_count().unwrap(), 0);
        assert_eq!(empty.required_byte_range().unwrap(), 0..0);
        let scalar = TensorDescriptor::contiguous(DataType::F64, vec![], Device::CPU);
        assert_eq!(scalar.element_count().unwrap(), 1);
        assert_eq!(scalar.required_byte_range().unwrap(), 0..8);
    }
}
