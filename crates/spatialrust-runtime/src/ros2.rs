//! ROS 2 adaptation contracts and CDR PointCloud2 codecs (no rclrs link).
//!
//! Native `rclrs` executors still require an installed ROS 2 toolchain and stay
//! outside this crate. Enabling `ros2` provides message negotiation, CDR LE
//! `sensor_msgs/msg/PointCloud2` XYZ codecs, and an in-process loopback node.

use crate::{RuntimeError, RuntimeResult};

/// Hint describing a ROS 2 message mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ros2MessageHint {
    /// Fully-qualified ROS type name, e.g. `sensor_msgs/msg/PointCloud2`.
    pub type_name: String,
    /// SpatialRust topic / schema id.
    pub spatial_topic: String,
}

/// Adapter interface for ROS 2 type negotiation.
pub trait Ros2Adapter {
    /// Returns supported type mappings.
    fn supported_types(&self) -> &[Ros2MessageHint];

    /// Negotiates a preferred mapping for one ROS type.
    fn negotiate(&self, ros_type: &str) -> Option<&Ros2MessageHint>;
}

/// In-memory catalog adapter used by default builds with `ros2` enabled.
#[derive(Clone, Debug, Default)]
pub struct CatalogRos2Adapter {
    hints: Vec<Ros2MessageHint>,
}

impl CatalogRos2Adapter {
    /// Creates an adapter from a catalog.
    #[must_use]
    pub fn new(hints: Vec<Ros2MessageHint>) -> Self {
        Self { hints }
    }

    /// Returns a catalog covering common XYZ point-cloud mappings.
    #[must_use]
    pub fn point_cloud2_xyz() -> Self {
        Self::new(vec![Ros2MessageHint {
            type_name: POINT_CLOUD2_TYPE.into(),
            spatial_topic: "point/xyz".into(),
        }])
    }
}

impl Ros2Adapter for CatalogRos2Adapter {
    fn supported_types(&self) -> &[Ros2MessageHint] {
        &self.hints
    }

    fn negotiate(&self, ros_type: &str) -> Option<&Ros2MessageHint> {
        self.hints.iter().find(|hint| hint.type_name == ros_type)
    }
}

/// Canonical ROS 2 type name for PointCloud2.
pub const POINT_CLOUD2_TYPE: &str = "sensor_msgs/msg/PointCloud2";

/// CDR encapsulation header for little-endian ROS 2 messages.
const CDR_LE_ENCAP: [u8; 4] = [0x00, 0x01, 0x00, 0x00];

/// Interleaved XYZ or XYZ-I PointCloud2 payload.
#[derive(Clone, Debug, PartialEq)]
pub struct PointCloud2Xyz {
    /// ROS frame id.
    pub frame_id: String,
    /// Header stamp seconds.
    pub stamp_sec: i32,
    /// Header stamp nanoseconds.
    pub stamp_nanosec: u32,
    /// Interleaved XYZ floats.
    pub xyz: Vec<f32>,
    /// Optional per-point float32 LiDAR intensity values.
    pub intensity: Option<Vec<f32>>,
}

impl PointCloud2Xyz {
    /// Creates a validated XYZ cloud (`xyz.len()` divisible by 3).
    pub fn try_new(
        frame_id: impl Into<String>,
        stamp_sec: i32,
        stamp_nanosec: u32,
        xyz: Vec<f32>,
    ) -> RuntimeResult<Self> {
        if xyz.len() % 3 != 0 {
            return Err(RuntimeError::InvalidConfiguration(
                "xyz length must be a multiple of 3".into(),
            ));
        }
        Ok(Self { frame_id: frame_id.into(), stamp_sec, stamp_nanosec, xyz, intensity: None })
    }

    /// Creates a validated XYZ-I cloud.
    pub fn try_new_with_intensity(
        frame_id: impl Into<String>,
        stamp_sec: i32,
        stamp_nanosec: u32,
        xyz: Vec<f32>,
        intensity: Vec<f32>,
    ) -> RuntimeResult<Self> {
        Self::try_new_with_optional_intensity(
            frame_id,
            stamp_sec,
            stamp_nanosec,
            xyz,
            Some(intensity),
        )
    }

    fn try_new_with_optional_intensity(
        frame_id: impl Into<String>,
        stamp_sec: i32,
        stamp_nanosec: u32,
        xyz: Vec<f32>,
        intensity: Option<Vec<f32>>,
    ) -> RuntimeResult<Self> {
        if xyz.len() % 3 != 0 {
            return Err(RuntimeError::InvalidConfiguration(
                "xyz length must be a multiple of 3".into(),
            ));
        }
        if intensity.as_ref().is_some_and(|values| values.len() != xyz.len() / 3) {
            return Err(RuntimeError::InvalidConfiguration(
                "intensity length must match the XYZ point count".into(),
            ));
        }
        Ok(Self { frame_id: frame_id.into(), stamp_sec, stamp_nanosec, xyz, intensity })
    }

    /// Returns point count.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.xyz.len() / 3
    }
}

/// Encodes an XYZ or XYZ-I PointCloud2 as ROS 2 CDR little-endian bytes.
pub fn encode_point_cloud2_xyz(msg: &PointCloud2Xyz) -> RuntimeResult<Vec<u8>> {
    let mut w = CdrWriter::new();
    w.write_encap();
    w.write_i32(msg.stamp_sec);
    w.write_u32(msg.stamp_nanosec);
    w.write_string(&msg.frame_id)?;
    let width = msg.point_count() as u32;
    w.write_u32(1); // height
    w.write_u32(width);
    let intensity = msg.intensity.as_deref();
    if intensity.is_some_and(|values| values.len() != msg.point_count()) {
        return Err(RuntimeError::InvalidConfiguration(
            "intensity length must match the XYZ point count".into(),
        ));
    }
    w.write_u32(if intensity.is_some() { 4 } else { 3 }); // fields length
    write_point_field(&mut w, "x", 0)?;
    write_point_field(&mut w, "y", 4)?;
    write_point_field(&mut w, "z", 8)?;
    if intensity.is_some() {
        write_point_field(&mut w, "intensity", 12)?;
    }
    w.write_bool(false); // is_bigendian
    let point_step = if intensity.is_some() { 16 } else { 12 };
    w.write_u32(point_step); // point_step
    w.write_u32(width.saturating_mul(point_step)); // row_step
    let mut data = Vec::with_capacity(msg.point_count() * point_step as usize);
    for (index, point) in msg.xyz.chunks_exact(3).enumerate() {
        for value in point {
            data.extend_from_slice(&value.to_le_bytes());
        }
        if let Some(intensity) = intensity {
            data.extend_from_slice(&intensity[index].to_le_bytes());
        }
    }
    let data_len = u32::try_from(data.len())
        .map_err(|_| RuntimeError::InvalidConfiguration("PointCloud2 data is too large".into()))?;
    w.write_u32(data_len);
    w.write_bytes(&data);
    w.write_bool(true); // is_dense
    Ok(w.into_bytes())
}

/// Inspects a PointCloud2 CDR header without materializing its point data.
pub fn point_cloud2_has_intensity(bytes: &[u8]) -> RuntimeResult<bool> {
    let mut r = CdrReader::new(bytes)?;
    r.expect_encap()?;
    let _stamp_sec = r.read_i32()?;
    let _stamp_nanosec = r.read_u32()?;
    let _frame_id = r.read_string()?;
    let _height = r.read_u32()?;
    let _width = r.read_u32()?;
    let field_count = r.read_u32()?;
    let mut has_intensity = false;
    for _ in 0..field_count {
        let name = r.read_string()?;
        let _offset = r.read_u32()?;
        let datatype = r.read_u8()?;
        r.align(4);
        let count = r.read_u32()?;
        if name.eq_ignore_ascii_case("intensity") {
            if count != 1 || datatype != 7 {
                return Err(RuntimeError::InvalidConfiguration(
                    "PointCloud2 intensity must be a scalar float32 field".into(),
                ));
            }
            has_intensity = true;
        }
    }
    Ok(has_intensity)
}

/// Decodes an XYZ or XYZ-I PointCloud2 from ROS 2 CDR little-endian bytes.
pub fn decode_point_cloud2_xyz(bytes: &[u8]) -> RuntimeResult<PointCloud2Xyz> {
    let mut r = CdrReader::new(bytes)?;
    r.expect_encap()?;
    let stamp_sec = r.read_i32()?;
    let stamp_nanosec = r.read_u32()?;
    let frame_id = r.read_string()?;
    let height = r.read_u32()?;
    let width = r.read_u32()?;
    let field_count = r.read_u32()?;
    let mut x_offset = None;
    let mut y_offset = None;
    let mut z_offset = None;
    let mut intensity_offset = None;
    for _ in 0..field_count {
        let name = r.read_string()?;
        let offset = r.read_u32()?;
        let datatype = r.read_u8()?;
        r.align(4);
        let count = r.read_u32()?;
        match name.to_ascii_lowercase().as_str() {
            "x" if count == 1 && datatype == 7 => x_offset = Some(offset),
            "y" if count == 1 && datatype == 7 => y_offset = Some(offset),
            "z" if count == 1 && datatype == 7 => z_offset = Some(offset),
            "intensity" => {
                if count != 1 || datatype != 7 {
                    return Err(RuntimeError::InvalidConfiguration(
                        "PointCloud2 intensity must be a scalar float32 field".into(),
                    ));
                }
                intensity_offset = Some(offset);
            }
            _ => {}
        }
    }
    let is_bigendian = r.read_bool()?;
    let point_step = r.read_u32()?;
    let row_step = r.read_u32()?;
    let data_len = r.read_u32()? as usize;
    let data = r.read_bytes(data_len)?;
    let _is_dense = r.read_bool()?;
    if height == 0 || width == 0 {
        return PointCloud2Xyz::try_new_with_optional_intensity(
            frame_id,
            stamp_sec,
            stamp_nanosec,
            Vec::new(),
            intensity_offset.map(|_| Vec::new()),
        );
    }
    let x_offset = x_offset.ok_or_else(|| {
        RuntimeError::InvalidConfiguration("PointCloud2 is missing float32 x field".into())
    })?;
    let y_offset = y_offset.ok_or_else(|| {
        RuntimeError::InvalidConfiguration("PointCloud2 is missing float32 y field".into())
    })?;
    let z_offset = z_offset.ok_or_else(|| {
        RuntimeError::InvalidConfiguration("PointCloud2 is missing float32 z field".into())
    })?;
    let last_offset = [x_offset, y_offset, z_offset]
        .into_iter()
        .chain(intensity_offset)
        .max()
        .expect("XYZ offsets are present");
    if u64::from(point_step) < u64::from(last_offset) + 4 {
        return Err(RuntimeError::InvalidConfiguration(
            "PointCloud2 point_step does not contain the declared fields".into(),
        ));
    }
    let height = height as usize;
    let width = width as usize;
    let point_step = point_step as usize;
    let row_step = row_step as usize;
    let row_bytes = width.checked_mul(point_step).ok_or_else(|| {
        RuntimeError::InvalidConfiguration("PointCloud2 row size overflow".into())
    })?;
    if row_step < row_bytes {
        return Err(RuntimeError::InvalidConfiguration(
            "PointCloud2 row_step is shorter than one row".into(),
        ));
    }
    let required_bytes = row_step.checked_mul(height).ok_or_else(|| {
        RuntimeError::InvalidConfiguration("PointCloud2 data size overflow".into())
    })?;
    if required_bytes > data.len() {
        return Err(RuntimeError::InvalidConfiguration(
            "PointCloud2 data is shorter than row_step × height".into(),
        ));
    }
    let points = height.checked_mul(width).ok_or_else(|| {
        RuntimeError::InvalidConfiguration("PointCloud2 point count overflow".into())
    })?;
    let xyz_capacity = points.checked_mul(3).ok_or_else(|| {
        RuntimeError::InvalidConfiguration("PointCloud2 XYZ capacity overflow".into())
    })?;
    let mut xyz = Vec::with_capacity(xyz_capacity);
    let mut intensity = intensity_offset.map(|_| Vec::with_capacity(points));
    for row in 0..height {
        for column in 0..width {
            let base = row
                .checked_mul(row_step)
                .and_then(|value| value.checked_add(column.checked_mul(point_step)?))
                .ok_or_else(|| {
                    RuntimeError::InvalidConfiguration("PointCloud2 point offset overflow".into())
                })?;
            for offset in [x_offset, y_offset, z_offset] {
                xyz.push(read_point_f32(data, base, offset, is_bigendian)?);
            }
            if let (Some(intensity), Some(offset)) = (&mut intensity, intensity_offset) {
                intensity.push(read_point_f32(data, base, offset, is_bigendian)?);
            }
        }
    }
    PointCloud2Xyz::try_new_with_optional_intensity(
        frame_id,
        stamp_sec,
        stamp_nanosec,
        xyz,
        intensity,
    )
}

fn read_point_f32(data: &[u8], base: usize, offset: u32, is_bigendian: bool) -> RuntimeResult<f32> {
    let start = base.checked_add(offset as usize).ok_or_else(|| {
        RuntimeError::InvalidConfiguration("PointCloud2 field offset overflow".into())
    })?;
    let end = start.checked_add(4).ok_or_else(|| {
        RuntimeError::InvalidConfiguration("PointCloud2 field end overflow".into())
    })?;
    if end > data.len() {
        return Err(RuntimeError::InvalidConfiguration(
            "PointCloud2 data shorter than field layout".into(),
        ));
    }
    let bytes = data[start..end].try_into().unwrap();
    Ok(if is_bigendian { f32::from_be_bytes(bytes) } else { f32::from_le_bytes(bytes) })
}

fn write_point_field(w: &mut CdrWriter, name: &str, offset: u32) -> RuntimeResult<()> {
    w.write_string(name)?;
    w.write_u32(offset);
    w.write_u8(7); // FLOAT32
    w.align(4);
    w.write_u32(1);
    Ok(())
}

/// In-process loopback node for testing ROS-shaped publish/subscribe without rclrs.
#[derive(Clone, Debug, Default)]
pub struct LoopbackRos2Node {
    topics: std::collections::BTreeMap<String, Vec<u8>>,
}

impl LoopbackRos2Node {
    /// Creates an empty node.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Publishes one CDR payload on a topic (replacing the previous sample).
    pub fn publish(&mut self, topic: impl Into<String>, payload: Vec<u8>) {
        self.topics.insert(topic.into(), payload);
    }

    /// Takes the latest payload for a topic, if any.
    pub fn take(&mut self, topic: &str) -> Option<Vec<u8>> {
        self.topics.remove(topic)
    }

    /// Returns whether a topic currently has a sample.
    #[must_use]
    pub fn has_topic(&self, topic: &str) -> bool {
        self.topics.contains_key(topic)
    }
}

struct CdrWriter {
    buf: Vec<u8>,
}

impl CdrWriter {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    fn write_encap(&mut self) {
        self.buf.extend_from_slice(&CDR_LE_ENCAP);
    }

    fn align(&mut self, n: usize) {
        while self.buf.len() % n != 0 {
            self.buf.push(0);
        }
    }

    fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn write_bool(&mut self, v: bool) {
        self.align(1);
        self.buf.push(u8::from(v));
    }

    fn write_i32(&mut self, v: i32) {
        self.align(4);
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_u32(&mut self, v: u32) {
        self.align(4);
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    fn write_string(&mut self, value: &str) -> RuntimeResult<()> {
        if value.len() >= u32::MAX as usize {
            return Err(RuntimeError::InvalidConfiguration("string too long".into()));
        }
        self.align(4);
        // ROS CDR strings include the trailing NUL in the length.
        let len = (value.len() + 1) as u32;
        self.write_u32(len);
        self.buf.extend_from_slice(value.as_bytes());
        self.buf.push(0);
        Ok(())
    }
}

struct CdrReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> CdrReader<'a> {
    fn new(buf: &'a [u8]) -> RuntimeResult<Self> {
        if buf.len() < 4 {
            return Err(RuntimeError::InvalidConfiguration("CDR buffer too short".into()));
        }
        Ok(Self { buf, pos: 0 })
    }

    fn expect_encap(&mut self) -> RuntimeResult<()> {
        if self.buf.len() < 4 || self.buf[..4] != CDR_LE_ENCAP {
            return Err(RuntimeError::InvalidConfiguration(
                "expected ROS 2 CDR little-endian encapsulation".into(),
            ));
        }
        self.pos = 4;
        Ok(())
    }

    fn align(&mut self, n: usize) {
        let rem = self.pos % n;
        if rem != 0 {
            self.pos += n - rem;
        }
    }

    fn read_u8(&mut self) -> RuntimeResult<u8> {
        let v = *self
            .buf
            .get(self.pos)
            .ok_or_else(|| RuntimeError::InvalidConfiguration("CDR truncated".into()))?;
        self.pos += 1;
        Ok(v)
    }

    fn read_bool(&mut self) -> RuntimeResult<bool> {
        Ok(self.read_u8()? != 0)
    }

    fn read_i32(&mut self) -> RuntimeResult<i32> {
        self.align(4);
        let bytes = self.read_exact(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> RuntimeResult<u32> {
        self.align(4);
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_bytes(&mut self, len: usize) -> RuntimeResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| RuntimeError::InvalidConfiguration("CDR overflow".into()))?;
        if end > self.buf.len() {
            return Err(RuntimeError::InvalidConfiguration("CDR truncated".into()));
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_exact(&mut self, len: usize) -> RuntimeResult<&'a [u8]> {
        self.read_bytes(len)
    }

    fn read_string(&mut self) -> RuntimeResult<String> {
        let len = self.read_u32()? as usize;
        if len == 0 {
            return Err(RuntimeError::InvalidConfiguration(
                "CDR string length must include NUL".into(),
            ));
        }
        let bytes = self.read_bytes(len)?;
        if bytes.last().copied() != Some(0) {
            return Err(RuntimeError::InvalidConfiguration(
                "CDR string missing NUL terminator".into(),
            ));
        }
        String::from_utf8(bytes[..len - 1].to_vec())
            .map_err(|_| RuntimeError::InvalidConfiguration("CDR string is not UTF-8".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        decode_point_cloud2_xyz, encode_point_cloud2_xyz, point_cloud2_has_intensity,
        write_point_field, CatalogRos2Adapter, CdrWriter, LoopbackRos2Node, PointCloud2Xyz,
        Ros2Adapter, POINT_CLOUD2_TYPE,
    };

    #[test]
    fn negotiates_point_cloud2() {
        let adapter = CatalogRos2Adapter::point_cloud2_xyz();
        assert_eq!(adapter.negotiate(POINT_CLOUD2_TYPE).unwrap().spatial_topic, "point/xyz");
    }

    #[test]
    fn roundtrips_xyz_cdr_and_loopback() {
        let msg =
            PointCloud2Xyz::try_new("lidar", 1, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let bytes = encode_point_cloud2_xyz(&msg).unwrap();
        assert!(!point_cloud2_has_intensity(&bytes).unwrap());
        let mut node = LoopbackRos2Node::new();
        node.publish("/points", bytes.clone());
        let taken = node.take("/points").unwrap();
        let decoded = decode_point_cloud2_xyz(&taken).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn roundtrips_xyzi_cdr_and_loopback() {
        let msg = PointCloud2Xyz::try_new_with_intensity(
            "lidar",
            1,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![10.0, 20.0],
        )
        .unwrap();
        let bytes = encode_point_cloud2_xyz(&msg).unwrap();
        assert!(point_cloud2_has_intensity(&bytes).unwrap());
        let decoded = decode_point_cloud2_xyz(&bytes).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn decodes_xyz_offsets_and_row_padding() {
        let mut writer = CdrWriter::new();
        writer.write_encap();
        writer.write_i32(3);
        writer.write_u32(4);
        writer.write_string("padded_lidar").unwrap();
        writer.write_u32(2); // height
        writer.write_u32(2); // width
        writer.write_u32(4); // fields
        write_point_field(&mut writer, "z", 8).unwrap();
        write_point_field(&mut writer, "intensity", 12).unwrap();
        write_point_field(&mut writer, "x", 0).unwrap();
        write_point_field(&mut writer, "y", 4).unwrap();
        writer.write_bool(false);
        writer.write_u32(16); // point_step
        writer.write_u32(40); // row_step includes eight bytes of row padding

        let mut data = Vec::new();
        for row in 0..2 {
            for column in 0..2 {
                let base = (row * 2 + column) as f32;
                for value in [base + 1.0, base + 2.0, base + 3.0, base + 4.0] {
                    data.extend_from_slice(&value.to_le_bytes());
                }
            }
            data.extend_from_slice(&[0; 8]);
        }
        writer.write_u32(data.len() as u32);
        writer.write_bytes(&data);
        writer.write_bool(true);

        let decoded = decode_point_cloud2_xyz(&writer.into_bytes()).unwrap();
        assert_eq!(decoded.frame_id, "padded_lidar");
        assert_eq!(decoded.stamp_sec, 3);
        assert_eq!(decoded.stamp_nanosec, 4);
        assert_eq!(decoded.xyz, vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0,]);
        assert_eq!(decoded.intensity, Some(vec![4.0, 5.0, 6.0, 7.0]));
    }
}
