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

/// Interleaved XYZ PointCloud2 payload.
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
        Ok(Self { frame_id: frame_id.into(), stamp_sec, stamp_nanosec, xyz })
    }

    /// Returns point count.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.xyz.len() / 3
    }
}

/// Encodes an XYZ PointCloud2 as ROS 2 CDR little-endian bytes.
pub fn encode_point_cloud2_xyz(msg: &PointCloud2Xyz) -> RuntimeResult<Vec<u8>> {
    let mut w = CdrWriter::new();
    w.write_encap();
    w.write_i32(msg.stamp_sec);
    w.write_u32(msg.stamp_nanosec);
    w.write_string(&msg.frame_id)?;
    let width = msg.point_count() as u32;
    w.write_u32(1); // height
    w.write_u32(width);
    w.write_u32(3); // fields length
    write_point_field(&mut w, "x", 0)?;
    write_point_field(&mut w, "y", 4)?;
    write_point_field(&mut w, "z", 8)?;
    w.write_bool(false); // is_bigendian
    w.write_u32(12); // point_step
    w.write_u32(width.saturating_mul(12)); // row_step
    let data: Vec<u8> = msg.xyz.iter().flat_map(|v| v.to_le_bytes()).collect();
    w.write_u32(data.len() as u32);
    w.write_bytes(&data);
    w.write_bool(true); // is_dense
    Ok(w.into_bytes())
}

/// Decodes an XYZ PointCloud2 from ROS 2 CDR little-endian bytes.
pub fn decode_point_cloud2_xyz(bytes: &[u8]) -> RuntimeResult<PointCloud2Xyz> {
    let mut r = CdrReader::new(bytes)?;
    r.expect_encap()?;
    let stamp_sec = r.read_i32()?;
    let stamp_nanosec = r.read_u32()?;
    let frame_id = r.read_string()?;
    let height = r.read_u32()?;
    let width = r.read_u32()?;
    let field_count = r.read_u32()?;
    for _ in 0..field_count {
        let _name = r.read_string()?;
        let _offset = r.read_u32()?;
        let _datatype = r.read_u8()?;
        r.align(4);
        let _count = r.read_u32()?;
    }
    let _is_bigendian = r.read_bool()?;
    let point_step = r.read_u32()?;
    let _row_step = r.read_u32()?;
    let data_len = r.read_u32()? as usize;
    let data = r.read_bytes(data_len)?;
    let _is_dense = r.read_bool()?;
    if height == 0 || width == 0 {
        return Ok(PointCloud2Xyz { frame_id, stamp_sec, stamp_nanosec, xyz: Vec::new() });
    }
    if point_step < 12 {
        return Err(RuntimeError::InvalidConfiguration(
            "point_step must be at least 12 for XYZ".into(),
        ));
    }
    let points = (height as usize).saturating_mul(width as usize);
    let mut xyz = Vec::with_capacity(points * 3);
    for i in 0..points {
        let base = i * point_step as usize;
        if base + 12 > data.len() {
            return Err(RuntimeError::InvalidConfiguration(
                "PointCloud2 data shorter than point_step layout".into(),
            ));
        }
        xyz.push(f32::from_le_bytes(data[base..base + 4].try_into().unwrap()));
        xyz.push(f32::from_le_bytes(data[base + 4..base + 8].try_into().unwrap()));
        xyz.push(f32::from_le_bytes(data[base + 8..base + 12].try_into().unwrap()));
    }
    PointCloud2Xyz::try_new(frame_id, stamp_sec, stamp_nanosec, xyz)
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
        decode_point_cloud2_xyz, encode_point_cloud2_xyz, CatalogRos2Adapter, LoopbackRos2Node,
        PointCloud2Xyz, Ros2Adapter, POINT_CLOUD2_TYPE,
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
        let mut node = LoopbackRos2Node::new();
        node.publish("/points", bytes.clone());
        let taken = node.take("/points").unwrap();
        let decoded = decode_point_cloud2_xyz(&taken).unwrap();
        assert_eq!(decoded, msg);
    }
}
