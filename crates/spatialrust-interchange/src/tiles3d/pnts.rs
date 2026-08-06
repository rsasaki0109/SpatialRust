//! OGC 3D Tiles 1.1 point (`pnts`) binary tile codec.

use crate::json::{parse_json, serialize_json, Json};
use crate::{InterchangeError, InterchangeResult};

const PNTS_MAGIC: &[u8; 4] = b"pnts";
const PNTS_VERSION: u32 = 1;
const HEADER_BYTES: usize = 28;
const POSITION_COMPONENTS: usize = 3;
const RGB_COMPONENTS: usize = 3;

/// Feature-table data carried by one `pnts` tile.
///
/// Positions are stored relative to `rtc_center` when present; this keeps
/// `f32` precision for tiles far from the coordinate origin.
#[derive(Clone, Debug, PartialEq)]
pub struct PntsFeatureTable {
    /// Interleaved `x,y,z` positions (N*3 values) in the tile local frame.
    pub positions: Vec<f32>,
    /// Optional interleaved `r,g,b` bytes (N*3 values, range 0–255).
    pub rgb: Option<Vec<u8>>,
    /// Optional relative-to-center vector written to the feature table.
    pub rtc_center: Option<[f64; 3]>,
}

impl PntsFeatureTable {
    /// Returns the number of points in this tile.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.positions.len() / POSITION_COMPONENTS
    }

    fn validate(&self) -> InterchangeResult<()> {
        if self.positions.len() % POSITION_COMPONENTS != 0 {
            return Err(InterchangeError::InvalidConfiguration(
                "pnts positions length must be a multiple of 3".into(),
            ));
        }
        if let Some(rgb) = &self.rgb {
            if rgb.len() != self.point_count() * RGB_COMPONENTS {
                return Err(InterchangeError::InvalidConfiguration(
                    "pnts RGB length must equal three times the point count".into(),
                ));
            }
        }
        if let Some(center) = self.rtc_center {
            if center.iter().any(|value| !value.is_finite()) {
                return Err(InterchangeError::InvalidConfiguration(
                    "pnts RTC_CENTER must contain finite values".into(),
                ));
            }
        }
        if self.positions.iter().any(|value| !value.is_finite()) {
            return Err(InterchangeError::InvalidConfiguration(
                "pnts positions must contain finite values".into(),
            ));
        }
        Ok(())
    }
}

/// Encodes a feature table into a complete `pnts` tile (header + JSON + binary).
pub fn encode_pnts(table: &PntsFeatureTable) -> InterchangeResult<Vec<u8>> {
    table.validate()?;
    let point_count = table.point_count();
    let position_bytes = point_count
        .checked_mul(POSITION_COMPONENTS)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| {
            InterchangeError::InvalidConfiguration("pnts position size overflow".into())
        })?;
    let rgb_bytes = table.rgb.as_ref().map_or(0, |rgb| rgb.len());
    let binary_len = position_bytes + rgb_bytes;

    let rgb_offset = position_bytes as u64;
    let mut members = vec![
        ("POINTS_LENGTH", Json::Number(point_count.to_string())),
        ("POSITION", Json::object(vec![("byteOffset", Json::Number("0".into()))])),
    ];
    if let Some(center) = table.rtc_center {
        members.push((
            "RTC_CENTER",
            Json::Array(center.iter().map(|v| Json::Number(format_f64(*v))).collect()),
        ));
    }
    if table.rgb.is_some() {
        members.push((
            "RGB",
            Json::object(vec![("byteOffset", Json::Number(rgb_offset.to_string()))]),
        ));
    }
    let mut feature_json = serialize_json(&Json::object(members)).into_bytes();
    pad_to_8(&mut feature_json);

    let total = HEADER_BYTES
        .checked_add(feature_json.len())
        .and_then(|n| n.checked_add(binary_len))
        .ok_or_else(|| {
            InterchangeError::InvalidConfiguration("pnts byte length overflow".into())
        })?;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(PNTS_MAGIC);
    out.extend_from_slice(&PNTS_VERSION.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(feature_json.len() as u32).to_le_bytes());
    out.extend_from_slice(&(binary_len as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // batch table JSON length
    out.extend_from_slice(&0u32.to_le_bytes()); // batch table binary length
    out.extend_from_slice(&feature_json);
    for value in &table.positions {
        out.extend_from_slice(&value.to_le_bytes());
    }
    if let Some(rgb) = &table.rgb {
        out.extend_from_slice(rgb);
    }
    Ok(out)
}

/// Decodes a complete `pnts` tile into its feature table.
pub fn decode_pnts(bytes: &[u8]) -> InterchangeResult<PntsFeatureTable> {
    if bytes.len() < HEADER_BYTES {
        return Err(InterchangeError::InvalidConfiguration(
            "pnts tile is shorter than its header".into(),
        ));
    }
    if &bytes[0..4] != PNTS_MAGIC {
        return Err(InterchangeError::InvalidConfiguration("pnts tile has invalid magic".into()));
    }
    let version = read_u32(bytes, 4)?;
    if version != PNTS_VERSION {
        return Err(InterchangeError::InvalidConfiguration(format!(
            "unsupported pnts version {version}"
        )));
    }
    let declared_len = read_u32(bytes, 8)? as usize;
    if declared_len != bytes.len() {
        return Err(InterchangeError::InvalidConfiguration(
            "pnts declared byte length does not match the input".into(),
        ));
    }
    let feature_json_len = read_u32(bytes, 12)? as usize;
    let feature_binary_len = read_u32(bytes, 16)? as usize;
    let batch_json_len = read_u32(bytes, 20)? as usize;
    let batch_binary_len = read_u32(bytes, 24)? as usize;

    let json_start = HEADER_BYTES;
    let json_end = json_start
        .checked_add(feature_json_len)
        .ok_or_else(|| InterchangeError::InvalidConfiguration("pnts JSON range overflow".into()))?;
    let binary_end = json_end.checked_add(feature_binary_len).ok_or_else(|| {
        InterchangeError::InvalidConfiguration("pnts binary range overflow".into())
    })?;
    let batch_end = binary_end
        .checked_add(batch_json_len)
        .and_then(|n| n.checked_add(batch_binary_len))
        .ok_or_else(|| {
            InterchangeError::InvalidConfiguration("pnts batch range overflow".into())
        })?;
    if batch_end > bytes.len() {
        return Err(InterchangeError::InvalidConfiguration(
            "pnts lengths exceed the tile byte length".into(),
        ));
    }

    let json_text = std::str::from_utf8(&bytes[json_start..json_end])
        .map_err(|_| InterchangeError::InvalidConfiguration("pnts JSON is not UTF-8".into()))?;
    let feature = parse_json(json_text)?;
    let point_count = feature
        .get("POINTS_LENGTH")
        .and_then(Json::as_u64)
        .ok_or_else(|| InterchangeError::InvalidConfiguration("missing POINTS_LENGTH".into()))?
        as usize;

    let position_offset = feature
        .get("POSITION")
        .and_then(|position| position.get("byteOffset"))
        .and_then(Json::as_u64)
        .ok_or_else(|| {
            InterchangeError::InvalidConfiguration("missing POSITION byteOffset".into())
        })? as usize;
    let position_bytes = point_count
        .checked_mul(POSITION_COMPONENTS)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| {
            InterchangeError::InvalidConfiguration("pnts position size overflow".into())
        })?;
    let position_end = position_offset.checked_add(position_bytes).ok_or_else(|| {
        InterchangeError::InvalidConfiguration("pnts position range overflow".into())
    })?;
    if position_end > feature_binary_len {
        return Err(InterchangeError::InvalidConfiguration(
            "pnts POSITION range exceeds the feature binary".into(),
        ));
    }

    let mut positions = Vec::with_capacity(position_bytes / 4);
    for chunk in bytes[json_end + position_offset..json_end + position_end].chunks_exact(4) {
        positions.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    let rgb = match feature.get("RGB") {
        None => None,
        Some(rgb) => {
            let rgb_offset = rgb.get("byteOffset").and_then(Json::as_u64).ok_or_else(|| {
                InterchangeError::InvalidConfiguration("missing RGB byteOffset".into())
            })? as usize;
            let rgb_bytes = point_count.checked_mul(RGB_COMPONENTS).ok_or_else(|| {
                InterchangeError::InvalidConfiguration("pnts RGB size overflow".into())
            })?;
            let rgb_end = rgb_offset.checked_add(rgb_bytes).ok_or_else(|| {
                InterchangeError::InvalidConfiguration("pnts RGB range overflow".into())
            })?;
            if rgb_end > feature_binary_len {
                return Err(InterchangeError::InvalidConfiguration(
                    "pnts RGB range exceeds the feature binary".into(),
                ));
            }
            Some(bytes[json_end + rgb_offset..json_end + rgb_end].to_vec())
        }
    };

    let rtc_center = match feature.get("RTC_CENTER") {
        None => None,
        Some(center) => {
            let values = center.as_array().ok_or_else(|| {
                InterchangeError::InvalidConfiguration("RTC_CENTER must be an array".into())
            })?;
            if values.len() != 3 {
                return Err(InterchangeError::InvalidConfiguration(
                    "RTC_CENTER must contain three values".into(),
                ));
            }
            let mut out = [0.0f64; 3];
            for (index, value) in values.iter().enumerate() {
                out[index] = value.as_f64().ok_or_else(|| {
                    InterchangeError::InvalidConfiguration("RTC_CENTER value is not numeric".into())
                })?;
            }
            Some(out)
        }
    };

    Ok(PntsFeatureTable { positions, rgb, rtc_center })
}

fn read_u32(bytes: &[u8], offset: usize) -> InterchangeResult<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| InterchangeError::InvalidConfiguration("pnts offset overflow".into()))?;
    let window = bytes
        .get(offset..end)
        .ok_or_else(|| InterchangeError::InvalidConfiguration("pnts header is truncated".into()))?;
    Ok(u32::from_le_bytes([window[0], window[1], window[2], window[3]]))
}

fn pad_to_8(bytes: &mut Vec<u8>) {
    while bytes.len() % 8 != 0 {
        bytes.push(b' ');
    }
}

fn format_f64(value: f64) -> String {
    if value == value.trunc() && value.abs() < 1e15 {
        format!("{value:.1}")
    } else {
        format!("{value}")
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_pnts, encode_pnts, PntsFeatureTable};

    #[test]
    fn round_trips_positions_only() {
        let table = PntsFeatureTable {
            positions: vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            rgb: None,
            rtc_center: None,
        };
        let encoded = encode_pnts(&table).unwrap();
        let decoded = decode_pnts(&encoded).unwrap();
        assert_eq!(decoded, table);
        assert_eq!(decoded.point_count(), 2);
    }

    #[test]
    fn round_trips_rgb_and_center() {
        let table = PntsFeatureTable {
            positions: vec![1.0, 2.0, 3.0],
            rgb: Some(vec![10, 20, 30]),
            rtc_center: Some([1000.0, 2000.0, 3000.0]),
        };
        let encoded = encode_pnts(&table).unwrap();
        let decoded = decode_pnts(&encoded).unwrap();
        assert_eq!(decoded, table);
    }

    #[test]
    fn rejects_invalid_magic() {
        let mut encoded = encode_pnts(&PntsFeatureTable {
            positions: vec![0.0, 0.0, 0.0],
            rgb: None,
            rtc_center: None,
        })
        .unwrap();
        encoded[0] = b'X';
        assert!(decode_pnts(&encoded).is_err());
    }

    #[test]
    fn rejects_truncated_tile() {
        let encoded = encode_pnts(&PntsFeatureTable {
            positions: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            rgb: None,
            rtc_center: None,
        })
        .unwrap();
        assert!(decode_pnts(&encoded[..encoded.len() - 1]).is_err());
        assert!(decode_pnts(&encoded[..10]).is_err());
    }

    #[test]
    fn rejects_mismatched_rgb_length() {
        let table = PntsFeatureTable {
            positions: vec![0.0, 0.0, 0.0],
            rgb: Some(vec![1, 2]),
            rtc_center: None,
        };
        assert!(encode_pnts(&table).is_err());
    }
}
