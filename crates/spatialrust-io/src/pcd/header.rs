use std::io::{BufRead, Read};

use crate::error::{pcd_format, pcd_parse, IoError};
use crate::pcd::schema::PcdFieldSpec;

/// PCD payload encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcdDataKind {
    /// ASCII payload.
    Ascii,
    /// Binary little-endian payload.
    Binary,
}

/// Parsed PCD header.
#[derive(Clone, Debug, PartialEq)]
pub struct PcdHeader {
    /// PCD version string.
    pub version: String,
    /// Field specifications.
    pub fields: Vec<PcdFieldSpec>,
    /// Cloud width.
    pub width: usize,
    /// Cloud height.
    pub height: usize,
    /// Viewpoint as seven float values.
    pub viewpoint: [f32; 7],
    /// Number of points.
    pub points: usize,
    /// Payload encoding.
    pub data: PcdDataKind,
}

impl PcdHeader {
    /// Returns the byte size of one point in binary PCD.
    #[must_use]
    pub fn point_step(&self) -> usize {
        self.fields.iter().map(PcdFieldSpec::byte_size).sum()
    }

    /// Parses a PCD header from a buffered reader.
    #[allow(unused_assignments)]
    pub fn parse<R: BufRead>(reader: &mut R) -> Result<(Self, usize), IoError> {
        let mut version = String::new();
        let mut field_names = Vec::new();
        let mut sizes = Vec::new();
        let mut kinds = Vec::new();
        let mut counts = Vec::new();
        let mut width = 0usize;
        let mut height = 0usize;
        let mut viewpoint = [0.0_f32; 7];
        let mut points = 0usize;
        let mut data: Option<PcdDataKind> = None;
        let mut header_bytes = 0usize;

        loop {
            let mut line = String::new();
            let read = reader.read_line(&mut line)?;
            if read == 0 {
                return Err(pcd_parse("unexpected EOF while reading PCD header"));
            }
            header_bytes += read;

            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            if let Some(rest) = trimmed.strip_prefix("VERSION ") {
                version = rest.trim().to_owned();
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("FIELDS ") {
                field_names = rest.split_whitespace().map(str::to_owned).collect();
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("SIZE ") {
                sizes = parse_usize_list(rest)?;
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("TYPE ") {
                kinds = parse_type_list(rest)?;
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("COUNT ") {
                counts = parse_usize_list(rest)?;
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("WIDTH ") {
                width = rest
                    .trim()
                    .parse()
                    .map_err(|_| pcd_parse(format!("invalid WIDTH `{rest}`")))?;
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("HEIGHT ") {
                height = rest
                    .trim()
                    .parse()
                    .map_err(|_| pcd_parse(format!("invalid HEIGHT `{rest}`")))?;
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("VIEWPOINT ") {
                let values = parse_f32_list(rest)?;
                if values.len() != 7 {
                    return Err(pcd_parse("VIEWPOINT must contain 7 values"));
                }
                viewpoint.copy_from_slice(&values);
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("POINTS ") {
                points = rest
                    .trim()
                    .parse()
                    .map_err(|_| pcd_parse(format!("invalid POINTS `{rest}`")))?;
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("DATA ") {
                data = Some(match rest.trim().to_ascii_lowercase().as_str() {
                    "ascii" => PcdDataKind::Ascii,
                    "binary" => PcdDataKind::Binary,
                    other => return Err(pcd_format(format!("unsupported DATA mode `{other}`"))),
                });
                break;
            }

            return Err(pcd_parse(format!("unexpected PCD header line `{trimmed}`")));
        }

        if version.is_empty() {
            version = "0.7".to_owned();
        }
        if field_names.is_empty() {
            return Err(pcd_parse("PCD header missing FIELDS"));
        }
        if sizes.len() != field_names.len()
            || kinds.len() != field_names.len()
            || counts.len() != field_names.len()
        {
            return Err(pcd_parse("FIELDS/SIZE/TYPE/COUNT length mismatch in PCD header"));
        }

        let fields = field_names
            .into_iter()
            .zip(sizes)
            .zip(kinds)
            .zip(counts)
            .map(|(((name, size), kind), count)| PcdFieldSpec { name, size, kind, count })
            .collect();

        let data_kind = data.ok_or_else(|| pcd_parse("PCD header missing DATA"))?;
        if points == 0 && width > 0 && height > 0 {
            points = width * height;
        }

        Ok((
            Self { version, fields, width, height, viewpoint, points, data: data_kind },
            header_bytes,
        ))
    }
}

fn parse_usize_list(input: &str) -> Result<Vec<usize>, IoError> {
    input
        .split_whitespace()
        .map(|value| {
            value.parse().map_err(|_| pcd_parse(format!("invalid integer `{value}` in PCD header")))
        })
        .collect()
}

fn parse_f32_list(input: &str) -> Result<Vec<f32>, IoError> {
    input
        .split_whitespace()
        .map(|value| {
            value.parse().map_err(|_| pcd_parse(format!("invalid float `{value}` in PCD header")))
        })
        .collect()
}

fn parse_type_list(input: &str) -> Result<Vec<crate::pcd::schema::PcdType>, IoError> {
    input
        .split_whitespace()
        .map(|value| match value {
            "I" => Ok(crate::pcd::schema::PcdType::I),
            "U" => Ok(crate::pcd::schema::PcdType::U),
            "F" => Ok(crate::pcd::schema::PcdType::F),
            _ => Err(pcd_parse(format!("invalid TYPE token `{value}`"))),
        })
        .collect()
}

/// Reads the remaining binary payload bytes after the header.
pub fn read_binary_payload<R: Read>(reader: &mut R, byte_len: usize) -> Result<Vec<u8>, IoError> {
    let mut data = vec![0_u8; byte_len];
    reader.read_exact(&mut data)?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::{PcdDataKind, PcdHeader};
    use std::io::Cursor;

    const SAMPLE_ASCII_HEADER: &str = "\
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 2
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 2
DATA ascii
";

    #[test]
    fn parses_ascii_header() {
        let mut reader = Cursor::new(SAMPLE_ASCII_HEADER);
        let (header, _) = PcdHeader::parse(&mut reader).unwrap();
        assert_eq!(header.points, 2);
        assert_eq!(header.data, PcdDataKind::Ascii);
        assert_eq!(header.fields.len(), 3);
    }
}
