use std::io::BufRead;

use crate::error::{ply_format, ply_parse, IoError};

/// PLY payload encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlyFormat {
    /// ASCII PLY.
    Ascii,
    /// Binary little-endian PLY.
    BinaryLittleEndian,
}

/// Supported PLY scalar property types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlyPropertyKind {
    /// Signed 8-bit integer.
    Char,
    /// Unsigned 8-bit integer.
    UChar,
    /// Signed 16-bit integer.
    Short,
    /// Unsigned 16-bit integer.
    UShort,
    /// Signed 32-bit integer.
    Int,
    /// Unsigned 32-bit integer.
    UInt,
    /// 32-bit float.
    Float,
    /// 64-bit float.
    Double,
}

impl PlyPropertyKind {
    /// Returns the size of one scalar value in bytes.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Char | Self::UChar => 1,
            Self::Short | Self::UShort => 2,
            Self::Int | Self::UInt | Self::Float => 4,
            Self::Double => 8,
        }
    }

    fn parse(token: &str) -> Result<Self, IoError> {
        match token {
            "char" | "int8" => Ok(Self::Char),
            "uchar" | "uint8" => Ok(Self::UChar),
            "short" | "int16" => Ok(Self::Short),
            "ushort" | "uint16" => Ok(Self::UShort),
            "int" | "int32" => Ok(Self::Int),
            "uint" | "uint32" => Ok(Self::UInt),
            "float" | "float32" => Ok(Self::Float),
            "double" | "float64" => Ok(Self::Double),
            _ => Err(ply_parse(format!("unsupported PLY property type `{token}`"))),
        }
    }

    fn as_token(self) -> &'static str {
        match self {
            Self::Char => "char",
            Self::UChar => "uchar",
            Self::Short => "short",
            Self::UShort => "ushort",
            Self::Int => "int",
            Self::UInt => "uint",
            Self::Float => "float",
            Self::Double => "double",
        }
    }
}

/// One vertex property in a PLY header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlyProperty {
    /// Property name.
    pub name: String,
    /// Scalar type.
    pub kind: PlyPropertyKind,
}

/// Parsed PLY header for vertex-only point clouds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlyHeader {
    /// File encoding.
    pub format: PlyFormat,
    /// Number of vertex records.
    pub vertex_count: usize,
    /// Vertex properties in declaration order.
    pub properties: Vec<PlyProperty>,
}

impl PlyHeader {
    /// Parses a PLY header from a buffered reader.
    pub fn parse<R: BufRead>(reader: &mut R) -> Result<Self, IoError> {
        let first = read_non_empty_line(reader)?;
        if first != "ply" {
            return Err(ply_format(format!("expected PLY magic, found `{first}`")));
        }

        let mut format = None;
        let mut vertex_count = None;
        let mut properties = Vec::new();
        let mut current_element: Option<String> = None;

        loop {
            let line = read_non_empty_line(reader)?;
            if line == "end_header" {
                break;
            }

            let mut parts = line.split_whitespace();
            let keyword = parts.next().ok_or_else(|| ply_parse("empty PLY header line"))?;
            match keyword {
                "format" => {
                    let token = parts.next().ok_or_else(|| ply_parse("missing PLY format token"))?;
                    let version = parts.next().ok_or_else(|| ply_parse("missing PLY format version"))?;
                    if version != "1.0" {
                        return Err(ply_format(format!("unsupported PLY version `{version}`")));
                    }
                    format = Some(match token {
                        "ascii" => PlyFormat::Ascii,
                        "binary_little_endian" => PlyFormat::BinaryLittleEndian,
                        "binary_big_endian" => {
                            return Err(ply_format("binary_big_endian PLY is not supported"));
                        }
                        _ => return Err(ply_format(format!("unsupported PLY format `{token}`"))),
                    });
                }
                "element" => {
                    let name = parts
                        .next()
                        .ok_or_else(|| ply_parse("missing PLY element name"))?
                        .to_owned();
                    let count = parts
                        .next()
                        .ok_or_else(|| ply_parse("missing PLY element count"))?
                        .parse::<usize>()
                        .map_err(|_| ply_parse("invalid PLY element count"))?;
                    if name == "vertex" {
                        if vertex_count.is_some() {
                            return Err(ply_format("duplicate vertex element in PLY header"));
                        }
                        vertex_count = Some(count);
                    } else {
                        return Err(ply_format(format!(
                            "unsupported PLY element `{name}` (only vertex is supported)"
                        )));
                    }
                    current_element = Some(name);
                }
                "property" => {
                    let element = current_element.as_deref().ok_or_else(|| {
                        ply_parse("PLY property declared before element")
                    })?;
                    if element != "vertex" {
                        return Err(ply_format("only vertex properties are supported"));
                    }
                    let kind_token = parts
                        .next()
                        .ok_or_else(|| ply_parse("missing PLY property type"))?;
                    let name = parts
                        .next()
                        .ok_or_else(|| ply_parse("missing PLY property name"))?
                        .to_owned();
                    properties.push(PlyProperty {
                        name,
                        kind: PlyPropertyKind::parse(kind_token)?,
                    });
                }
                "comment" | "obj_info" => {}
                _ => return Err(ply_parse(format!("unsupported PLY header keyword `{keyword}`"))),
            }
        }

        Ok(Self {
            format: format.ok_or_else(|| ply_format("missing PLY format line"))?,
            vertex_count: vertex_count.ok_or_else(|| ply_format("missing vertex element"))?,
            properties,
        })
    }

    /// Returns the byte size of one binary vertex record.
    #[must_use]
    pub fn vertex_stride(&self) -> usize {
        self.properties.iter().map(|property| property.kind.size_bytes()).sum()
    }

    /// Writes a vertex-only PLY header.
    pub fn write_header<W: std::io::Write>(
        &self,
        writer: &mut W,
    ) -> Result<(), IoError> {
        writeln!(writer, "ply")?;
        let format = match self.format {
            PlyFormat::Ascii => "ascii 1.0",
            PlyFormat::BinaryLittleEndian => "binary_little_endian 1.0",
        };
        writeln!(writer, "format {format}")?;
        writeln!(writer, "element vertex {}", self.vertex_count)?;
        for property in &self.properties {
            writeln!(
                writer,
                "property {} {}",
                property.kind.as_token(),
                property.name
            )?;
        }
        writeln!(writer, "end_header")?;
        Ok(())
    }
}

fn read_non_empty_line<R: BufRead>(reader: &mut R) -> Result<String, IoError> {
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            return Err(ply_parse("unexpected EOF while reading PLY header"));
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        return Ok(trimmed.to_owned());
    }
}
