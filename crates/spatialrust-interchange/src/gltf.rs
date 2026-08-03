//! Minimal glTF 2.0 JSON export/import for triangle meshes (no external crate).

use spatialrust_scene::TriangleMesh;

use crate::{InterchangeError, InterchangeResult};

/// Exports a triangle mesh to a minimal glTF 2.0 JSON document (embedded base64 positions/indices).
pub fn export_triangle_mesh_gltf_json(mesh: &TriangleMesh) -> InterchangeResult<String> {
    if mesh.positions.len() % 3 != 0 {
        return Err(InterchangeError::InvalidConfiguration(
            "mesh positions length must be a multiple of 3".into(),
        ));
    }
    if mesh.indices.len() % 3 != 0 {
        return Err(InterchangeError::InvalidConfiguration(
            "mesh indices length must be a multiple of 3".into(),
        ));
    }
    let pos_bytes = f32_slice_as_bytes(&mesh.positions);
    let pos_b64 = base64_encode(&pos_bytes);
    let idx_bytes: Vec<u8> = mesh.indices.iter().flat_map(|v| v.to_le_bytes()).collect();
    let idx_b64 = base64_encode(&idx_bytes);
    let vertex_count = mesh.vertex_count();
    let index_count = mesh.indices.len();
    // Hand-written minimal glTF JSON without serde.
    Ok(format!(
        r#"{{"asset":{{"version":"2.0","generator":"spatialrust-interchange"}},"buffers":[{{"byteLength":{pos_len},"uri":"data:application/octet-stream;base64,{pos_b64}"}},{{"byteLength":{idx_len},"uri":"data:application/octet-stream;base64,{idx_b64}"}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{pos_len},"target":34962}},{{"buffer":1,"byteOffset":0,"byteLength":{idx_len},"target":34963}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":{vertex_count},"type":"VEC3"}},{{"bufferView":1,"componentType":5125,"count":{index_count},"type":"SCALAR"}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1}}}}],"nodes":[{{"mesh":0}}],"scenes":[{{"nodes":[0]}}],"scene":0}}"#,
        pos_len = mesh.positions.len() * 4,
        idx_len = idx_bytes.len(),
        pos_b64 = pos_b64,
        idx_b64 = idx_b64,
        vertex_count = vertex_count,
        index_count = index_count,
    ))
}

/// Imports vertex/index counts from a SpatialRust-exported glTF JSON fragment.
///
/// Full binary decode is intentionally limited to validating SpatialRust-authored payloads
/// that embed `accessors` counts.
pub fn import_triangle_mesh_gltf_json(json: &str) -> InterchangeResult<(usize, usize)> {
    let mesh = decode_triangle_mesh_gltf_json(json)?;
    Ok((mesh.vertex_count(), mesh.indices.len()))
}

/// Decodes a SpatialRust-exported glTF JSON mesh with embedded base64 buffers.
///
/// The portable interchange boundary intentionally accepts only the minimal
/// glTF shape emitted by [`export_triangle_mesh_gltf_json`]. It does not fetch
/// external buffers, interpret arbitrary glTF scenes, or apply transforms.
pub fn decode_triangle_mesh_gltf_json(json: &str) -> InterchangeResult<TriangleMesh> {
    if !json.contains(r#""version":"2.0""#) {
        return Err(InterchangeError::InvalidConfiguration(
            "missing glTF 2.0 asset version".into(),
        ));
    }
    let vertex_count = extract_accessor_count(json, "VEC3")?;
    let index_count = extract_accessor_count(json, "SCALAR")?;
    let buffers = extract_embedded_buffers(json)?;
    if buffers.len() < 2 {
        return Err(InterchangeError::InvalidConfiguration(
            "glTF mesh requires embedded position and index buffers".into(),
        ));
    }
    let position_bytes = base64_decode(buffers[0])?;
    let index_bytes = base64_decode(buffers[1])?;
    let expected_position_bytes = vertex_count
        .checked_mul(3)
        .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| {
            InterchangeError::InvalidConfiguration("position byte count overflow".into())
        })?;
    let expected_index_bytes =
        index_count.checked_mul(std::mem::size_of::<u32>()).ok_or_else(|| {
            InterchangeError::InvalidConfiguration("index byte count overflow".into())
        })?;
    if position_bytes.len() != expected_position_bytes {
        return Err(InterchangeError::InvalidConfiguration(format!(
            "position buffer has {} bytes; expected {}",
            position_bytes.len(),
            expected_position_bytes
        )));
    }
    if index_bytes.len() != expected_index_bytes {
        return Err(InterchangeError::InvalidConfiguration(format!(
            "index buffer has {} bytes; expected {}",
            index_bytes.len(),
            expected_index_bytes
        )));
    }

    let mut positions = Vec::with_capacity(vertex_count * 3);
    for chunk in position_bytes.chunks_exact(4) {
        positions.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let mut indices = Vec::with_capacity(index_count);
    for chunk in index_bytes.chunks_exact(4) {
        let index = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if usize::try_from(index).map_or(true, |index| index >= vertex_count) {
            return Err(InterchangeError::InvalidConfiguration(format!(
                "mesh index {} is outside {} vertices",
                index, vertex_count
            )));
        }
        indices.push(index);
    }
    Ok(TriangleMesh { positions, indices })
}

fn extract_accessor_count(json: &str, value_type: &str) -> InterchangeResult<usize> {
    let marker = format!(r#""type":"{value_type}""#);
    let idx = json
        .find(&marker)
        .ok_or_else(|| InterchangeError::InvalidConfiguration(format!("missing {marker}")))?;
    let before = &json[..idx];
    let key = "\"count\":";
    let count_idx = before
        .rfind(key)
        .ok_or_else(|| InterchangeError::InvalidConfiguration("missing accessor count".into()))?;
    let rest = &before[count_idx + key.len()..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().map_err(|_| {
        InterchangeError::InvalidConfiguration("accessor count is not an integer".into())
    })
}

fn extract_embedded_buffers(json: &str) -> InterchangeResult<Vec<&str>> {
    let marker = r#""uri":"data:application/octet-stream;base64,"#;
    let buffers: Vec<_> =
        json.split(marker).skip(1).filter_map(|rest| rest.split('"').next()).collect();
    if buffers.iter().any(|buffer| buffer.is_empty() && !json.contains(r#""byteLength":0"#)) {
        return Err(InterchangeError::InvalidConfiguration(
            "embedded glTF buffer URI is empty or malformed".into(),
        ));
    }
    Ok(buffers)
}

fn base64_decode(value: &str) -> InterchangeResult<Vec<u8>> {
    if value.len() % 4 != 0 {
        return Err(InterchangeError::InvalidConfiguration(
            "base64 buffer length must be a multiple of four".into(),
        ));
    }
    let mut output = Vec::with_capacity(value.len() / 4 * 3);
    for (chunk_index, chunk) in value.as_bytes().chunks_exact(4).enumerate() {
        let a = base64_value(chunk[0]).ok_or_else(|| invalid_base64(chunk_index))?;
        let b = base64_value(chunk[1]).ok_or_else(|| invalid_base64(chunk_index))?;
        let c = if chunk[2] == b'=' {
            0
        } else {
            base64_value(chunk[2]).ok_or_else(|| invalid_base64(chunk_index))?
        };
        let d = if chunk[3] == b'=' {
            0
        } else {
            base64_value(chunk[3]).ok_or_else(|| invalid_base64(chunk_index))?
        };
        output.push((a << 2) | (b >> 4));
        if chunk[2] != b'=' {
            output.push((b << 4) | (c >> 2));
        }
        if chunk[3] != b'=' {
            if chunk[2] == b'=' {
                return Err(invalid_base64(chunk_index));
            }
            output.push((c << 6) | d);
        }
        if (chunk[2] == b'=' || chunk[3] == b'=') && chunk_index + 1 != value.len() / 4 {
            return Err(invalid_base64(chunk_index));
        }
    }
    Ok(output)
}

fn base64_value(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn invalid_base64(chunk_index: usize) -> InterchangeError {
    InterchangeError::InvalidConfiguration(format!("invalid base64 buffer at chunk {chunk_index}"))
}

fn f32_slice_as_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((triple >> 18) & 63) as usize] as char);
        out.push(TABLE[((triple >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[((triple >> 6) & 63) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(triple & 63) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        decode_triangle_mesh_gltf_json, export_triangle_mesh_gltf_json,
        import_triangle_mesh_gltf_json,
    };
    use spatialrust_scene::TriangleMesh;

    #[test]
    fn roundtrips_counts() {
        let mesh = TriangleMesh {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            indices: vec![0, 1, 2],
        };
        let json = export_triangle_mesh_gltf_json(&mesh).unwrap();
        let (vertices, indices) = import_triangle_mesh_gltf_json(&json).unwrap();
        assert_eq!(vertices, 3);
        assert_eq!(indices, 3);
        assert_eq!(decode_triangle_mesh_gltf_json(&json).unwrap(), mesh);
    }

    #[test]
    fn rejects_invalid_embedded_index() {
        let mesh = TriangleMesh { positions: vec![0.0, 0.0, 0.0], indices: vec![] };
        let json = export_triangle_mesh_gltf_json(&mesh).unwrap();
        let invalid = json.replace("\"byteLength\":0", "\"byteLength\":4");
        assert!(decode_triangle_mesh_gltf_json(&invalid).is_err());
    }
}
