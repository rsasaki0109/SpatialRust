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
    let vertex_count = extract_count(json, "\"type\":\"VEC3\"")?;
    let index_count = extract_count(json, "\"type\":\"SCALAR\"")?;
    Ok((vertex_count, index_count))
}

fn extract_count(json: &str, marker: &str) -> InterchangeResult<usize> {
    let idx = json
        .find(marker)
        .ok_or_else(|| InterchangeError::InvalidConfiguration(format!("missing {marker}")))?;
    let before = &json[..idx];
    let key = "\"count\":";
    let count_idx = before
        .rfind(key)
        .ok_or_else(|| InterchangeError::InvalidConfiguration("missing count".into()))?;
    let rest = &before[count_idx + key.len()..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits
        .parse()
        .map_err(|_| InterchangeError::InvalidConfiguration("count is not an integer".into()))
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
    use super::{export_triangle_mesh_gltf_json, import_triangle_mesh_gltf_json};
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
    }
}
