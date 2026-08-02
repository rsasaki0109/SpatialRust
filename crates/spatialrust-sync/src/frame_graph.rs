//! Calibrated frame graph with parent→child isometries.

use std::collections::{HashMap, HashSet, VecDeque};

use spatialrust_core::{
    FieldSemantic, FrameId, HasNormals3, HasPositions3, PointBuffer, PointBufferSet, PointCloud,
};
use spatialrust_math::{Isometry3, TransformPoint, Vec3};
use spatialrust_records::SpatialRecord;

use crate::{SyncError, SyncResult};

/// One directed edge: `child_T_parent` transform.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameEdge {
    /// Parent frame.
    pub parent: FrameId,
    /// Child frame.
    pub child: FrameId,
    /// Transform that maps parent coordinates into child coordinates.
    pub child_t_parent: Isometry3<f32>,
}

/// Directed calibrated frame graph.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameGraph {
    /// Adjacency: parent → (child, child_T_parent).
    edges: HashMap<String, Vec<(String, Isometry3<f32>)>>,
    /// Reverse adjacency for lookups towards parents.
    reverse: HashMap<String, Vec<(String, Isometry3<f32>)>>,
}

impl FrameGraph {
    /// Creates an empty frame graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces a parent→child edge.
    pub fn insert_edge(&mut self, edge: FrameEdge) -> SyncResult<()> {
        if edge.parent.0 == edge.child.0 {
            return Err(SyncError::InvalidConfiguration(
                "frame edge parent and child must differ".into(),
            ));
        }
        self.edges
            .entry(edge.parent.0.clone())
            .or_default()
            .retain(|(child, _)| child != &edge.child.0);
        self.edges
            .entry(edge.parent.0.clone())
            .or_default()
            .push((edge.child.0.clone(), edge.child_t_parent));

        let parent_t_child = edge.child_t_parent.inverse();
        self.reverse
            .entry(edge.child.0.clone())
            .or_default()
            .retain(|(parent, _)| parent != &edge.parent.0);
        self.reverse.entry(edge.child.0.clone()).or_default().push((edge.parent.0, parent_t_child));
        Ok(())
    }

    /// Looks up a transform that maps `from` coordinates into `to` coordinates.
    pub fn lookup(&self, from: &FrameId, to: &FrameId) -> SyncResult<Isometry3<f32>> {
        if from == to {
            return Ok(Isometry3::identity());
        }
        let mut queue = VecDeque::from([(from.0.clone(), Isometry3::identity())]);
        let mut visited = HashSet::from([from.0.clone()]);
        while let Some((node, acc)) = queue.pop_front() {
            for (next, edge) in self.neighbors(&node) {
                if !visited.insert(next.clone()) {
                    continue;
                }
                let composed = edge.compose(acc);
                if next == to.0 {
                    return Ok(composed);
                }
                queue.push_back((next, composed));
            }
        }
        Err(SyncError::NoTransformPath { from: from.0.clone(), to: to.0.clone() })
    }

    /// Transforms a record from its metadata frame into `target`.
    ///
    /// Positions and complete normal triplets are transformed by the rigid
    /// frame path. Other columns, timestamps, schema, and source provenance
    /// are preserved. The returned record owns a new CPU cloud, so no hidden
    /// device transfer or in-place mutation occurs.
    pub fn transform_record_to(
        &self,
        record: &SpatialRecord,
        target: &FrameId,
    ) -> SyncResult<SpatialRecord> {
        let source = &record.metadata().frame_id;
        let transform = self.lookup(source, target)?;
        let cloud = record.cloud();
        let (x, y, z) = cloud.positions3()?;
        let mut positions = Vec::with_capacity(cloud.len());
        for index in 0..cloud.len() {
            positions.push(transform.transform_point(Vec3::new(x[index], y[index], z[index])));
        }

        let normal_names = normal_field_names(cloud)?;
        let normals = if normal_names.is_some() {
            let (x, y, z) = cloud.normals3()?;
            Some(
                (0..cloud.len())
                    .map(|index| {
                        transform
                            .transform_vector(Vec3::new(x[index], y[index], z[index]))
                            .normalize()
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        let position_names = position_field_names(cloud);
        let mut buffers = PointBufferSet::new();
        for field in cloud.schema().fields() {
            let buffer = if field.name == position_names.0 {
                PointBuffer::from_f32(positions.iter().map(|point| point.x).collect())
            } else if field.name == position_names.1 {
                PointBuffer::from_f32(positions.iter().map(|point| point.y).collect())
            } else if field.name == position_names.2 {
                PointBuffer::from_f32(positions.iter().map(|point| point.z).collect())
            } else if let Some((names, values)) = normal_names.as_ref().zip(normals.as_ref()) {
                if field.name == names.0 {
                    PointBuffer::from_f32(values.iter().map(|normal| normal.x).collect())
                } else if field.name == names.1 {
                    PointBuffer::from_f32(values.iter().map(|normal| normal.y).collect())
                } else if field.name == names.2 {
                    PointBuffer::from_f32(values.iter().map(|normal| normal.z).collect())
                } else {
                    clone_buffer(cloud.field(&field.name)?)
                }
            } else {
                clone_buffer(cloud.field(&field.name)?)
            };
            buffers.insert(field.name.clone(), buffer);
        }

        let mut metadata = cloud.metadata().clone();
        metadata.frame_id = target.clone();
        metadata.sensor_origin =
            metadata.sensor_origin.map(|origin| transform.transform_point(origin));
        let transformed_cloud =
            PointCloud::try_from_parts(cloud.schema().clone(), buffers, metadata)?;
        Ok(SpatialRecord::try_new_with_provenance(
            record.schema().clone(),
            transformed_cloud,
            record.provenance().clone(),
        )?)
    }

    fn neighbors(&self, node: &str) -> Vec<(String, Isometry3<f32>)> {
        let mut out = Vec::new();
        if let Some(forward) = self.edges.get(node) {
            out.extend(forward.iter().cloned());
        }
        if let Some(back) = self.reverse.get(node) {
            out.extend(back.iter().cloned());
        }
        out
    }
}

fn position_field_names(cloud: &PointCloud) -> (String, String, String) {
    (
        cloud
            .schema()
            .find_semantic(FieldSemantic::PositionX)
            .expect("positions3 validates position fields")
            .name
            .clone(),
        cloud
            .schema()
            .find_semantic(FieldSemantic::PositionY)
            .expect("positions3 validates position fields")
            .name
            .clone(),
        cloud
            .schema()
            .find_semantic(FieldSemantic::PositionZ)
            .expect("positions3 validates position fields")
            .name
            .clone(),
    )
}

fn normal_field_names(cloud: &PointCloud) -> SyncResult<Option<(String, String, String)>> {
    let names = [
        (FieldSemantic::NormalX, "normal x"),
        (FieldSemantic::NormalY, "normal y"),
        (FieldSemantic::NormalZ, "normal z"),
    ]
    .map(|(semantic, label)| {
        let fields: Vec<&str> = cloud
            .schema()
            .fields()
            .iter()
            .filter(|field| field.semantic == semantic)
            .map(|field| field.name.as_str())
            .collect();
        (label, fields)
    });
    let present = names.iter().filter(|(_, fields)| !fields.is_empty()).count();
    if present == 0 {
        return Ok(None);
    }
    if names.iter().any(|(_, fields)| fields.len() != 1) {
        return Err(SyncError::InvalidConfiguration(
            "normal fields must contain exactly one x/y/z semantic each".into(),
        ));
    }
    Ok(Some((names[0].1[0].to_owned(), names[1].1[0].to_owned(), names[2].1[0].to_owned())))
}

fn clone_buffer(buffer: &PointBuffer) -> PointBuffer {
    match buffer {
        PointBuffer::F32(values) => PointBuffer::from_f32(values.clone()),
        PointBuffer::F64(values) => PointBuffer::F64(values.clone()),
        PointBuffer::U8(values) => PointBuffer::U8(values.clone()),
        PointBuffer::U16(values) => PointBuffer::U16(values.clone()),
        PointBuffer::U32(values) => PointBuffer::U32(values.clone()),
        PointBuffer::I32(values) => PointBuffer::I32(values.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameEdge, FrameGraph};
    use spatialrust_core::{
        FrameId, HasNormals3, HasPositions3, PointBuffer, PointBufferSet, PointCloud,
        SpatialMetadata, StandardSchemas, Timestamp,
    };
    use spatialrust_math::{Isometry3, Quat, TransformPoint, Vec3};
    use spatialrust_records::{RecordProvenance, SchemaVersion, SpatialRecord};

    #[test]
    fn composes_chain_base_to_lidar() {
        let mut graph = FrameGraph::new();
        graph
            .insert_edge(FrameEdge {
                parent: FrameId::new("base"),
                child: FrameId::new("sensor"),
                child_t_parent: Isometry3::new(
                    Quat::new(0.0, 0.0, 0.0, 1.0),
                    Vec3::new(1.0, 0.0, 0.0),
                ),
            })
            .unwrap();
        graph
            .insert_edge(FrameEdge {
                parent: FrameId::new("sensor"),
                child: FrameId::new("lidar"),
                child_t_parent: Isometry3::new(
                    Quat::new(0.0, 0.0, 0.0, 1.0),
                    Vec3::new(0.0, 2.0, 0.0),
                ),
            })
            .unwrap();
        let t = graph.lookup(&FrameId::new("base"), &FrameId::new("lidar")).unwrap();
        let p = t.transform_point(Vec3::new(0.0, 0.0, 0.0));
        assert!((p.x - 1.0).abs() < 1e-5);
        assert!((p.y - 2.0).abs() < 1e-5);
    }

    #[test]
    fn transforms_record_and_preserves_non_geometry_data() {
        let mut graph = FrameGraph::new();
        graph
            .insert_edge(FrameEdge {
                parent: FrameId::new("base"),
                child: FrameId::new("sensor"),
                child_t_parent: Isometry3::new(
                    Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2),
                    Vec3::new(1.0, 2.0, 0.0),
                ),
            })
            .unwrap();

        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![1.0]));
        buffers.insert("y", PointBuffer::from_f32(vec![0.0]));
        buffers.insert("z", PointBuffer::from_f32(vec![0.0]));
        buffers.insert("intensity", PointBuffer::from_f32(vec![7.0]));
        buffers.insert("normal_x", PointBuffer::from_f32(vec![1.0]));
        buffers.insert("normal_y", PointBuffer::from_f32(vec![0.0]));
        buffers.insert("normal_z", PointBuffer::from_f32(vec![0.0]));
        let mut metadata = SpatialMetadata::new("base", Timestamp::from_nanos(42));
        metadata.sensor_origin = Some(Vec3::new(0.0, 0.0, 0.0));
        let cloud =
            PointCloud::try_from_parts(StandardSchemas::point_xyzinormal(), buffers, metadata)
                .unwrap();
        let provenance = RecordProvenance::try_new("bag")
            .unwrap()
            .with_stream_id("/lidar")
            .with_sequence(Some(9));
        let record = SpatialRecord::try_from_cloud_with_provenance(
            "point",
            SchemaVersion::new(1, 0),
            cloud,
            provenance.clone(),
        )
        .unwrap();

        let transformed = graph.transform_record_to(&record, &FrameId::new("sensor")).unwrap();
        let (x, y, z) = transformed.cloud().positions3().unwrap();
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!((y[0] - 3.0).abs() < 1e-5);
        assert!((z[0]).abs() < 1e-5);
        let (nx, ny, nz) = transformed.cloud().normals3().unwrap();
        assert!(nx[0].abs() < 1e-5);
        assert!((ny[0] - 1.0).abs() < 1e-5);
        assert!(nz[0].abs() < 1e-5);
        assert_eq!(transformed.metadata().frame_id, FrameId::new("sensor"));
        assert_eq!(transformed.metadata().sensor_origin, Some(Vec3::new(1.0, 2.0, 0.0)));
        assert_eq!(transformed.cloud().field("intensity").unwrap().as_f32().unwrap(), &[7.0]);
        assert_eq!(transformed.provenance(), &provenance);
    }

    #[test]
    fn transform_record_rejects_missing_path() {
        let mut buffers = PointBufferSet::new();
        buffers.insert("x", PointBuffer::from_f32(vec![0.0]));
        buffers.insert("y", PointBuffer::from_f32(vec![0.0]));
        buffers.insert("z", PointBuffer::from_f32(vec![0.0]));
        let cloud = PointCloud::try_from_parts(
            StandardSchemas::point_xyz(),
            buffers,
            SpatialMetadata::new("source", Timestamp::from_nanos(0)),
        )
        .unwrap();
        let record =
            SpatialRecord::try_from_cloud("point", SchemaVersion::new(1, 0), cloud).unwrap();
        let error =
            FrameGraph::new().transform_record_to(&record, &FrameId::new("target")).unwrap_err();
        assert!(matches!(error, crate::SyncError::NoTransformPath { .. }));
    }
}
