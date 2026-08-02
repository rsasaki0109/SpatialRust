use crate::{LodError, LodIndex, LodResult, NodeId};

/// Converts selected index node bounds into one bounded COPC resolution query.
///
/// The caller remains responsible for executing and cancelling the query. This
/// adapter does not open a file, issue a range request, or materialize points.
pub fn copc_query_for_nodes(
    index: &LodIndex,
    nodes: &[NodeId],
    max_resolution: f64,
) -> LodResult<spatialrust_io::CopcQuery> {
    if nodes.is_empty() {
        return Err(LodError::Copc("at least one selected node is required".into()));
    }
    if !max_resolution.is_finite() || max_resolution <= 0.0 {
        return Err(LodError::Copc("COPC max resolution must be finite and positive".into()));
    }
    let first = index.node(nodes[0]).ok_or(LodError::UnknownNode(nodes[0].0))?;
    let mut bounds = first.bounds;
    for id in &nodes[1..] {
        let node = index.node(*id).ok_or(LodError::UnknownNode(id.0))?;
        bounds = bounds.union(node.bounds);
    }
    let copc_bounds = spatialrust_io::CopcBounds::new(
        [bounds.min.x as f64, bounds.min.y as f64, bounds.min.z as f64],
        [bounds.max.x as f64, bounds.max.y as f64, bounds.max.z as f64],
    );
    let query = spatialrust_io::CopcQuery::with_resolution(copc_bounds, max_resolution);
    query.validate().map_err(|error| LodError::Copc(error.to_string()))?;
    Ok(query)
}

#[cfg(test)]
mod tests {
    use spatialrust_math::Vec3;

    use crate::{LodBounds, LodIndex, LodNode, NodeId};

    #[test]
    fn selected_nodes_become_one_bounded_resolution_query() {
        let index = LodIndex::try_new([
            LodNode {
                id: NodeId(0),
                parent: None,
                children: vec![NodeId(1)],
                bounds: LodBounds::try_new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0))
                    .unwrap(),
                geometric_error: 1.0,
                point_count: 10,
                host_bytes: 40,
                upload_bytes: 40,
                gpu_bytes: 40,
            },
            LodNode {
                id: NodeId(1),
                parent: Some(NodeId(0)),
                children: Vec::new(),
                bounds: LodBounds::try_new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0))
                    .unwrap(),
                geometric_error: 0.5,
                point_count: 5,
                host_bytes: 20,
                upload_bytes: 20,
                gpu_bytes: 20,
            },
        ])
        .unwrap();
        let query = super::copc_query_for_nodes(&index, &[NodeId(1)], 0.25).unwrap();
        assert_eq!(query.bounds.min, [0.0, 0.0, 0.0]);
        assert_eq!(query.bounds.max, [1.0, 1.0, 1.0]);
        assert_eq!(query.max_resolution, Some(0.25));
        assert!(super::copc_query_for_nodes(&index, &[], 0.25).is_err());
    }
}
