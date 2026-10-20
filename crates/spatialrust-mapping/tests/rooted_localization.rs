//! Independent public-API checks for anchored, atomic SE3 propagation.

use spatialrust_core::Timestamp;
use spatialrust_mapping::{MappingError, PoseGraph, PoseGraphEdge, PoseNodeId, StampedPose};
use spatialrust_math::{Isometry3, Pose3, Quat, TransformPoint, Vec3};
use spatialrust_sync::{ClockDomain, StampedTime};

fn pose(rotation: Quat<f32>, translation: [f32; 3], nanos: u64) -> StampedPose {
    StampedPose::new(
        StampedTime::exact("graph", ClockDomain::HostSteady, Timestamp::from_nanos(nanos)),
        Pose3::new(Isometry3::new(
            rotation,
            Vec3::new(translation[0], translation[1], translation[2]),
        )),
    )
}

fn identity(nanos: u64) -> StampedPose {
    pose(Quat::<f32>::identity(), [0.0; 3], nanos)
}

fn edge(from: &str, to: &str, rotation: Quat<f32>, translation: [f32; 3]) -> PoseGraphEdge {
    PoseGraphEdge {
        from: PoseNodeId::new(from),
        to: PoseNodeId::new(to),
        to_t_from: Isometry3::new(
            rotation,
            Vec3::new(translation[0], translation[1], translation[2]),
        ),
        loop_closure: false,
    }
}

fn shift(from: &str, to: &str, x: f32) -> PoseGraphEdge {
    edge(from, to, Quat::<f32>::identity(), [x, 0.0, 0.0])
}

fn quarter_turn(axis: [f32; 3]) -> Quat<f32> {
    Quat::from_axis_angle(Vec3::new(axis[0], axis[1], axis[2]), std::f32::consts::FRAC_PI_2)
}

fn assert_point(graph: &PoseGraph, id: &str, point: [f32; 3], expected: [f32; 3]) {
    let actual =
        graph.nodes()[id].pose.isometry.transform_point(Vec3::new(point[0], point[1], point[2]));
    for (actual, expected) in [actual.x, actual.y, actual.z].into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5, "{id}: {actual} != {expected}");
    }
}

fn assert_graph_failure_is_atomic(graph: &mut PoseGraph) {
    let before = graph.clone();
    assert!(matches!(
        graph.localize_from_root(&PoseNodeId::new("root")),
        Err(MappingError::Graph(_))
    ));
    assert_eq!(*graph, before);
}

#[test]
fn pure_rotation_updates_basis_without_translation() {
    let mut graph = PoseGraph::new();
    graph.upsert_node("root", identity(10));
    graph.upsert_node("child", identity(20));
    graph.add_edge(edge("root", "child", quarter_turn([0.0, 0.0, 1.0]), [0.0; 3])).unwrap();
    let stamp = graph.nodes()["child"].stamp.clone();
    graph.localize_from_root(&PoseNodeId::new("root")).unwrap();
    assert_point(&graph, "child", [0.0; 3], [0.0; 3]);
    assert_point(&graph, "child", [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert_point(&graph, "child", [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]);
    assert_eq!(graph.nodes()["child"].stamp, stamp);
}

#[test]
fn noncommuting_rotations_keep_left_composition_in_either_edge_order() {
    for reverse in [false, true] {
        let mut graph = PoseGraph::new();
        let root = pose(quarter_turn([0.0, 0.0, 1.0]), [1.0, 2.0, 3.0], 1);
        graph.upsert_node("root", root.clone());
        // Translation is already correct; the missing orientation must still change.
        graph.upsert_node("b", pose(Quat::<f32>::identity(), [5.0, 2.0, 8.0], 2));
        graph.upsert_node("c", identity(3));
        let mut edges = vec![
            edge("root", "b", quarter_turn([1.0, 0.0, 0.0]), [4.0, 5.0, 6.0]),
            edge("b", "c", quarter_turn([0.0, 1.0, 0.0]), [-2.0, 1.0, 0.0]),
        ];
        if reverse {
            edges.reverse();
        }
        for edge in edges {
            graph.add_edge(edge).unwrap();
        }
        let before_edges = graph.edges().to_vec();
        graph.localize_from_root(&PoseNodeId::new("root")).unwrap();
        // Hand-computed Ry(90) * Rx(90) * Rz(90), not a compose-based oracle.
        assert_point(&graph, "c", [0.0; 3], [6.0, 3.0, -5.0]);
        assert_point(&graph, "c", [1.0, 0.0, 0.0], [7.0, 3.0, -5.0]);
        assert_point(&graph, "c", [0.0, 1.0, 0.0], [6.0, 3.0, -4.0]);
        assert_point(&graph, "c", [0.0, 0.0, 1.0], [6.0, 2.0, -5.0]);
        assert_eq!(graph.nodes()["root"], root);
        assert_eq!(graph.nodes()["b"].stamp.as_nanos(), 2);
        assert_eq!(graph.nodes()["c"].stamp.as_nanos(), 3);
        assert_eq!(graph.edges(), before_edges);
        let localized = graph.clone();
        graph.localize_from_root(&PoseNodeId::new("root")).unwrap();
        assert_eq!(graph, localized);
    }
}

#[test]
fn tiny_translation_is_not_discarded_as_converged() {
    let mut graph = PoseGraph::new();
    graph.upsert_node("root", identity(0));
    graph.upsert_node("child", identity(1));
    graph.add_edge(shift("root", "child", 2e-6)).unwrap();
    graph.localize_from_root(&PoseNodeId::new("root")).unwrap();
    assert_eq!(graph.nodes()["child"].pose.isometry.translation().x, 2e-6);
}

#[test]
fn unchanged_parent_still_reaches_its_descendants() {
    let mut graph = PoseGraph::new();
    for (nanos, id) in ["root", "a", "b"].into_iter().enumerate() {
        graph.upsert_node(id, identity(nanos as u64));
    }
    graph.add_edge(shift("a", "b", 3.0)).unwrap();
    graph.add_edge(shift("root", "a", 0.0)).unwrap();
    graph.localize_from_root(&PoseNodeId::new("root")).unwrap();
    assert_point(&graph, "b", [0.0; 3], [3.0, 0.0, 0.0]);
}

#[test]
fn disconnected_component_is_not_localized() {
    let mut graph = PoseGraph::new();
    graph.upsert_node("root", identity(0));
    graph.upsert_node("u", pose(Quat::<f32>::identity(), [7.0, 0.0, 0.0], 1));
    graph.upsert_node("v", identity(2));
    graph.add_edge(shift("u", "v", 2.0)).unwrap();
    let before = graph.clone();
    graph.localize_from_root(&PoseNodeId::new("root")).unwrap();
    assert_eq!(graph, before);
}

#[test]
fn disconnected_incoming_edge_cannot_move_root_or_descendants() {
    let mut graph = PoseGraph::new();
    let root = pose(quarter_turn([0.0, 0.0, 1.0]), [1.0, 2.0, 3.0], 10);
    graph.upsert_node("root", root.clone());
    graph.upsert_node("u", identity(20));
    graph.upsert_node("child", identity(30));
    graph.add_edge(shift("u", "root", 99.0)).unwrap();
    graph.add_edge(shift("root", "child", 0.0)).unwrap();
    graph.localize_from_root(&PoseNodeId::new("root")).unwrap();
    assert_eq!(graph.nodes()["root"], root);
    assert_point(&graph, "child", [1.0, 0.0, 0.0], [1.0, 3.0, 3.0]);
}

#[test]
fn disconnected_cycles_are_outside_the_rooted_component() {
    let mut graph = PoseGraph::new();
    for (nanos, id) in ["root", "u", "v"].into_iter().enumerate() {
        graph.upsert_node(id, identity(nanos as u64));
    }
    graph.add_edge(shift("u", "v", 2.0)).unwrap();
    graph.add_edge(shift("v", "u", -2.0)).unwrap();
    let before = graph.clone();
    graph.localize_from_root(&PoseNodeId::new("root")).unwrap();
    assert_eq!(graph, before);
}

#[test]
fn reachable_cycles_and_self_edges_fail_atomically() {
    for target in ["root", "a", "b"] {
        let mut graph = PoseGraph::new();
        for (nanos, id) in ["root", "a", "b"].into_iter().enumerate() {
            graph.upsert_node(id, identity(nanos as u64));
        }
        graph.add_edge(shift("root", "a", 1.0)).unwrap();
        graph.add_edge(shift("a", "b", 2.0)).unwrap();
        graph.add_edge(shift("b", target, 3.0)).unwrap();
        assert_graph_failure_is_atomic(&mut graph);
    }
}

#[test]
fn duplicate_constraints_fail_atomically_even_when_equal() {
    let mut graph = PoseGraph::new();
    graph.upsert_node("root", identity(0));
    graph.upsert_node("a", identity(1));
    graph.add_edge(shift("root", "a", 1.0)).unwrap();
    graph.add_edge(shift("root", "a", 1.0)).unwrap();
    assert_graph_failure_is_atomic(&mut graph);
}

#[test]
fn multiple_incoming_paths_fail_atomically() {
    let mut graph = PoseGraph::new();
    for (nanos, id) in ["root", "a", "b", "c"].into_iter().enumerate() {
        graph.upsert_node(id, identity(nanos as u64));
    }
    for (from, to) in [("root", "a"), ("root", "b"), ("a", "c"), ("b", "c")] {
        graph.add_edge(shift(from, to, 1.0)).unwrap();
    }
    assert_graph_failure_is_atomic(&mut graph);
}

#[test]
fn loop_closure_edges_remain_stored_without_propagation() {
    let mut graph = PoseGraph::new();
    graph.upsert_node("root", identity(0));
    graph.upsert_node("a", identity(1));
    graph.add_edge(shift("root", "a", 2.0)).unwrap();
    let mut closure = shift("a", "root", 99.0);
    closure.loop_closure = true;
    graph.add_edge(closure).unwrap();
    let edges = graph.edges().to_vec();
    graph.localize_from_root(&PoseNodeId::new("root")).unwrap();
    assert_point(&graph, "root", [0.0; 3], [0.0; 3]);
    assert_point(&graph, "a", [0.0; 3], [2.0, 0.0, 0.0]);
    assert_eq!(graph.edges(), edges);
}

#[test]
fn missing_root_fails_without_mutation() {
    let mut graph = PoseGraph::new();
    graph.upsert_node("a", identity(1));
    let before = graph.clone();
    assert!(matches!(
        graph.localize_from_root(&PoseNodeId::new("missing")),
        Err(MappingError::Missing(_))
    ));
    assert_eq!(graph, before);
}

#[test]
fn nonfinite_root_is_rejected_before_any_update() {
    let mut graph = PoseGraph::new();
    graph.upsert_node("root", pose(Quat::<f32>::identity(), [f32::NAN, 0.0, 0.0], 0));
    graph.upsert_node("a", identity(1));
    graph.add_edge(shift("root", "a", 1.0)).unwrap();
    let child = graph.nodes()["a"].clone();
    assert!(matches!(
        graph.localize_from_root(&PoseNodeId::new("root")),
        Err(MappingError::Graph(_))
    ));
    assert!(graph.nodes()["root"].pose.isometry.translation().x.is_nan());
    assert_eq!(graph.nodes()["a"], child);
}

#[test]
fn nonfinite_reachable_edge_leaves_all_node_poses_unchanged() {
    let mut graph = PoseGraph::new();
    for (nanos, id) in ["root", "a", "b"].into_iter().enumerate() {
        graph.upsert_node(id, identity(nanos as u64));
    }
    graph.add_edge(shift("root", "a", 1.0)).unwrap();
    graph.add_edge(shift("a", "b", f32::NAN)).unwrap();
    let nodes = graph.nodes().clone();
    assert!(matches!(
        graph.localize_from_root(&PoseNodeId::new("root")),
        Err(MappingError::Graph(_))
    ));
    assert_eq!(graph.nodes(), &nodes);
    assert!(graph.edges()[1].to_t_from.translation().x.is_nan());
}

#[test]
fn finite_input_overflow_fails_atomically() {
    let mut graph = PoseGraph::new();
    graph.upsert_node("root", pose(Quat::<f32>::identity(), [3e38, 0.0, 0.0], 0));
    graph.upsert_node("a", identity(1));
    graph.add_edge(shift("root", "a", 3e38)).unwrap();
    assert_graph_failure_is_atomic(&mut graph);
}

#[test]
fn reversed_long_chain_propagates_without_recursion() {
    let mut graph = PoseGraph::new();
    const COUNT: usize = 4096;
    for index in 0..=COUNT {
        graph.upsert_node(format!("n{index}"), identity(index as u64));
    }
    for index in (0..COUNT).rev() {
        graph.add_edge(shift(&format!("n{index}"), &format!("n{}", index + 1), 1.0)).unwrap();
    }
    graph.localize_from_root(&PoseNodeId::new("n0")).unwrap();
    assert_point(&graph, &format!("n{COUNT}"), [0.0; 3], [COUNT as f32, 0.0, 0.0]);
    assert_eq!(graph.nodes()[&format!("n{COUNT}")].stamp.as_nanos(), COUNT as u64);
}
