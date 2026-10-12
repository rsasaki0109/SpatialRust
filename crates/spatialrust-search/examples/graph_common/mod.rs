#[path = "../../tests/fixtures/graph_serial.rs"]
mod previous;

use spatialrust_core::{PointCloud, PointCloudBuilder};
use spatialrust_search::{
    knn_graph, knn_graph_parallel, radius_graph, radius_graph_parallel, NeighborGraph,
};

#[derive(Clone, Copy, Debug)]
pub enum Kind {
    Knn8,
    Knn32,
    Radius,
}
#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Previous,
    Serial,
    Parallel,
}

pub fn fixture(points: usize, duplicates: bool) -> PointCloud {
    let mut builder = PointCloudBuilder::xyz();
    for i in 0..points {
        let j = if duplicates { i / 8 } else { i };
        let spacing = if duplicates { 3.0 } else { 1.0 };
        builder
            .push_point([
                (j % 64) as f32 * spacing - 96.0,
                ((j / 64) % 64) as f32 * spacing - 96.0,
                (j / 4096) as f32 * spacing - 12.0,
            ])
            .unwrap();
    }
    builder.build().unwrap()
}

pub fn run(cloud: &PointCloud, kind: Kind, mode: Mode) -> NeighborGraph {
    match (kind, mode) {
        (Kind::Knn8, Mode::Previous) => previous::knn_graph(cloud, 8),
        (Kind::Knn32, Mode::Previous) => previous::knn_graph(cloud, 32),
        (Kind::Radius, Mode::Previous) => previous::radius_graph(cloud, 1.01),
        (Kind::Knn8, Mode::Serial) => knn_graph(cloud, 8),
        (Kind::Knn32, Mode::Serial) => knn_graph(cloud, 32),
        (Kind::Radius, Mode::Serial) => radius_graph(cloud, 1.01),
        (Kind::Knn8, Mode::Parallel) => knn_graph_parallel(cloud, 8),
        (Kind::Knn32, Mode::Parallel) => knn_graph_parallel(cloud, 32),
        (Kind::Radius, Mode::Parallel) => radius_graph_parallel(cloud, 1.01),
    }
    .unwrap()
}
