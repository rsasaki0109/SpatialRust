use spatialrust_core::{PointCloud, PointCloudBuilder, SpatialTensor};
use spatialrust_search::{
    nearest_k_spatial_tensor, nearest_k_spatial_tensor_parallel_into, radius_search_spatial_tensor,
    radius_search_spatial_tensor_parallel_into, ChunkedNearestNeighborIndex,
    ChunkedRadiusSearchIndex, KdTree, Neighbor,
};

#[derive(Clone, Copy, Debug)]
pub enum Kind {
    Radius,
    Knn,
}

#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Serial,
    Bounded,
    Legacy,
}

pub struct Fixture {
    cloud: PointCloud,
    tree: KdTree,
    chunk_size: usize,
}

pub struct Query<'a> {
    tree: &'a KdTree,
    x: &'a [f32],
    y: &'a [f32],
    z: &'a [f32],
    tensor: SpatialTensor<'a>,
}

impl Fixture {
    pub fn new(points: usize, chunk_size: usize) -> Self {
        let mut builder = PointCloudBuilder::xyz();
        for i in 0..points {
            builder.push_point([(i % 256) as f32, (i / 256) as f32, 0.0]).unwrap();
        }
        let cloud = builder.build().unwrap();
        let tree = KdTree::from_point_cloud(&cloud).unwrap();
        Self { cloud, tree, chunk_size }
    }

    pub fn query(&self) -> Query<'_> {
        Query {
            tree: &self.tree,
            x: self.cloud.field("x").unwrap().as_f32().unwrap(),
            y: self.cloud.field("y").unwrap().as_f32().unwrap(),
            z: self.cloud.field("z").unwrap().as_f32().unwrap(),
            tensor: self.cloud.spatial_tensor_chunks(self.chunk_size).unwrap(),
        }
    }
}

impl Query<'_> {
    pub fn run(&self, kind: Kind, mode: Mode) -> Vec<(usize, Neighbor)> {
        let mut out = Vec::new();
        match (kind, mode) {
            (Kind::Radius, Mode::Serial) => radius_search_spatial_tensor(
                self.tree,
                self.x,
                self.y,
                self.z,
                &self.tensor,
                1.01,
                &mut out,
            ),
            (Kind::Knn, Mode::Serial) => nearest_k_spatial_tensor(
                self.tree,
                self.x,
                self.y,
                self.z,
                &self.tensor,
                8,
                &mut out,
            ),
            (Kind::Radius, Mode::Bounded) => radius_search_spatial_tensor_parallel_into(
                self.tree,
                self.x,
                self.y,
                self.z,
                &self.tensor,
                1.01,
                &mut out,
            ),
            (Kind::Knn, Mode::Bounded) => nearest_k_spatial_tensor_parallel_into(
                self.tree,
                self.x,
                self.y,
                self.z,
                &self.tensor,
                8,
                &mut out,
            ),
            (_, Mode::Legacy) => {
                // Reproduce the previous thread-per-chunk implementation only
                // for modest chunk counts. Never run it on tiny-chunk fixtures.
                assert!(self.tensor.len().div_ceil(self.tensor.chunk_size()) <= 256);
                let ranges: Vec<_> = self.tensor.chunks().map(|chunk| chunk.range()).collect();
                std::thread::scope(|scope| {
                    let mut handles = Vec::with_capacity(ranges.len());
                    for range in ranges {
                        handles.push(scope.spawn(move || {
                            let mut local = Vec::new();
                            match kind {
                                Kind::Radius => self.tree.radius_search_chunk_into(
                                    self.x, self.y, self.z, range, 1.01, &mut local,
                                ),
                                Kind::Knn => self.tree.nearest_k_chunk_into(
                                    self.x, self.y, self.z, range, 8, &mut local,
                                ),
                            }
                            local
                        }));
                    }
                    for handle in handles {
                        out.extend(handle.join().expect("legacy reference worker failed"));
                    }
                });
            }
        }
        out
    }
}
