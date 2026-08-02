use criterion::{black_box, criterion_group, criterion_main, Criterion};
use spatialrust_distribute::BackpressurePolicy;
use spatialrust_runtime::{FnOperator, GraphNodeSpec, SpatialExecutionGraph};

fn benchmark_graph(c: &mut Criterion) {
    let mut graph = SpatialExecutionGraph::new(BackpressurePolicy::try_new(8, 16).unwrap());
    for index in 0..8 {
        let id = format!("stage-{index}");
        graph
            .add_node(
                GraphNodeSpec::try_new(&id, "cpu", true).unwrap(),
                FnOperator::new(|value: u64| Ok(value.wrapping_add(1))),
            )
            .unwrap();
        if index > 0 {
            graph.connect(format!("stage-{}", index - 1), id).unwrap();
        }
    }
    let mut graph = graph.compile().unwrap();
    c.bench_function("fused_graph_8_stages", |b| {
        b.iter(|| {
            graph.try_submit(black_box(0)).unwrap();
            black_box(graph.run_next().unwrap().unwrap())
        })
    });
}

criterion_group!(benches, benchmark_graph);
criterion_main!(benches);
