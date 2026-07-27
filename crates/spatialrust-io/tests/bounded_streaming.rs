#![cfg(all(
    feature = "streaming",
    feature = "io-pcd",
    feature = "io-ply",
    feature = "io-las",
    feature = "io-copc"
))]

use std::io::Cursor;

use spatialrust_core::{PointCloud, PointCloudBuilder};
use spatialrust_io::{
    read_copc_file, read_las_file, read_pcd, read_ply, write_copc_stream, BoundedSpool,
    CopcChunkSource, CopcWriterParams, LasChunkSink, LasWriteFormat, PcdChunkSink, PcdChunkSource,
    PcdWriteFormat, PlyChunkSink, PlyChunkSource, PlyWriteFormat, SpoolOptions,
};
use spatialrust_records::{
    BoundedSpatialRecordSink, BoundedSpatialRecordSource, CancellationToken, MemoryBudget,
    RecyclingMemoryChunkSource, SchemaDescriptor, SchemaVersion, StreamOptions,
};

fn cloud(points: usize) -> PointCloud {
    let mut builder = PointCloudBuilder::xyz();
    for index in 0..points {
        builder.push_point([index as f32, index as f32 * 2.0, -(index as f32)]).unwrap();
    }
    builder.build().unwrap()
}

fn options(chunk_points: usize, bytes: u64) -> StreamOptions {
    StreamOptions::new(chunk_points, MemoryBudget::new(bytes).unwrap()).unwrap()
}

fn descriptor(cloud: &PointCloud) -> SchemaDescriptor {
    SchemaDescriptor::try_new("test.xyz", SchemaVersion::new(1, 0), cloud.schema().clone()).unwrap()
}

fn collect_lengths(source: &mut impl BoundedSpatialRecordSource) -> Vec<(u64, u64, usize)> {
    let mut chunks = Vec::new();
    while let Some(chunk) = source.next_chunk() {
        let chunk = chunk.unwrap();
        chunks.push((
            chunk.identity().sequence,
            chunk.identity().point_offset,
            chunk.record().cloud().len(),
        ));
    }
    chunks
}

#[test]
fn pcd_and_ply_sources_chunk_without_materializing_the_file() {
    let input = cloud(5);
    let mut pcd = Vec::new();
    spatialrust_io::write_pcd(&mut pcd, &input, PcdWriteFormat::Binary).unwrap();
    let mut pcd_source =
        PcdChunkSource::new(Cursor::new(pcd), options(2, 48), CancellationToken::default())
            .unwrap();
    assert_eq!(collect_lengths(&mut pcd_source), vec![(0, 0, 2), (1, 2, 2), (2, 4, 1)]);
    assert_eq!(pcd_source.memory_tracker().snapshot().peak_bytes, 48);

    let mut ply = Vec::new();
    spatialrust_io::write_ply(&mut ply, &input, PlyWriteFormat::BinaryLittleEndian).unwrap();
    let mut ply_source =
        PlyChunkSource::new(Cursor::new(ply), options(2, 48), CancellationToken::default())
            .unwrap();
    assert_eq!(collect_lengths(&mut ply_source), vec![(0, 0, 2), (1, 2, 2), (2, 4, 1)]);
    assert_eq!(ply_source.memory_tracker().snapshot().peak_bytes, 48);
}

#[test]
fn ascii_sources_chunk_and_reject_oversize_records() {
    let input = cloud(3);
    let mut pcd = Vec::new();
    spatialrust_io::write_pcd(&mut pcd, &input, PcdWriteFormat::Ascii).unwrap();
    let mut source =
        PcdChunkSource::new(Cursor::new(pcd), options(2, 24), CancellationToken::default())
            .unwrap();
    assert_eq!(collect_lengths(&mut source), vec![(0, 0, 2), (1, 2, 1)]);

    let mut ply = Vec::new();
    spatialrust_io::write_ply(&mut ply, &input, PlyWriteFormat::Ascii).unwrap();
    let mut source =
        PlyChunkSource::new(Cursor::new(ply), options(2, 24), CancellationToken::default())
            .unwrap();
    assert_eq!(collect_lengths(&mut source), vec![(0, 0, 2), (1, 2, 1)]);

    let mut oversized = b"VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA ascii\n".to_vec();
    oversized.extend(std::iter::repeat_n(b'1', 17 * 1024));
    oversized.push(b'\n');
    let mut source =
        PcdChunkSource::new(Cursor::new(oversized), options(1, 12), CancellationToken::default())
            .unwrap();
    assert!(source.next_chunk().unwrap().is_err());
    assert_eq!(source.memory_tracker().snapshot().current_bytes, 0);
}

#[test]
fn pcd_and_ply_sinks_enforce_counts_and_round_trip() {
    let input = cloud(5);
    let schema = descriptor(&input);

    let mut pcd_bytes = Vec::new();
    {
        let mut source = RecyclingMemoryChunkSource::try_new(
            schema.clone(),
            input,
            options(2, 24),
            CancellationToken::default(),
        )
        .unwrap();
        let mut sink =
            PcdChunkSink::new(&mut pcd_bytes, schema.clone(), 5, PcdWriteFormat::Binary).unwrap();
        while let Some(chunk) = source.next_chunk() {
            sink.write_chunk(&chunk.unwrap()).unwrap();
        }
        sink.finish().unwrap();
    }
    assert_eq!(read_pcd(&mut Cursor::new(pcd_bytes)).unwrap().len(), 5);

    let input = cloud(5);
    let mut ply_bytes = Vec::new();
    {
        let mut source = RecyclingMemoryChunkSource::try_new(
            schema.clone(),
            input,
            options(2, 24),
            CancellationToken::default(),
        )
        .unwrap();
        let mut sink =
            PlyChunkSink::new(&mut ply_bytes, schema, 5, PlyWriteFormat::BinaryLittleEndian)
                .unwrap();
        while let Some(chunk) = source.next_chunk() {
            sink.write_chunk(&chunk.unwrap()).unwrap();
        }
        sink.finish().unwrap();
    }
    assert_eq!(read_ply(&mut Cursor::new(ply_bytes)).unwrap().len(), 5);
}

#[test]
fn las_sink_and_source_round_trip_in_chunks() {
    let input = cloud(5);
    let schema = descriptor(&input);
    let path = std::env::temp_dir().join(format!("spatialrust_bounded_{}.las", std::process::id()));
    let mut source = RecyclingMemoryChunkSource::try_new(
        schema.clone(),
        input,
        options(2, 24),
        CancellationToken::default(),
    )
    .unwrap();
    let mut sink = LasChunkSink::create(&path, schema, 5, LasWriteFormat::Las).unwrap();
    while let Some(chunk) = source.next_chunk() {
        sink.write_chunk(&chunk.unwrap()).unwrap();
    }
    sink.finish().unwrap();
    drop(sink);

    assert_eq!(read_las_file(&path).unwrap().len(), 5);
    let mut source =
        spatialrust_io::LasChunkSource::open(&path, options(2, 34), CancellationToken::default())
            .unwrap();
    assert_eq!(collect_lengths(&mut source), vec![(0, 0, 2), (1, 2, 2), (2, 4, 1)]);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn copc_stream_writer_and_local_source_are_bounded_and_deterministic() {
    let input = cloud(12);
    let schema = descriptor(&input);
    let mut source = RecyclingMemoryChunkSource::try_new(
        schema,
        input,
        options(3, 36),
        CancellationToken::default(),
    )
    .unwrap();
    let path =
        std::env::temp_dir().join(format!("spatialrust_bounded_{}.copc.laz", std::process::id()));
    let spool = SpoolOptions::new(std::env::temp_dir(), 1024 * 1024).unwrap();
    let receipt =
        write_copc_stream(&path, &mut source, 12, &CopcWriterParams::default(), &spool).unwrap();
    assert_eq!(receipt.points, 12);
    assert!(receipt.spill_bytes > 0);
    assert_eq!(read_copc_file(&path).unwrap().len(), 12);

    let mut source =
        CopcChunkSource::open(&path, None, options(4, 1024 * 1024), CancellationToken::default())
            .unwrap();
    let chunks = collect_lengths(&mut source);
    assert_eq!(chunks.iter().map(|chunk| chunk.2).sum::<usize>(), 12);
    assert!(chunks.iter().enumerate().all(|(index, chunk)| chunk.0 == index as u64));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn bounded_spool_preflights_growth_and_removes_partial_file() {
    use std::io::Write;

    let options = SpoolOptions::new(std::env::temp_dir(), 3).unwrap();
    let path;
    {
        let mut spool = BoundedSpool::create(&options, "integration").unwrap();
        path = spool.path().to_path_buf();
        spool.write_all(b"123").unwrap();
        assert!(spool.write_all(b"4").is_err());
    }
    assert!(!path.exists());
}

#[test]
fn cancellation_and_spill_limits_fail_before_source_progress() {
    let input = cloud(4);
    let schema = descriptor(&input);
    let cancellation = CancellationToken::default();
    cancellation.cancel();
    let mut cancelled =
        RecyclingMemoryChunkSource::try_new(schema.clone(), input, options(2, 24), cancellation)
            .unwrap();
    assert!(matches!(
        cancelled.next_chunk().unwrap(),
        Err(spatialrust_records::RecordsError::Cancelled)
    ));
    assert_eq!(cancelled.memory_tracker().snapshot().peak_bytes, 0);

    let input = cloud(4);
    let mut source = RecyclingMemoryChunkSource::try_new(
        schema,
        input,
        options(2, 24),
        CancellationToken::default(),
    )
    .unwrap();
    let path =
        std::env::temp_dir().join(format!("spatialrust_rejected_{}.copc.laz", std::process::id()));
    let spool = SpoolOptions::new(std::env::temp_dir(), 1).unwrap();
    assert!(write_copc_stream(&path, &mut source, 4, &CopcWriterParams::default(), &spool).is_err());
    assert_eq!(source.memory_tracker().snapshot().peak_bytes, 0);
    assert!(!path.exists());
}

#[test]
fn streaming_sink_rejects_a_short_final_count() {
    let input = cloud(2);
    let schema = descriptor(&input);
    let mut bytes = Vec::new();
    let mut source = RecyclingMemoryChunkSource::try_new(
        schema.clone(),
        input,
        options(2, 24),
        CancellationToken::default(),
    )
    .unwrap();
    let mut sink = PcdChunkSink::new(&mut bytes, schema, 3, PcdWriteFormat::Binary).unwrap();
    sink.write_chunk(&source.next_chunk().unwrap().unwrap()).unwrap();
    assert!(sink.finish().is_err());
}
