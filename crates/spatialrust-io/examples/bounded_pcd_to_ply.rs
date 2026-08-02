use spatialrust_io::{PcdChunkSource, PlyChunkSink, PlyWriteFormat};
use spatialrust_records::{
    BoundedSpatialRecordSink, BoundedSpatialRecordSource, CancellationToken, MemoryBudget,
    StreamOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let input = arguments.next().ok_or("usage: bounded_pcd_to_ply INPUT.pcd OUTPUT.ply")?;
    let output = arguments.next().ok_or("usage: bounded_pcd_to_ply INPUT.pcd OUTPUT.ply")?;

    let options = StreamOptions::new(64 * 1024, MemoryBudget::new(64 * 1024 * 1024)?)?;
    let cancellation = CancellationToken::default();
    let mut source = PcdChunkSource::open(input, options, cancellation)?;
    let point_count = source.header().points as u64;
    let mut sink = PlyChunkSink::create(
        output,
        source.schema().clone(),
        point_count,
        PlyWriteFormat::BinaryLittleEndian,
    )?;

    while let Some(chunk) = source.next_chunk() {
        sink.write_chunk(&chunk?)?;
    }
    sink.finish()?;
    Ok(())
}
