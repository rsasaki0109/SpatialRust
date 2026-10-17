use super::*;
use std::{hint::black_box, time::Instant};

fn stage_samples(
    r: &WgpuRuntime,
    prior: &ParentPipelines,
    candidate: &ParentPipelines,
    input: &[Entry],
    expected: &[Entry],
) -> [[Vec<f64>; 2]; 2] {
    let count = input.len() as u32;
    let seed = buffer(r, bytemuck::cast_slice(input));
    let work = [Work::new(r, input), Work::new(r, input)];
    let pipelines =
        [prior.for_groups(count.div_ceil(256)), candidate.for_groups(count.div_ceil(256))];
    let mut samples: [[Vec<f64>; 2]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    for round in 0..33 {
        let mut profiler = GpuStageProfiler::new(r, 144).unwrap();
        let mut encoder = r.device().create_command_encoder(&Default::default());
        for offset in 0..2 {
            let mode = (round + offset) % 2;
            encoder.copy_buffer_to_buffer(&seed, 0, &work[mode].entries[0], 0, seed.size());
            work[mode].encode(&mut encoder, pipelines[mode], &mut Some(&mut profiler));
        }
        r.queue().submit(Some(encoder.finish()));
        let report = profiler.collect(r).unwrap();
        assert_eq!(report.stages.len(), 144);
        assert_eq!(report.device_to_host_bytes, 2304);
        if round >= 3 {
            for (sequence, stages) in report.stages.chunks_exact(72).enumerate() {
                let mode = (round + sequence) % 2;
                samples[0][mode].push(
                    stages
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| i % 3 == 1)
                        .map(|(_, s)| s.elapsed_ns)
                        .sum::<f64>()
                        / 1_000_000.,
                );
                samples[1][mode]
                    .push(stages.iter().map(|s| s.elapsed_ns).sum::<f64>() / 1_000_000.);
            }
        }
    }
    for w in &work {
        assert_eq!(read_words(r, &w.entries[0]), bytemuck::cast_slice::<Entry, u32>(expected));
    }
    samples
}

fn measure(
    r: &WgpuRuntime,
    prior: &ParentPipelines,
    candidate: &ParentPipelines,
    count: usize,
    pattern: usize,
) {
    let (input, expected) = stable_inputs(count, pattern);
    let mut wall: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    for round in 0..33 {
        for offset in 0..4 {
            let variant = (round + offset) % 4;
            let current = variant % 2 == 1;
            let full = variant >= 2;
            let uploaded = (!full).then(|| buffer(r, bytemuck::cast_slice(&input)));
            r.wait_idle();
            let start = Instant::now();
            let entries = uploaded
                .unwrap_or_else(|| buffer(black_box(r), bytemuck::cast_slice(black_box(&input))));
            let output = if current {
                old::previous(candidate, black_box(r), entries, count as u32, None)
            } else {
                old::previous(prior, black_box(r), entries, count as u32, None)
            };
            let downloaded = if full {
                Some(read_words(r, &output))
            } else {
                r.wait_idle();
                None
            };
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            let downloaded = downloaded.unwrap_or_else(|| read_words(r, &output));
            assert_eq!(downloaded, bytemuck::cast_slice::<Entry, u32>(&expected));
            if round >= 3 {
                wall[variant].push(elapsed);
            }
        }
    }
    let stages = stage_samples(r, prior, candidate, &input, &expected);
    println!("RADIX_FOUR_RECORD {{\"count\":{count},\"pattern\":{pattern},\"cached\":{},\"prefix_old_ms\":{:?},\"prefix_ms\":{:?},\"sort_gpu_old_ms\":{:?},\"sort_gpu_ms\":{:?},\"resident_old_ms\":{:?},\"resident_ms\":{:?},\"full_old_ms\":{:?},\"full_ms\":{:?}}}",count>524288,stages[0][0],stages[0][1],stages[1][0],stages[1][1],wall[0],wall[1],wall[2],wall[3]);
}

#[test]
#[ignore = "paired prefix/all-pass timestamps and completed private sort/upload-sort-readback; run alone in release mode"]
fn benchmark_complete_voxel_radix_four_calls() {
    let r =
        WgpuRuntime::new_headless_with_timestamps(WgpuPowerPreference::HighPerformance).unwrap();
    assert!(r.timestamp_queries_enabled());
    let prior = ParentPipelines::new(&r);
    let candidate = ParentPipelines::candidate(&r);
    println!("RADIX_FOUR_ENV backend={} device_type={} available_workers={} warmups=3 pairs=30 routes=prefix_gpu_sort_gpu_resident_sort_and_upload_sort_readback checks_drop_excluded=true",r.adapter_info().backend,r.adapter_info().device_type,std::thread::available_parallelism().map_or(1,|n|n.get()));
    for count in [257, 65537, 524288] {
        measure(&r, &prior, &candidate, count, 0);
    }
    for count in [524289, 1000003, 2000003] {
        for pattern in 0..3 {
            measure(&r, &prior, &candidate, count, pattern);
        }
    }
}
