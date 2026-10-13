//! Isolated paired completed selection and full/prepared moment calls.
use super::*;
use std::time::Instant;

fn upload_matches(runtime: &WgpuRuntime, words: &[[u32; 2]]) -> wgpu::Buffer {
    runtime.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(words),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn selection_case(
    runtimes: [&WgpuRuntime; 2],
    caches: [&TrimPipelines; 2],
    n: usize,
    profile: usize,
) {
    let words: Vec<[u32; 2]> = (0..n)
        .map(|i| {
            let distance = if profile == 0 { (i * 7919 % 1024) as f32 / 1024.0 } else { 0.25 };
            [if profile == 0 && i % 7 == 0 { u32::MAX } else { 0 }, distance.to_bits()]
        })
        .collect();
    let inputs = runtimes.map(|runtime| upload_matches(runtime, &words));
    let params = TrimParams::new(
        n as u32,
        plan_trim(n as u64, &runtimes[0].device().limits()).unwrap(),
        0.8,
        0.5,
        1,
    );
    let expected = selection_snapshot(runtimes[0], caches[0], &inputs[0], params);
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| vec![]);
    for round in 0..23 {
        for offset in 0..4 {
            let stage = (round + offset) % 4;
            let variant = stage % 2;
            let runtime = runtimes[variant];
            let start = Instant::now();
            let uploaded = (stage < 2).then(|| upload_matches(runtime, &words));
            let output = selection_resident(
                runtime,
                caches[variant],
                uploaded.as_ref().unwrap_or(&inputs[variant]),
                params,
            );
            let actual;
            let elapsed;
            if stage < 2 {
                actual = read(runtime, &output, 32);
                drop(output);
                drop(uploaded);
                elapsed = start.elapsed();
            } else {
                runtime.wait_idle();
                elapsed = start.elapsed();
                actual = read(runtime, &output, 32);
            }
            assert_eq!(actual, expected, "n={n} profile={profile}");
            if round >= 3 {
                samples[stage].push(elapsed.as_secs_f64() * 1000.0);
            }
        }
    }
    println!("GPU_ICP_SELECTOR_BENCH count={n} profile={profile} groups={} gated={} retained={} old_full_ms={:?} new_full_ms={:?} old_resident_ms={:?} new_resident_ms={:?}",params.groups,expected[0],expected[1],samples[0],samples[1],samples[2],samples[3]);
}

fn moment_case(runtimes: [&WgpuRuntime; 2], n: usize, profile: usize) {
    let source: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            [
                0.125,
                ((i * 17 % 9) as f32 - 4.0) / 64.0,
                if profile == 0 || i % 7 != 0 { ((i * 31 % 7) as f32 - 3.0) / 64.0 } else { 1.0 },
            ]
        })
        .collect();
    let target = tree(&[[0.0; 3]]);
    let inputs = runtimes.map(|runtime| GpuIcpInput::upload(runtime, &source, &target).unwrap());
    let expected = inputs[0]
        .trimmed_moments_resident(runtimes[0], Isometry3::identity(), 0.25, 0.8)
        .unwrap()
        .download(runtimes[0])
        .unwrap();
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| vec![]);
    for round in 0..23 {
        for offset in 0..4 {
            let stage = (round + offset) % 4;
            let variant = stage % 2;
            let runtime = runtimes[variant];
            let start = Instant::now();
            let uploaded = (stage < 2)
                .then(|| GpuIcpInput::upload(runtime, &source, &tree(&[[0.0; 3]])).unwrap());
            let input = uploaded.as_ref().unwrap_or(&inputs[variant]);
            let output =
                input.trimmed_moments_resident(runtime, Isometry3::identity(), 0.25, 0.8).unwrap();
            let actual;
            let elapsed;
            if stage < 2 {
                actual = output.download(runtime).unwrap();
                drop(output);
                drop(uploaded);
                elapsed = start.elapsed();
            } else {
                runtime.wait_idle();
                elapsed = start.elapsed();
                actual = output.download(runtime).unwrap();
            }
            assert_eq!(actual, expected, "moments n={n} profile={profile}");
            assert_eq!(actual.moments.transfers.device_to_host_bytes(), 176);
            if round >= 3 {
                samples[stage].push(elapsed.as_secs_f64() * 1000.0);
            }
        }
    }
    println!("GPU_ICP_SELECTOR_MOMENTS count={n} profile={profile} gated={} retained={} h2d={} d2h=176 old_full_ms={:?} new_full_ms={:?} old_resident_ms={:?} new_resident_ms={:?}",expected.evaluated_correspondences,expected.moments.correspondences,inputs[0].upload_transfers().host_to_device_bytes(),samples[0],samples[1],samples[2],samples[3]);
}

#[test]
#[ignore = "paired completed selector/moment survey; run this exact test alone in release mode"]
fn completed_selector_and_moment_calls() {
    let old = WgpuRuntime::new_headless().unwrap();
    let new = WgpuRuntime::new_headless().unwrap();
    old.icp_trim_pipelines.set(frozen_cache(&old)).unwrap_or_else(|_| panic!("fresh cache"));
    let current = new.icp_trim_pipelines.get_or_init(|| TrimPipelines::new(new.device()));
    let previous = old.icp_trim_pipelines.get().unwrap();
    let info = old.adapter_info();
    assert_eq!(info.backend, new.adapter_info().backend);
    assert_eq!(info.device_type, new.adapter_info().device_type);
    println!("GPU_ICP_SELECTOR_ENV backend={} device_type={}", info.backend, info.device_type);
    for n in [257, 4097, 16385, 65537, 262145, 1_000_003] {
        for profile in 0..2 {
            selection_case([&old, &new], [previous, current], n, profile);
        }
    }
    for n in [257, 4097, 65537, 262145] {
        for profile in 0..2 {
            moment_case([&old, &new], n, profile);
        }
    }
}
