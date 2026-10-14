//! Frozen sort, current serial and automatic bounded parallel median calls.
use super::*;
use std::time::Instant;

fn measure<const C: usize>(w: usize, h: usize, kernel: usize, profile: usize) {
    let image = bytes::<C>(w, h, profile);
    let expected = previous(image.view(), kernel, BorderMode::Replicate).unwrap();
    let mut samples: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
    for round in 0..23 {
        for offset in 0..3 {
            let stage = (round + offset) % 3;
            let start = Instant::now();
            let actual = match stage {
                0 => previous(image.view(), kernel, BorderMode::Replicate),
                1 => run_with_workers(image.view(), kernel, BorderMode::Replicate, Some(1)),
                _ => run(image.view(), kernel, BorderMode::Replicate),
            }
            .unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(actual, expected);
            if round >= 3 {
                samples[stage].push(elapsed);
            }
        }
    }
    println!("CPU_MEDIAN_RECORD {{\"width\":{w},\"height\":{h},\"channels\":{C},\"kernel\":{kernel},\"profile\":{profile},\"workers\":{},\"old_ms\":{:?},\"serial_ms\":{:?},\"parallel_ms\":{:?}}}",worker_count(w*h*C,h,rayon::current_num_threads()),samples[0],samples[1],samples[2]);
}

#[test]
#[ignore = "requires isolated release survey; exact test only"]
fn completed_cpu_median_calls() {
    println!(
        "CPU_MEDIAN_ENV available_workers={} pool_workers={}",
        std::thread::available_parallelism().map_or(1, |n| n.get()),
        rayon::current_num_threads()
    );
    measure::<1>(65, 33, 3, 0);
    measure::<3>(257, 193, 5, 0);
    measure::<1>(511, 513, 3, 0);
    measure::<1>(512, 512, 3, 0);
    measure::<1>(513, 512, 3, 0);
    measure::<1>(1, 262_145, 3, 0);
    measure::<1>(262_145, 1, 3, 0);
    for kernel in [3, 5] {
        measure::<1>(1024, 1024, kernel, 0);
        measure::<3>(1024, 1024, kernel, 0);
    }
    measure::<3>(1024, 1024, 5, 2);
    measure::<3>(1920, 1080, 5, 0);
    measure::<3>(257, 193, 9, 0);
}
