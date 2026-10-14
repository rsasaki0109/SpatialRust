use super::*;
use std::time::Instant;
fn measure<const C: usize>(w: usize, h: usize, d: usize, p: usize) {
    let im = fixture::<C>(w, h, p);
    let expected = previous(im.view(), d, 40.0, 3.0, BorderMode::Replicate).unwrap();
    let mut samples: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
    for round in 0..23 {
        for offset in 0..3 {
            let stage = (round + offset) % 3;
            let start = Instant::now();
            let actual = match stage {
                0 => previous(im.view(), d, 40.0, 3.0, BorderMode::Replicate),
                1 => run_with_workers(im.view(), d, 40.0, 3.0, BorderMode::Replicate, Some(1)),
                _ => crate::bilateral_filter(im.view(), d, 40.0, 3.0, BorderMode::Replicate),
            }
            .unwrap();
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(actual, expected);
            if round >= 3 {
                samples[stage].push(ms);
            }
        }
    }
    println!("CPU_BILATERAL_RECORD {{\"width\":{w},\"height\":{h},\"channels\":{C},\"diameter\":{d},\"profile\":{p},\"old_ms\":{:?},\"serial_ms\":{:?},\"parallel_ms\":{:?}}}",samples[0],samples[1],samples[2]);
}
#[test]
#[ignore = "requires isolated native release measurement"]
fn completed_cpu_bilateral_calls() {
    println!(
        "CPU_BILATERAL_ENV available_workers={} pool_workers={}",
        std::thread::available_parallelism().map_or(1, |n| n.get()),
        rayon::current_num_threads()
    );
    measure::<1>(65, 33, 3, 0);
    measure::<3>(257, 193, 5, 0);
    measure::<1>(511, 513, 3, 0);
    measure::<1>(512, 512, 3, 0);
    measure::<3>(1024, 1024, 5, 0);
    measure::<3>(1920, 1080, 5, 0);
    measure::<3>(1024, 1024, 5, 2);
    measure::<1>(1024, 1024, 7, 0);
}
