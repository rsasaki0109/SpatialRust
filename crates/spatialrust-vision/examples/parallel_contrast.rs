//! Compare contrast worker counts in interleaved rounds on a shared desktop.
//! `cargo run -p spatialrust-vision --release --features imgproc-analysis --example parallel_contrast`

use std::time::Instant;

use spatialrust_image::Image;
use spatialrust_vision::{clahe, equalize_histogram};

const ROUNDS: usize = 12;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let counts = [1, 2, 6];
    let pools: Vec<_> = counts
        .iter()
        .map(|&threads| rayon::ThreadPoolBuilder::new().num_threads(threads).build())
        .collect::<Result<_, _>>()?;
    println!("Interleaved completed calls; {ROUNDS} rounds; pool setup/input generation/checks excluded; CPU load is uncontrolled.");
    for (name, width, height) in [("1080p", 1920, 1080), ("4k", 3840, 2160)] {
        let image = Image::<u8, 1>::try_new(
            width,
            height,
            (0..width * height).map(|i| (i & 255) as u8).collect(),
        )?;
        for operation in ["equalize_histogram", "clahe_8x8"] {
            let run = || match operation {
                "equalize_histogram" => equalize_histogram(image.view()),
                _ => clahe(image.view(), 2.0, 8, 8),
            };
            let expected = pools[0].install(run)?;
            for pool in &pools {
                for _ in 0..2 {
                    let output = pool.install(run)?;
                    assert!(
                        output.as_slice() == expected.as_slice(),
                        "warmup pixel parity {operation}/{name}"
                    );
                }
            }
            let mut samples: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(ROUNDS));
            for round in 0..ROUNDS {
                // Rotate order to distribute changing background load between pools.
                for offset in 0..pools.len() {
                    let worker = (round + offset) % pools.len();
                    let started = Instant::now();
                    let output = pools[worker].install(run)?;
                    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                    samples[worker].push(elapsed_ms);
                    assert!(
                        output.as_slice() == expected.as_slice(),
                        "pixel parity {operation}/{name}, workers={}",
                        counts[worker]
                    );
                    assert_eq!(output.metadata(), image.metadata());
                }
            }
            for (worker, mut values) in samples.into_iter().enumerate() {
                values.sort_by(f64::total_cmp);
                let median = (values[ROUNDS / 2 - 1] + values[ROUNDS / 2]) / 2.0;
                println!("{operation}/{name}, workers={}: median={median:.3} ms, range=[{:.3},{:.3}] ms, pixel parity=exact",
                    counts[worker], values[0], values[ROUNDS - 1]);
            }
        }
    }
    Ok(())
}
