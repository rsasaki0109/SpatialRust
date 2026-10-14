use super::super::aos_upload_reference::previous_prepare as previous;
use super::*;

fn fixture(n: usize, profile: usize) -> Vec<[f32; 3]> {
    let mut state = 0x8a74_92d1_u32;
    (0..n)
        .map(|i| {
            std::array::from_fn(|a| {
                let zero = f32::from_bits(((i + a) as u32 & 1) << 31);
                match profile {
                    0 => ((i * (a + 1)) % 997) as f32 * 0.125,
                    1 => zero,
                    2 => f32::from_bits((i as u32 % 31) | (((i + a) as u32 & 1) << 31)),
                    _ => {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        f32::from_bits(if state & 0x7f80_0000 == 0x7f80_0000 {
                            state ^ 0x0080_0000
                        } else {
                            state
                        })
                    }
                }
            })
        })
        .collect()
}

#[test]
fn original_finite_words_and_zero_padding_match_every_worker_shape() {
    let mut words = 0;
    for n in [0, 1, 17, 65_537, 524_287, 524_288, 524_289, 1_048_577] {
        for profile in 0..4 {
            let input = fixture(n, profile);
            let old = previous(&input).unwrap();
            assert_eq!(old.len(), n);
            for available in [0, 1, 2, 3, 4, 12, usize::MAX] {
                let new = prepare_with_workers(&input, available).unwrap();
                assert_eq!(
                    bytemuck::cast_slice::<_, u32>(&old),
                    bytemuck::cast_slice::<_, u32>(&new)
                );
                for (row, p) in new.iter().zip(&input) {
                    assert_eq!(
                        row.map(f32::to_bits),
                        [p[0].to_bits(), p[1].to_bits(), p[2].to_bits(), 0]
                    );
                }
                words += new.len() * 4;
            }
            let actual = prepare(&input).unwrap();
            assert_eq!(
                bytemuck::cast_slice::<_, u32>(&old),
                bytemuck::cast_slice::<_, u32>(&actual)
            );
        }
    }
    println!("AOS_QUERY_HOST words={words}");
}

#[test]
fn every_original_component_and_worker_edge_rejects_nonfinite_words() {
    let mut input = fixture(1_048_577, 0);
    let mut errors = 0;
    for index in [0, 262_143, 262_144, 262_145, 524_288, 524_290, 786_435, input.len() - 1] {
        for a in 0..3 {
            let old = input[index][a];
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(0x7f80_0001)] {
                input[index][a] = value;
                let expected = previous(&input).unwrap_err();
                for available in [1, 2, 3, 4, 12] {
                    assert_eq!(prepare_with_workers(&input, available).unwrap_err(), expected);
                    errors += 1;
                }
                input[index][a] = old;
            }
        }
    }
    println!("AOS_QUERY_ERRORS errors={errors}");
}

#[test]
fn bounded_policy_and_concurrent_calls_keep_private_outputs() {
    for (n, available, expected) in [
        (0, 12, 1),
        (524_287, 12, 1),
        (524_288, 0, 1),
        (524_288, 12, 2),
        (1_048_575, 12, 3),
        (1_048_576, 12, 4),
        (usize::MAX, usize::MAX, 4),
    ] {
        assert_eq!(workers(n, available), expected);
    }
    let input = fixture(524_289, 2);
    let expected = previous(&input).unwrap();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..3).map(|_| scope.spawn(|| prepare(&input).unwrap())).collect();
        for handle in handles {
            let actual = handle.join().unwrap();
            assert_eq!(
                bytemuck::cast_slice::<_, u32>(&actual),
                bytemuck::cast_slice::<_, u32>(&expected)
            );
        }
    });
}

#[path = "aos_query_staging_benchmark.rs"]
mod benchmark;
