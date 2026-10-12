use super::row_tests::{fixture, previous};
use super::{euclidean_cluster_roots, grid_bounds, uniform_grid_fits, MAX_UNIFORM_GRID_CELLS};
use spatialrust_core::SpatialError;

#[test]
fn overflowed_float_span_returns_a_structured_error() {
    assert!(matches!(
        grid_bounds(&[f32::MIN, f32::MAX], &[0.0; 2], &[0.0; 2], 1.0),
        Err(SpatialError::InvalidArgument(_))
    ));
}

#[test]
fn extreme_grid_spans_return_errors_instead_of_overflow_or_wrapped_dimensions() {
    let cases = [
        ([vec![0.0, 4_294_967_296.0], vec![0.0; 2], vec![0.0; 2]], 1.0),
        ([vec![f32::MIN, f32::MAX], vec![0.0; 2], vec![0.0; 2]], 1.0),
        ([vec![0.0, 1.0e20], vec![0.0; 2], vec![0.0; 2]], 1.0),
        ([vec![0.0, 2_147_483_648.0], vec![0.0, 2_147_483_648.0], vec![0.0, 2_147_483_648.0]], 1.0),
        ([vec![0.0, 1.0], vec![0.0; 2], vec![0.0; 2]], f32::MIN_POSITIVE),
        ([vec![0.0, 399.0], vec![0.0, 399.0], vec![0.0, 1.0e18]], 1.0),
        ([vec![0.0, f32::MAX], vec![0.0; 2], vec![0.0; 2]], 0.5),
    ];
    for (xyz, radius) in cases {
        assert!(
            matches!(
                grid_bounds(&xyz[0], &xyz[1], &xyz[2], radius),
                Err(SpatialError::InvalidArgument(_))
            ),
            "radius={radius}, x={:?}",
            xyz[0]
        );
        assert!(!uniform_grid_fits(&xyz[0], &xyz[1], &xyz[2], radius));
        assert!(matches!(
            euclidean_cluster_roots(&xyz[0], &xyz[1], &xyz[2], radius),
            Err(SpatialError::InvalidArgument(_))
        ));
    }
}

#[test]
fn malformed_axes_coordinates_and_cell_scales_are_rejected() {
    for lengths in [[0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [2, 1, 2]] {
        let xyz = lengths.map(|n| vec![0.0; n]);
        assert!(matches!(
            grid_bounds(&xyz[0], &xyz[1], &xyz[2], 1.0),
            Err(SpatialError::InvalidArgument(_))
        ));
        assert!(!uniform_grid_fits(&xyz[0], &xyz[1], &xyz[2], 1.0));
    }
    for radius in [
        0.0,
        -0.0,
        -1.0,
        f32::NAN,
        f32::NEG_INFINITY,
        f32::from_bits(1),
        f32::from_bits(0x00200000),
    ] {
        for n in [0, 2] {
            let zeros = vec![0.0; n];
            assert!(matches!(
                grid_bounds(&zeros, &zeros, &zeros, radius),
                Err(SpatialError::InvalidArgument(_))
            ));
            assert!(!uniform_grid_fits(&zeros, &zeros, &zeros, radius));
        }
    }
    for axis in 0..3 {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut xyz = [vec![0.0; 3], vec![0.0; 3], vec![0.0; 3]];
            xyz[axis][1] = bad;
            for radius in [1.0, f32::INFINITY] {
                assert!(matches!(
                    grid_bounds(&xyz[0], &xyz[1], &xyz[2], radius),
                    Err(SpatialError::InvalidArgument(_))
                ));
                assert!(!uniform_grid_fits(&xyz[0], &xyz[1], &xyz[2], radius));
            }
        }
    }
}

#[test]
fn accepted_bounds_preserve_frozen_f32_words_and_dimensions() {
    let mut cases = 0;
    for n in [0_usize, 1, 2, 3, 17, 255, 256, 257, 1025, 4097] {
        for kind in 0..5 {
            let xyz = fixture(n, kind);
            for radius in [0.25, 0.3, 0.75, 1.0, 2.0, f32::INFINITY] {
                let (origin, dims) = grid_bounds(&xyz[0], &xyz[1], &xyz[2], radius).unwrap();
                let (old_origin, old_dims) =
                    previous::grid_bounds(&xyz[0], &xyz[1], &xyz[2], radius).unwrap();
                assert_eq!(origin.map(f32::to_bits), old_origin.map(f32::to_bits));
                assert_eq!(dims, old_dims);
                assert!(dims.iter().all(|&dim| dim > 0));
                assert!(dims.into_iter().map(u64::from).product::<u64>() <= MAX_UNIFORM_GRID_CELLS);
                assert!(uniform_grid_fits(&xyz[0], &xyz[1], &xyz[2], radius));
                cases += 1;
            }
        }
    }
    let (origin, dims) = grid_bounds(&[-0.0], &[-0.0], &[-0.0], 1.0).unwrap();
    assert_eq!(origin.map(f32::to_bits), [(-0.0_f32).to_bits(); 3]);
    assert_eq!(dims, [1; 3]);
    eprintln!("GRID_BOUNDS verified {cases} frozen origin/dimension cases.");
}

#[test]
fn cell_cap_and_reciprocal_boundaries_have_independent_expected_dimensions() {
    // Exact 400^3 cap, without allocating its grid.
    assert_eq!(
        grid_bounds(&[0.0, 399.0], &[0.0, 399.0], &[0.0, 399.0], 1.0).unwrap(),
        ([0.0; 3], [400; 3])
    );
    assert!(grid_bounds(&[0.0, 399.0], &[0.0, 399.0], &[0.0, 400.0], 1.0).is_err());
    let below = f32::from_bits((MAX_UNIFORM_GRID_CELLS as f32).to_bits() - 1);
    assert_eq!(
        grid_bounds(&[0.0, below], &[0.0; 2], &[0.0; 2], 1.0).unwrap().1,
        [63_999_997, 1, 1]
    );
    assert!(grid_bounds(&[0.0, MAX_UNIFORM_GRID_CELLS as f32], &[0.0; 2], &[0.0; 2], 1.0).is_err());
    assert_eq!(
        grid_bounds(&[0.0, f32::MIN_POSITIVE], &[0.0; 2], &[0.0; 2], f32::MIN_POSITIVE).unwrap().1,
        [2, 1, 1]
    );
    let tiny_finite_scale = f32::from_bits(0x00200001);
    assert!((1.0 / tiny_finite_scale).is_finite());
    assert_eq!(grid_bounds(&[1.0], &[1.0], &[1.0], tiny_finite_scale).unwrap().1, [1; 3]);
    // The existing positive-infinity radius remains an unbounded one-cell grid.
    assert_eq!(
        grid_bounds(&[f32::MIN, f32::MAX], &[0.0; 2], &[0.0; 2], f32::INFINITY).unwrap(),
        ([f32::MIN, 0.0, 0.0], [1; 3])
    );
    for radius in [1.0, f32::INFINITY] {
        assert_eq!(grid_bounds(&[], &[], &[], radius).unwrap(), ([f32::INFINITY; 3], [1; 3]));
    }
}
