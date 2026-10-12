use spatialrust::search::{euclidean_cluster_roots, grid_bounds, uniform_grid_fits};

#[test]
fn facade_grid_bounds_reject_wrapped_spans_and_malformed_axes() {
    for max in [4_294_967_296.0, 1.0e20, f32::MAX] {
        assert!(grid_bounds(&[0.0, max], &[0.0; 2], &[0.0; 2], 1.0).is_err());
        assert!(!uniform_grid_fits(&[0.0, max], &[0.0; 2], &[0.0; 2], 1.0));
        assert!(euclidean_cluster_roots(&[0.0, max], &[0.0; 2], &[0.0; 2], 1.0).is_err());
    }
    assert!(!uniform_grid_fits(&[], &[0.0], &[], 1.0));
    assert!(!uniform_grid_fits(&[f32::NAN], &[0.0], &[0.0], 1.0));
    assert!(!uniform_grid_fits(&[0.0], &[0.0], &[0.0], f32::from_bits(1)));
}

#[test]
fn facade_grid_bounds_keep_exact_cap_and_infinite_radius_contract() {
    assert_eq!(
        grid_bounds(&[0.0, 399.0], &[0.0, 399.0], &[0.0, 399.0], 1.0).unwrap(),
        ([0.0; 3], [400; 3])
    );
    assert!(grid_bounds(&[0.0, 399.0], &[0.0, 399.0], &[0.0, 400.0], 1.0).is_err());
    assert_eq!(
        grid_bounds(&[f32::MIN, f32::MAX], &[0.0; 2], &[0.0; 2], f32::INFINITY).unwrap(),
        ([f32::MIN, 0.0, 0.0], [1; 3])
    );
}
