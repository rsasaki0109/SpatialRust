use proptest::prelude::*;
use spatialrust_tensor::{DataType, Device, TensorDescriptor, TensorView};

proptest! {
    #[test]
    fn compact_range_matches_checked_product(shape in prop::collection::vec(0usize..32, 0..6)) {
        let descriptor = TensorDescriptor::contiguous(DataType::F32, shape.clone(), Device::CPU);
        let count = shape.iter().try_fold(1usize, |value, dimension| value.checked_mul(*dimension)).unwrap();
        let expected = count * 4;
        prop_assert_eq!(descriptor.required_byte_range().unwrap(), 0..expected);
        prop_assert!(TensorView::try_new(&vec![0; expected], descriptor).is_ok());
    }

    #[test]
    fn reversed_vector_span_stays_inside_allocation(length in 1usize..2048) {
        let descriptor = TensorDescriptor::try_strided(
            DataType::U16,
            vec![length],
            vec![-1],
            (length - 1) * 2,
            Device::CPU,
        ).unwrap();
        prop_assert_eq!(descriptor.required_byte_range().unwrap(), 0..length * 2);
        prop_assert!(TensorView::try_new(&vec![0; length * 2], descriptor).is_ok());
    }
}
