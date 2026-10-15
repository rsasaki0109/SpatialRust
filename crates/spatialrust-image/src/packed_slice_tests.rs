use super::*;

#[test]
fn packed_slice_borrows_only_logical_pixels_and_preserves_scalar_bits() {
    let words = [0x8000_0000, 0x7fc0_0123, 0x7f80_0000, 0xff80_0000, 7, 9];
    let data = words.map(f32::from_bits);
    let view = ImageView::<_, 2>::new(2, 1, 4, &data).unwrap();
    let pixels = view.as_packed_slice().unwrap();
    assert_eq!(pixels.as_ptr(), data.as_ptr());
    assert_eq!(pixels.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), words[..4]);
    let dense = ImageView::<_, 1>::new(2, 2, 2, &data).unwrap();
    assert_eq!(dense.as_packed_slice().unwrap().len(), 4);
}

#[test]
fn packed_slice_handles_cropped_rows_without_exposing_neighbors() {
    let data: Vec<_> = (0..8 * 5).collect();
    let parent = ImageView::<_, 2>::new(4, 5, 8, &data).unwrap();
    let dense = parent.subview(ImageRegion::new(0, 1, 4, 3)).unwrap();
    let pixels = dense.as_packed_slice().unwrap();
    assert_eq!(pixels, &data[8..32]);
    assert_eq!(pixels.as_ptr(), data[8..].as_ptr());
    let narrow = parent.subview(ImageRegion::new(1, 1, 2, 3)).unwrap();
    assert!(narrow.as_packed_slice().is_none());
    let row = narrow.subview(ImageRegion::new(0, 1, 2, 1)).unwrap();
    assert_eq!(row.as_packed_slice().unwrap(), &data[18..22]);
}

#[test]
fn packed_slice_rejects_multirow_padding_and_accepts_one_row_with_huge_stride() {
    let data = [1, 2, 99, 3, 4, 88];
    assert!(ImageView::<_, 1>::new(2, 2, 3, &data).unwrap().as_packed_slice().is_none());
    let row = ImageView::<_, 1>::new(2, 1, usize::MAX, &data).unwrap();
    assert_eq!(row.as_packed_slice().unwrap(), &data[..2]);
    assert_eq!(row.as_packed_slice().unwrap().as_ptr(), data.as_ptr());
}

#[test]
fn packed_slice_empty_views_do_not_multiply_retained_dimensions_or_stride() {
    let horizontal = ImageView::<u8, 1>::new(usize::MAX, 0, usize::MAX, &[]).unwrap();
    let vertical = ImageView::<u8, 4>::new(0, usize::MAX, 0, &[]).unwrap();
    for pixels in [horizontal.as_packed_slice(), vertical.as_packed_slice()] {
        assert_eq!(pixels, Some(&[][..]));
    }
    let empty = vertical.subview(ImageRegion::new(0, 1, 0, usize::MAX - 1)).unwrap();
    assert_eq!(empty.as_packed_slice(), Some(&[][..]));
    let data = [1_u8; 16];
    let parent = ImageView::<_, 4>::new(2, 2, 8, &data).unwrap();
    for region in [ImageRegion::new(2, 0, 0, 2), ImageRegion::new(0, 2, 2, 0)] {
        assert_eq!(parent.subview(region).unwrap().as_packed_slice(), Some(&[][..]));
    }
}

#[test]
fn packed_slice_requires_no_scalar_clone_or_copy() {
    struct Scalar(String);
    let data = [Scalar("first".into()), Scalar("second".into())];
    let view = ImageView::<_, 1>::new(2, 1, 2, &data).unwrap();
    let pixels = view.as_packed_slice().unwrap();
    assert_eq!(pixels.as_ptr(), data.as_ptr());
    assert_eq!(pixels[1].0, "second");
    assert_eq!(view.as_packed_slice().unwrap().len(), 2);
}
