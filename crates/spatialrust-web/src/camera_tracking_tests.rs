use super::*;

const W: usize = 160;
const H: usize = 128;

fn texture(dx: isize, dy: isize) -> Vec<u8> {
    (0..W * H)
        .map(|i| {
            let x = (i % W) as isize - dx;
            let y = (i / W) as isize - dy;
            if !(0..W as isize).contains(&x) || !(0..H as isize).contains(&y) {
                return 0;
            }
            let hash = (x as u32).wrapping_mul(73856093) ^ (y as u32).wrapping_mul(19349663);
            ((hash ^ (hash >> 13)) & 255) as u8
        })
        .collect()
}

#[test]
fn rejects_unbounded_or_tiny_shapes() {
    for (w, h) in [(0, H), (63, H), (W, 63), (321, H), (W, usize::MAX)] {
        assert!(CameraTracker::try_new(w, h).is_err());
    }
    assert!(CameraTracker::try_new(320, 320).is_ok());
}

#[test]
fn first_frame_has_no_invented_motion() {
    let mut t = CameraTracker::try_new(W, H).unwrap();
    let r = t.track_gray_copy(&texture(0, 0), &[], 0.).unwrap();
    assert_eq!(r.status, CameraTrackingStatus::Warming);
    assert_eq!(r.background_flow_px, None);
    assert!(r.tracks.is_empty());
    assert!(r.seeded_features >= 8 && r.seeded_features <= 80);
}

#[test]
fn tracks_known_two_dimensional_translation() {
    let mut t = CameraTracker::try_new(W, H).unwrap();
    t.track_gray_copy(&texture(0, 0), &[], 0.).unwrap();
    let r = t.track_gray_copy(&texture(2, -1), &[], 100.).unwrap();
    assert_eq!(r.status, CameraTrackingStatus::Tracking, "tracks={}", r.tracks.len());
    let flow = r.background_flow_px.unwrap();
    assert!((flow[0] - 2.).abs() < 0.35 && (flow[1] + 1.).abs() < 0.35, "{flow:?}");
    assert!(r.tracks.len() <= 80);
}

#[test]
fn stationary_texture_has_negligible_flow() {
    let mut t = CameraTracker::try_new(W, H).unwrap();
    let p = texture(0, 0);
    t.track_gray_copy(&p, &[], 0.).unwrap();
    let r = t.track_gray_copy(&p, &[], 100.).unwrap();
    let f = r.background_flow_px.unwrap();
    assert!(f[0].abs() < 0.1 && f[1].abs() < 0.1);
}

#[test]
fn no_texture_does_not_claim_zero_motion() {
    let mut t = CameraTracker::try_new(W, H).unwrap();
    let p = vec![127; W * H];
    t.track_gray_copy(&p, &[], 0.).unwrap();
    let r = t.track_gray_copy(&p, &[], 100.).unwrap();
    assert_eq!(r.status, CameraTrackingStatus::Lost);
    assert_eq!(r.background_flow_px, None);
}

#[test]
fn complete_person_mask_excludes_every_point() {
    let mut t = CameraTracker::try_new(W, H).unwrap();
    let p = texture(0, 0);
    let mask = vec![1; W * H];
    let r = t.track_gray_copy(&p, &mask, 0.).unwrap();
    assert_eq!(r.seeded_features, 0);
    let r = t.track_gray_copy(&p, &mask, 100.).unwrap();
    assert!(r.tracks.is_empty());
    assert_eq!(r.person_fraction, 1.);
    assert_eq!(r.background_flow_px, None);
}

#[test]
fn current_person_mask_rejects_arriving_track_neighborhoods() {
    let mut t = CameraTracker::try_new(W, H).unwrap();
    let p = texture(0, 0);
    t.track_gray_copy(&p, &[], 0.).unwrap();
    let mask: Vec<_> = (0..W * H).map(|i| u8::from(i % W >= W / 2)).collect();
    let r = t.track_gray_copy(&p, &mask, 100.).unwrap();
    assert!(!r.tracks.is_empty());
    assert!(r.tracks.iter().all(|p| p.to[0] + 22. < (W / 2) as f64));
    assert_eq!(r.person_fraction, 0.5);
}

#[test]
fn source_mask_is_kept_until_the_next_pair() {
    let mut t = CameraTracker::try_new(W, H).unwrap();
    let p = texture(0, 0);
    let mask: Vec<_> = (0..W * H).map(|i| u8::from(i % W >= W / 2)).collect();
    t.track_gray_copy(&p, &mask, 0.).unwrap();
    let r = t.track_gray_copy(&p, &[], 100.).unwrap();
    assert!(!r.tracks.is_empty());
    assert!(r.tracks.iter().all(|p| p.from[0] + 22. < (W / 2) as f64));
}

#[test]
fn invalid_inputs_leave_the_previous_pair_intact() {
    let mut t = CameraTracker::try_new(W, H).unwrap();
    let p = texture(0, 0);
    t.track_gray_copy(&p, &[], 100.).unwrap();
    assert!(t.track_gray_copy(&p[..10], &[], 110.).is_err());
    assert!(t.track_gray_copy(&p, &[0; 3], 110.).is_err());
    assert!(t.track_gray_copy(&p, &vec![2; W * H], 110.).is_err());
    for ts in [f64::NAN, f64::INFINITY, -1., 100., 99.] {
        assert!(t.track_gray_copy(&p, &[], ts).is_err());
    }
    let r = t.track_gray_copy(&p, &[], 110.).unwrap();
    assert_eq!(r.status, CameraTrackingStatus::Tracking);
}

#[test]
fn large_time_gap_reseeds_without_a_disconnected_motion() {
    let mut t = CameraTracker::try_new(W, H).unwrap();
    let p = texture(0, 0);
    t.track_gray_copy(&p, &[], 0.).unwrap();
    let r = t.track_gray_copy(&p, &[], 501.).unwrap();
    assert_eq!(r.status, CameraTrackingStatus::Reseeded);
    assert!(r.tracks.is_empty());
    assert_eq!(r.background_flow_px, None);
    assert_eq!(t.track_gray_copy(&p, &[], 600.).unwrap().status, CameraTrackingStatus::Tracking);
}

#[test]
fn reset_drops_frames_and_clock_state() {
    let mut t = CameraTracker::try_new(W, H).unwrap();
    let p = texture(0, 0);
    t.track_gray_copy(&p, &[], 100.).unwrap();
    t.reset();
    let r = t.track_gray_copy(&p, &[], 0.).unwrap();
    assert_eq!(r.status, CameraTrackingStatus::Warming);
}

#[test]
fn reports_are_finite_bounded_and_contain_no_pixels() {
    let mut t = CameraTracker::try_new(W, H).unwrap();
    let p = texture(0, 0);
    t.track_gray_copy(&p, &[], 0.).unwrap();
    let r = t.track_gray_copy(&p, &[], 100.).unwrap();
    let json = serde_json::to_string(&r).unwrap();
    assert!(json.len() < 32768);
    assert!(!json.contains("previous") && !json.contains("gray"));
    assert!(r.tracks.iter().all(|p| p.from.iter().chain(p.to.iter()).all(|v| v.is_finite())));
}

// Frozen parent support predicate; preserve its inclusive square and truncation.
fn brute_support(width: usize, height: usize, mask: &[u8], x: f64, y: f64) -> bool {
    if !x.is_finite()
        || !y.is_finite()
        || x < 22.
        || y < 22.
        || x >= (width - 22) as f64
        || y >= (height - 22) as f64
    {
        return false;
    }
    if mask.is_empty() {
        return true;
    }
    let (x, y) = (x as usize, y as usize);
    (y - 22..=y + 22).all(|row| (x - 22..=x + 22).all(|column| mask[row * width + column] == 0))
}
fn support_queries(width: usize, height: usize, mask: &[u8], points: &[(f64, f64)]) {
    let tracker = CameraTracker::try_new(width, height).unwrap();
    let support = MaskSupport::from_validated(
        mask,
        width,
        height,
        mask.iter().filter(|&&value| value == 1).count(),
    );
    for &(x, y) in points {
        assert_eq!(
            tracker.background(&support, x, y),
            brute_support(width, height, mask, x, y),
            "{width}x{height} at ({x},{y})"
        );
    }
}
#[test]
fn support_keeps_finite_fractional_and_strict_boundary_admission() {
    for (w, h) in [(64, 64), (65, 79), (192, 144), (320, 320), (64, 320), (320, 64)] {
        let values = [
            f64::NAN,
            f64::NEG_INFINITY,
            f64::INFINITY,
            -1.,
            -0.,
            0.,
            21.999999999,
            22.,
            22.5,
            23.,
            (w - 23) as f64 + 0.999999999,
            (w - 22) as f64,
            (h - 23) as f64 + 0.999999999,
            (h - 22) as f64,
        ];
        let points: Vec<_> =
            values.iter().flat_map(|&x| values.iter().map(move |&y| (x, y))).collect();
        for mask in [
            Vec::new(),
            vec![0; w * h],
            vec![1; w * h],
            (0..w * h).map(|i| u8::from(i == (h / 2) * w + w / 2)).collect(),
        ] {
            support_queries(w, h, &mask, &points);
        }
    }
}
#[test]
fn support_matches_every_small_frame_pixel_with_one_exclusion() {
    let (w, h) = (64, 65);
    let points: Vec<_> = (0..w * h).map(|i| ((i % w) as f64, (i / w) as f64)).collect();
    for person in [0, 21 * w + 21, 22 * w + 22, 32 * w + 32, 43 * w + 43, w * h - 1] {
        let mut mask = vec![0; w * h];
        mask[person] = 1;
        support_queries(w, h, &mask, &points);
    }
}
#[test]
fn support_matches_deterministic_mixed_masks_and_subpixel_queries() {
    for (w, h) in [(65, 79), (160, 128), (256, 192), (320, 320)] {
        let points: Vec<_> = (0..2048)
            .map(|i| {
                (
                    ((i * 73) % w) as f64 + (i % 4) as f64 / 4.,
                    ((i * 97) % h) as f64 + (i % 3) as f64 / 3.,
                )
            })
            .collect();
        for divisor in [2, 19, 997, usize::MAX] {
            let mask: Vec<_> = (0..w * h)
                .map(|i: usize| {
                    u8::from(i.wrapping_mul(1103515245).wrapping_add(12345) % divisor == 0)
                })
                .collect();
            support_queries(w, h, &mask, &points);
        }
    }
}
#[test]
fn changing_caller_mask_does_not_change_retained_support() {
    let image = texture(0, 0);
    let mut caller = vec![0; W * H];
    for row in 0..H {
        for column in W / 2..W {
            caller[row * W + column] = 1;
        }
    }
    let mut a = CameraTracker::try_new(W, H).unwrap();
    let mut b = CameraTracker::try_new(W, H).unwrap();
    let frozen = caller.clone();
    a.track_gray_copy(&image, &caller, 0.).unwrap();
    b.track_gray_copy(&image, &frozen, 0.).unwrap();
    caller.fill(0);
    assert_eq!(
        a.track_gray_copy(&image, &caller, 125.).unwrap(),
        b.track_gray_copy(&image, &caller, 125.).unwrap()
    );
}

#[test]
fn uniform_support_needs_no_prefix_storage() {
    for (mask, count, empty) in
        [(vec![], 0, true), (vec![0; W * H], 0, true), (vec![1; W * H], W * H, false)]
    {
        let support = MaskSupport::from_validated(&mask, W, H, count);
        assert!(if empty {
            matches!(support, MaskSupport::Empty)
        } else {
            matches!(support, MaskSupport::Full)
        });
    }
}

#[test]
fn largest_mixed_prefix_keeps_zero_border_and_exact_pixel_mass() {
    let (w, h) = (MAX_AXIS, MAX_AXIS);
    let mut mask = vec![1; w * h];
    mask[0] = 0;
    let MaskSupport::Prefix { stride, sums } = MaskSupport::from_validated(&mask, w, h, w * h - 1)
    else {
        panic!("mixed mask requires prefix storage");
    };
    assert_eq!(stride, 321);
    assert_eq!(sums.len() * std::mem::size_of::<u32>(), 412_164);
    assert!(sums[..stride].iter().all(|&value| value == 0));
    assert!((0..=h).all(|row| sums[row * stride] == 0));
    for row in 1..=h {
        for column in 1..=w {
            assert_eq!(sums[row * stride + column], (row * column - 1) as u32);
        }
    }
}

#[test]
fn rejected_mask_or_clock_keeps_mixed_support_and_reset_releases_it() {
    let image = texture(0, 0);
    let mask: Vec<_> = (0..W * H).map(|i| u8::from(i % W >= W / 2)).collect();
    let mut actual = CameraTracker::try_new(W, H).unwrap();
    let mut control = CameraTracker::try_new(W, H).unwrap();
    actual.track_gray_copy(&image, &mask, 0.).unwrap();
    control.track_gray_copy(&image, &mask, 0.).unwrap();
    let mut invalid_mask = vec![0; W * H];
    invalid_mask[W * H - 1] = 2;
    assert!(actual.track_gray_copy(&image, &invalid_mask, 125.).is_err());
    assert!(actual.track_gray_copy(&image, &vec![1; W * H], f64::NAN).is_err());
    assert!(matches!(actual.previous_mask, MaskSupport::Prefix { .. }));
    assert_eq!(
        actual.track_gray_copy(&image, &[], 125.).unwrap(),
        control.track_gray_copy(&image, &[], 125.).unwrap()
    );
    actual.track_gray_copy(&image, &mask, 250.).unwrap();
    actual.reset();
    assert!(matches!(actual.previous_mask, MaskSupport::Empty));
    assert_eq!(
        actual.track_gray_copy(&image, &[], 0.).unwrap().status,
        CameraTrackingStatus::Warming
    );
}
