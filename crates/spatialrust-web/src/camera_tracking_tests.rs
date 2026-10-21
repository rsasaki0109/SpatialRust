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
