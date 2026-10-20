//! Synthetic portable receipt checks; no measured calibration evidence.

use super::{tests::complete_state, *};

fn rebuild(state: FullBagMappingState) -> ViewerResult<FullBagMappingState> {
    FullBagMappingState::try_new(
        state.title,
        state.source,
        state.calibration,
        state.source_summary,
        state.odometry,
        state.tsdf,
        state.artifacts,
        state.summary.clock_applied,
        state.summary.frame_graph_applied,
        state.blockers,
    )
}

fn scans(state: &mut FullBagMappingState, count: u64) {
    let odometry = state.odometry.as_mut().unwrap();
    odometry.scan_count = count;
    odometry.motion_count = count.saturating_sub(1);
    odometry.pose_graph_node_count = count;
    odometry.pose_graph_edge_count = odometry.motion_count;
}

#[test]
fn complete_source_requires_message_cardinality() {
    for front in [true, false] {
        let mut state = complete_state();
        if front {
            state.source_summary.front_bag_message_count += 1;
        } else {
            state.source_summary.rear_bag_message_count += 1;
        }
        assert!(state.source_summary.validate().is_err());
    }
}

#[test]
fn complete_source_requires_corrected_bounds() {
    let mut source = complete_state().source_summary;
    source.start_nanos = None;
    source.end_nanos = None;
    assert!(source.validate().is_err());
}

#[test]
fn complete_source_requires_nonempty_scans() {
    let mut source = complete_state().source_summary;
    source.total_point_count = source.total_record_count - 1;
    assert!(source.validate().is_err());
}

#[test]
fn partial_source_and_finished_stage_diagnostics_remain_valid() {
    let mut state = complete_state();
    state.source_summary.full_bag_processed = false;
    state.source_summary.front_bag_message_count += 1;
    state.source_summary.rear_bag_message_count += 2;
    let odometry = state.odometry.as_mut().unwrap();
    odometry.complete = false;
    odometry.truncated = true;
    let state = rebuild(state).unwrap();
    assert!(!state.summary.mapping_admitted);
    assert!(state.summary.tsdf_complete);
    state.validate().unwrap();
}

#[test]
fn complete_front_with_partial_rear_remains_valid() {
    let mut state = complete_state();
    state.source_summary.full_bag_processed = false;
    state.source_summary.rear_bag_message_count += 2;
    let state = rebuild(state).unwrap();
    assert!(state.summary.odometry_complete);
    assert!(!state.summary.mapping_admitted);
}

#[test]
fn complete_odometry_cannot_omit_retained_front_scans() {
    let mut state = complete_state();
    state.source_summary.front_record_count += 1;
    state.source_summary.front_bag_message_count += 1;
    state.source_summary.front_chunk_count += 1;
    state.source_summary.total_record_count += 1;
    state.source_summary.validate().unwrap();
    state.odometry.as_ref().unwrap().validate().unwrap();
    assert!(rebuild(state).is_err());
}

#[test]
fn odometry_cannot_exceed_retained_front_scans() {
    for complete in [true, false] {
        let mut state = complete_state();
        scans(&mut state, 3);
        state.odometry.as_mut().unwrap().complete = complete;
        state.odometry.as_ref().unwrap().validate().unwrap();
        assert!(rebuild(state).is_err());
    }
}

#[test]
fn complete_odometry_cannot_claim_unretained_front_messages() {
    let mut state = complete_state();
    state.source_summary.full_bag_processed = false;
    state.source_summary.front_bag_message_count += 1;
    state.source_summary.validate().unwrap();
    assert!(rebuild(state).is_err());
}

#[test]
fn partial_odometry_prefix_remains_valid() {
    let mut state = complete_state();
    scans(&mut state, 1);
    state.odometry.as_mut().unwrap().complete = false;
    let state = rebuild(state).unwrap();
    assert!(!state.summary.mapping_admitted);
    assert_eq!(state.odometry.as_ref().unwrap().scan_count, 1);
}

#[test]
fn odometry_must_use_selected_front_topic() {
    let mut state = complete_state();
    state.odometry.as_mut().unwrap().topic = state.source_summary.rear_topic.clone();
    assert!(rebuild(state).is_err());
}

#[test]
fn odometry_frame_must_match_registered_front_role() {
    let mut state = complete_state();
    state.odometry.as_mut().unwrap().source_frame = "rear_frame".into();
    assert!(rebuild(state).is_err());
}

#[test]
fn odometry_root_must_match_registered_root() {
    let mut state = complete_state();
    state.odometry.as_mut().unwrap().root_frame = "other_root".into();
    assert!(rebuild(state).is_err());
}

#[test]
fn odometry_clock_must_match_registered_target() {
    let mut state = complete_state();
    state.odometry.as_mut().unwrap().clock_id = "other_clock".into();
    assert!(rebuild(state).is_err());
}

#[test]
fn descriptive_clock_label_is_not_a_typed_identity() {
    let mut state = complete_state();
    state.odometry.as_mut().unwrap().clock_domain = "fixture corrected time label".into();
    assert!(rebuild(state).unwrap().summary.mapping_admitted);
}

#[test]
fn volume_frame_must_match_registered_root() {
    let mut state = complete_state();
    state.tsdf.as_mut().unwrap().frame_id = "other_root".into();
    assert!(rebuild(state).is_err());
}

#[test]
fn volume_frame_must_match_odometry_even_without_calibration() {
    let mut state = complete_state();
    state.calibration = None;
    state.summary.clock_applied = false;
    state.summary.frame_graph_applied = false;
    state.tsdf.as_mut().unwrap().frame_id = "other_root".into();
    assert!(rebuild(state).is_err());
}

#[test]
fn missing_calibration_with_consistent_stages_remains_blocked() {
    let mut state = complete_state();
    state.calibration = None;
    state.summary.clock_applied = false;
    state.summary.frame_graph_applied = false;
    let state = rebuild(state).unwrap();
    assert!(!state.summary.calibration_registered);
    assert!(!state.summary.mapping_admitted);
}

#[test]
fn unmatched_rear_records_do_not_require_visit_equality() {
    let mut state = complete_state();
    let volume = state.tsdf.as_mut().unwrap();
    volume.integrated_record_count = 3;
    volume.integrated_point_count = 9;
    let state = rebuild(state).unwrap();
    assert!(state.summary.mapping_admitted);
    assert!(
        state.tsdf.as_ref().unwrap().integrated_record_count
            < state.source_summary.total_record_count
    );
}

#[test]
fn reused_rear_records_may_exceed_unique_source_totals() {
    let mut state = complete_state();
    state.source_summary.front_record_count = 3;
    state.source_summary.front_bag_message_count = 3;
    state.source_summary.front_chunk_count = 3;
    state.source_summary.rear_record_count = 1;
    state.source_summary.rear_bag_message_count = 1;
    state.source_summary.rear_chunk_count = 1;
    scans(&mut state, 3);
    let volume = state.tsdf.as_mut().unwrap();
    volume.integrated_record_count = 6;
    volume.integrated_point_count = 18;
    let state = rebuild(state).unwrap();
    assert!(state.summary.mapping_admitted);
    assert!(
        state.tsdf.as_ref().unwrap().integrated_record_count
            > state.source_summary.total_record_count
    );
    assert!(
        state.tsdf.as_ref().unwrap().integrated_point_count
            > state.source_summary.total_point_count
    );
}

#[test]
fn source_total_overflow_stays_rejected() {
    let mut source = complete_state().source_summary;
    source.front_record_count = u64::MAX;
    source.rear_record_count = 1;
    assert!(source.validate().is_err());
}

#[test]
fn completed_graph_motion_overflow_stays_rejected() {
    let mut odometry = complete_state().odometry.unwrap();
    odometry.scan_count = u64::MAX;
    odometry.motion_count = u64::MAX;
    odometry.pose_graph_node_count = u64::MAX;
    odometry.pose_graph_edge_count = u64::MAX;
    assert!(odometry.validate().is_err());
}

#[test]
fn constructor_rejects_cross_stage_mismatch() {
    let mut state = complete_state();
    state.odometry.as_mut().unwrap().clock_id = "unregistered".into();
    assert!(rebuild(state).is_err());
}

#[cfg(feature = "serde")]
#[test]
fn schema_one_round_trip_retains_complete_identity() {
    let original = complete_state();
    let json = serde_json::to_value(&original).unwrap();
    assert_eq!(json["version"], 1);
    let decoded: FullBagMappingState = serde_json::from_value(json).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded, original);
}

#[cfg(feature = "serde")]
#[test]
fn decoded_forged_trajectory_count_is_rejected() {
    let mut json = serde_json::to_value(complete_state()).unwrap();
    json["odometry"]["scan_count"] = 3.into();
    json["odometry"]["motion_count"] = 2.into();
    json["odometry"]["pose_graph_node_count"] = 3.into();
    json["odometry"]["pose_graph_edge_count"] = 2.into();
    let decoded: FullBagMappingState = serde_json::from_value(json).unwrap();
    decoded.odometry.as_ref().unwrap().validate().unwrap();
    assert!(decoded.validate().is_err());
}

#[cfg(feature = "serde")]
#[test]
fn decoded_forged_clock_is_rejected() {
    let mut json = serde_json::to_value(complete_state()).unwrap();
    json["odometry"]["clock_id"] = "unregistered".into();
    let decoded: FullBagMappingState = serde_json::from_value(json).unwrap();
    assert!(decoded.summary.mapping_admitted);
    assert!(decoded.validate().is_err());
}

#[cfg(feature = "serde")]
#[test]
fn schema_one_unknown_fields_stay_rejected() {
    let mut json = serde_json::to_value(complete_state()).unwrap();
    json["unique_physical_coverage_proven"] = true.into();
    assert!(serde_json::from_value::<FullBagMappingState>(json).is_err());
}
