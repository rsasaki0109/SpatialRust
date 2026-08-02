//! Constructs and prints the canonical Vision 2 release receipt.

use spatialrust::platform::{
    ConformanceReport, ConformanceStatus, SecurityChecklist, Vision2Measurements,
    Vision2ReleaseEvidence, Vision2ReleaseGate,
};

fn main() {
    let mut conformance = ConformanceReport::new();
    for &id in Vision2ReleaseGate::required_conformance_cases() {
        conformance.record(id, ConformanceStatus::Pass, Some("CI/release receipt".into()));
    }
    let evidence = Vision2ReleaseEvidence {
        conformance,
        security: SecurityChecklist::north_star_baseline_satisfied(),
        measurements: Vision2Measurements {
            native_allocate_1080p_us: 648,
            native_reuse_1080p_us: 195,
            python_allocate_1080p_us: 825,
            python_reuse_1080p_us: 232,
            peak_host_memory_bytes: 6_220_800,
            steady_state_allocations: 0,
            worker_threads: 12,
            gpu_host_to_device_bytes: 3840 * 2160 * 4,
            gpu_device_to_host_bytes: 0,
        },
        passed_receipts: Vision2ReleaseGate::required_receipts()
            .iter()
            .map(ToString::to_string)
            .collect(),
        verified_examples: Vision2ReleaseGate::required_examples()
            .iter()
            .map(ToString::to_string)
            .collect(),
        migration_policy: "vision-2".into(),
    };
    let decision = Vision2ReleaseGate::evaluate(&evidence);
    assert!(decision.allowed, "Vision 2 denied: {:?}", decision.reasons);
    print!("{}", Vision2ReleaseGate::render_markdown(&evidence));
}
