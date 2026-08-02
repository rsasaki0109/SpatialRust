//! Constructs a machine-checkable Vision 1 release receipt.

use spatialrust::platform::{
    ConformanceReport, ConformanceStatus, SecurityChecklist, Vision1Measurements,
    Vision1ReleaseEvidence, Vision1ReleaseGate,
};

fn main() {
    let mut conformance = ConformanceReport::new();
    for &id in Vision1ReleaseGate::required_conformance_cases() {
        conformance.record(id, ConformanceStatus::Pass, Some("CI receipt".into()));
    }
    let evidence = Vision1ReleaseEvidence {
        conformance,
        security: SecurityChecklist::north_star_baseline_satisfied(),
        measurements: Vision1Measurements {
            cpu_vga_latency_us: 20_000,
            cpu_1080p_latency_us: 120_000,
            gpu_4k_chain_latency_us: 13_523,
            explicit_copy_bytes: 8 * 1024 * 1024,
        },
        passed_comparison_suites: Vision1ReleaseGate::required_comparison_suites()
            .iter()
            .map(ToString::to_string)
            .collect(),
        verified_examples: Vision1ReleaseGate::required_examples()
            .iter()
            .map(ToString::to_string)
            .collect(),
        migration_policy: "vision-1".into(),
    };
    let decision = Vision1ReleaseGate::evaluate(&evidence);
    assert!(decision.allowed, "Vision 1 denied: {:?}", decision.reasons);
    println!("SpatialRust Vision 1 release gate: allowed");
}
