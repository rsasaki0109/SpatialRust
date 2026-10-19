"""Audit actual prior receipts and reject misleading hardware coverage claims."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_public_gpu_conformance import audit_runs

MANIFEST = "f0c2fe90f845de3bb434b7d0e24b10b4ddfe8754bb39455841bcaa654c98a7ae"


def prior_runs(name="windows_reports"):
    receipt = json.loads((ROOT / "docs/receipts/2026-10-19_public_multi_adapter_rgbd.json").read_text())
    return [{"preference": item["report"]["preference"], "exit_status": 0,
             "report": copy.deepcopy(item["report"])} for item in receipt[name]]


def test_actual_windows_hardware_reports_pass_all_fixed_gates():
    result = audit_runs(prior_runs(), MANIFEST, "windows")
    assert result["hardware_coverage_accepted"]
    assert result["issues"] == []


def test_actual_linux_software_reports_pass_operations_but_not_hardware():
    result = audit_runs(prior_runs("linux_reports"), MANIFEST)
    assert result["all_operation_reports_passed"]
    assert not result["hardware_coverage_accepted"]
    assert result["accepted_hardware_classes"] == []


def test_actual_isolated_gallium_operations_do_not_promote_other_class():
    receipt = json.loads((ROOT / "docs/receipts/2026-10-20_isolated_gallium_gles.json")
                         .read_text(encoding="utf-8"))
    runs = receipt["runs"]
    assert {run["report"]["device_type"] for run in runs} == {"Other"}
    result = audit_runs(runs, MANIFEST, "linux", expected_backend="Gl")
    assert result["all_operation_reports_passed"]
    assert not result["hardware_coverage_accepted"]
    assert result["accepted_hardware_classes"] == []


def gl_schema_runs():
    # Synthetic backend metadata over frozen measurements tests validation only.
    # These rows are never presented as evidence of actual GL execution.
    runs = prior_runs()
    for run in runs:
        run['report'].update(backend='Gl', requested_backend='gl')
    return runs


def test_explicit_gl_expectation_is_separate_from_default_vulkan():
    runs = gl_schema_runs()
    assert audit_runs(runs, MANIFEST, 'windows', expected_backend='Gl')['hardware_coverage_accepted']
    assert not audit_runs(runs, MANIFEST, 'windows')['all_operation_reports_passed']


@pytest.mark.parametrize('selection_request', [None, 'primary', True])
def test_gl_receipt_requires_the_explicit_request(selection_request):
    runs = gl_schema_runs()
    if selection_request is None:
        runs[0]['report'].pop('requested_backend')
    else:
        runs[0]['report']['requested_backend'] = selection_request
    result = audit_runs(runs, MANIFEST, 'windows', expected_backend='Gl')
    assert not result['all_operation_reports_passed']


def test_actual_backend_cannot_be_substituted_by_the_requested_label():
    runs = gl_schema_runs()
    runs[0]['report']['backend'] = 'Vulkan'
    assert not audit_runs(runs, MANIFEST, 'windows', expected_backend='Gl')['all_operation_reports_passed']


@pytest.mark.parametrize('backend', ['gl', 'Dx12', None, False])
def test_unknown_or_ambiguous_expected_backend_is_rejected(backend):
    with pytest.raises(ValueError):
        audit_runs(prior_runs(), MANIFEST, 'windows', expected_backend=backend)


def test_gl_other_class_does_not_become_verified_hardware():
    runs = gl_schema_runs()
    for run in runs:
        run['report']['device_type'] = 'Other'
    result = audit_runs(runs, MANIFEST, 'windows', expected_backend='Gl')
    assert result['all_operation_reports_passed']
    assert not result['hardware_coverage_accepted']
    assert result['accepted_hardware_classes'] == []


@pytest.mark.parametrize("change", [
    lambda r: r.pop(),
    lambda r: r.reverse(),
    lambda r: r[0].update(exit_status=1),
    lambda r: r[0].update(exit_status=False),
    lambda r: r[0].update(report=None),
    lambda r: r[0]["report"].update(manifest_sha256="0" * 64),
    lambda r: r[1]["report"].update(device_type="DiscreteGpu"),
    lambda r: r[1]["report"].update(device_type=[]),
    lambda r: r[0]["report"]["frames"].pop(),
    lambda r: r[0]["report"]["frames"][0]["image"].update(gray_mismatches=1),
    lambda r: r[0]["report"]["frames"][0]["image"].update(gray_mismatches=False),
    lambda r: r[0]["report"]["frames"][0]["image"].update(gray_sha256="0" * 64),
    lambda r: r[0]["report"]["frames"][0]["voxel"].update(maximum_centroid_coordinate_error_m=float("nan")),
    lambda r: r[0]["report"]["frames"][0]["voxel"].update(maximum_centroid_coordinate_error_m=-1.),
    lambda r: r[0]["report"]["frames"][0]["voxel"].update(maximum_centroid_coordinate_error_m=10**400),
    lambda r: r[0]["report"]["frames"][0]["voxel"].update(maximum_centroid_coordinate_error_m=1.1e-5),
    lambda r: r[0]["report"]["frames"][0]["voxel"].update(centroid_tolerance_m=.1),
    lambda r: r[0]["report"]["frames"][0]["radius"].update(count_mismatches=1),
    lambda r: r[0]["report"]["frames"][0]["radius"].update(search_device_to_host_bytes=4),
    lambda r: r[0]["report"]["frames"][0]["pools"]["image"].update(cached_bytes=67108865),
])
def test_incomplete_or_inconsistent_receipts_cannot_claim_hardware_coverage(change):
    runs = prior_runs()
    change(runs)
    assert not audit_runs(runs, MANIFEST, "windows")["hardware_coverage_accepted"]


def test_failed_frame_is_retained_and_rejected_without_assuming_payload_exists():
    runs = prior_runs()
    runs[0]["report"]["frames"][0] = {"index": 0, "accepted": False, "error": "device failure"}
    result = audit_runs(runs, MANIFEST, "windows")
    assert not result["hardware_coverage_accepted"]
    assert any("frame 0" in issue for issue in result["issues"])


def test_initialization_failures_have_no_observed_centroid_error():
    runs = [{"preference": preference, "exit_status": 1, "report": None}
            for preference in ("high", "low")]
    result = audit_runs(runs, MANIFEST)
    assert not result["hardware_coverage_accepted"]
    assert result["maximum_centroid_coordinate_error_m"] is None
