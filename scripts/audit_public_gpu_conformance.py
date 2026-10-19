"""Data-only audit of fixed public RGB-D operation and hardware coverage receipts.

This checks report consistency, not authenticity of GPU execution. Bind process,
binary, input and driver hashes separately. No device enumeration occurs here.
"""
import math
import re


FRAME_COUNTS = {0: (229875, 5666, 25093),
                399: (231941, 4399, 38648),
                797: (230416, 2425, 51389)}
HARDWARE_CLASSES = {"DiscreteGpu", "IntegratedGpu"}


def audit_runs(runs, manifest_sha256, expected_os="linux", *, expected_backend="Vulkan"):
    """Return all issues; failed processes/frames cannot disappear from coverage."""
    if not isinstance(manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
        raise ValueError("Expected input manifest SHA-256 is required")
    if type(expected_backend) is not str or expected_backend not in ("Vulkan", "Gl"):
        raise ValueError("Explicit expected backend must be Vulkan or Gl")
    issues = []
    classes = set()
    maximum_error = None

    def require(ok, label):
        if not ok:
            issues.append(label)

    def integer(obj, key, expected=None):
        value = obj.get(key)
        return type(value) is int and value >= 0 and (expected is None or value == expected)

    def equal(obj, key, value):
        actual = obj.get(key)
        return type(actual) is type(value) and actual == value

    def object_value(obj, key):
        value = obj.get(key)
        return value if isinstance(value, dict) else {}

    require(type(runs) is list and len(runs) == 2, "exactly two preference runs required")
    if type(runs) is not list:
        runs = []
    require([r.get("preference") if isinstance(r, dict) else None for r in runs] == ["high", "low"],
            "both ordered preferences required")
    image_hashes = []
    for number, run in enumerate(runs):
        label = "run {}: ".format(number)
        before = len(issues)
        if not isinstance(run, dict):
            require(False, label + "run object required")
            continue
        require(integer(run, "exit_status", 0), label + "process must exit zero")
        report = object_value(run, "report")
        for key, value in (("schema", "spatialrust.public-gpu-conformance.v1"),
                           ("manifest_sha256", manifest_sha256), ("os", expected_os),
                           ("backend", expected_backend), ("accepted", True)):
            require(equal(report, key, value), label + key)
        if expected_backend == "Gl":
            require(equal(report, "requested_backend", "gl"), label + "explicit GL request required")
        require(report.get("preference") == run.get("preference"), label + "report preference")
        frames = report.get("frames")
        require(type(frames) is list and
                [f.get("index") if isinstance(f, dict) else None for f in frames] == list(FRAME_COUNTS),
                label + "every ordered fixed frame required")
        if type(frames) is not list:
            frames = []
        hashes = []
        for frame in frames:
            if not isinstance(frame, dict) or type(frame.get("index")) is not int or frame["index"] not in FRAME_COUNTS:
                require(False, label + "invalid frame")
                continue
            prefix = label + "frame {}: ".format(frame["index"])
            points, cells, neighbors = FRAME_COUNTS[frame["index"]]
            require(frame.get("accepted") is True, prefix + "accepted")
            image = object_value(frame, "image")
            for key, value in (("gray_mismatches", 0), ("box_11x11_mismatches", 0),
                               ("box_259x3_mismatches", 0), ("histogram_bin_mismatches", 0),
                               ("histogram_pixels", 307200), ("host_to_device_bytes", 1228800),
                               ("explicit_device_to_host_bytes", 3687424),
                               ("readbacks_before_device_consumers", 0)):
                require(integer(image, key, value), prefix + "image." + key)
            for key in ("gray_sha256", "box_11x11_sha256", "box_259x3_sha256"):
                digest = image.get(key)
                require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                        prefix + "image." + key)
                hashes.append(digest)
            voxel = object_value(frame, "voxel")
            for key, value in (("point_count", points), ("occupied_cells", cells),
                               ("metadata_mismatches", 0), ("host_to_device_payload_bytes", points * 12),
                               ("construction_device_to_host_bytes", 4),
                               ("centroid_device_to_host_bytes", cells * 12),
                               ("metadata_device_to_host_bytes", points * 4 + cells * 20)):
                require(integer(voxel, key, value), prefix + "voxel." + key)
            require(equal(voxel, "centroid_tolerance_m", 1e-5), prefix + "fixed centroid tolerance")
            error = voxel.get("maximum_centroid_coordinate_error_m")
            valid_error = type(error) in (int, float) and 0 <= error <= 1e-5 and math.isfinite(error)
            require(valid_error, prefix + "finite centroid error within tolerance")
            if valid_error:
                maximum_error = error if maximum_error is None else max(maximum_error, error)
            radius = object_value(frame, "radius")
            for key, value in (("queries", 64), ("count_mismatches", 0), ("total_neighbors", neighbors),
                               ("queries_host_to_device_bytes", 1024), ("explicit_device_to_host_bytes", 512),
                               ("search_host_to_device_bytes", 0), ("search_device_to_host_bytes", 0)):
                require(integer(radius, key, value), prefix + "radius." + key)
            require(equal(radius, "radius_m", 0.03125), prefix + "fixed radius")
            require(radius.get("complete_row_access_rejected") is True, prefix + "count-only overflow")
            require(integer(radius, "tree_host_to_device_bytes") and radius["tree_host_to_device_bytes"] > 0,
                    prefix + "explicit tree upload")
            pools = object_value(frame, "pools")
            for name in ("buffer", "image", "scratch"):
                pool = object_value(pools, name)
                require(integer(pool, "budget_bytes", 67108864) and integer(pool, "cached_bytes") and
                        pool["cached_bytes"] <= pool["budget_bytes"], prefix + "pool." + name)
        image_hashes.append(hashes)
        if len(issues) == before and type(report.get("device_type")) is str and report["device_type"] in HARDWARE_CLASSES:
            classes.add(report["device_type"])
    if len(image_hashes) == 2:
        require(image_hashes[0] == image_hashes[1], "cross-preference image hashes")
    operation_passed = not issues
    require(classes == HARDWARE_CLASSES, "distinct integrated and discrete hardware classes required")
    return {"schema": "spatialrust.public-gpu-report-audit.v1",
            "all_operation_reports_passed": operation_passed,
            "accepted_hardware_classes": sorted(classes),
            "hardware_coverage_accepted": not issues,
            "maximum_centroid_coordinate_error_m": maximum_error,
            "issues": issues}
