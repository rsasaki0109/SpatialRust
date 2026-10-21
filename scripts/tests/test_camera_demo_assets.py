import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("camera_builder", ROOT / "scripts" / "build_camera_demo.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CameraAssetAdmissionTests(unittest.TestCase):
    def manifest(self):
        return {"schema": "spatialrust.camera-assets.v1", "max_total_bytes": 1024,
                "assets": [{"path": "vendor/a.wasm", "bytes": 4, "sha256": "0" * 64,
                            "url": "https://cdn.jsdelivr.net/npm/package@1/a.wasm"}]}

    def test_accepts_fixed_manifest(self):
        self.assertEqual(len(MODULE.validate_assets(self.manifest())), 1)

    def test_rejects_traversal_or_absolute_paths(self):
        for name in ["../a", "vendor/../a", "vendor/./a", "/vendor/a", "vendor\\a"]:
            document = self.manifest();document["assets"][0]["path"] = name
            with self.subTest(name=name), self.assertRaises(ValueError): MODULE.validate_assets(document)

    def test_rejects_duplicate_outputs(self):
        document = self.manifest();document["assets"].append(copy.deepcopy(document["assets"][0]))
        with self.assertRaises(ValueError): MODULE.validate_assets(document)

    def test_rejects_nonpublisher_or_insecure_urls(self):
        for url in ["http://cdn.jsdelivr.net/x", "https://example.com/x", "https://user:secret@cdn.jsdelivr.net/x", "https://cdn.jsdelivr.net/x#fragment"]:
            document = self.manifest();document["assets"][0]["url"] = url
            with self.subTest(url=url), self.assertRaises(ValueError): MODULE.validate_assets(document)

    def test_rejects_invalid_sizes_or_unpinned_hashes(self):
        for key, value in [("bytes", 0), ("bytes", -1), ("bytes", True), ("bytes", 1025), ("sha256", "latest"), ("sha256", "a" * 63)]:
            document = self.manifest();document["assets"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): MODULE.validate_assets(document)

    def test_rejects_invalid_global_limits(self):
        for value in [0, -1, True, MODULE.MAX_BYTES + 1]:
            document = self.manifest();document["max_total_bytes"] = value
            with self.subTest(value=value), self.assertRaises(ValueError): MODULE.validate_assets(document)


if __name__ == "__main__": unittest.main()
