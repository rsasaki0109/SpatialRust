"""Regression checks for observed Windows and WASM CI failures (stdlib only)."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('wasm_version', SCRIPTS / 'wasm_bindgen_version.py')
wasm_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wasm_version)


class RuntimeCiTests(unittest.TestCase):
    def test_resolved_version_and_ambiguous_or_missing_version(self):
        with tempfile.TemporaryDirectory() as directory:
            lockfile = Path(directory) / 'Cargo.lock'
            lockfile.write_text('[[package]]\nname = "wasm-bindgen"\nversion = "0.2.129"\n', encoding='utf-8')
            self.assertEqual(wasm_version.required_version(lockfile), '0.2.129')
            lockfile.write_text(lockfile.read_text(encoding='utf-8') + '[[package]]\nname = "wasm-bindgen"\nversion = "0.2.126"\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                wasm_version.required_version(lockfile)
            lockfile.write_text('version = 4\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                wasm_version.required_version(lockfile)

    def test_junit_annotations_survive_redirected_cp1252_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'junit.xml'
            report.write_text('<testsuite><testcase name="windows"><failure>target → source 50%\n::warning::nested</failure></testcase></testsuite>', encoding='utf-8')
            result = subprocess.run([sys.executable, str(SCRIPTS / 'report_junit_failures.py'), str(report)],
                                    env=dict(os.environ, PYTHONIOENCODING='cp1252', PYTHONUTF8='0'),
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.decode('utf-8').strip(),
                             '::error::windows%0Atarget → source 50%25%0A::warning::nested')


if __name__ == '__main__':
    unittest.main()
