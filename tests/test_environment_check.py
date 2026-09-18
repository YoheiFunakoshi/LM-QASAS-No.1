"""Hardware-free tests: all GPU responses here are explicitly simulated."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "environment_check.py"
SPEC = importlib.util.spec_from_file_location("environment_check", SCRIPT)
environment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(environment)


class EnvironmentCheckTests(unittest.TestCase):
    def test_nvidia_probe_handles_absence_failure_timeout_and_success(self):
        with patch.object(environment.shutil, "which", return_value=None):
            self.assertEqual(environment.nvidia_smi_info()["reason"], "command_not_found")
        cases = [
            (subprocess.CompletedProcess([], 1, "", "simulated failure"), "command_failed"),
            (subprocess.TimeoutExpired("nvidia-smi", 15), "command_timeout"),
            (subprocess.CompletedProcess([], 0, "malformed\n", ""), "unexpected_command_output"),
            (subprocess.CompletedProcess([], 0, "Simulated GPU, 8192, 000.00\n", ""), None),
        ]
        for response, reason in cases:
            with self.subTest(reason=reason), patch.object(environment.shutil, "which", return_value="nvidia-smi"):
                kwargs = {"side_effect": response} if isinstance(response, Exception) else {"return_value": response}
                with patch.object(environment.subprocess, "run", **kwargs) as run:
                    report = environment.nvidia_smi_info()
                    self.assertEqual(run.call_args.kwargs["timeout"], 15)
                    if reason is not None:
                        self.assertEqual(report["reason"], reason)
                    else:
                        self.assertEqual(report["devices"][0]["vram_mib"], 8192)

    def test_required_import_failure_is_saved_and_existing_output_is_preserved(self):
        simulated = {"packages": {"torch": {"status": "error"}, "ablang2": {"status": "ok"}}}
        # Normal inherited permissions also work under restricted Windows accounts.
        directory = Path.cwd() / f".test-env-{uuid.uuid4().hex}"
        directory.mkdir()
        output = directory / "report.json"
        try:
            with patch.object(environment, "collect_report", return_value=simulated), patch("builtins.print"):
                result = environment.main(["--output", str(output), "--require-packages"])
                self.assertEqual(result, 1)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8")), simulated)
                first = output.read_bytes()
                self.assertEqual(environment.main(["--output", str(output)]), 2)
                self.assertEqual(output.read_bytes(), first)
        finally:
            output.unlink(missing_ok=True)
            directory.rmdir()


if __name__ == "__main__":
    unittest.main()
