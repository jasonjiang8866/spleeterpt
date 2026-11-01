"""Exercise worker setup without installing packages or renting compute."""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import worker


class WorkerDependencyTests(unittest.TestCase):
    def commands(self, task, profile="four-stem"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "report.json").write_text(json.dumps({"passed": True}))
            environment = {
                "SPT_TASK": task,
                "SPT_BENCHMARK_PROFILE": profile,
                "SPT_EXPIRY": str(int(time.time()) + 600),
                "SPT_RESULT_TOKEN": "never-pass-to-installers",
            }
            with (
                patch.dict(os.environ, environment),
                patch.object(worker, "Path", side_effect=lambda path: root / Path(path).name),
                patch.object(
                    worker.subprocess, "Popen", return_value=Mock(wait=lambda **_: 0)
                ) as run,
            ):
                worker.run()
            self.assertEqual(worker.RESULT["status"], "passed")
            for call in run.call_args_list:
                self.assertNotIn("SPT_RESULT_TOKEN", call.kwargs["env"])
            return [call.args[0] for call in run.call_args_list]

    def test_cuda_profiles_keep_interpreter_and_explicit_backend(self):
        for task, profile in [
            ("quality", "four-stem"),
            ("benchmark", "four-stem"),
            ("benchmark", "vocal"),
            ("benchmark", "convergence"),
            ("benchmark", "bounded"),
        ]:
            with self.subTest(task=task, profile=profile):
                commands = self.commands(task, profile)
                venv = next(command for command in commands if "venv" in command)
                self.assertEqual(venv[venv.index("--python") + 1], sys.executable)
                installs = [command for command in commands if "install" in command][1:]
                self.assertIn("torch==2.8.0", installs[0])
                self.assertIn("https://download.pytorch.org/whl/cu128", installs[0])
                for command in installs:
                    self.assertIn("uv", command)
                    self.assertIn("--no-sources", command)
                    self.assertEqual(
                        command[command.index("--python") + 1], "/workspace/eval-venv/bin/python"
                    )
                if profile == "four-stem" and task == "benchmark":
                    demucs = next(
                        command for command in installs if any("demucs @" in arg for arg in command)
                    )
                    self.assertIn("--no-deps", demucs)

    def test_smoke_preserves_image_environment(self):
        commands = self.commands("smoke")
        installs = [command for command in commands if "install" in command]
        self.assertIn("uv==0.13.4", installs[0])
        self.assertIn("--system", installs[1])
        self.assertFalse(any("torch==" in arg for arg in installs[1]))

    def test_exports_need_no_package_installation(self):
        for task in ("global-export", "global-training-export"):
            commands = self.commands(task)
            self.assertEqual(len(commands), 1)
            self.assertIn("runpod/global_inputs.py", commands[0])
            self.assertNotIn("install", commands[0])


if __name__ == "__main__":
    unittest.main()
