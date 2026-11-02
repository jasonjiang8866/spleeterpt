import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bounded import measured_command


class Tests(unittest.TestCase):
    def test_collects_only_complete_numerical_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "report.json"
            args = (
                "python",
                "benchmark",
                "--durations",
                1,
                12,
                "--input",
                "x",
                "--output",
                destination,
            )
            with patch(
                "bounded.command", side_effect=subprocess.CalledProcessError(1, "benchmark")
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    measured_command(destination, *args, collect_rejection=True)
                report = {
                    "numerical_pass": False,
                    "cases": [{"seconds": 1, "stems": {}, "measurements": []}],
                }
                destination.write_text(json.dumps(report))
                with self.assertRaises(subprocess.CalledProcessError):
                    measured_command(destination, *args, collect_rejection=True)
                report["cases"].append({"seconds": 12, "stems": {}, "measurements": []})
                destination.write_text(json.dumps(report))
                self.assertFalse(measured_command(destination, *args, collect_rejection=True))
                with self.assertRaises(subprocess.CalledProcessError):
                    measured_command(destination, *args)


if __name__ == "__main__":
    unittest.main()
