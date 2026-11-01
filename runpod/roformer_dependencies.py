"""Fail before model work if the pinned vocal runtime or scoring tools drift."""

import importlib.metadata
import subprocess
import sys

from benchmark_dependencies import check_scoring_runtime
from quality import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from vocal_metrics import DEPENDENCIES


def check():
    check_scoring_runtime()
    for name, version in {**DEPENDENCIES, "torch": "2.8.0+cu128"}.items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Vocal dependency drift: " + name)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print("PASS: pinned vocal dependencies and scoring runtime", flush=True)


if __name__ == "__main__":
    check()
