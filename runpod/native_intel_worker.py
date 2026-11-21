"""Probe the processor before installing anything, then replay the frozen CPU calibration."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import worker

SOURCE = "e63c8987fdf18a30c8d85785f1d680e08bc9430d"
UPSTREAM = "c8854001ac8acad34a9bc2bd15f28475541828b1"


def cpu_identity(cpuinfo):
    fields = dict(
        line.split(":", 1) for line in cpuinfo.split("\n\n")[0].splitlines() if ":" in line
    )
    fields = {key.strip(): value.strip() for key, value in fields.items()}
    return {"vendor": fields["vendor_id"], "model": fields["model name"]}


def measure():
    cpu = cpu_identity(Path("/proc/cpuinfo").read_text())
    if cpu["vendor"] != "GenuineIntel":
        return {"status": "other-cpu", "cpu": cpu}
    worker.RESULT = {"status": "running", "cpu": cpu}
    root = Path("/tmp/native-intel")
    root.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "SPT_RESULT_TOKEN"}
    env.update(
        TF_ENABLE_ONEDNN_OPTS="0",
        TF_CPP_MIN_LOG_LEVEL="2",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="2",
    )

    def run(*args, cwd=root, check=True):
        return subprocess.run(args, cwd=cwd, env=env, check=check)

    source = root / "source"
    run("git", "worktree", "add", "--detach", str(source), SOURCE, cwd=Path.cwd())
    run("git", "clone", "-q", "https://github.com/deezer/spleeter.git", "upstream")
    run("git", "-C", "upstream", "checkout", "-q", UPSTREAM)
    run(sys.executable, "-m", "pip", "install", "uv==0.13.4")
    run(
        "uv",
        "sync",
        "--python",
        "3.12.16",
        "--locked",
        "--group",
        "dev",
        "--group",
        "release",
        cwd=source,
    )
    python = source / ".venv/bin/python"
    weights = root / "weights"
    run(python, "scripts/prepare_five_stem.py", "--work", weights, cwd=source)
    report = root / "native.json"
    proc = run(
        python,
        "scripts/verify_native_five_stem.py",
        "--weights",
        weights,
        "--upstream-root",
        root / "upstream",
        "--output",
        report,
        cwd=source,
        check=False,
    )
    return {
        "status": "passed" if proc.returncode == 0 else "failed",
        "cpu": cpu,
        "report": json.loads(report.read_text()) if report.exists() else None,
    }


if __name__ == "__main__":
    server = worker.ThreadingHTTPServer(("0.0.0.0", 8888), worker.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        worker.RESULT = measure()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        worker.RESULT = {**worker.RESULT, "status": "failed", "error": type(exc).__name__}
    threading.Event().wait()
