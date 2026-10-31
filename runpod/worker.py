"""Serve a token-protected, bounded result over the Runpod HTTPS proxy."""

import hmac
import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RESULT = {"status": "running"}
TOKEN = os.environ.get("SPT_RESULT_TOKEN", "")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not TOKEN or not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + TOKEN
        ):
            self.send_error(401)
            return
        if self.path != "/result":
            self.send_error(404)
            return
        data = json.dumps(RESULT).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def run():
    global RESULT
    log = Path("/workspace/smoke.log")
    report = Path("/workspace/report.json")
    # Never pass the response token into training or dependency installation.
    env = {k: v for k, v in os.environ.items() if k != "SPT_RESULT_TOKEN"}
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    expiry = int(os.environ["SPT_EXPIRY"])
    try:
        task = os.environ.get("SPT_TASK", "smoke")
        commands = [
            ["python", "-m", "pip", "install", "-e", "."],
            ["python", "runpod/gpu_smoke.py", "--output", str(report)],
        ]
        if task in {"quality", "training"}:
            python = "/workspace/eval-venv/bin/python"
            commands = [
                [
                    "bash",
                    "-lc",
                    "command -v ffmpeg && command -v ffprobe || (apt-get update && apt-get install -y --no-install-recommends ffmpeg)",
                ],
                ["python", "-m", "venv", "/workspace/eval-venv"],
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "torch==2.8.0",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu128",
                ],
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "-e",
                    ".[evaluate]",
                    "numpy==2.0.2",
                    "scipy==1.15.3",
                ],
                [python, "-m", "pip", "check"],
                [python, f"runpod/{task}.py", "--output", str(report)],
            ]
        elif task != "smoke":
            raise ValueError("Unknown worker task")
        with log.open("w") as stream:
            for index, cmd in enumerate(commands):
                RESULT = {
                    "status": "running",
                    "phase": f"{task}: command {index + 1}/{len(commands)}",
                }
                process = subprocess.Popen(
                    cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True
                )
                try:
                    code = process.wait(timeout=max(1, expiry - time.time() - 180))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise RuntimeError("Worker deadline exceeded") from None
                if code:
                    raise RuntimeError(f"Worker command failed with exit code {code}")
        result = {"status": "passed", "report": json.loads(report.read_text())}
    except Exception as error:  # noqa: BLE001 -- always publish a sanitized failure result
        result = {"status": "failed", "error": type(error).__name__}
        if report.exists():
            result["report"] = json.loads(report.read_text())
    result["log_tail"] = log.read_text(errors="replace")[-100_000:] if log.exists() else ""
    RESULT = result


if __name__ == "__main__":
    if len(TOKEN) < 32:
        raise SystemExit("Missing result token")
    server = ThreadingHTTPServer(("0.0.0.0", 8888), Handler)
    threading.Thread(target=run, daemon=True).start()
    server.serve_forever()
