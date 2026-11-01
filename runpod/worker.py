"""Serve a token-protected, bounded result over the Runpod HTTPS proxy."""

import base64
import hashlib
import hmac
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RESULT = {"status": "running"}
TOKEN = os.environ.get("SPT_RESULT_TOKEN", "")
RESTORE_LOCK = threading.Lock()
EXPORT_FILE = Path("/workspace/checkpoint.tar")
RESTORE_FILE = Path("/workspace/restore.tar")


PROGRESS = re.compile(
    r"(?:Progress: (?:benchmark (?:official_converted|htdemucs|official_2stems|melband_kim) songs \d+/(?:1|10)|convergence (?:official_converted|lr1e5_step(?:90|450|900)|lr3e6_step(?:90|450|900)) songs \d+/10|convergence (?:lr1e5|lr3e6) step (?:90|450|900) complete|training arm (?:normal|frozen) complete|global input files \d+/\d+|archive chunks \d+/\d+|verifying audio \d+/100|evaluated validation songs \d+/10)|Prepared training-pool track \d+/100|Durable checkpoint committed: step \d+)"
)


def progress(path=Path("/workspace/smoke.log")):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            lines = stream.read(65536).decode(errors="replace").splitlines()
        return next((line for line in reversed(lines) if PROGRESS.fullmatch(line)), None)
    except OSError:
        return None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not TOKEN or not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + TOKEN
        ):
            self.send_error(401)
            return
        route = urllib.parse.urlsplit(self.path)
        if route.path == "/checkpoint":
            from transfer import PROBE_BYTES

            query = urllib.parse.parse_qs(route.query)
            try:
                if set(query) - {"offset", "length", "probe"} or any(
                    len(v) != 1 for v in query.values()
                ):
                    raise ValueError("Unexpected chunk query")
                offset, length = int(query["offset"][0]), int(query["length"][0])
                probe = query.get("probe") == ["1"]
                if not probe and (
                    RESULT.get("status") not in {"passed", "failed"} or not EXPORT_FILE.is_file()
                ):
                    self.send_error(404)
                    return
                total = len(PROBE_BYTES) if probe else EXPORT_FILE.stat().st_size
                if not 0 < length <= 4 * 1024 * 1024 or not 0 <= offset < offset + length <= total:
                    raise ValueError("Invalid chunk bounds")
                if probe:
                    chunk = PROBE_BYTES[offset : offset + length]
                else:
                    with EXPORT_FILE.open("rb") as stream:
                        stream.seek(offset)
                        chunk = stream.read(length)
                data = json.dumps(
                    {
                        "offset": offset,
                        "bytes": len(chunk),
                        "sha256": hashlib.sha256(chunk).hexdigest(),
                        "data": base64.b64encode(chunk).decode("ascii"),
                    }
                ).encode()
            except (ValueError, KeyError):
                self.send_error(400)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path != "/result":
            self.send_error(404)
            return
        result = dict(RESULT)
        if result.get("status") == "running":
            result["progress"] = progress()
        data = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_PUT(self):
        from recovery_result import MAX_BUNDLE

        if not TOKEN or not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + TOKEN
        ):
            self.send_error(401)
            return
        route = urllib.parse.urlsplit(self.path)
        if route.path != "/restore" or os.environ.get("SPT_TASK") != "recovery-resume":
            self.send_error(404)
            return
        with RESTORE_LOCK:
            temporary = RESTORE_FILE.with_suffix(".part")
            try:
                spec = json.loads(os.environ["SPT_RESTORE_DESCRIPTOR"])
                size = int(self.headers.get("Content-Length", "-1"))
                offset = int(urllib.parse.parse_qs(route.query).get("offset", ["0"])[0])
                existing = temporary.stat().st_size if temporary.exists() else 0
                if (
                    not 0 < size <= 8 * 1024 * 1024
                    or not 0 < spec["bytes"] <= MAX_BUNDLE
                    or offset != existing
                    or offset + size > spec["bytes"]
                    or RESTORE_FILE.exists()
                ):
                    self.send_error(409)
                    return
                self.connection.settimeout(120)
                chunk = self.rfile.read(size)
                if len(chunk) != size:
                    raise ValueError("Short upload")
                with temporary.open("ab") as stream:
                    stream.write(chunk)
                reply = {"received": offset + size}
                if offset + size == spec["bytes"]:
                    h = hashlib.sha256()
                    with temporary.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            h.update(block)
                    if h.hexdigest() != spec["sha256"]:
                        temporary.unlink()
                        raise ValueError("Upload checksum mismatch")
                    os.replace(temporary, RESTORE_FILE)
                    reply["sha256"] = h.hexdigest()
                data = json.dumps(reply).encode()
                self.send_response(201)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, OSError):
                self.send_error(400)

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
        if task in {
            "quality",
            "finetune",
            "bn",
            "frozen",
            "training",
            "recovery-export",
            "recovery-resume",
            "persistent-prepare",
            "persistent-train",
            "persistent-resume",
        }:
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
                (
                    [
                        python,
                        "runpod/recovery.py",
                        "--stage",
                        task.split("-", 1)[1],
                        "--output",
                        str(report),
                    ]
                    if task.startswith("recovery-")
                    else (
                        [
                            python,
                            "runpod/persistent.py",
                            "--stage",
                            task.split("-", 1)[1],
                            "--output",
                            str(report),
                        ]
                        if task.startswith("persistent-")
                        else [python, f"runpod/{task}.py", "--output", str(report)]
                    )
                ),
            ]
        elif task == "benchmark":
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
                    "torchaudio==2.8.0",
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
                    "dora-search==0.1.12",
                    "einops==0.8.1",
                    "julius==0.2.7",
                    "lameenc==1.8.1",
                    "openunmix==1.3.0",
                    "pyyaml==6.0.2",
                    "tqdm==4.67.1",
                ],
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "demucs @ git+https://github.com/facebookresearch/demucs.git@e976d93ecc3865e5757426930257e200846a520a",
                ],
                [python, "runpod/benchmark_dependencies.py"],
                [python, "runpod/benchmark.py", "--output", str(report)],
            ]
            profile = os.environ.get("SPT_BENCHMARK_PROFILE", "four-stem")
            if profile == "vocal":
                commands = commands[:3] + [
                    [
                        python,
                        "-m",
                        "pip",
                        "install",
                        "--no-cache-dir",
                        "-e",
                        ".[evaluate]",
                        "-r",
                        "runpod/roformer-requirements.txt",
                    ],
                    [python, "runpod/roformer_dependencies.py"],
                    [python, "runpod/benchmark.py", "--profile", "vocal", "--output", str(report)],
                ]
            elif profile == "convergence":
                commands = commands[:3] + [
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
                    [python, "runpod/convergence.py", "--output", str(report)],
                ]
            elif profile != "four-stem":
                raise ValueError("Unknown benchmark profile")
            env.update(
                OPENBLAS_NUM_THREADS="2",
                OMP_NUM_THREADS="2",
                TF_NUM_INTRAOP_THREADS="2",
                TF_NUM_INTEROP_THREADS="2",
                TF_CPP_MIN_LOG_LEVEL="2",
            )
        elif task in {"global-export", "global-training-export"}:
            commands = [["python", "runpod/global_inputs.py", "--output", str(report)]]
            if task == "global-training-export":
                commands[0].append("--training")
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
