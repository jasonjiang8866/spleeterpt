"""Verify exact pilot scope and independently recompute downloaded summaries."""

import json
import sys
from pathlib import Path


def validate_quality(result, sha, output):
    if result.get("status") != "passed":
        raise RuntimeError("GPU quality worker failed; inspect result.json log_tail")
    report = result.get("report", {})
    if report.get("kind") != "quality-preview" or report.get("commit") != sha:
        raise RuntimeError("Quality result identity mismatch")
    if report.get("torch") != "2.8.0+cu128" or report.get("gpu") != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("Quality runtime differs from pinned stable CUDA environment")
    names = {"manifest.json", "status.json", "summary.json", "runtime.json"}
    names |= {f"tracks/{i:04d}.json" for i in range(5)}
    expected = {f"{stems}stems/{name}" for stems in (2, 4) for name in names}
    files = report.get("files", {})
    if set(files) != expected or any(not isinstance(text, str) for text in files.values()):
        raise RuntimeError("Missing or unexpected quality files")
    root = Path(__file__).resolve().parents[1]
    assets = json.loads((root / "runpod/quality-assets.json").read_text())
    sys.path.insert(0, str(root / "scripts"))
    from musdb_evaluation import VARIANTS, validate_run

    for name, text in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for stems in (2, 4):
        folder = output / f"{stems}stems"
        manifest = json.loads((folder / "manifest.json").read_text())
        env = manifest["environment"]
        if env["packages"]["torch"] != "2.8.0+cu128" or env["device"] != "cuda":
            raise RuntimeError("Evaluation did not use stable CUDA torch")
        if env["variant_devices"] != {
            v: "cpu" if v == "tensorflow_current" else "cuda" for v in VARIANTS
        }:
            raise RuntimeError("Unexpected comparison devices")
        if manifest["code"]["spleeterpt"]["commit"] != sha:
            raise RuntimeError("Evaluation source commit mismatch")
        for key in ("upstream", "legacy"):
            if manifest["code"][key]["commit"] != assets[key + "_commit"]:
                raise RuntimeError("Reference source commit mismatch")
        if (
            len(manifest["instruments"]) != stems
            or manifest["start_seconds"] != 0
            or manifest["duration_seconds"] is not None
        ):
            raise RuntimeError("Unexpected stems or evaluation interval")
        expected_tracks = {
            "test/" + name.removesuffix(".stem.mp4"): {"test/" + name: h}
            for name, h in assets["tracks"].items()
        }
        if {r["id"]: r["sha256"] for r in manifest["selection"]} != expected_tracks:
            raise RuntimeError("Preview selection or hashes differ from pinned inputs")
        summary = validate_run(folder)
        if summary["track_count"] != 5:
            raise RuntimeError("Incomplete pilot")
        print(f"PASS: {stems}-stem, 5 previews, four variants; summary recomputed", flush=True)
