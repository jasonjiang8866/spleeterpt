"""Validate the opt-in CPU separator and optionally score the registered TRIOS mixtures."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from register_trios import PIECES, register
from score_piano_holdout import assess, read
from study_audio_cross import array_hash, environment, waveform_metrics
from verify_audio_parity import OfficialGraph, audio_cases

from spleeterpt.audio import read_audio, separate
from spleeterpt.checkpoint import load_model
from spleeterpt.pretrained import digest


def comparisons(reference, outputs):
    return {s: waveform_metrics(ref, outputs[s].numpy()) for s, ref in reference.items()}


def passed(rows):
    return all(row["allclose_atol_1e-5_rtol_1e-4"] for row in rows.values())


def benchmark(model, waveform):
    # Per-mode warmup; alternate execution order to limit warm-cache/order effects.
    samples = {"default": [], "candidate": []}
    for mode in samples:
        separate(model, waveform, five_stem_cpu_precision=mode == "candidate")
    for index in range(6):
        for mode in reversed(samples) if index % 2 else samples:
            start = time.perf_counter()
            separate(model, waveform, five_stem_cpu_precision=mode == "candidate")
            samples[mode].append(time.perf_counter() - start)
    return samples


def score_holdout(archive, manifest, oracle, model):
    if digest(archive) != manifest["trios_archive"]["sha256"]:
        raise ValueError("TRIOS archive identity mismatch")
    results = {}
    with tempfile.TemporaryDirectory() as tmp, ZipFile(archive) as zf:
        root = Path(tmp)
        # Verify both recordings before running either through the model.
        for item in manifest["records"]:
            if "trios" in item:
                name = item["recording_id"]
                piece = name.removeprefix("trios-")
                if register(zf, piece, PIECES[piece], root / name) != item["trios"]:
                    raise ValueError(f"TRIOS reference mismatch for {piece}")
        for piece in PIECES:
            name = f"trios-{piece}"
            inputs = {role: read(root, name, role) for role in ("mix", "piano", "other")}
            audio = torch.from_numpy(np.ascontiguousarray(inputs["mix"].T))
            reference = oracle.separate(audio)
            outputs = separate(model, audio, five_stem_cpu_precision=True)
            results[name] = {
                "input_pcm_sha256": array_hash(inputs["mix"]),
                "waveforms": comparisons(reference, outputs),
                "quality": assess(
                    inputs,
                    {s: v.T for s, v in reference.items()},
                    {s: v.numpy().T for s, v in outputs.items()},
                ),
            }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trios-archive", type=Path)
    args = parser.parse_args()
    if os.environ.get("TF_ENABLE_ONEDNN_OPTS") != "0":
        raise ValueError("set TF_ENABLE_ONEDNN_OPTS=0 before starting Python")
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    pins = json.loads((root / "scripts/five_stem_source.json").read_text())
    history = json.loads(
        (root / "docs/results/five-stem-20261102/5stems-strict-final.json").read_text()
    )
    source, checkpoint = args.upstream_root / "audio_example.mp3", args.weights / "5stems.pt"
    upstream = subprocess.check_output(
        ["git", "-C", args.upstream_root, "rev-parse", "HEAD"], text=True
    ).strip()
    tf_prefix = args.weights / "5stems/model"
    tf_hashes = {
        suffix: digest(Path(str(tf_prefix) + suffix)) for suffix in history["tf_checkpoint_files"]
    }
    if (upstream, digest(source), digest(checkpoint)) != (
        pins["upstream_commit"],
        pins["audio_sha256"],
        pins["converted_sha256"],
    ) or tf_hashes != history["tf_checkpoint_files"]:
        raise ValueError("source or checkpoint differs from frozen calibration")
    model = load_model(checkpoint).eval()
    wave = read_audio(source, model.config.sample_rate)
    report = {
        "source_commit": subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True
        ).strip(),
        "upstream_commit": upstream,
        "checkpoint_sha256": digest(checkpoint),
        "tf_checkpoint_files": tf_hashes,
        "environment": environment(),
        "thresholds": {"atol": 1e-5, "rtol": 1e-4},
        "calibration": {},
        "holdout": None,
    }
    oracle = OfficialGraph(tf_prefix, model.config.instruments, args.upstream_root)
    try:
        for name, audio in audio_cases(wave, model.config, True).items():
            if array_hash(audio.numpy()) != history["case_provenance"][name]["decoded_pcm_sha256"]:
                raise ValueError("calibration PCM mismatch")
            ref = oracle.separate(audio)
            report["calibration"][name] = {
                "input_pcm_sha256": array_hash(audio.numpy()),
                "reference_sha256": {s: array_hash(v) for s, v in ref.items()},
                "candidate": comparisons(ref, separate(model, audio, five_stem_cpu_precision=True)),
                "default": comparisons(ref, separate(model, audio)),
            }
        report["calibration_passed"] = all(
            passed(c["candidate"]) for c in report["calibration"].values()
        )
        report["timing_music_original_seconds"] = benchmark(model, wave)
        if args.trios_archive and report["calibration_passed"]:
            manifest = root / "scripts/five_stem_corpus.json"
            report["holdout_manifest_sha256"] = digest(manifest)
            report["holdout"] = score_holdout(
                args.trios_archive, json.loads(manifest.read_text()), oracle, model
            )
    finally:
        oracle.close()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["calibration_passed"]:
        raise SystemExit("historical strict calibration failed")
    if report["holdout"] and not all(
        passed(r["waveforms"]) and all(q["passed"] for q in r["quality"].values())
        for r in report["holdout"].values()
    ):
        raise SystemExit("holdout evaluation completed with failures")


if __name__ == "__main__":
    main()
