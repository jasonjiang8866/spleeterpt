"""Three preselected piano excerpts: TF parity and legacy/bounded agreement.

One independent recording, no isolated stem targets and no quality acceptance.
No audio is copied into the report. Source metadata licenses disagree on version;
retain both notices and do not redistribute audio/derived stems with the repo.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import tensorflow as tf
import torch
from benchmark_bounded import acceptance, paired_metrics
from verify_audio_parity import OfficialGraph, agreement

from spleeterpt.audio import read_audio, separate
from spleeterpt.checkpoint import load_model
from spleeterpt.pretrained import digest
from spleeterpt.streaming import separate_file

DATA_COMMIT = "38f4b06556fa0ff1acda5e677d8ba05d1bc0fff0"
FILENAME = "442789__lena-orsa__happy-music-pistachio-ice-cream-ragtime.hq.ogg"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data-root", "upstream-root", "tf-checkpoint", "checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    ref = subprocess.check_output(
        ["git", "-C", str(args.data_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if ref != DATA_COMMIT:
        raise ValueError("Unpinned librosa data revision")
    source = args.data_root / "audio" / FILENAME
    if digest(source) != "1cc83b775e640a8dd409da9e29f5b5e0124bc5ec9bb4fda975919d31d1d893b8":
        raise ValueError("Piano source hash mismatch")
    upstream_ref = subprocess.check_output(
        ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if upstream_ref != "c8854001ac8acad34a9bc2bd15f28475541828b1":
        raise ValueError("Unpinned upstream audio implementation")
    torch.set_num_threads(2)
    model = load_model(args.checkpoint)
    if list(model.config.instruments) != ["vocals", "piano", "drums", "bass", "other"]:
        raise ValueError("Expected official five-stem order")
    audio = read_audio(source, model.config.sample_rate)
    oracle = OfficialGraph(args.tf_checkpoint, model.config.instruments, args.upstream_root)
    report = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_sha256": digest(args.checkpoint),
        "input": {
            "repository": "https://github.com/librosa/data",
            "commit": ref,
            "path": "audio/" + FILENAME,
            "sha256": digest(source),
            "attribution": "The Piano Lady (Lena Orsa), Pistachio Ice Cream Ragtime",
            "original_url": "https://freesound.org/people/lena_orsa/sounds/442789/",
            "text_notice": source.name.removesuffix(".hq.ogg") + ".txt",
        },
        "scope": "Three 12-second excerpts from ONE solo-piano recording; no ground-truth quality scores. RMS allocation is descriptive, not instrument accuracy.",
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "device": "cpu",
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "tf_onednn": os.environ.get("TF_ENABLE_ONEDNN_OPTS", "default"),
            "threads": 2,
        },
        "upstream_commit": upstream_ref,
        "working_tree_modified": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "config": model.config.to_dict(),
        "tf_checkpoint_files": {
            suffix: digest(Path(str(args.tf_checkpoint) + suffix))
            for suffix in (".meta", ".index", ".data-00000-of-00001")
        },
        "cases": {},
        "strict_tf_passed": True,
        "bounded_passed": True,
    }
    # Metadata applies to basename before .hq, not the audio codec suffix.
    base = source.name.removesuffix(".hq.ogg")
    report["input"]["license_notices"] = {
        ext: (source.parent / (base + ext)).read_text() for ext in (".txt", ".toml")
    }
    try:
        for start in (0, 15, 30):
            clip = audio[:, start * 44100 : (start + 12) * 44100]
            if clip.shape[-1] != 12 * 44100:
                raise ValueError("Incomplete preselected excerpt")
            expected = oracle.separate(clip)
            actual = separate(model, clip)
            strict = {
                name: agreement(expected[name], actual[name].numpy())
                for name in model.config.instruments
            }
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                sf.write(root / "input.wav", clip.numpy().T, 44100, subtype="FLOAT")
                for name, value in actual.items():
                    sf.write(root / f"{name}.wav", value.numpy().T, 44100, subtype="FLOAT")
                separate_file(model, root / "input.wav", root / "bounded")
                bounded = {
                    name: paired_metrics(root / f"{name}.wav", root / "bounded" / f"{name}.wav")
                    for name in model.config.instruments
                }
                for value in bounded.values():
                    value["acceptance"] = acceptance(
                        value, SimpleNamespace(max_abs_error=2e-6, min_si_sdr=100)
                    )
                report["cases"][str(start)] = {
                    "start_seconds": start,
                    "duration_seconds": 12,
                    "wav_sha256": digest(root / "input.wav"),
                    "tf_reference": strict,
                    "legacy_reference": bounded,
                }
            report["strict_tf_passed"] &= all(
                v["allclose_atol_1e-5_rtol_1e-4"] for v in strict.values()
            )
            report["bounded_passed"] &= all(
                not v["acceptance"].startswith("failed") for v in bounded.values()
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"piano excerpt {start}s recorded", flush=True)
    finally:
        oracle.close()
    if not (report["strict_tf_passed"] and report["bounded_passed"]):
        raise SystemExit("FAIL: piano clip agreement; original thresholds retained")


if __name__ == "__main__":
    main()
