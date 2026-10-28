"""Small deterministic MUSDB preview check, NOT an official MUSDB benchmark.

Select first N lexicographically sorted test/*.stem.mp4 files. Decode streams
according to sigsep-mus-db/musdb/configs/mus.yaml. No training or downloads.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from diagnose_legacy_port import aligned_legacy, load_legacy, use_four_stem_activations
from verify_audio_parity import OfficialGraph, agreement

from spleeterpt.audio import separate
from spleeterpt.checkpoint import load_model


def decode_stream(path, index):
    data = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            f"0:a:{index}",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "2",
            "-ar",
            "44100",
            "pipe:1",
        ]
    )
    return torch.from_numpy(np.frombuffer(data, dtype="<f4").reshape(-1, 2).T.copy())


def si_sdr(reference, estimate):
    """Global stereo SI-SDR: flatten channels, remove mean, fit one scalar.

    No time shift, source permutation, or distortion filter is fitted. This is
    not museval/BSSEval-v4 SDR and must not be compared with its published scores.
    """
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    got = np.asarray(estimate, dtype=np.float64).reshape(-1)
    ref, got = ref - ref.mean(), got - got.mean()
    energy = np.dot(ref, ref)
    if energy < 1e-12:
        return None
    projected = ref * (np.dot(got, ref) / energy)
    error = got - projected
    return float(
        10 * np.log10((np.dot(projected, projected) + 1e-20) / (np.dot(error, error) + 1e-20))
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--tf-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracks", type=int, default=5)
    args = parser.parse_args()
    if args.tracks < 1:
        raise ValueError("tracks must be positive")
    tracks = sorted((args.root / "test").glob("*.stem.mp4"))[: args.tracks]
    if len(tracks) != args.tracks:
        raise ValueError("not enough test tracks")
    torch.set_num_threads(2)
    model = load_model(args.checkpoint)
    config = model.config
    stems = len(config.instruments)
    if stems not in (2, 4):
        raise ValueError("MUSDB has no isolated piano target")
    legacy = load_legacy(args.legacy_root.resolve(), args.tf_checkpoint, stems)
    corrected = load_legacy(args.legacy_root.resolve(), args.tf_checkpoint, stems)
    if stems == 4:
        use_four_stem_activations(corrected)
    oracle = OfficialGraph(args.tf_checkpoint, config.instruments, args.upstream_root)
    report = {
        "dataset": "MUSDB18-7-STEMS preview, test subset",
        "selection": f"first {args.tracks} lexicographically sorted test filenames, selected before inference",
        "metric": "global stereo SI-SDR (dB), not BSSEval-v4 SDR",
        "aggregation": "median per instrument across selected tracks",
        "instruments": list(config.instruments),
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "legacy_commit": subprocess.check_output(
            ["git", "-C", str(args.legacy_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "tracks": [],
    }
    try:
        for path in tracks:
            streams = [decode_stream(path, index) for index in range(5)]
            length = min(item.shape[-1] for item in streams)
            mixture, drums, bass, other, vocals = [item[:, :length] for item in streams]
            targets = {
                "vocals": vocals,
                "drums": drums,
                "bass": bass,
                "other": other,
                "accompaniment": drums + bass + other,
            }
            variants = {
                "tensorflow_current": oracle.separate(mixture),
                "spleeterpt": {
                    name: value.numpy() for name, value in separate(model, mixture).items()
                },
                "legacy_unmodified": dict(
                    zip(config.instruments, [value.numpy() for value in legacy(mixture)])
                ),
                "legacy_corrected": dict(
                    zip(
                        config.instruments,
                        [value.numpy() for value in aligned_legacy(corrected, mixture)],
                    )
                ),
            }
            # Fair common crop for all scores; legacy drops the fractional-hop tail.
            common = min(
                value.shape[-1] for outputs in variants.values() for value in outputs.values()
            )
            scores = {
                variant: {
                    name: si_sdr(targets[name][:, :common], outputs[name][:, :common])
                    for name in config.instruments
                }
                for variant, outputs in variants.items()
            }
            parity = {
                name: agreement(variants["tensorflow_current"][name], variants["spleeterpt"][name])
                for name in config.instruments
            }
            record = {
                "track": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "decoded_samples": length,
                "evaluated_samples": common,
                "si_sdr_db": scores,
                "torch_vs_tf": parity,
            }
            report["tracks"].append(record)
            print(json.dumps({"track": path.name, "si_sdr_db": scores}), flush=True)
    finally:
        oracle.close()
    report["median_si_sdr_db"] = {
        variant: {
            name: float(
                np.median(
                    [
                        track["si_sdr_db"][variant][name]
                        for track in report["tracks"]
                        if track["si_sdr_db"][variant][name] is not None
                    ]
                )
            )
            for name in config.instruments
        }
        for variant in ("tensorflow_current", "spleeterpt", "legacy_unmodified", "legacy_corrected")
    }
    report["max_abs_torch_tf_score_difference_db"] = max(
        abs(track["si_sdr_db"]["tensorflow_current"][name] - track["si_sdr_db"]["spleeterpt"][name])
        for track in report["tracks"]
        for name in config.instruments
        if track["si_sdr_db"]["tensorflow_current"][name] is not None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"median_si_sdr_db": report["median_si_sdr_db"]}), flush=True)


if __name__ == "__main__":
    main()
