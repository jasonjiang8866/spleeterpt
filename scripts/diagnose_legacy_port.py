"""Controlled ablation of fwcd/spleeter-pytorch's 2/4-stem pipeline.

Uses an explicitly supplied, separately cloned repository. Does not edit it or
vendor its implementation. Bypasses only its CLI initializer/torchaudio loader;
all variants receive the same decoded float32 PCM and official checkpoint.
"""

import argparse
import importlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from verify_audio_parity import OfficialGraph, agreement, audio_cases

from spleeterpt.audio import read_audio, separate, stft
from spleeterpt.checkpoint import load_model


def load_legacy(root, checkpoint, stems):
    # Only skip __init__.py, which imports CLI audio I/O and torchaudio. The exact
    # original estimator, separator, U-Net, and conversion modules are loaded.
    package = types.ModuleType("spleeter_pytorch")
    package.__path__ = [str(root / "spleeter_pytorch")]
    sys.modules["spleeter_pytorch"] = package
    module = importlib.import_module("spleeter_pytorch.estimator")
    return module.Estimator(stems, str(checkpoint)).eval()


@torch.inference_mode()
def aligned_legacy(estimator, audio):
    """Only pad/crop around the original estimator; leave its internals unchanged.

    Legacy center=True internally pads n_fft/2; the external leading n_fft/2
    restores upstream's total n_fft. Tail padding reproduces TF pad_end frames.
    """
    fft, hop = estimator.win_length, estimator.hop_length
    length = audio.shape[-1]
    frames = math.ceil((length + fft) / hop)
    left = fft // 2
    right = (frames - 1) * hop - length - left
    padded = F.pad(audio, (left, right))
    return [item[:, left : left + length] for item in estimator(padded)]


@torch.inference_mode()
def length_only_legacy(estimator, audio):
    # Keep original analysis/masks; only supply the missing output length argument.
    spectrum, magnitude = estimator.compute_stft(audio)
    masks = estimator.separator(magnitude)
    results = []
    for mask in masks:
        masked = spectrum * mask
        missing = estimator.win_length // 2 + 1 - masked.shape[1]
        masked = torch.view_as_complex(F.pad(masked, (0, 0, 0, 0, 0, missing)))
        results.append(
            torch.istft(
                masked,
                estimator.win_length,
                hop_length=estimator.hop_length,
                window=estimator.win,
                center=True,
                length=audio.shape[-1],
            )
        )
    return results


def use_four_stem_activations(estimator):
    for net in estimator.separator.instruments:
        for index in range(1, 7):
            getattr(net, f"down{index}_act")[1] = nn.ELU()
            getattr(net, f"up{index}")[2] = nn.ELU()


def record_variant(reference, outputs, names):
    return {
        name: agreement(reference[name], result.numpy()) for name, result in zip(names, outputs)
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--tf-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-multiwindow", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_model(args.checkpoint)
    config = model.config
    stems = len(config.instruments)
    if stems not in (2, 4):
        raise ValueError("this ablation supports 2 or 4 stems; legacy has no 5-stem softmax")
    legacy = load_legacy(args.legacy_root.resolve(), args.tf_checkpoint, stems)
    audio = read_audio(args.input, config.sample_rate)
    oracle = OfficialGraph(args.tf_checkpoint, config.instruments, args.upstream_root)
    report = {
        "legacy_repository": "https://github.com/fwcd/spleeter-pytorch",
        "legacy_commit": subprocess.check_output(
            ["git", "-C", str(args.legacy_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "input": args.input.name,
        "sample_rate": config.sample_rate,
        "instruments": list(config.instruments),
        "metric_note": "Reference SNR is agreement with TF estimates, NOT separation SDR.",
        "cases": {},
    }
    cases = audio_cases(audio, config, args.include_multiwindow)
    cases = {
        key: value for key, value in cases.items() if key in {"music_original", "music_multiwindow"}
    }
    references = {}
    try:
        for label, waveform in cases.items():
            reference = oracle.separate(waveform)
            references[label] = reference
            result = {}
            # Same exact spectral tensor isolates network + weight conversion.
            magnitude = stft(waveform, config, inference=True).abs()[..., : config.frequency_bins]
            magnitude = magnitude[..., : config.segment_frames, :]
            magnitude = F.pad(magnitude, (0, 0, 0, config.segment_frames - magnitude.shape[-2]))[
                None
            ]
            ours_spec = model(magnitude).numpy()
            old_spec = torch.stack(
                [net(magnitude) for net in legacy.separator.instruments], dim=1
            ).numpy()
            result["same_input_network_relative_l2"] = float(
                np.linalg.norm((old_spec - ours_spec).ravel()) / np.linalg.norm(ours_spec.ravel())
            )
            for name, outputs in (
                ("legacy_unmodified", legacy(waveform)),
                ("legacy_length_only", length_only_legacy(legacy, waveform)),
                ("legacy_aligned_padding", aligned_legacy(legacy, waveform)),
                ("spleeterpt", list(separate(model, waveform).values())),
            ):
                result[name] = record_variant(reference, outputs, config.instruments)
            report["cases"][label] = result
            print(json.dumps({"case": label, "results": result}), flush=True)
        if stems == 4:
            use_four_stem_activations(legacy)
            for label, waveform in cases.items():
                result = report["cases"][label]
                for name, outputs in (
                    ("legacy_activation_only", legacy(waveform)),
                    ("legacy_alignment_and_activation", aligned_legacy(legacy, waveform)),
                ):
                    result[name] = record_variant(references[label], outputs, config.instruments)
                print(json.dumps({"case": label, "activation_ablation": result}), flush=True)
    finally:
        oracle.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
