"""Offline milestone pairing and a synthetic loss/reconstruction counterexample.

No weights, MUSDB audio, test split, network or paid compute are accessed.
The counterexample proves a possibility, not the cause of the measured regression.
"""

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

import torch
from evaluate_preview import si_sdr

from spleeterpt.audio import separate
from spleeterpt.config import ModelConfig
from spleeterpt.train import spectral_loss

STEPS = (90, 450, 900)
ARMS = ("lr1e5", "lr3e6")
STEMS = ("vocals", "drums", "bass", "other")


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected finite numeric metric")
    return value


def paired_change(before, after):
    """Pair by unique track ID; never subtract independently aggregated medians."""

    def index(records):
        indexed = {r["id"]: r for r in records}
        if not indexed or len(indexed) != len(records):
            raise ValueError("Empty or duplicate track IDs")
        return indexed

    first, last = index(before), index(after)
    if first.keys() != last.keys():
        raise ValueError("Track sets differ")
    values = {s: {} for s in STEMS}
    for key in sorted(first):
        if first[key]["samples"] != last[key]["samples"]:
            raise ValueError("Track lengths differ")
        for s in STEMS:
            values[s][key] = finite(last[key]["scores"][s]) - finite(first[key]["scores"][s])
    return {
        s: {
            "median_paired_delta_db": statistics.median(v.values()),
            "improved_tracks": sum(x > 0 for x in v.values()),
            "worsened_tracks": sum(x < 0 for x in v.values()),
            "unchanged_tracks": sum(x == 0 for x in v.values()),
            "per_track_delta_db": v,
        }
        for s, v in values.items()
    }


def analyze(data):
    if data["kind"] != "convergence-pilot-summary" or data["complete"] is not True:
        raise ValueError("Expected complete retained convergence summary")
    result = {
        "source_commit": data["commit"],
        "experiment": data["experiment"],
        "source_worker_status": data["worker_status"],
        "scope": "descriptive; 3 matched milestones per arm, reused 10-song validation set",
        "baseline_crop_loss_available": False,
        "same_crop_audio_quality_available": False,
        "per_track_crop_loss_available": False,
        "arms": {},
    }
    curves = data["curves"]
    for arm in ARMS:
        records = data["arms"][arm]["epochs"]
        epochs = {e["step"]: e for e in records}
        if len(epochs) != len(records):
            raise ValueError("Duplicate loss steps")
        losses = {step: finite(epochs[step]["validation_loss"]) for step in STEPS}
        if any(x < 0 for x in losses.values()):
            raise ValueError("Negative loss")
        transitions = []
        for start, end in itertools.pairwise(STEPS):
            quality = paired_change(
                curves[f"{arm}_step{start}"]["tracks"],
                curves[f"{arm}_step{end}"]["tracks"],
            )
            transitions.append(
                {
                    "start_step": start,
                    "end_step": end,
                    "validation_loss_delta": losses[end] - losses[start],
                    "si_sdr": quality,
                    "lower_loss_but_worse_median_stems": [
                        s
                        for s in STEMS
                        if losses[end] < losses[start] and quality[s]["median_paired_delta_db"] < 0
                    ],
                }
            )
        result["arms"][arm] = {
            "milestones": [
                {
                    "step": step,
                    "validation_loss": losses[step],
                    "si_sdr_vs_official": paired_change(
                        curves["official_converted"]["tracks"],
                        curves[f"{arm}_step{step}"]["tracks"],
                    ),
                }
                for step in STEPS
            ],
            "transitions": transitions,
            "step90_to900_validation_loss_relative_change": (
                (losses[900] - losses[90]) / losses[90] if losses[90] else None
            ),
        }
    return result


class FixedMagnitudes(torch.nn.Module):
    """Hand-constructed bounded magnitude estimates, not trained network weights."""

    def __init__(self, config, fractions, *, square_root):
        super().__init__()
        self.config = config
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.register_buffer("gains", fractions.sqrt() if square_root else fractions)

    def forward(self, magnitude):
        return magnitude.unsqueeze(1) * self.gains[None, :, None, None, :]


@torch.inference_mode()
def counterexample():
    """Use production spectral_loss AND separate, with two overlapping low-band tones."""
    config = ModelConfig(
        instruments=STEMS,
        filters=(2, 3, 4, 5, 6, 7),
        sample_rate=8000,
        n_fft=256,
        hop_length=64,
        segment_frames=64,
        frequency_bins=128,
    )
    t = torch.arange(config.segment_samples, dtype=torch.float64)
    tones = torch.stack([torch.sin(2 * math.pi * k * t / config.n_fft) for k in (16, 64)])
    coefficients = torch.tensor([[0.55, 0.10], [0.25, 0.20], [0.15, 0.30], [0.05, 0.40]])
    targets = (coefficients.double() @ tones).float()[:, None, :].repeat(1, 2, 1)
    mixture = targets.sum(0)
    bins = torch.arange(config.frequency_bins)
    fractions = torch.where(bins[None, :] <= 40, coefficients[:, :1], coefficients[:, 1:])
    result = {
        "scope": "synthetic possibility proof only; not the measured MUSDB cause",
        "config": config.to_dict(),
        "tone_bins": [16, 64],
        "source_coefficients": coefficients.tolist(),
        "candidates": {},
    }
    for name, root in (("ratio_aligned", True), ("magnitude_aligned", False)):
        model = FixedMagnitudes(config, fractions, square_root=root)
        loss = spectral_loss(
            model,
            mixture[None],
            targets[None],
            config,
            precision="fp32",
            device=torch.device("cpu"),
        ).item()
        output = separate(model, mixture)
        result["candidates"][name] = {
            "summed_magnitude_l1": loss,
            "global_stereo_si_sdr_db": {
                s: si_sdr(targets[i].numpy(), output[s].numpy()) for i, s in enumerate(STEMS)
            },
            "waveform_mae": {
                s: (targets[i] - output[s]).abs().mean().item() for i, s in enumerate(STEMS)
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    raw = args.summary.read_bytes()
    result = analyze(json.loads(raw))
    result["source_sha256"] = hashlib.sha256(raw).hexdigest()
    result["torch_version"] = torch.__version__
    result["synthetic_counterexample"] = counterexample()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
