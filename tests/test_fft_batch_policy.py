"""Candidate preserves the model's zero context and bounds FFT allocations."""

import math
import runpy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from spleeterpt import audio, streaming
from spleeterpt.config import ModelConfig
from spleeterpt.model import Spleeter


def collect(model, waveform, policy="auto"):
    blocks = list(streaming.separate_iter(model, waveform, fft_batch_policy=policy))
    return {s: torch.cat([b[s] for b in blocks], -1) for s in model.config.instruments}


@pytest.mark.parametrize("stems", [2, 4, 5])
@pytest.mark.parametrize("remainder", [1, 2, 7, 28, 31, 32, 63, 64])
def test_long_tail_preserves_network_context(tiny_config, stems, remainder):
    preset = ModelConfig.preset(stems)
    config = replace(
        tiny_config,
        instruments=preset.instruments,
        encoder_activation=preset.encoder_activation,
        decoder_activation=preset.decoder_activation,
        mask_activation=preset.mask_activation,
    )
    torch.manual_seed(42)
    model = Spleeter(config).eval()
    length = (config.segment_frames + remainder) * config.hop_length - config.n_fft
    waveform = torch.randn(2, length) * 0.1
    reference = audio.separate(model, waveform)
    candidate = collect(model, waveform, "pad-long-tail")
    for name in config.instruments:
        assert candidate[name].shape == (2, length)
        torch.testing.assert_close(candidate[name], reference[name], atol=2e-6, rtol=0)


@pytest.mark.parametrize("stems", [2, 4, 5])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_auto_policy_selection(stems, device):
    config = ModelConfig.preset(stems)
    assert streaming._use_padded_tail("auto", config, torch.device(device)) == (
        device == "cuda" and stems == 5
    )


def test_file_jobs_reuse_model_without_sharing_tail_policy(tiny_config, monkeypatch, tmp_path):
    import soundfile as sf

    model = Spleeter(tiny_config)
    seen = []
    original = streaming.stft

    def record(samples, config):
        seen.append(1 + (samples.shape[-1] - config.n_fft) // config.hop_length)
        return original(samples, config)

    monkeypatch.setattr(streaming, "stft", record)
    window = tiny_config.segment_frames
    for job, (frames, policy, expected) in enumerate(
        [
            (window + 28, "pad-long-tail", [window, window]),
            (28, "pad-long-tail", [28]),
            (window * 2, "pad-long-tail", [window, window]),
            (window + 28, "native", [window, 28]),
            (window + 28, "auto", [window, 28]),  # CPU default
        ]
    ):
        length = frames * tiny_config.hop_length - tiny_config.n_fft
        source = tmp_path / "input.wav"
        sf.write(source, torch.ones(length).numpy(), tiny_config.sample_rate, subtype="FLOAT")
        seen.clear()
        paths = streaming.separate_file(model, source, tmp_path / str(job), fft_batch_policy=policy)
        assert seen == expected
        assert all(sf.info(path).frames == length for path in paths.values())


def test_unknown_policy_rejected(tiny_config):
    with pytest.raises(ValueError, match="unknown FFT batch policy"):
        collect(Spleeter(tiny_config), torch.zeros(2, 129), "invalid")


def test_extra_fft_frames_never_reach_network(tiny_config):
    model = Spleeter(tiny_config).eval()
    inputs = []
    hook = model.register_forward_pre_hook(lambda module, args: inputs.append(args[0].clone()))
    length = (64 + 28) * 32 - 128
    try:
        collect(model, torch.ones(2, length), "pad-long-tail")
    finally:
        hook.remove()
    assert len(inputs) == 2
    assert torch.count_nonzero(inputs[-1][..., 28:, :]) == 0
    assert torch.count_nonzero(inputs[-1][..., :28, :]) > 0


def test_fft_matrix_measures_both_policies_on_partial_second_block():
    script = Path(__file__).resolve().parents[1] / "scripts/verify_fft_batches.py"
    matrix = runpy.run_path(str(script))
    length = (512 + 7) * 1024 - 4096
    report = matrix["run"]("cpu", lengths=[length])
    (case,) = report["cases"]
    assert (case["samples"], case["frames"]) == (length, 519)
    assert set(case["errors"]) == {"native", "pad-long-tail"}
    for maxima in case["errors"].values():
        assert len(maxima) == 2
        assert all(math.isfinite(error) and error >= 0 for error in maxima)
