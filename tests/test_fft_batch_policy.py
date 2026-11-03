"""Candidate preserves the model's zero context and bounds FFT allocations."""

import importlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from spleeterpt import audio, streaming
from spleeterpt.config import ModelConfig
from spleeterpt.model import Spleeter


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("fft_batch_policy").fft_batch_policy


def collect(model, waveform):
    blocks = list(streaming.separate_iter(model, waveform))
    return {s: torch.cat([b[s] for b in blocks], -1) for s in model.config.instruments}


@pytest.mark.parametrize("stems", [2, 4, 5])
@pytest.mark.parametrize("remainder", [1, 2, 7, 28, 31, 32, 63, 64])
def test_long_tail_preserves_network_context(policy, tiny_config, stems, remainder):
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
    original = streaming.stft
    with policy("pad-long-tail", config, length):
        candidate = collect(model, waveform)
    assert streaming.stft is original
    for name in config.instruments:
        assert candidate[name].shape == (2, length)
        torch.testing.assert_close(candidate[name], reference[name], atol=2e-6, rtol=0)


@pytest.mark.parametrize("length", [1, 31, 32, 33, 127, 128, 129, 1919, 1920])
def test_single_window_and_native_unchanged(policy, tiny_config, length):
    original = streaming.stft
    for name in ("native", "pad-long-tail"):
        with policy(name, tiny_config, length):
            assert streaming.stft is original


def test_padding_crops_before_model_and_restores_on_error(policy, tiny_config, monkeypatch):
    seen = []
    original = streaming.stft

    def record(samples, config):
        seen.append(samples.shape[-1])
        return original(samples, config)

    monkeypatch.setattr(streaming, "stft", record)
    samples = torch.randn(2, tiny_config.n_fft + 27 * tiny_config.hop_length)
    expected = original(samples, tiny_config)
    with pytest.raises(RuntimeError, match="consumer"), policy("pad-long-tail", tiny_config, 8000):
        actual = streaming.stft(samples, tiny_config)
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        assert seen == [tiny_config.n_fft + 63 * tiny_config.hop_length]
        raise RuntimeError("consumer")
    assert streaming.stft is record


def test_invalid_grid_rejected(policy, tiny_config):
    with policy("pad-long-tail", tiny_config, 8000), pytest.raises(ValueError, match="grid"):
        streaming.stft(torch.zeros(2, 129), tiny_config)


def test_extra_fft_frames_never_reach_network(policy, tiny_config):
    model = Spleeter(tiny_config).eval()
    inputs = []
    hook = model.register_forward_pre_hook(lambda module, args: inputs.append(args[0].clone()))
    length = (64 + 28) * 32 - 128
    try:
        with policy("pad-long-tail", tiny_config, length):
            collect(model, torch.ones(2, length))
    finally:
        hook.remove()
    assert len(inputs) == 2
    assert torch.count_nonzero(inputs[-1][..., 28:, :]) == 0
    assert torch.count_nonzero(inputs[-1][..., :28, :]) > 0
