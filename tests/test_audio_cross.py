"""Check diagnostic cuts against the real native path and signed contrasts."""

import importlib
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from spleeterpt.audio import istft, separate, stft
from spleeterpt.model import Spleeter

pytest.importorskip("tensorflow")


@pytest.fixture
def cross(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("study_audio_cross")


def test_signed_interaction_distinguishes_additive_and_joint_errors(cross):
    base = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    front = np.array([[0.25, -0.5], [0.25, -0.5]])
    inverse = np.array([[0.5, 0.25], [-0.5, -0.25]])
    additive = cross.factorial_contrasts(
        base, base, base + front, base + inverse, base + front + inverse
    )
    assert additive["interaction"]["max_absolute_delta"] == 0
    assert additive["frontend_at_tf_inverse"]["max_absolute_delta"] == 0.5
    joint = cross.factorial_contrasts(
        base, base, base + front, base + inverse, base + front + inverse + 0.125
    )
    assert joint["interaction"]["rms_delta"] == 0.125
    assert joint["interaction"]["max_delta_over_reference_tolerance"] == pytest.approx(12500)


@pytest.mark.parametrize("bad", [np.full((2, 2), np.nan), np.zeros((1, 2))])
def test_contrasts_reject_invalid_arrays(cross, bad):
    base = np.zeros((2, 2))
    with pytest.raises(ValueError, match="matching finite"):
        cross.factorial_contrasts(base, base, base, base, bad)


def test_waveform_metrics_do_not_accept_truncated_audio(cross):
    with pytest.raises(ValueError, match="shape mismatch"):
        cross.waveform_metrics(np.zeros((2, 8)), np.zeros((2, 7)))


@pytest.mark.parametrize("samples", [31, 1920, 4097])
@pytest.mark.parametrize("stems", [2, 4, 5])
def test_native_audio_replay_is_bitwise_equal(cross, tiny_config, samples, stems):
    # Covers short/padded, exactly one window and multiple windows, using small
    # generated weights. The real official checkpoint is checked by the study.
    preset = tiny_config.preset(stems)
    config = replace(
        tiny_config,
        instruments=preset.instruments,
        mask_activation=preset.mask_activation,
        encoder_activation=preset.encoder_activation,
        decoder_activation=preset.decoder_activation,
    )
    torch.manual_seed(501)
    model = Spleeter(config).eval()
    waveform = torch.randn(2, samples) * 0.1
    magnitudes, stem_logits = [], {s: [] for s in config.instruments}
    with torch.inference_mode(), ExitStack() as stack:
        handle = model.register_forward_pre_hook(
            lambda _, inputs: magnitudes.append(cross.nhwc(inputs[0]).copy())
        )
        stack.callback(handle.remove)
        for s, net in model.unets.items():
            handle = net.register_forward_hook(
                lambda _, __, output, s=s: stem_logits[s].append(cross.nhwc(output).copy())
            )
            stack.callback(handle.remove)
        expected = separate(model, waveform)
    with torch.inference_mode():
        spectrum = stft(waveform, config, inference=True)
        magnitude = cross.native_magnitude(spectrum, config)
        np.testing.assert_array_equal(magnitude, np.concatenate(magnitudes))
        logits = np.stack([np.concatenate(stem_logits[s]) for s in config.instruments], -1)
        masked = cross.native_masked_spectra(logits, magnitude, spectrum, config)
        actual = istft(masked, config, samples)
        for index, stem in enumerate(config.instruments):
            assert torch.equal(actual[index], expected[stem])
