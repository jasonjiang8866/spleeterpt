from dataclasses import replace

import pytest
import torch

from spleeterpt.audio import istft, ratio_masks, separate, stft
from spleeterpt.config import ModelConfig
from spleeterpt.model import SameConv2d, SameConvTranspose2d, Spleeter


@pytest.mark.parametrize("stems", [2, 4, 5])
def test_trainable_stem_presets(tiny_config, stems):
    preset = ModelConfig.preset(stems)
    config = replace(
        tiny_config,
        instruments=preset.instruments,
        encoder_activation=preset.encoder_activation,
        decoder_activation=preset.decoder_activation,
        mask_activation=preset.mask_activation,
    )
    model = Spleeter(config)
    magnitude = torch.rand(2, 2, 64, 64)
    predicted = model(magnitude)
    assert predicted.shape == (2, stems, 2, 64, 64)
    assert (predicted >= 0).all()
    assert (predicted <= magnitude[:, None] + 1e-6).all()
    if stems == 5:
        torch.testing.assert_close(predicted.sum(1), magnitude)
    predicted.mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_same_convolution_is_adjoint_of_transpose():
    # A spatial shift in transpose cropping fails this inner-product identity.
    conv = SameConv2d(2, 3, 5, stride=2, bias=False).double()
    transposed = SameConvTranspose2d(3, 2).double()
    with torch.no_grad():
        transposed.weight.copy_(conv.weight)
        transposed.bias.zero_()
    x = torch.randn(1, 2, 8, 10, dtype=torch.float64)
    y = torch.randn(1, 3, 4, 5, dtype=torch.float64)
    torch.testing.assert_close((conv(x) * y).sum(), (x * transposed(y)).sum())


@pytest.mark.parametrize("length", [1, 31, 128, 2179, 8193])
def test_stft_reconstruction_and_boundaries(tiny_config, length):
    audio = torch.randn(2, length)
    reconstructed = istft(stft(audio, tiny_config, inference=True), tiny_config, length)
    torch.testing.assert_close(reconstructed, audio, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("extension", ["zeros", "average"])
def test_masks_silence_and_extension(tiny_config, extension):
    config = replace(tiny_config, mask_extension=extension)
    masks = ratio_masks(torch.zeros(1, 2, 2, 64, 64), config)
    assert torch.isfinite(masks).all()
    torch.testing.assert_close(masks[..., :64], torch.full_like(masks[..., :64], 0.5))
    assert masks.shape[-1] == 65
    expected = 0.0 if extension == "zeros" else 0.5
    torch.testing.assert_close(masks[..., 64:], torch.full_like(masks[..., 64:], expected))


def test_separator_length_silence_and_mixture_consistency(tiny_config):
    config = replace(tiny_config, mask_extension="average")
    model = Spleeter(config)
    audio = torch.randn(2, 4099)
    results = separate(model, audio)
    assert set(results) == {"vocals", "accompaniment"}
    torch.testing.assert_close(sum(results.values()), audio, atol=2e-6, rtol=1e-5)
    silent = separate(model, torch.zeros(2, 1))
    assert all(value.shape == (2, 1) and value.count_nonzero() == 0 for value in silent.values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"segment_frames": 65},
        {"frequency_bins": 256},
        {"hop_length": 16},
        {"instruments": ["vocals", "vocals"]},
        {"mask_activation": "wrong"},
    ],
)
def test_invalid_config(tiny_config, kwargs):
    with pytest.raises(ValueError):
        replace(tiny_config, **kwargs)
