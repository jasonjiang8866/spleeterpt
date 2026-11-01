from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
import torch

from spleeterpt.audio import read_audio, separate
from spleeterpt.checkpoint import FORMAT_VERSION, atomic_save
from spleeterpt.cli import main
from spleeterpt.config import ModelConfig
from spleeterpt.model import Spleeter
from spleeterpt.streaming import separate_file, separate_iter


def collect(model, audio):
    blocks = list(separate_iter(model, audio))
    assert all(
        0
        < next(iter(b.values())).shape[-1]
        <= model.config.segment_frames * model.config.hop_length
        for b in blocks
    )
    return {name: torch.cat([b[name] for b in blocks], dim=-1) for name in model.config.instruments}


@pytest.mark.parametrize("stems", [2, 4, 5])
@pytest.mark.parametrize("extension", ["zeros", "average"])
@pytest.mark.parametrize(
    "length",
    [1, 31, 32, 33, 127, 128, 129, 1919, 1920, 1921, 2047, 2048, 2049, 3967, 3968, 3969, 20609],
)
def test_matches_full_path(tiny_config, stems, extension, length):
    preset = ModelConfig.preset(stems)
    config = replace(
        preset,
        filters=tiny_config.filters,
        sample_rate=8000,
        n_fft=128,
        hop_length=32,
        segment_frames=64,
        frequency_bins=64,
        mask_extension=extension,
    )
    torch.manual_seed(41)
    model = Spleeter(config)
    audio = torch.randn(2, length) * 0.1
    # Explicit impulses beside model-window joins expose a shifted/missing carry.
    if length > 1921:
        audio[:, 1919:1922] = 0.8
    expected = separate(model, audio)
    actual = collect(model, audio)
    for name in expected:
        assert actual[name].shape == (2, length)
        torch.testing.assert_close(actual[name], expected[name], atol=2e-6, rtol=1e-5)
    if extension == "average":
        torch.testing.assert_close(sum(actual.values()), audio, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("stems", [2, 4, 5])
@pytest.mark.parametrize("length", [1, 1920, 1921, 4099])
def test_silence(tiny_config, stems, length):
    config = replace(tiny_config, instruments=ModelConfig.preset(stems).instruments)
    result = collect(Spleeter(config), torch.zeros(2, length))
    assert all(x.shape == (2, length) and x.count_nonzero() == 0 for x in result.values())


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("format", ["WAV", "FLAC"])
def test_file_and_cli(tiny_config, tmp_path, channels, format):
    model = Spleeter(tiny_config)
    source = tmp_path / f"input.{format.lower()}"
    sf.write(
        source,
        np.random.default_rng(1).normal(0, 0.1, (8193, channels)),
        8000,
        format=format,
        subtype="PCM_16",
    )
    expected = separate(model, read_audio(source, 8000))
    paths = separate_file(model, source, tmp_path / "out")
    for name, path in paths.items():
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        assert rate == 8000 and sf.info(path).subtype == "FLOAT"
        torch.testing.assert_close(torch.from_numpy(audio.T), expected[name], atol=2e-6, rtol=1e-5)
    with pytest.raises(FileExistsError):
        separate_file(model, source, tmp_path / "out")
    checkpoint = tmp_path / "model.pt"
    atomic_save(
        {
            "format_version": FORMAT_VERSION,
            "config": tiny_config.to_dict(),
            "model": model.state_dict(),
        },
        checkpoint,
    )
    main(
        [
            "separate",
            str(source),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "cli"),
            "--bounded-memory",
        ]
    )
    # Float WAV PEAK chunks may contain wall-clock timestamps; compare audio.
    np.testing.assert_array_equal(
        sf.read(tmp_path / "cli" / "vocals.wav")[0], sf.read(paths["vocals"])[0]
    )


def test_mono_iterator_and_inference_context(tiny_config):
    model = Spleeter(tiny_config)
    audio = torch.randn(4099)
    expected = separate(model, audio.expand(2, -1))
    for mono in (audio, audio[None]):
        actual = collect(model, mono)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=2e-6, rtol=1e-5)
    iterator = separate_iter(model, audio)
    next(iterator)
    assert not torch.is_inference_mode_enabled()
    (torch.ones(1, requires_grad=True) * 2).sum().backward()
    iterator.close()


@pytest.mark.parametrize(
    "audio",
    [
        torch.zeros(2, 0),
        torch.zeros(3, 3),
        torch.zeros(1, 2, 3),
        torch.ones(2, 3, dtype=torch.int16),
        torch.full((2, 3), float("nan")),
        torch.full((2, 3), float("inf")),
    ],
)
def test_invalid_tensor(tiny_config, audio):
    with pytest.raises(ValueError):
        list(separate_iter(Spleeter(tiny_config), audio))


def test_file_validation_and_cleanup(tiny_config, tmp_path):
    model = Spleeter(tiny_config)
    source = tmp_path / "input.wav"
    output = tmp_path / "output"
    sf.write(source, np.zeros((100, 2)), 44100)
    with pytest.raises(ValueError, match="8000 Hz"):
        separate_file(model, source, output)
    sf.write(source, np.empty((0, 2)), 8000)
    with pytest.raises(ValueError, match="empty"):
        separate_file(model, source, output)
    sf.write(source, np.zeros((100, 3)), 8000)
    with pytest.raises(ValueError, match="mono/stereo"):
        separate_file(model, source, output)
    # Fail after at least one block was written, not just on preflight.
    audio = np.zeros((8193, 2), dtype="float32")
    audio[-1] = np.nan
    sf.write(source, audio, 8000, subtype="FLOAT")
    with pytest.raises(ValueError, match="NaN/Inf"):
        separate_file(model, source, output)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [source]


def test_file_read_and_fft_sizes_are_bounded(tiny_config, tmp_path, monkeypatch):
    from spleeterpt import streaming

    source = tmp_path / "long.wav"
    sf.write(source, np.zeros((tiny_config.segment_samples * 30 + 1, 2)), 8000)
    reads, fft_samples = [], []
    original_read, original_stft = sf.SoundFile.read, streaming.stft

    def read(self, frames=-1, *args, **kwargs):
        reads.append(frames)
        return original_read(self, frames, *args, **kwargs)

    def stft(audio, config):
        fft_samples.append(audio.shape[-1])
        return original_stft(audio, config)

    monkeypatch.setattr(sf.SoundFile, "read", read)
    monkeypatch.setattr(streaming, "stft", stft)
    separate_file(Spleeter(tiny_config), source, tmp_path / "out")
    assert len(reads) > 20 and len(fft_samples) == len(reads)
    assert 0 < min(reads) <= max(reads) <= tiny_config.segment_samples
    assert max(fft_samples) == tiny_config.segment_samples


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
@pytest.mark.parametrize("stems", [2, 4, 5])
def test_cuda_parity_and_input_guard(tiny_config, stems):
    preset = ModelConfig.preset(stems)
    config = replace(
        tiny_config,
        instruments=preset.instruments,
        encoder_activation=preset.encoder_activation,
        decoder_activation=preset.decoder_activation,
        mask_activation=preset.mask_activation,
    )
    model = Spleeter(config).cuda()
    audio = torch.randn(2, 20609)
    expected, actual = separate(model, audio), collect(model, audio)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=2e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="CPU"):
        next(separate_iter(model, audio.cuda()))
