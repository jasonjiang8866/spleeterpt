"""Torch STFT and Spleeter's Hann overlap-add convention (not centered librosa STFT)."""

import math

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.nn import functional as F

from .config import ModelConfig


def stft(waveform: torch.Tensor, config: ModelConfig, *, inference=False):
    """[..., samples] -> [..., frames, frequencies]. Float32/float64 only."""
    if waveform.shape[-1] == 0:
        raise ValueError("audio must not be empty")
    if inference:
        waveform = F.pad(waveform, (config.n_fft, 0))
        frames = math.ceil(waveform.shape[-1] / config.hop_length)
        needed = config.n_fft + (frames - 1) * config.hop_length
        waveform = F.pad(waveform, (0, needed - waveform.shape[-1]))
    elif waveform.shape[-1] < config.n_fft:
        raise ValueError("training waveform shorter than n_fft")
    shape = waveform.shape[:-1]
    flat = waveform.reshape(-1, waveform.shape[-1])
    window = torch.hann_window(config.n_fft, dtype=flat.dtype, device=flat.device)
    spec = torch.stft(
        flat, config.n_fft, config.hop_length, window=window, center=False, return_complex=True
    ).transpose(-2, -1)
    return spec.reshape(*shape, *spec.shape[-2:])


def istft(spec: torch.Tensor, config: ModelConfig, length: int):
    """Invert inference STFT, removing its leading padding and restoring exact length."""
    shape = spec.shape[:-2]
    frames = spec.shape[-2]
    time_frames = torch.fft.irfft(spec, n=config.n_fft, dim=-1)
    window = torch.hann_window(config.n_fft, dtype=time_frames.dtype, device=spec.device)
    time_frames = (time_frames * window).reshape(-1, frames, config.n_fft).transpose(1, 2)
    total = config.n_fft + (frames - 1) * config.hop_length
    audio = F.fold(
        time_frames,
        output_size=(1, total),
        kernel_size=(1, config.n_fft),
        stride=(1, config.hop_length),
    ).reshape(*shape, total)
    return audio[..., config.n_fft : config.n_fft + length] * (2.0 / 3.0)


def ratio_masks(estimates: torch.Tensor, config: ModelConfig):
    # Compute in float32 even with AMP; epsilon and powers must not underflow in fp16.
    power = estimates.float().pow(config.separation_exponent)
    masks = (power + 1e-10 / len(config.instruments)) / (power.sum(dim=1, keepdim=True) + 1e-10)
    missing = config.n_fft // 2 + 1 - masks.shape[-1]
    if config.mask_extension == "average":
        return torch.cat(
            (masks, masks.mean(dim=-1, keepdim=True).expand(*masks.shape[:-1], missing)), dim=-1
        )
    return F.pad(masks, (0, missing))


def read_audio(path, sample_rate):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[0] == 0 or not np.isfinite(audio).all():
        raise ValueError("audio must be nonempty and finite")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if audio.shape[1] != 2:
        raise ValueError("only mono/stereo input is supported")
    if rate != sample_rate:
        divisor = math.gcd(rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, rate // divisor, axis=0)
    return torch.from_numpy(np.ascontiguousarray(audio.T))


@torch.inference_mode()
def separate(model, waveform: torch.Tensor):
    """Separate stereo audio [2, samples] in bounded-size network windows.

    The full STFT is resident on the model device; this is not streaming inference.
    """
    if waveform.ndim != 2 or waveform.shape[0] != 2:
        raise ValueError("expected stereo audio [2, samples]")
    if not torch.isfinite(waveform).all():
        raise ValueError("audio contains NaN/Inf")
    config = model.config
    device = next(model.parameters()).device
    model.eval()
    spectrum = stft(waveform.to(device=device, dtype=torch.float32), config, inference=True)
    frames = spectrum.shape[-2]
    estimated = torch.empty(
        (len(config.instruments), *spectrum.shape), device=device, dtype=spectrum.dtype
    )
    for start in range(0, frames, config.segment_frames):
        chunk = spectrum[:, start : start + config.segment_frames]
        actual = chunk.shape[-2]
        magnitude = F.pad(
            chunk.abs()[..., : config.frequency_bins], (0, 0, 0, config.segment_frames - actual)
        )
        masks = ratio_masks(model(magnitude.unsqueeze(0)), config)[0, :, :, :actual]
        estimated[:, :, start : start + actual] = masks * chunk.unsqueeze(0)
    audio = istft(estimated, config, waveform.shape[-1]).cpu()
    return dict(zip(config.instruments, audio))
