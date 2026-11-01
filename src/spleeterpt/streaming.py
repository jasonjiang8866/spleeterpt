"""Opt-in bounded-memory separation with the existing Spleeter frame grid.

No centered STFT, window crossfade, per-block resampling, or changed model context.
The file API bounds host and device working memory; the tensor iterator excludes
its caller-owned input and any output blocks retained by the caller.
"""

import math
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from .audio import ratio_masks, stft


def _separate_blocks(model, length, read):
    """read(start, stop) returns CPU stereo samples at the model sample rate."""
    if length <= 0:
        raise ValueError("audio must not be empty")
    config = model.config
    device = next(model.parameters()).device
    model.eval()
    n_fft, hop = config.n_fft, config.hop_length
    frames = math.ceil((length + n_fft) / hop)
    overlap = n_fft - hop
    tail = None
    for start in range(0, frames, config.segment_frames):
        # Scope inference_mode to computation: a context around yield would leak
        # thread-local inference mode into the consumer's unrelated operations.
        with torch.inference_mode():
            actual = min(config.segment_frames, frames - start)
            origin = start * hop
            left = origin - n_fft
            right = left + n_fft + (actual - 1) * hop
            lo, hi = max(0, left), min(length, right)
            if hi > lo:
                samples = read(lo, hi).to(device=device, dtype=torch.float32)
                if samples.shape != (2, hi - lo):
                    raise ValueError("audio source returned an incomplete stereo block")
                if not torch.isfinite(samples).all():
                    raise ValueError("audio contains NaN/Inf")
                samples = F.pad(samples, (lo - left, right - hi))
            else:
                samples = torch.zeros((2, right - left), device=device)
            spectrum = stft(samples, config)
            magnitude = F.pad(
                spectrum.abs()[..., : config.frequency_bins],
                (0, 0, 0, config.segment_frames - actual),
            )
            masks = ratio_masks(model(magnitude.unsqueeze(0)), config)[0, :, :, :actual]
            estimated = masks * spectrum.unsqueeze(0)
            time_frames = torch.fft.irfft(estimated, n=n_fft, dim=-1)
            window = torch.hann_window(n_fft, device=device, dtype=time_frames.dtype)
            time_frames = (time_frames * window).reshape(-1, actual, n_fft).transpose(1, 2)
            audio = F.fold(
                time_frames,
                output_size=(1, actual * hop + overlap),
                kernel_size=(1, n_fft),
                stride=(1, hop),
            ).reshape(len(config.instruments), 2, -1)
            if tail is not None:
                audio[..., :overlap] += tail
            # Future frames cannot affect samples before the next frame origin.
            # Clone so this tiny carry does not retain the entire block storage.
            tail = audio[..., actual * hop :].clone()
            begin = max(n_fft - origin, 0)
            end = min(actual * hop, n_fft + length - origin)
            block = None
            if end > begin:
                block = (audio[..., begin:end] * (2.0 / 3.0)).cpu()
            # Release device temporaries before yielding to a potentially slow sink.
            del samples, spectrum, magnitude, masks, estimated, time_frames, audio, window
        if block is not None:
            yield dict(zip(config.instruments, block))
    # ceil((length+n_fft)/hop)*hop covers the requested crop. The remaining
    # overlap tail is wholly outside the original audio and must be discarded.


def separate_iter(model, waveform: torch.Tensor):
    """Yield consecutive CPU float32 stereo stem blocks, without collecting them.

    Accepts CPU [samples], [1, samples], or [2, samples] already at the model
    sample rate. Mono is duplicated exactly as in read_audio(). Input storage is
    still O(duration); use separate_file() to bound end-to-end host memory too.
    Consume and release blocks instead of list()/cat() for bounded output memory.
    """
    if waveform.device.type != "cpu":
        raise ValueError("bounded-memory input must be on CPU; use separate_file for disk input")
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
        raise ValueError("expected mono/stereo audio [channels, samples]")
    if not waveform.is_floating_point():
        raise ValueError("audio must be floating point")
    if waveform.shape[0] == 1:
        waveform = waveform.expand(2, -1)
    yield from _separate_blocks(model, waveform.shape[-1], lambda lo, hi: waveform[:, lo:hi])


def separate_file(model, input_path, output_dir):
    """Separate a seekable file with bounded CPU/GPU memory into float WAV stems.

    Requires the model sample rate (no implicit full-file resampling). output_dir
    must not exist. Publish the directory only after every stem closes cleanly;
    an error removes temporary files. RF64 is used when a stem would exceed 4 GiB.
    Returns paths, never full-length waveform tensors. Calls are synchronous.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("output directory already exists; choose another directory")
    config = model.config
    with sf.SoundFile(input_path) as source:
        if source.samplerate != config.sample_rate:
            raise ValueError(
                f"bounded-memory input must be {config.sample_rate} Hz; "
                "resample to a file first or use the default separate path"
            )
        if source.channels not in (1, 2) or not source.seekable():
            raise ValueError("bounded-memory input must be seekable mono/stereo audio")
        if len(source) <= 0:
            raise ValueError("audio must not be empty")

        def read(lo, hi):
            source.seek(lo)
            samples = source.read(hi - lo, dtype="float32", always_2d=True)
            if source.channels == 1:
                samples = np.repeat(samples, 2, axis=1)
            return torch.from_numpy(np.ascontiguousarray(samples.T))

        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
            staging = Path(temp) / "stems"
            staging.mkdir()
            with ExitStack() as stack:
                writers = {
                    name: stack.enter_context(
                        sf.SoundFile(
                            staging / f"{name}.wav",
                            mode="w",
                            samplerate=config.sample_rate,
                            channels=2,
                            subtype="FLOAT",
                            format="RF64" if len(source) * 8 >= 2**32 - 4096 else "WAV",
                        )
                    )
                    for name in config.instruments
                }
                for block in _separate_blocks(model, len(source), read):
                    for name, samples in block.items():
                        writers[name].write(samples.numpy().T)
            if output.exists():
                raise FileExistsError("output directory appeared during separation")
            os.rename(staging, output)
    return {name: output / f"{name}.wav" for name in config.instruments}
