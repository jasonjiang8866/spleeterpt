"""Process-local experimental FFT policy; never installed as a production default.

Only use in single-threaded measurement processes. The context temporarily replaces
streaming.stft, leaving the full-file reference frontend untouched.
"""

import math
from contextlib import contextmanager

from torch.nn import functional as F

from spleeterpt import streaming


@contextmanager
def fft_batch_policy(policy, config, length):
    if policy not in ("native", "pad-long-tail"):
        raise ValueError("unknown FFT batch policy")
    if length <= 0:
        raise ValueError("audio must not be empty")
    original = streaming.stft
    total = math.ceil((length + config.n_fft) / config.hop_length)
    if policy == "native" or total <= config.segment_frames:
        yield
        return

    def padded(samples, block_config, *, inference=False):
        if inference or block_config != config:
            raise ValueError("experimental policy requires the bounded frame grid")
        delta = samples.shape[-1] - config.n_fft
        if delta < 0 or delta % config.hop_length:
            raise ValueError("incomplete FFT frame grid")
        actual = 1 + delta // config.hop_length
        if not 1 <= actual <= config.segment_frames:
            raise ValueError("FFT block exceeds model window")
        if actual == config.segment_frames:
            return original(samples, config)
        extended = F.pad(samples, (0, (config.segment_frames - actual) * config.hop_length))
        # Crop BEFORE magnitude/model padding: extra FFT frames may contain the
        # overlapping signal tail and must never become extra network context.
        return original(extended, config)[..., :actual, :]

    streaming.stft = padded
    try:
        yield
    finally:
        streaming.stft = original
