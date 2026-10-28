"""Explicit, checkpoint-serializable model and audio settings."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    instruments: tuple[str, ...] = ("vocals", "accompaniment")
    filters: tuple[int, ...] = (16, 32, 64, 128, 256, 512)
    sample_rate: int = 44100
    n_fft: int = 4096
    hop_length: int = 1024
    segment_frames: int = 512
    frequency_bins: int = 1024
    encoder_activation: str = "leaky_relu"
    decoder_activation: str = "relu"
    mask_activation: str = "sigmoid"
    mask_extension: str = "zeros"
    separation_exponent: float = 2.0

    def __post_init__(self):
        object.__setattr__(self, "instruments", tuple(self.instruments))
        object.__setattr__(self, "filters", tuple(self.filters))
        if len(self.filters) != 6 or any(n < 1 for n in self.filters):
            raise ValueError("filters must contain six positive channel counts")
        if not self.instruments or len(set(self.instruments)) != len(self.instruments):
            raise ValueError("instruments must be nonempty and unique")
        if any(not name.isidentifier() for name in self.instruments):
            raise ValueError("instrument names must be Python identifiers")
        if self.sample_rate <= 0 or self.n_fft < 4 or self.n_fft % 4:
            raise ValueError("positive sample_rate and n_fft divisible by four required")
        if self.hop_length != self.n_fft // 4:
            raise ValueError("Spleeter Hann synthesis requires hop_length == n_fft / 4")
        if self.segment_frames < 64 or self.segment_frames % 64:
            raise ValueError("segment_frames must be a positive multiple of 64")
        if self.frequency_bins < 64 or self.frequency_bins % 64:
            raise ValueError("frequency_bins must be a positive multiple of 64")
        if self.frequency_bins > self.n_fft // 2 + 1:
            raise ValueError("frequency_bins exceeds the one-sided FFT size")
        if self.encoder_activation not in {"leaky_relu", "relu", "elu"}:
            raise ValueError("invalid encoder activation")
        if self.decoder_activation not in {"leaky_relu", "relu", "elu"}:
            raise ValueError("invalid decoder activation")
        if self.mask_activation not in {"sigmoid", "softmax"}:
            raise ValueError("invalid mask activation")
        if self.mask_extension not in {"zeros", "average"}:
            raise ValueError("invalid mask extension")
        if self.separation_exponent <= 0:
            raise ValueError("separation_exponent must be positive")

    @property
    def segment_samples(self):
        # A center=False training STFT produces exactly segment_frames frames.
        return self.n_fft + (self.segment_frames - 1) * self.hop_length

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def preset(cls, stems: int):
        if stems == 2:
            return cls()
        if stems == 4:
            return cls(
                instruments=("vocals", "drums", "bass", "other"),
                encoder_activation="elu",
                decoder_activation="elu",
            )
        if stems == 5:
            return cls(
                instruments=("vocals", "piano", "drums", "bass", "other"),
                encoder_activation="elu",
                decoder_activation="elu",
                mask_activation="softmax",
            )
        raise ValueError("supported presets: 2, 4, 5")
