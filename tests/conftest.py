import json

import numpy as np
import pytest
import soundfile as sf
import torch

from spleeterpt.config import ModelConfig


@pytest.fixture(autouse=True)
def small_thread_pool():
    torch.set_num_threads(2)


@pytest.fixture
def tiny_config():
    return ModelConfig(
        filters=(2, 3, 4, 5, 6, 7),
        sample_rate=8000,
        n_fft=128,
        hop_length=32,
        segment_frames=64,
        frequency_bins=64,
    )


@pytest.fixture
def audio_dataset(tmp_path, tiny_config):
    rng = np.random.default_rng(12)
    tracks = []
    for i in range(3):
        track = tmp_path / f"track-{i}"
        track.mkdir()
        sources = rng.normal(0, 0.02, (4, tiny_config.segment_samples * 2, 2)).astype("float32")
        for name, audio in zip(("vocals", "drums", "bass", "other"), sources):
            sf.write(track / f"{name}.wav", audio, tiny_config.sample_rate, subtype="FLOAT")
        sf.write(
            track / "mixture.wav", sources.sum(axis=0), tiny_config.sample_rate, subtype="FLOAT"
        )
        tracks.append(track)
    train, val = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    train.write_text("".join(json.dumps({"track": str(p)}) + "\n" for p in tracks[:2]))
    val.write_text(json.dumps({"track": str(tracks[2])}) + "\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(tiny_config.to_dict()))
    return config, train, val, tracks
