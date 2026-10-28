"""Aligned WAV stems, deterministic crops, and explicit train/validation manifests."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .config import ModelConfig


def read_manifest(path):
    """JSONL records: {\"track\": \"path/to/track-directory\"}; paths relative to manifest."""
    path = Path(path).resolve()
    tracks = []
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            tracks.append((path.parent / record["track"]).resolve())
    if not tracks or len(set(tracks)) != len(tracks):
        raise ValueError("manifest must contain nonempty, unique track paths")
    return tracks


def manifest_fingerprint(paths):
    # Include paths and audio metadata so resume cannot silently change the dataset.
    records = []
    for path in paths:
        files = [
            (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(path.glob("*.wav"))
        ]
        records.append((str(path), files))
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def make_manifests(root, output, validation_tracks, seed=0):
    """Split by track, never by audio chunk. Caller supplies only the training pool."""
    root, output = Path(root).resolve(), Path(output).resolve()
    tracks = sorted(p.parent for p in root.glob("*/mixture.wav"))
    if not 1 <= validation_tracks < len(tracks):
        raise ValueError("need at least one training and one validation track")
    random.Random(seed).shuffle(tracks)
    output.mkdir(parents=True, exist_ok=True)
    destinations = [output / "train.jsonl", output / "validation.jsonl"]
    if any(p.exists() for p in destinations):
        raise FileExistsError("manifest exists; use a new output directory")
    for dest, subset in zip(destinations, (tracks[validation_tracks:], tracks[:validation_tracks])):
        dest.write_text("".join(json.dumps({"track": str(p)}) + "\n" for p in subset))
    return destinations


class StemDataset(Dataset):
    def __init__(self, manifest, config: ModelConfig, *, training, samples_per_track=1, seed=0):
        self.tracks = read_manifest(manifest)
        self.config = config
        self.training = training
        self.samples_per_track = samples_per_track
        self.seed = seed
        self.epoch = 0
        if samples_per_track < 1:
            raise ValueError("samples_per_track must be positive")
        self.records = []
        for track in self.tracks:
            mixture = track / "mixture.wav"
            targets = []
            for name in config.instruments:
                target = track / f"{name}.wav"
                if name == "accompaniment" and not target.exists():
                    # MUSDB four-stem layout. Sum actual stems; do not use mix - vocals.
                    targets.append([track / f"{s}.wav" for s in ("drums", "bass", "other")])
                else:
                    targets.append([target])
            info = sf.info(mixture)
            if info.frames < 1:
                raise ValueError(f"empty audio: {mixture}")
            for file in [mixture] + [f for group in targets for f in group]:
                item = sf.info(file)
                if (item.samplerate, item.channels, item.frames) != (
                    config.sample_rate,
                    2,
                    info.frames,
                ):
                    raise ValueError(
                        f"unaligned audio or wrong sample rate/channels: {file}; "
                        f"expected {config.sample_rate} Hz stereo, {info.frames} samples"
                    )
            self.records.append((mixture, targets, info.frames))

    def __len__(self):
        return len(self.records) * self.samples_per_track

    def __getitem__(self, index):
        track_index, crop_index = divmod(index, self.samples_per_track)
        mixture, targets, frames = self.records[track_index]
        length = self.config.segment_samples
        latest = max(frames - length, 0)
        if self.training:
            rng = random.Random(f"{self.seed}/{self.epoch}/{index}")
            start = rng.randint(0, latest)
        else:
            # Fixed, evenly spaced validation crops, independent of epoch/workers.
            start = round(latest * (crop_index + 0.5) / self.samples_per_track)

        def read(file):
            audio, _ = sf.read(file, start=start, frames=length, dtype="float32", always_2d=True)
            if not np.isfinite(audio).all():
                raise ValueError(f"nonfinite samples in {file}")
            audio = np.pad(audio, ((0, length - len(audio)), (0, 0)))
            return torch.from_numpy(np.ascontiguousarray(audio.T))

        return read(mixture), torch.stack([sum(read(f) for f in group) for group in targets])
