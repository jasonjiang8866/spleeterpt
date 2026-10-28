"""Versioned, atomic checkpoints loadable with torch weights_only=True."""

import os
import tempfile
from pathlib import Path

import torch

from .config import ModelConfig
from .model import Spleeter

FORMAT_VERSION = 1


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint version")
    return checkpoint


def load_model(path, device="cpu"):
    checkpoint = read_checkpoint(path)
    model = Spleeter(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()
