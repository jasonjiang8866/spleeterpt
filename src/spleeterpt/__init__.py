"""Trainable PyTorch port of Spleeter. No bundled pretrained weights."""

from .config import ModelConfig
from .model import Spleeter

__all__ = ["ModelConfig", "Spleeter"]
