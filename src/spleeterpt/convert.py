"""Strict importer for Deezer's v1.4.0 TF1 name-based pretrained checkpoints.

TensorFlow is imported only here; converted checkpoints train/infer with torch alone.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch

from .checkpoint import FORMAT_VERSION, atomic_save
from .config import ModelConfig
from .model import Spleeter


def layer_name(base, index):
    return base if index == 0 else f"{base}_{index}"


def import_weights(reader, model):
    """Accept a TF checkpoint reader; validate every model tensor and naming layout."""
    variables = reader.get_variable_to_shape_map()
    consumed = set()

    def copy(destination, name, *, kernel=False):
        if name not in variables:
            raise ValueError(f"missing checkpoint tensor: {name}")
        value = reader.get_tensor(name)
        if kernel:
            # Conv: HWIO -> OIHW. Transpose conv: HWOI -> IOHW.
            value = value.transpose(3, 2, 0, 1)
        if tuple(value.shape) != tuple(destination.shape):
            raise ValueError(
                f"shape mismatch for {name}: {value.shape} != {tuple(destination.shape)}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite tensor: {name}")
        destination.copy_(torch.from_numpy(np.ascontiguousarray(value)))
        consumed.add(name)

    def conv(module, base, index):
        name = layer_name(base, index)
        copy(module.weight, f"{name}/kernel", kernel=True)
        copy(module.bias, f"{name}/bias")

    def norm(module, index):
        name = layer_name("batch_normalization", index)
        for target, source in (
            ("weight", "gamma"),
            ("bias", "beta"),
            ("running_mean", "moving_mean"),
            ("running_var", "moving_variance"),
        ):
            copy(getattr(module, target), f"{name}/{source}")

    with torch.no_grad():
        for stem, net in enumerate(model.unets.values()):
            for i, layer in enumerate(net.encoder):
                conv(layer, "conv2d", stem * 7 + i)
            for i, layer in enumerate(net.encoder_norm):
                norm(layer, stem * 12 + i)
            for i, layer in enumerate(net.decoder):
                conv(layer, "conv2d_transpose", stem * 6 + i)
                norm(net.decoder_norm[i], stem * 12 + 6 + i)
            conv(net.output, "conv2d", stem * 7 + 6)
    # These upstream batch norms are disconnected from the prediction graph.
    ignored = {
        f"{layer_name('batch_normalization', stem * 12 + 5)}/{field}"
        for stem in range(len(model.unets))
        for field in ("gamma", "beta", "moving_mean", "moving_variance")
    }
    # Permit training bookkeeping, but reject unexpected model scopes or extra stems.
    model_fields = {"kernel", "bias", "gamma", "beta", "moving_mean", "moving_variance"}
    unexpected = (
        {name for name in variables if name.split("/")[-1] in model_fields} - consumed - ignored
    )
    if unexpected:
        raise ValueError(
            f"unexpected model tensors (wrong preset/checkpoint layout): {sorted(unexpected)}"
        )
    return len(consumed)


def convert_checkpoint(prefix, stems, output):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "conversion requires the optional extra: pip install '.[convert]'"
        ) from exc
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    prefix = Path(prefix)
    config = ModelConfig.preset(stems)
    reader = tf.train.load_checkpoint(str(prefix))
    model = Spleeter(config)
    count = import_weights(reader, model)
    source_files = [
        Path(str(prefix) + ".index"),
        *sorted(prefix.parent.glob(prefix.name + ".data-*")),
    ]
    hashes = {}
    for file in source_files:
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[file.name] = digest.hexdigest()
    atomic_save(
        {
            "format_version": FORMAT_VERSION,
            "config": config.to_dict(),
            "model": model.state_dict(),
            "source": "deezer/spleeter TF1 checkpoint",
            "source_sha256": hashes,
        },
        output,
    )
    return count
