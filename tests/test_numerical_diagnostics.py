"""Validate diagnostic gates, tensor layouts and the independent FP64 topology."""

import importlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from spleeterpt.model import Spleeter

tf = pytest.importorskip("tensorflow")


@pytest.fixture
def diagnostics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("diagnose_numerics")


def test_diagnostic_counts_strict_sample_failures(diagnostics):
    reference = np.array([0.0, 1.0, -2.0, 0.0])
    candidate = reference + np.array([2e-5, 2e-4, 1e-4, 0.0])
    result = diagnostics.metrics(reference, candidate)
    assert result["strict_failures"] == 2
    assert result["elements"] == 4
    assert result["max_tolerance_ratio"] == pytest.approx(2.0)
    assert diagnostics.metrics(reference, reference)["strict_failures"] == 0
    with pytest.raises(ValueError, match="shape mismatch"):
        diagnostics.metrics(reference, candidate[:3])


def test_native_softmax_uses_actual_five_stem_layout(diagnostics):
    # Distinct channel/time/frequency sizes catch incorrect transpose/softmax axes.
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(2, 3, 7, 2, 5)).astype(np.float32)
    logits += np.float32(1000)
    result = diagnostics.native_softmax(logits)
    np.testing.assert_allclose(result.sum(-1), 1, atol=3e-7, rtol=0)
    np.testing.assert_allclose(result, tf.nn.softmax(logits, -1).numpy(), atol=3e-7, rtol=0)


def test_fp64_reference_reconstructs_five_unets(diagnostics, tiny_config):
    config = replace(
        tiny_config,
        instruments=("vocals", "piano", "drums", "bass", "other"),
        encoder_activation="elu",
        decoder_activation="elu",
        mask_activation="softmax",
    )
    torch.manual_seed(432)
    model = Spleeter(config).double().eval()
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.running_mean.uniform_(-1, 1)
                layer.running_var.uniform_(0.2, 2)
                layer.weight.uniform_(0.5, 1.5)
                layer.bias.uniform_(-1, 1)
        magnitude = np.random.default_rng(432).uniform(0, 50, (1, 64, 64, 2))
        expected = diagnostics.tf64_logits(model, magnitude)
        actual = np.stack(
            [diagnostics.nhwc(net(diagnostics.nchw(magnitude))) for net in model.unets.values()], -1
        )
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
