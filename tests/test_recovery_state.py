"""Recovery hashes compare tensor state rather than serialization/container bytes."""

import importlib.util
import sys
from pathlib import Path

import torch


def test_recovery_hash_preserves_tensor_bytes_shape_type_and_state(tmp_path, monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "runpod"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("recovery_hash_test", folder / "recovery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = {
        "model": {"w": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        "step": 3,
        "rng": torch.tensor([1, 2], dtype=torch.uint8),
    }
    p = tmp_path / "checkpoint.pt"
    torch.save(value, p)
    loaded = torch.load(p, weights_only=True)
    assert module.state_hash(value) == module.state_hash(loaded)
    assert module.state_hash(value) == module.state_hash(dict(reversed(list(loaded.items()))))
    loaded["model"]["w"][0, 0] += 1
    assert module.state_hash(value) != module.state_hash(loaded)
    tensor = value["model"]["w"]
    assert module.state_hash(tensor) != module.state_hash(tensor.reshape(4))
    assert module.state_hash(tensor) != module.state_hash(tensor.double())
    assert module.state_hash([1, 2]) != module.state_hash((1, 2))
    # Imported helper modules are ordinary repo code; don't retain test module aliases.
    sys.modules.pop("recovery_hash_test", None)
