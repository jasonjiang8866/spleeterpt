import copy
import importlib
import json
from pathlib import Path

import pytest


def test_holdout_identity_and_incomplete_piano_coverage(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(root))
    validate = importlib.import_module("prepare_five_stem_corpus").validate
    manifest = json.loads((root / "five_stem_corpus.json").read_text())
    assert validate(manifest) is False
    with pytest.raises(ValueError, match="piano coverage"):
        validate(manifest, require_piano_coverage=True)
    duplicate = copy.deepcopy(manifest)
    duplicate["records"].append(dict(duplicate["records"][0], split="holdout"))
    with pytest.raises(ValueError, match="leakage"):
        validate(duplicate)
    observed = copy.deepcopy(manifest)
    observed["records"][1]["inference_observed"] = True
    with pytest.raises(ValueError, match="observed"):
        validate(observed)
