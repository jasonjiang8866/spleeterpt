import importlib
from pathlib import Path

import numpy as np


def test_per_target_improvement_and_candidate_degradation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    scoring = importlib.import_module("score_piano_holdout")
    rng = np.random.default_rng(41)
    piano, other = rng.normal(size=(2, 2048, 2))
    mix = piano + other
    refs = {"piano": piano, "other": other, "mix": mix}
    oracle = {"piano": piano + 0.1 * other, "other": other + 0.1 * piano}
    ok = scoring.assess(refs, oracle, oracle)
    assert all(row["passed"] for row in ok.values())
    assert ok["other"]["leakage"]["tf"]["status"] == "finite"
    assert scoring.score(np.zeros_like(piano), mix) is None
    failed = scoring.assess(refs, oracle, {**oracle, "other": mix})
    assert failed["piano"]["passed"]
    assert not failed["other"]["passed"]
