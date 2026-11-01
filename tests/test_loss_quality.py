"""Guard the pairing semantics and exercise the actual loss/synthesis mismatch."""

import copy
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("diagnose_loss_quality")


def tracks(values):
    return [
        {
            "id": str(i),
            "samples": 100,
            "scores": dict.fromkeys(("vocals", "drums", "bass", "other"), v),
        }
        for i, v in enumerate(values)
    ]


def test_pairs_ids_before_medians(diagnostic):
    before = tracks([0, 10, 11])
    after = tracks([9, 8, 100])[::-1]
    result = diagnostic.paired_change(before, after)["vocals"]
    # median([9, -2, 89]) == 9; median(after)-median(before) == -1.
    assert result["median_paired_delta_db"] == 9
    assert result["improved_tracks"] == 2
    assert result["worsened_tracks"] == 1


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "length", "nan"])
def test_rejects_unpairable_scores(diagnostic, mutation):
    before, after = tracks([1, 2]), tracks([2, 3])
    if mutation == "duplicate":
        after.append(copy.deepcopy(after[0]))
    elif mutation == "missing":
        after.pop()
    elif mutation == "length":
        after[0]["samples"] = 99
    else:
        after[0]["scores"]["vocals"] = float("nan")
    with pytest.raises(ValueError):
        diagnostic.paired_change(before, after)


def test_real_summary_milestones_and_missing_measurements(diagnostic):
    source = json.loads((ROOT / "docs/results/convergence-pilot-41290811204.json").read_text())
    report = diagnostic.analyze(source)
    assert report["source_worker_status"] == "failed"
    assert not report["baseline_crop_loss_available"]
    assert not report["same_crop_audio_quality_available"]
    for arm in report["arms"].values():
        assert [m["step"] for m in arm["milestones"]] == [90, 450, 900]
        for transition in arm["transitions"]:
            for stem in transition["si_sdr"].values():
                assert len(stem["per_track_delta_db"]) == 10


def test_lower_production_loss_can_worsen_production_audio(diagnostic):
    candidates = diagnostic.counterexample()["candidates"]
    a, b = candidates["ratio_aligned"], candidates["magnitude_aligned"]
    assert b["summed_magnitude_l1"] < a["summed_magnitude_l1"] * 0.001
    for stem in a["global_stereo_si_sdr_db"]:
        assert b["global_stereo_si_sdr_db"][stem] < a["global_stereo_si_sdr_db"][stem] - 10
        assert b["waveform_mae"][stem] > a["waveform_mae"][stem] * 100
