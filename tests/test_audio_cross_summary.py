"""Reject incomplete or incomparable replicate evidence, even if rows pass."""

import copy
import importlib
import itertools
import json
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def summary(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("summarize_audio_cross").summarize


@pytest.fixture
def reports():
    row = {
        "strict_failed_samples": 0,
        "max_absolute_error": 0.0,
        "max_tolerance_ratio": 0.0,
        "allclose_atol_1e-5_rtol_1e-4": True,
        "output_sha256": "same",
    }
    cells = {
        f"magnitude_{a}__postlogit_{b}__carrier_{c}__inverse_{d}": {"piano": dict(row)}
        for a, b, c, d in itertools.product(("tf", "pt"), repeat=4)
    }
    case = {
        "decoded_pcm_sha256": "pcm",
        "samples": 8,
        "frames": 1,
        "windows": 1,
        "last_window_frames": 1,
        "attribution_valid": True,
        "cells": cells,
        "controls": {
            k: True
            for k in (
                "official_masked_refeed_exact",
                "native_hybrid_replay_exact",
                "tf_post_refeed_tf_tf_exact",
                "tf_post_refeed_tf_pt_exact",
                "tf_post_refeed_pt_tf_exact",
                "tf_post_refeed_pt_pt_exact",
            )
        },
    }
    report = {
        k: "fixed"
        for k in (
            "schema_version",
            "source_commit",
            "code_sha256",
            "upstream_commit",
            "checkpoint_sha256",
            "tf_checkpoint_sha256",
            "audio_sha256",
            "thresholds",
            "fixed_network",
            "primary_factors",
            "postlogit_scope",
            "bridge_factors",
        )
    }
    report.update(
        config={"instruments": ["piano"]},
        holdouts_used=False,
        release_eligible=False,
        environment={"cpu": "test"},
        cases={k: copy.deepcopy(case) for k in ("music_original", "music_multiwindow")},
    )
    return [dict(copy.deepcopy(report), replicate=str(i)) for i in range(3)]


def test_one_pass_cannot_hide_repeat_failure(summary, reports):
    cell = next(iter(reports[1]["cases"]["music_original"]["cells"].values()))
    cell["piano"].update(
        {
            "strict_failed_samples": 2,
            "allclose_atol_1e-5_rtol_1e-4": False,
            "output_sha256": "other",
        }
    )
    result = summary(reports)
    row = result["rows"][0]
    assert row["strict_failed_samples"] == [0, 2, 0]
    assert not row["all_replicates_passed"] and not row["bitwise_repeatable"]
    assert not result["release_eligible"]


@pytest.mark.parametrize(
    "field", ["source_commit", "code_sha256", "thresholds", "checkpoint_sha256"]
)
def test_provenance_changes_are_not_pooled(summary, reports, field):
    reports[1][field] = "changed"
    with pytest.raises(ValueError, match="provenance mismatch"):
        summary(reports)


def test_missing_cells_in_every_repeat_are_rejected(summary, reports):
    for report in reports:
        report["cases"]["music_original"]["cells"].popitem()
    with pytest.raises(ValueError, match="incomplete cross cells"):
        summary(reports)


def test_summary_checks_actual_replay_controls(summary, reports):
    reports[0]["cases"]["music_original"]["controls"]["native_hybrid_replay_exact"] = False
    with pytest.raises(ValueError, match="replay controls invalid"):
        summary(reports)


def test_same_process_label_cannot_count_as_replication(summary, reports):
    with pytest.raises(ValueError, match="three distinct"):
        summary([reports[0]] * 3)


def test_environment_difference_remains_explicit(summary, reports):
    reports[1]["environment"] = {"cpu": "different"}
    assert not summary(reports)["identical_recorded_environment"]


def test_magnitude_archive_preserves_repeated_drums_failure(summary):
    archive = Path(__file__).resolve().parents[1] / "docs/results/magnitude-20261109/evidence.zip"
    with zipfile.ZipFile(archive) as zipped:
        reports = [json.loads(zipped.read(f"cpu-{i}.json")) for i in (1, 2, 3)]
    result = summary(reports)
    assert len(result["rows"]) == 80
    assert result["identical_recorded_environment"]
    assert all(row["bitwise_repeatable"] for row in result["rows"])
    failures = [row for row in result["rows"] if not row["all_replicates_passed"]]
    assert {row["cell"] for row in failures} == {
        "fft_pt__hann_tf__abs_tf",
        "fft_pt__hann_tf__abs_pt",
    }
    assert all(row["case"] == "music_multiwindow" and row["stem"] == "drums" for row in failures)
    assert all(row["strict_failed_samples"] == [1, 1, 1] for row in failures)
    assert not result["release_eligible"]


def test_fft_head_rejects_duplicate_coordinates_even_when_count_matches(summary, reports):
    for report in reports:
        report["primary_factors"] = "analysis RFFT x output head on widened D6"
        case = report["cases"]["music_original"]
        row = next(iter(case["cells"].values()))["piano"]
        row.update(failed_channel_sample=[[0, 4], [0, 4]], strict_failed_samples=2)
        implementations = ("pt32", "tf32", "pt64_cast")
        case["cells"] = {
            f"fft_{a}__head_{b}": {"piano": row}
            for a, b in itertools.product(implementations, repeat=2)
        }
        case["controls"] = dict.fromkeys(
            (
                "tf_head_refeed_exact",
                "native_waveform_replay_exact",
                "pt_rfft_matches_native_stft",
                "tf_rfft_matches_tf_stft",
                "windowed_frame_bytes_equal",
            ),
            True,
        )
        case.update(
            frame_sha256="frame",
            reference_sha256={"piano": "ref"},
            spectrum_sha256=dict.fromkeys(implementations, "spectrum"),
        )
        report["cases"] = dict.fromkeys(
            ("music_original", "music_multiwindow", "music_short", "silence", "boundary_impulses"),
            case,
        )
    with pytest.raises(ValueError, match="invalid failing sample coordinates"):
        summary(reports)
