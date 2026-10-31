"""Small synthetic CPU checks; never download data or instantiate pretrained models."""

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def ev(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("musdb_evaluation")


def make_wav_track(root, name, samples=44100, rate=44100):
    folder = root / "test" / name
    folder.mkdir(parents=True)
    rng = np.random.default_rng(4)
    sources = {
        n: rng.normal(0, 0.05, (samples, 2)).astype(np.float32)
        for n in ("drums", "bass", "other", "vocals")
    }
    sources["mixture"] = sum(sources.values())
    for source, audio in sources.items():
        sf.write(folder / f"{source}.wav", audio, rate, subtype="FLOAT")
    return sources


def test_selection_and_manifest_order(ev, tmp_path):
    for name in ("z", "a", "é", "b"):
        make_wav_track(tmp_path, name, samples=8)
    all_tracks = ev.discover(tmp_path, "wav", ["test"])
    assert [r["id"] for r in all_tracks] == ["test/a", "test/b", "test/z", "test/é"]
    chosen = ev.discover(tmp_path, "wav", ["test"], limit=2, seed=81)
    assert chosen == ev.discover(tmp_path, "wav", ["test"], limit=2, seed=81)
    assert len(chosen) == 2
    assert ev.discover(tmp_path, "wav", ["test"], ["test/z", "test/a"]) == [
        all_tracks[0],
        all_tracks[2],
    ]
    for kwargs in (
        {"limit": 8},
        {"limit": 0},
        {"track_ids": ["test/missing"]},
        {"track_ids": ["test/a", "test/a"]},
        {"seed": 1},
    ):
        with pytest.raises(ValueError):
            ev.discover(tmp_path, "wav", ["test"], **kwargs)
    (tmp_path / "test" / "a" / "other.wav").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        ev.discover(tmp_path, "wav", ["test"])


def test_wav_targets_and_reject_bad_references(ev, tmp_path):
    originals = make_wav_track(tmp_path, "a", samples=30)
    record = ev.discover(tmp_path, "wav", ["test"])[0]
    targets = ev.decode_track(tmp_path, record, "wav")
    np.testing.assert_array_equal(targets["vocals"], originals["vocals"].T)
    np.testing.assert_array_equal(
        targets["accompaniment"], targets["drums"] + targets["bass"] + targets["other"]
    )
    for audio, rate, error in (
        (np.ones((20, 2)), 44100, "lengths differ"),
        (np.ones((30, 1)), 44100, "stereo"),
        (np.ones((30, 2)), 22050, "Hz"),
    ):
        sf.write(tmp_path / "test/a/drums.wav", audio, rate, subtype="FLOAT")
        with pytest.raises(ValueError, match=error):
            ev.decode_track(tmp_path, record, "wav")


def test_stems_mapping_uses_preview_decoder(ev, tmp_path, monkeypatch):
    preview = importlib.import_module("evaluate_preview")
    folder = tmp_path / "test"
    folder.mkdir()
    (folder / "one.stem.mp4").touch()
    calls = []

    def decode(path, index):
        calls.append(index)
        return SimpleNamespace(numpy=lambda: np.full((2, 40), index, np.float32))

    monkeypatch.setattr(preview, "decode_stream", decode)
    record = ev.discover(tmp_path, "stems", ["test"])[0]
    targets = ev.decode_track(tmp_path, record, "stems")
    assert calls == [0, 1, 2, 3, 4]
    assert targets["vocals"][0, 0] == 4
    assert targets["accompaniment"][0, 0] == 6


def test_length_policy_never_repairs_outputs(ev):
    targets = {"mixture": np.ones((2, 2000))}
    variants = {
        v: {n: np.ones((2, 2000)) for n in ("vocals", "accompaniment")} for v in ev.VARIANTS
    }
    variants["legacy_unmodified"] = {n: a[:, :-9] for n, a in variants["legacy_unmodified"].items()}
    common, lengths = ev.validate_outputs(
        targets, variants, ["vocals", "accompaniment"], 1024, "common-crop"
    )
    assert common == 1991 and lengths["spleeterpt"]["vocals"] == 2000
    with pytest.raises(ValueError, match="length"):
        ev.validate_outputs(targets, variants, ["vocals", "accompaniment"], 1024, "strict")
    for bad in (np.ones((2, 500)), np.ones((1, 2000)), np.full((2, 2000), np.nan)):
        variants["spleeterpt"]["vocals"] = bad
        with pytest.raises(ValueError):
            ev.validate_outputs(targets, variants, ["vocals", "accompaniment"], 1024, "common-crop")


@pytest.mark.parametrize(
    "instruments", [["vocals", "accompaniment"], ["vocals", "drums", "bass", "other"]]
)
def test_real_museval_matches_official_track_api(ev, instruments, monkeypatch):
    museval = pytest.importorskip("museval")
    rng = np.random.default_rng(32)
    # Includes two complete windows and a fractional tail (not a third BSS frame).
    samples = 2 * ev.RATE + 137
    targets = {n: rng.normal(size=(2, samples)).astype(np.float32) for n in instruments}
    targets["mixture"] = sum(targets.values())
    estimates = {
        n: a + rng.normal(0, 0.1, a.shape).astype(np.float32)
        for n, a in targets.items()
        if n in instruments
    }
    variants = {v: estimates for v in ev.VARIANTS}
    record = ev.score_variants(targets, variants, instruments)
    calls = []
    evaluate = museval.evaluate

    def capture(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(museval, "evaluate", capture)
    track = SimpleNamespace(
        name="synthetic",
        rate=ev.RATE,
        targets={n: SimpleNamespace(audio=targets[n].T.astype(np.float64)) for n in instruments},
    )
    native = museval.eval_mus_track(
        track, {n: a.T.astype(np.float64) for n, a in estimates.items()}
    )
    for item in native.scores["targets"]:
        name = item["name"]
        actual = record["scores"]["spleeterpt"][name]["bsseval_v4"]["sdr"]
        # TrackStore rounds to five decimals. Compare its raw evaluation result
        # before serialization, without loosening any numerical threshold.
        expected = calls[-1][0][instruments.index(name)]
        np.testing.assert_array_equal(actual, expected)
        assert len(actual) == 2
        assert actual[0] > 10
    assert record["scores"]["spleeterpt"] == record["scores"]["tensorflow_current"]


def fake_record(ev, identifier, frame_values):
    return {
        "id": identifier,
        "scores": {
            v: {
                "vocals": {
                    "global_stereo_si_sdr_db": 3.0,
                    "bsseval_v4": {m: frame_values for m in ev.BSS_METRICS},
                }
            }
            for v in ev.VARIANTS
        },
    }


def test_track_weighted_aggregation_and_undefined_counts(ev):
    records = [fake_record(ev, "test/a", [1, 2, 99]), fake_record(ev, "test/b", [10])]
    records[0]["scores"]["legacy_unmodified"]["vocals"]["bsseval_v4"]["sdr"] = [-1, 0, 97]
    summary = ev.summarize(records, ["vocals"])
    stats = summary["aggregates"]["bsseval_v4_sdr_db"]["spleeterpt"]["vocals"]
    assert stats["median"] == 6  # median([median([1, 2, 99]), 10]), not pooled frames
    pair = summary["paired_differences"]["bsseval_v4_sdr_db"]["spleeterpt_minus_legacy_unmodified"][
        "vocals"
    ]
    assert pair["median"] == 1 and pair["candidate_better"] == 1 and pair["equal"] == 1
    assert ev.statistics([None, "+inf", "-inf", 2])["finite"] == 1
    assert ev.statistics([None, None])["median"] is None
    assert ev.statistics(["+inf"])["median"] == "+inf"


def create_fake_run(ev, folder):
    (folder / "tracks").mkdir()
    instruments = ["vocals", "accompaniment"]
    selection = [{"id": "test/a", "sha256": {"test/a.stem.mp4": "input-hash"}}]
    manifest = {
        "instruments": instruments,
        "selection": selection,
        "model_config": {"hop_length": 1024},
        "length_policy": "common-crop",
        "start_seconds": 0,
        "duration_seconds": None,
    }
    ev.atomic_json(folder / "manifest.json", manifest)
    digest = ev.sha256(folder / "manifest.json")
    record = fake_record(ev, "test/a", [2])
    for v in ev.VARIANTS:
        record["scores"][v]["accompaniment"] = copy.deepcopy(record["scores"][v]["vocals"])
    record.update(
        input_sha256=selection[0]["sha256"],
        manifest_sha256=digest,
        evaluated_samples=44100,
        decoded_samples=44100,
        segment_samples=44100,
        segment_start_sample=0,
        discarded_reference_tail_samples=0,
        output_samples={v: {n: 44100 for n in instruments} for v in ev.VARIANTS},
    )
    ev.atomic_json(folder / "tracks/0000.json", record)
    ev.atomic_json(folder / "summary.json", ev.summarize([record], instruments))
    status = {
        "state": "complete",
        "manifest_sha256": digest,
        "record_sha256": {"0000.json": ev.sha256(folder / "tracks/0000.json")},
        "summary_sha256": ev.sha256(folder / "summary.json"),
    }
    ev.atomic_json(folder / "status.json", status)
    return record, status


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "extra",
        "summary",
        "record",
        "manifest",
        "failed",
        "target",
        "frame",
        "hash_inventory",
    ],
)
def test_integrity_fails_closed(ev, tmp_path, corruption):
    record, status = create_fake_run(ev, tmp_path)
    assert ev.validate_run(tmp_path)["track_count"] == 1
    if corruption == "missing":
        (tmp_path / "tracks/0000.json").unlink()
    elif corruption == "extra":
        ev.atomic_json(tmp_path / "tracks/0001.json", record)
    elif corruption == "summary":
        summary = json.loads((tmp_path / "summary.json").read_text())
        summary["track_count"] = 2
        ev.atomic_json(tmp_path / "summary.json", summary)
        status["summary_sha256"] = ev.sha256(tmp_path / "summary.json")
    elif corruption == "manifest":
        (tmp_path / "manifest.json").write_text("{}")
    elif corruption == "failed":
        status["state"] = "failed"
    elif corruption == "hash_inventory":
        status["record_sha256"] = {}
    else:
        if corruption == "target":
            del record["scores"]["spleeterpt"]["vocals"]
        elif corruption == "frame":
            record["scores"]["spleeterpt"]["vocals"]["bsseval_v4"]["sdr"] = []
        else:
            record["id"] = "test/other"
        ev.atomic_json(tmp_path / "tracks/0000.json", record)
        status["record_sha256"]["0000.json"] = ev.sha256(tmp_path / "tracks/0000.json")
    ev.atomic_json(tmp_path / "status.json", status)
    with pytest.raises((ValueError, KeyError)):
        ev.validate_run(tmp_path)


def test_whole_silent_source_is_explicitly_undefined(ev):
    pytest.importorskip("museval")
    targets = {"vocals": np.zeros((2, 2048)), "accompaniment": np.ones((2, 2048))}
    targets["mixture"] = targets["accompaniment"]
    variants = {v: {n: targets[n] for n in ("vocals", "accompaniment")} for v in ev.VARIANTS}
    record = ev.score_variants(targets, variants, ["vocals", "accompaniment"])
    for variant in ev.VARIANTS:
        score = record["scores"][variant]["vocals"]
        assert score["global_stereo_si_sdr_db"] is None
        assert score["bsseval_v4"]["sdr"] == [None]
        assert "non-silent" in score["bsseval_v4_error"]


def test_musdb_rejects_piano_target(ev, tmp_path):
    create_fake_run(ev, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["instruments"] = ["vocals", "piano", "drums", "bass", "other"]
    ev.atomic_json(tmp_path / "manifest.json", manifest)
    status = json.loads((tmp_path / "status.json").read_text())
    status["manifest_sha256"] = ev.sha256(tmp_path / "manifest.json")
    ev.atomic_json(tmp_path / "status.json", status)
    with pytest.raises(ValueError, match="target set"):
        ev.validate_run(tmp_path)


@pytest.mark.parametrize("fail_second", [False, True])
def test_runner_atomic_completion_and_failed_track(ev, tmp_path, monkeypatch, fail_second):
    """Exercise orchestration with tiny injected models, real WAV I/O and result files.

    BSSEval's actual numerical adapter is tested independently above.
    """
    import sys

    import torch
    from torch import nn

    import spleeterpt.audio
    import spleeterpt.checkpoint
    import spleeterpt.config
    import spleeterpt.convert
    import spleeterpt.model

    runner = importlib.import_module("evaluate_musdb")
    root = tmp_path / "data"
    make_wav_track(root, "a", samples=2048)
    make_wav_track(root, "b", samples=2048)
    config = spleeterpt.config.ModelConfig.preset(2)

    class Model(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.weight = nn.Parameter(torch.ones(1))

    monkeypatch.setattr(spleeterpt.checkpoint, "load_model", lambda path: Model(config))
    monkeypatch.setattr(spleeterpt.model, "Spleeter", Model)
    monkeypatch.setattr(spleeterpt.convert, "import_weights", lambda reader, model: None)
    monkeypatch.setitem(
        sys.modules,
        "tensorflow",
        SimpleNamespace(
            __spec__=importlib.machinery.ModuleSpec("tensorflow", loader=None),
            train=SimpleNamespace(load_checkpoint=lambda path: object()),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "threadpoolctl",
        SimpleNamespace(
            threadpool_limits=lambda **kw: SimpleNamespace(restore_original_limits=lambda: None)
        ),
    )

    def outputs(audio):
        return {name: audio * 0.5 for name in config.instruments}

    monkeypatch.setattr(spleeterpt.audio, "separate", lambda model, audio: outputs(audio))
    monkeypatch.setitem(
        sys.modules,
        "diagnose_legacy_port",
        SimpleNamespace(
            load_legacy=lambda *args: lambda audio: list(outputs(audio).values()),
            aligned_legacy=lambda model, audio: model(audio),
            use_four_stem_activations=lambda model: None,
        ),
    )
    closed = []

    class Oracle:
        def __init__(self, *args):
            self.calls = 0

        def separate(self, audio):
            self.calls += 1
            if fail_second and self.calls == 2:
                raise RuntimeError("injected model failure")
            return {name: value.numpy() for name, value in outputs(audio).items()}

        def close(self):
            closed.append(True)

    monkeypatch.setitem(sys.modules, "verify_audio_parity", SimpleNamespace(OfficialGraph=Oracle))
    monkeypatch.setattr(runner, "provenance", lambda path: {"synthetic_test": True})
    version = runner.importlib.metadata.version
    monkeypatch.setattr(
        runner.importlib.metadata,
        "version",
        lambda p: "2.18.1" if p == "tensorflow-cpu" else version(p),
    )

    def scores(targets, variants, names, hop, policy):
        common, lengths = ev.validate_outputs(targets, variants, names, hop, policy)
        return {
            "evaluated_samples": common,
            "discarded_reference_tail_samples": 0,
            "output_samples": lengths,
            "scores": {
                v: {
                    name: {
                        "global_stereo_si_sdr_db": 1.0,
                        "bsseval_v4": {m: [2.0] for m in ev.BSS_METRICS},
                    }
                    for name in names
                }
                for v in ev.VARIANTS
            },
        }

    monkeypatch.setattr(runner, "score_variants", scores)
    for suffix in (".pt", ".index", ".meta", ".data-00000-of-00001"):
        (tmp_path / ("weights" + suffix)).write_bytes(b"synthetic fixture")
    output = tmp_path / "run"
    args = runner.build_parser().parse_args(
        [
            "--root",
            str(root),
            "--layout",
            "wav",
            "--dataset-label",
            "synthetic-test",
            "--upstream-root",
            str(tmp_path),
            "--legacy-root",
            str(tmp_path),
            "--tf-checkpoint",
            str(tmp_path / "weights"),
            "--checkpoint",
            str(tmp_path / "weights.pt"),
            "--output",
            str(output),
        ]
    )
    try:
        if fail_second:
            with pytest.raises(RuntimeError, match="injected"):
                runner.run(args)
            assert json.loads((output / "status.json").read_text())["state"] == "failed"
            assert len(list((output / "tracks").glob("*.json"))) == 1
            assert not (output / "summary.json").exists()
            with pytest.raises(ValueError, match="not complete"):
                ev.validate_run(output)
        else:
            runner.run(args)
            assert ev.validate_run(output)["track_count"] == 2
            with pytest.raises(FileExistsError):
                runner.run(args)
        assert closed == [True]
    finally:
        torch.use_deterministic_algorithms(False)


def test_missing_ffmpeg_fails_before_model_loading(ev, monkeypatch):
    runner = importlib.import_module("evaluate_musdb")
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="including WAV"):
        runner.run(SimpleNamespace())
