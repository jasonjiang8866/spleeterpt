import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from spleeterpt.checkpoint import atomic_save
from spleeterpt.model import Spleeter


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("study_cuda_settings")


def test_real_cli_warms_loaded_model(probe, monkeypatch, tmp_path, tiny_config, capsys):
    config = replace(tiny_config, instruments=probe.CONFIG.instruments)
    checkpoint = tmp_path / "model.pt"
    atomic_save(
        {"format_version": 1, "config": config.to_dict(), "model": Spleeter(config).state_dict()},
        checkpoint,
    )
    source = tmp_path / "mix.wav"
    sf.write(source, np.zeros((400, 2)), config.sample_rate, subtype="FLOAT")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test CPU")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    seen = []
    separate = probe.separate

    def tracked(model, audio, **kwargs):
        seen.append((id(model), audio.shape[1]))
        return separate(model, audio, **kwargs)

    monkeypatch.setattr(probe, "separate", tracked)
    monkeypatch.setattr(probe.cli, "separate", tracked)
    for profile, count in (("default", 0), ("warmup", 1)):
        seen.clear()
        args = probe.cli_args(checkpoint, source, tmp_path / profile, "full")
        args[args.index("cuda")] = "cpu"
        probe.run_cli(profile, args)
        assert json.loads(capsys.readouterr().out)["warmup_windows"] == count
        assert [size for _, size in seen] == [config.hop_length] * count + [400]
        assert len({identity for identity, _ in seen}) == 1
        assert len(list((tmp_path / profile / "full").glob("*.wav"))) == 5


def test_study_repeats_partial_recovery_and_numeric_gates(probe, tmp_path, monkeypatch):
    instruments = probe.CONFIG.instruments
    audio = torch.zeros((2, 80))
    ref = dict.fromkeys(instruments, audio.numpy())
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    dest = tmp_path / "study.json"

    def short_input(path, config, seconds, real):
        assert seconds == 60
        sf.write(path, audio.T.numpy(), config.sample_rate, subtype="FLOAT")

    monkeypatch.setattr(probe, "make_input", short_input)
    calls = []

    def simulate(command, text):
        _, _, profile, *args = command
        out = Path(args[args.index("--output") + 1])
        if len(calls) == 2 and interrupt:
            raise RuntimeError("interrupted")
        calls.append(profile)
        out.mkdir()
        for index, stem in enumerate(instruments):
            data = np.zeros((80, 2), dtype="float32")
            if profile == "default" and index == 0 and out.name == "bounded":
                data[3, 0] = 0.01
            sf.write(out / (stem + ".wav"), data, probe.CONFIG.sample_rate, subtype="FLOAT")
        return "{}\n"

    monkeypatch.setattr(probe.subprocess, "check_output", simulate)
    thresholds = SimpleNamespace(max_abs_error=2e-6, min_si_sdr=100)
    interrupt = True
    with pytest.raises(RuntimeError, match="interrupted"):
        probe.study(checkpoint, audio, ref, None, thresholds, dest, "test")
    partial = json.loads(dest.read_text())
    assert not partial["complete"]
    assert len(partial["cases"]["music_original"]["runs"]) == 1
    interrupt = False
    calls.clear()
    probe.study(checkpoint, audio, ref, None, thresholds, dest, "test")
    result = json.loads(dest.read_text())
    assert result["complete"] and not result["release_eligible"]
    assert len(calls) == 72 and len(set(calls)) == 9
    for case in result["cases"].values():
        assert [r["profile"] for r in case["runs"]] == probe.PROFILES + probe.PROFILES[::-1]
    baseline = result["summary"]["default"]
    assert baseline["real60"]["bounded_stems_passed"] == [4, 4]
    assert baseline["music_original"]["tf_stems_passed"] == {"full": [5, 5], "bounded": [4, 4]}
    assert result["summary"]["benchmark"]["music_original"]["identical_repeat_pcm"]
    assert result["summary"]["benchmark"]["real60"]["bounded_stems_passed"] == [5, 5]
