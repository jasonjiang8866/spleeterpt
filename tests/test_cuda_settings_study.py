import importlib
import json
from dataclasses import replace
from functools import partial
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


@pytest.mark.parametrize("precision", [False, True])
def test_study_repeats_partial_recovery_and_numeric_gates(probe, tmp_path, monkeypatch, precision):
    profiles = probe.PRECISION_PROFILES if precision else probe.PROFILES
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
            if profile == profiles[0] and index == 0 and out.name == "bounded":
                data[3, 0] = 0.01
            sf.write(out / (stem + ".wav"), data, probe.CONFIG.sample_rate, subtype="FLOAT")
        return "{}\n"

    monkeypatch.setattr(probe.subprocess, "check_output", simulate)
    thresholds = SimpleNamespace(max_abs_error=2e-6, min_si_sdr=100)
    interrupt = True
    with pytest.raises(RuntimeError, match="interrupted"):
        probe.study(checkpoint, audio, ref, None, thresholds, dest, "test", precision=precision)
    partial = json.loads(dest.read_text())
    assert not partial["complete"]
    assert len(partial["cases"]["music_original"]["runs"]) == 1
    interrupt = False
    calls.clear()
    probe.study(checkpoint, audio, ref, None, thresholds, dest, "test", precision=precision)
    result = json.loads(dest.read_text())
    assert result["complete"] and not result["release_eligible"]
    assert len(calls) == len(profiles) * 8 and set(calls) == set(profiles)
    if precision:
        assert result["screen_passed_profiles"] == profiles[1:]
    for case in result["cases"].values():
        assert [r["profile"] for r in case["runs"]] == profiles + profiles[::-1]
    baseline = result["summary"][profiles[0]]
    assert baseline["real60"]["bounded_stems_passed"] == [4, 4]
    assert baseline["music_original"]["tf_stems_passed"] == {"full": [5, 5], "bounded": [4, 4]}
    assert result["summary"][profiles[-1]]["music_original"]["identical_repeat_pcm"]
    assert result["summary"][profiles[-1]]["real60"]["bounded_stems_passed"] == [5, 5]


def test_precision_profiles_exercise_selected_operations_on_both_cli_modes(
    probe, monkeypatch, tmp_path, tiny_config, capsys, request
):
    config = replace(tiny_config, instruments=probe.CONFIG.instruments)
    checkpoint = tmp_path / "model.pt"
    model = Spleeter(config).eval()
    atomic_save(
        {"format_version": 1, "config": config.to_dict(), "model": model.state_dict()}, checkpoint
    )
    source = tmp_path / "mix.wav"
    # Traverse multiple windows and a partial tail.
    waveform = torch.randn(2, config.segment_samples + 123)
    sf.write(source, waveform.T.numpy(), config.sample_rate, subtype="FLOAT")
    reference = probe.separate(model, waveform, five_stem_cpu_precision=True)

    seen = set()
    stft = probe.audio.stft

    def fft(*args, **kwargs):
        if kwargs.get("wide_fft"):
            seen.add("fft64")
        return stft(*args, **kwargs)

    monkeypatch.setattr(probe.audio, "stft", fft)
    monkeypatch.setattr(probe.streaming, "stft", fft)
    load = probe.cli.load_model

    def load_traced(*args):
        loaded = load(*args)
        for net in loaded.unets.values():
            # Observe the wide flag inside forward rather than in a pre-hook:
            # the diagnostic binds it on the instance's forward callable.
            for label, layer in (("decoder64", net.decoder[-1]), ("output64", net.output)):

                def traced(x, *, forward=layer.forward, name=label, wide=False):
                    if wide:
                        seen.add(name)
                    return forward(x, wide=wide)

                layer.forward = traced
        return loaded

    monkeypatch.setattr(probe.cli, "load_model", load_traced)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test CPU")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    # Restore process-global flags after exercising the worker in-process.
    for backend, name in (
        (torch.backends.cudnn, "allow_tf32"),
        (torch.backends.cuda.matmul, "allow_tf32"),
        (torch.backends.cudnn, "deterministic"),
    ):
        request.addfinalizer(partial(setattr, backend, name, getattr(backend, name)))
    assert len(set(probe.PRECISION_PROFILES)) == 8
    for profile in probe.PRECISION_PROFILES:
        expected = set(profile.split("+")).intersection(probe.WIDE_OPS)
        for mode in probe.MODES:
            seen.clear()
            root = tmp_path / profile
            args = probe.cli_args(checkpoint, source, root, mode)
            args[args.index("cuda")] = "cpu"
            probe.run_cli(profile, args)
            settings = json.loads(capsys.readouterr().out)
            assert settings["cudnn_deterministic"] and not settings["cudnn_tf32"]
            assert not settings["matmul_tf32"] and settings["warmup_windows"] == 0
            assert seen == expected == set(settings["fp64_operations"])
            if expected == set(probe.WIDE_OPS) and mode == "full":
                for stem, wave in reference.items():
                    got, _ = sf.read(root / mode / (stem + ".wav"), dtype="float32")
                    np.testing.assert_array_equal(got.T, wave.numpy())


@pytest.mark.parametrize("mode", ["full", "bounded"])
def test_public_precision_option_matches_screen_on_cpu_fixture(
    probe, monkeypatch, tmp_path, tiny_config, mode
):
    # Keep the CLI's device selection, substitute CPU only for model loading.
    config = replace(tiny_config, instruments=probe.CONFIG.instruments)
    model = Spleeter(config).eval()
    monkeypatch.setattr(probe.cli, "load_model", lambda checkpoint, device: model)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU fixture")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    source = tmp_path / "music.wav"
    sf.write(
        source,
        torch.randn(config.segment_samples + 123, 2).numpy(),
        config.sample_rate,
        subtype="FLOAT",
    )

    def backend():
        return (
            torch.get_float32_matmul_precision(),
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        )

    # Keep the diagnostic's process-global changes inside this test.
    with probe.cli._cuda_precision():
        probe.run_cli(
            "no_tf32+deterministic+fft64",
            probe.cli_args("weights", source, tmp_path / "study", mode),
        )

    states = []
    original = probe.audio.stft

    def observe(*args, **kwargs):
        states.append((backend(), kwargs.get("wide_fft", False)))
        return original(*args, **kwargs)

    monkeypatch.setattr(probe.audio, "stft", observe)
    monkeypatch.setattr(probe.streaming, "stft", observe)
    before = backend()
    args = probe.cli_args("weights", source, tmp_path / "public", mode, profile="fft64")
    probe.cli.main(args)
    assert states and all(state == (("highest", False, True, False), True) for state in states)
    for stem in config.instruments:
        name = stem + ".wav"
        actual = sf.read(tmp_path / "public" / mode / name, dtype="float32")[0]
        reference = sf.read(tmp_path / "study" / mode / name, dtype="float32")[0]
        np.testing.assert_array_equal(actual, reference)
    assert backend() == before
    with pytest.raises(FileExistsError):
        probe.cli.main(args)
    assert backend() == before


def test_public_precision_rejects_wrong_device_and_weights(
    probe, monkeypatch, tmp_path, tiny_config
):
    args = probe.cli_args("weights", "music.wav", tmp_path, "full", profile="fft64")
    cpu = ["cpu" if arg == "cuda" else arg for arg in args]
    with pytest.raises(ValueError, match="device cuda"):
        probe.cli.main(cpu)
    released = [
        "--model" if arg == "--checkpoint" else "2stems" if arg == "weights" else arg
        for arg in args
    ]
    with pytest.raises(ValueError, match="checkpoint"):
        probe.cli.main(released)
    monkeypatch.setattr(probe.cli, "load_model", lambda *a: Spleeter(tiny_config))
    with pytest.raises(ValueError, match="five-stem checkpoint"):
        probe.cli.main(args)
    assert not (tmp_path / "full").exists()
