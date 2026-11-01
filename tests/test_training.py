import json

import pytest
import soundfile as sf
import torch

from spleeterpt.checkpoint import load_model, read_checkpoint
from spleeterpt.cli import main, parser
from spleeterpt.data import StemDataset
from spleeterpt.train import run_training


def test_frozen_bn_keeps_dropout_training_and_affine_gradients():
    from spleeterpt.train import training_mode

    model = torch.nn.Sequential(torch.nn.BatchNorm1d(8), torch.nn.Dropout(0.5))
    before = {k: t.clone() for k, t in model[0].named_buffers()}
    model.eval()
    training_mode(model, freeze_bn_stats=True)
    assert model.training and model[1].training and not model[0].training
    x = torch.randn(16, 8)
    y = model(x)
    assert not torch.equal(y, model(x))
    y.square().mean().backward()
    assert all(p.requires_grad and p.grad.abs().sum() > 0 for p in model[0].parameters())
    assert all(torch.equal(before[k], t) for k, t in model[0].named_buffers())


def test_frozen_bn_resume_and_policy_guard(audio_dataset, tmp_path):
    complete, resumed = tmp_path / "frozen", tmp_path / "frozen-resume"

    def args(out, *extra):
        return training_args(audio_dataset, out, "--freeze-bn-stats", *extra)

    run_training(args(complete))
    run_training(args(resumed, "--max-steps", "2"))
    checkpoint = resumed / "last.pt"
    with pytest.raises(ValueError, match="resume configuration/data differs"):
        run_training(training_args(audio_dataset, resumed, "--resume", str(checkpoint)))
    run_training(args(resumed, "--resume", str(checkpoint)))
    a, b = read_checkpoint(complete / "last.pt"), read_checkpoint(checkpoint)
    assert a["signature"]["freeze_bn_stats"] is True
    for k, t in a["model"].items():
        assert torch.equal(t, b["model"][k])
        if k.endswith("num_batches_tracked"):
            assert t.item() == 0
        elif k.endswith("running_mean"):
            assert torch.count_nonzero(t) == 0
        elif k.endswith("running_var"):
            assert torch.equal(t, torch.ones_like(t))


def training_args(audio_dataset, output, *extra):
    config, train, val, _ = audio_dataset
    return parser().parse_args(
        [
            "train",
            "--config",
            str(config),
            "--train-manifest",
            str(train),
            "--val-manifest",
            str(val),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--crops-per-track",
            "2",
            "--validation-crops",
            "1",
            "--workers",
            "0",
            "--save-every",
            "2",
            "--deterministic",
            *extra,
        ]
    )


def test_aligned_targets_and_reproducible_crops(audio_dataset, tiny_config):
    _, train, _, _ = audio_dataset
    data = StemDataset(train, tiny_config, training=True, samples_per_track=2)
    mixture, targets = data[0]
    torch.testing.assert_close(targets.sum(0), mixture, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(data[0][0], mixture, atol=0, rtol=0)
    data.epoch = 1
    assert not torch.equal(data[0][0], mixture)
    validation = StemDataset(train, tiny_config, training=False, samples_per_track=2)
    first = validation[0][0]
    validation.epoch = 7
    torch.testing.assert_close(first, validation[0][0], atol=0, rtol=0)


def test_training_resume_matches_uninterrupted_and_cli_inference(audio_dataset, tmp_path):
    complete, resumed = tmp_path / "complete", tmp_path / "resumed"
    result = run_training(training_args(audio_dataset, complete))
    assert result["step"] == 4
    run_training(training_args(audio_dataset, resumed, "--max-steps", "2"))
    checkpoint = resumed / "last.pt"
    assert read_checkpoint(checkpoint)["next_batch"] == 2
    run_training(training_args(audio_dataset, resumed, "--resume", str(checkpoint)))
    baseline, restored = read_checkpoint(complete / "last.pt"), read_checkpoint(checkpoint)
    assert baseline["step"] == restored["step"] == 4
    assert baseline["best"] == restored["best"]
    for name, weights in baseline["model"].items():
        torch.testing.assert_close(weights, restored["model"][name], atol=0, rtol=0)
    assert (resumed / "best.pt").exists()
    model = load_model(checkpoint)
    assert model.config.sample_rate == 8000
    destination = tmp_path / "separated"
    source = audio_dataset[3][2] / "mixture.wav"
    main(
        [
            "separate",
            str(source),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(destination),
            "--device",
            "cpu",
        ]
    )
    for name in model.config.instruments:
        info = sf.info(destination / f"{name}.wav")
        assert info.frames == sf.info(source).frames
        assert info.samplerate == 8000 and info.channels == 2
    with pytest.raises(FileExistsError):
        run_training(training_args(audio_dataset, complete))


def test_resume_rejects_changed_configuration(audio_dataset, tmp_path):
    run = tmp_path / "run"
    run_training(training_args(audio_dataset, run, "--max-steps", "1"))
    with pytest.raises(ValueError, match="resume configuration/data differs"):
        run_training(
            training_args(audio_dataset, run, "--resume", str(run / "last.pt"), "--batch-size", "2")
        )


def test_overlapping_validation_rejected(audio_dataset, tmp_path):
    args = training_args(audio_dataset, tmp_path / "run")
    args.val_manifest = args.train_manifest
    with pytest.raises(ValueError, match="overlap"):
        run_training(args)


def test_misaligned_audio_rejected(audio_dataset, tiny_config):
    _, train, _, tracks = audio_dataset
    file = tracks[0] / "vocals.wav"
    audio, sample_rate = sf.read(file)
    sf.write(file, audio[:-1], sample_rate)
    with pytest.raises(ValueError, match="unaligned"):
        StemDataset(train, tiny_config, training=True)


def test_manifest_split_is_track_disjoint(audio_dataset, tmp_path):
    dest = tmp_path / "manifests"
    main(["manifest", "--root", str(tmp_path), "--output", str(dest), "--validation-tracks", "1"])
    train = [json.loads(line)["track"] for line in (dest / "train.jsonl").read_text().splitlines()]
    val = [
        json.loads(line)["track"] for line in (dest / "validation.jsonl").read_text().splitlines()
    ]
    assert len(train) == 2 and len(val) == 1 and not set(train) & set(val)


def test_frozen_multiepoch_resume_matches_full_training(audio_dataset, tmp_path):
    """Epoch-end validation/dropout/RNG and optimizer survive several milestone restores."""
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runpod"))
    from training import equal_state

    full, staged = tmp_path / "full", tmp_path / "staged"
    common = ("--freeze-bn-stats", "--validation-crops", "4", "--lr", "3e-6")
    run_training(training_args(audio_dataset, full, *common, "--epochs", "5"))
    resume = ()
    for epoch in (1, 3, 5):
        args = training_args(audio_dataset, staged, *common, "--epochs", str(epoch), *resume)
        # The real pilot starts a new interpreter at every milestone.
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json,sys,torch; from argparse import Namespace; "
                    "from spleeterpt.train import run_training; torch.set_num_threads(2); "
                    "run_training(Namespace(**json.loads(sys.argv[1])))"
                ),
                json.dumps(vars(args)),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        checkpoint = read_checkpoint(staged / "last.pt")
        assert [checkpoint[k] for k in ("step", "epoch", "next_batch")] == [4 * epoch, epoch, 0]
        restored = tmp_path / f"restored-{epoch}.pt"
        shutil.copyfile(staged / "last.pt", restored)
        resume = ("--resume", str(restored))
    assert equal_state(read_checkpoint(full / "last.pt"), read_checkpoint(staged / "last.pt"))
    assert equal_state(read_checkpoint(full / "best.pt"), read_checkpoint(staged / "best.pt"))
    assert (full / "metrics.jsonl").read_bytes() == (staged / "metrics.jsonl").read_bytes()


def test_crop_start_matches_actual_read_and_union_coverage(audio_dataset, tiny_config):
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runpod"))
    from convergence import coverage

    data = StemDataset(audio_dataset[1], tiny_config, training=True, seed=17)
    report = coverage(data, 10)
    for index, track in enumerate(report["tracks"]):
        covered = set()
        for epoch, start in enumerate(track["starts"]):
            data.epoch = epoch
            assert start == data.crop_start(index)
            audio, _ = sf.read(
                data.records[index][0],
                start=start,
                frames=tiny_config.segment_samples,
                dtype="float32",
                always_2d=True,
            )
            np.testing.assert_array_equal(data[index][0].numpy(), audio.T)
            covered.update(range(start, min(track["samples"], start + tiny_config.segment_samples)))
        assert track["covered_samples"] == len(covered)
        assert len(set(track["starts"])) > 1


def test_spectral_loss_is_sum_of_stem_means_with_gradients(tiny_config):
    from spleeterpt.audio import stft
    from spleeterpt.train import spectral_loss

    class Gain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(0.3))

        def forward(self, x):
            return x.unsqueeze(1).expand(-1, len(tiny_config.instruments), -1, -1, -1) * self.gain

    model = Gain()
    mix = torch.randn(2, 2, tiny_config.segment_samples)
    target = torch.randn(2, len(tiny_config.instruments), 2, tiny_config.segment_samples)
    loss = spectral_loss(
        model, mix, target, tiny_config, precision="fp32", device=torch.device("cpu")
    )
    x = stft(mix, tiny_config).abs()[..., : tiny_config.frequency_bins]
    y = stft(target, tiny_config).abs()[..., : tiny_config.frequency_bins]
    expected = sum((0.3 * x - y[:, i]).abs().mean() for i in range(len(tiny_config.instruments)))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(model.gain.grad) and model.gain.grad.abs() > 0


def test_silent_crop_has_valid_zero_loss_and_gradient(audio_dataset, tmp_path):
    """Finite zero loss/norm is legitimate; Adam still records the optimizer step."""
    import numpy as np

    for path in audio_dataset[3][0].glob("*.wav"):
        info = sf.info(path)
        sf.write(
            path, np.zeros((info.frames, 2), dtype="float32"), info.samplerate, subtype="FLOAT"
        )
    output = tmp_path / "silent-run"
    run_training(training_args(audio_dataset, output, "--freeze-bn-stats"))
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    zeros = [m for m in metrics if m["kind"] == "train" and m["loss"] == 0.0]
    assert len(zeros) == 2
    assert all(m["grad_norm"] == 0.0 and not m["update_skipped"] for m in zeros)
    checkpoint = read_checkpoint(output / "last.pt")
    assert checkpoint["step"] == 4
    assert {int(v["step"]) for v in checkpoint["optimizer"]["state"].values()} == {4}
