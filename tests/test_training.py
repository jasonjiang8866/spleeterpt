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
