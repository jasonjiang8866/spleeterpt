import json
from unittest.mock import patch

import pytest
import torch
from test_training import training_args

from spleeterpt import backup
from spleeterpt.checkpoint import read_checkpoint
from spleeterpt.train import run_training


def test_backup_restore_replays_only_committed_history(audio_dataset, tmp_path):
    complete, interrupted, restored, store = (
        tmp_path / n for n in ("complete", "interrupted", "restored", "store")
    )
    run_training(training_args(audio_dataset, complete))
    run_training(
        training_args(audio_dataset, interrupted, "--max-steps", "2", "--backup-dir", str(store))
    )
    with (interrupted / "metrics.jsonl").open("a") as f:
        f.write('{"kind":"train","step":3,"loss":999}\n')
    checkpoint, info = backup.restore(store, restored)
    assert info["step"] == 2
    run_training(
        training_args(
            audio_dataset, restored, "--resume", str(checkpoint), "--backup-dir", str(store)
        )
    )
    a, b = read_checkpoint(complete / "last.pt"), read_checkpoint(restored / "last.pt")
    for k, v in a["model"].items():
        torch.testing.assert_close(v, b["model"][k], atol=0, rtol=0)
    assert (complete / "metrics.jsonl").read_bytes() == (restored / "metrics.jsonl").read_bytes()
    assert backup.validate(store)[1]["step"] == 4
    assert len(list(store.glob("step-*"))) <= 3


def test_interrupted_publish_preserves_previous_committed_generation(tmp_path):
    source, store = tmp_path / "source", tmp_path / "store"
    source.mkdir()
    (source / "last.pt").write_bytes(b"old")
    (source / "metrics.jsonl").write_text("first\n")
    old = backup.publish(source, store, 1)
    (source / "last.pt").write_bytes(b"new")
    with (
        patch.object(backup, "synced_copy", side_effect=OSError("disk full")),
        pytest.raises(OSError),
    ):
        backup.publish(source, store, 2)
    assert backup.validate(store)[1] == old
    folder, _ = backup.validate(store)
    (folder / "last.pt").write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        backup.restore(store, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_backup_pointer_cannot_escape_storage(tmp_path):
    (tmp_path / "latest.json").write_text(json.dumps({"generation": "../foreign"}))
    with pytest.raises(ValueError, match="generation"):
        backup.validate(tmp_path)
