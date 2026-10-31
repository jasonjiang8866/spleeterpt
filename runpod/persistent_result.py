"""Fail-closed acceptance for the full-training-pool storage pilot."""

import json
from pathlib import Path

from training_result import number, require


def validate_persistent(result, sha, task, volume_id, pod_id):
    r = result.get("report", {})
    stage = task.removeprefix("persistent-")
    require(result.get("status") == "passed", "persistent worker failed")
    require(
        r.get("kind") == "persistent-pilot"
        and r.get("stage") == stage
        and r.get("commit") == sha
        and r.get("volume_id") == volume_id
        and r.get("pod_id") == pod_id,
        "persistent identity mismatch",
    )
    d = r.get("dataset", {})
    require(
        d.get("archive_sha256")
        == "2765dd51fed264bbe2dac5c382f7120b7cf1932698fc67d100aa0f3811479152"
        and d.get("train_tracks") == 90
        and d.get("validation_tracks") == 10
        and d.get("verified_audio_files") == 500
        and d.get("test_audio_extracted") is False,
        "incomplete dataset evidence",
    )
    require(number(r.get("seconds"), positive=True), "invalid duration")
    if stage != "prepare":
        final = stage == "resume"
        step = 90 if final else 10
        require(
            r.get("torch") == "2.8.0+cu128"
            and r.get("cuda") == "12.8"
            and r.get("gpu") == "NVIDIA GeForce RTX 4090",
            "unexpected runtime",
        )
        require(
            json.loads(json.dumps(r.get("config")))
            == json.loads(
                (Path(__file__).resolve().parents[1] / "configs/4stems.json").read_text()
            ),
            "wrong config",
        )
        require(
            r.get("cursor") == ([90, 1, 0] if final else [10, 0, 10])
            and r.get("optimizer_steps") == [step]
            and r.get("weights_finite") is True,
            "incomplete training",
        )
        metrics = r.get("metrics", [])
        require(
            [(m.get("kind"), m.get("step")) for m in metrics]
            == [("train", i) for i in range(1, step + 1)] + ([("validation", 90)] if final else []),
            "metric history incomplete or duplicated",
        )
        require(
            all(
                number(m.get("loss"))
                and (
                    m["kind"] != "train"
                    or (number(m.get("grad_norm")) and m.get("update_skipped") is False)
                )
                for m in metrics
            ),
            "nonfinite training metrics",
        )
        require(r.get("backup", {}).get("step") == step, "durable checkpoint behind training")
        require(
            {"last.pt", "metrics.jsonl"} | ({"best.pt"} if final else set())
            <= r["backup"].get("files", {}).keys(),
            "missing durable files",
        )
        if final:
            require(
                r.get("source_pod") != pod_id and r.get("restored", {}).get("step") == 10,
                "replacement Pod restore missing",
            )
    print("PASS: " + task + " data/storage/training evidence", flush=True)
