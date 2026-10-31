"""Fail closed on incomplete real-music training or resume evidence (stdlib only)."""

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_KEYS = {
    "model",
    "optimizer",
    "scaler",
    "rng_cpu",
    "rng_cuda",
    "signature",
    "step",
    "epoch",
    "next_batch",
    "best",
    "config",
}


def require(condition, message):
    if not condition:
        raise RuntimeError("Training acceptance: " + message)


def number(value, *, positive=False):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def validate_training(result, sha):
    require(result.get("status") == "passed", "worker failed")
    report = result.get("report", {})
    require(
        report.get("kind") == "training-preview" and report.get("commit") == sha,
        "identity mismatch",
    )
    require(
        report.get("torch") == "2.8.0+cu128"
        and report.get("cuda") == "12.8"
        and report.get("gpu") == "NVIDIA GeForce RTX 4090",
        "unexpected runtime",
    )
    assets = json.loads((ROOT / "runpod/training-assets.json").read_text())
    require(report.get("assets") == assets, "training pool selection mismatch")
    require(set(assets["train"]).isdisjoint(assets["validation"]), "overlapping split")
    require(
        all(n.startswith("train/") for group in assets.values() for n in group),
        "test-set contamination",
    )
    decoded = report.get("decoded", {})
    require(
        set(decoded) == set(assets["train"]) | set(assets["validation"]), "missing decoded tracks"
    )
    for item in decoded.values():
        require(
            type(item.get("samples")) is int and 6 * 44100 < item["samples"] < 8 * 44100,
            "invalid audio length",
        )
        hashes = item.get("wav_sha256", {})
        require(
            set(hashes) == {"mixture", "drums", "bass", "other", "vocals"}
            and all(
                isinstance(h, str) and re.fullmatch(r"[a-f0-9]{64}", h) for h in hashes.values()
            ),
            "decoded hashes missing",
        )
    runs = report.get("runs", [])
    require(
        len(runs) == 2 and [r.get("stems") for r in runs] == [2, 4], "incomplete preset coverage"
    )
    for r in runs:
        expected = json.loads((ROOT / f"configs/{r['stems']}stems.json").read_text())
        require(
            json.loads(json.dumps(r.get("config"))) == expected
            and r.get("expected_config_matches") is True,
            "not the full preset",
        )
        require(
            r.get("precision") == "fp32"
            and r.get("lr") == 1e-5
            and r.get("batch_size") == 1
            and r.get("epochs") == 2,
            "unexpected training setup",
        )
        require(
            r.get("interrupted_cursor") == [3, 0, 3] and r.get("final_cursor") == [8, 2, 0],
            "resume cursor mismatch",
        )
        require(
            r.get("state_equal") == dict.fromkeys(STATE_KEYS, True),
            "continuous/resumed state differs",
        )
        for key in (
            "best_state_equal",
            "metrics_equal",
            "weights_changed",
            "weights_finite",
            "inference_finite",
        ):
            require(r.get(key) is True, key)
        require(r.get("optimizer_steps") == [8], "optimizer did not update all eight steps")
        metrics = r.get("metrics", {})
        require(
            set(metrics) == {"continuous", "resumed"}
            and metrics["continuous"] == metrics["resumed"],
            "metric histories differ",
        )
        rows = metrics["resumed"]
        expected_rows = [("train", i, (i - 1) // 4) for i in range(1, 5)] + [("validation", 4, 0)]
        expected_rows += [("train", i, (i - 1) // 4) for i in range(5, 9)] + [("validation", 8, 1)]
        require(
            [(x.get("kind"), x.get("step"), x.get("epoch")) for x in rows] == expected_rows,
            "missing/repeated train or validation records",
        )
        for row in rows:
            require(number(row.get("loss")), "nonfinite loss")
            if row["kind"] == "train":
                require(
                    number(row.get("grad_norm")) and row.get("update_skipped") is False,
                    "invalid gradient or skipped update",
                )
        timings = r.get("timings", {})
        require(set(timings) == {"continuous", "interrupted", "resumed"}, "missing process timings")
        for timing in timings.values():
            require(
                number(timing.get("seconds"), positive=True)
                and number(timing.get("peak_allocated_bytes"), positive=True),
                "invalid timing/memory",
            )
        for key in ("initial_sha256", "last_sha256", "best_sha256"):
            require(
                isinstance(r.get(key), str) and re.fullmatch(r"[a-f0-9]{64}", r[key]),
                "missing checkpoint hash",
            )
    require(number(report.get("seconds"), positive=True), "invalid total duration")
    print(
        "PASS: 2/4-stem real training, validation, exact fresh-process resume and inference",
        flush=True,
    )
