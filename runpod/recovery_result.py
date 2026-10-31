"""Stdlib validation for bounded checkpoint transport and two-Pod recovery evidence."""

import hashlib
import json
import re
import tarfile
from pathlib import Path

from training_result import STATE_KEYS, number, require

MAX_BUNDLE = 1_500_000_000
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_descriptor(descriptor, *, complete):
    names = {"last.pt", "metrics.jsonl", "recovery.json"} | ({"best.pt"} if complete else set())
    require(
        type(descriptor.get("bytes")) is int and 0 < descriptor["bytes"] <= MAX_BUNDLE,
        "invalid bundle size",
    )
    require(
        bool(re.fullmatch(r"[a-f0-9]{64}", descriptor.get("sha256", ""))), "invalid bundle hash"
    )
    files = descriptor.get("files", {})
    require(set(files) == names, "unexpected bundle paths")
    for item in files.values():
        require(
            type(item.get("bytes")) is int and 0 < item["bytes"] <= MAX_BUNDLE,
            "invalid member size",
        )
        require(bool(re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", ""))), "invalid member hash")


def validate_bundle(path, descriptor, *, complete):
    validate_descriptor(descriptor, complete=complete)
    path = Path(path)
    require(
        path.stat().st_size == descriptor["bytes"] and digest(path) == descriptor["sha256"],
        "bundle corrupted",
    )
    with tarfile.open(path, "r:") as tar:
        members = tar.getmembers()
        require(
            len(members) == len(descriptor["files"])
            and {m.name for m in members} == set(descriptor["files"]),
            "unexpected/duplicate archive members",
        )
        for m in members:
            item = descriptor["files"][m.name]
            require(m.isfile() and m.size == item["bytes"], "links or invalid member sizes")
            h = hashlib.sha256()
            with tar.extractfile(m) as stream:
                for b in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(b)
            require(h.hexdigest() == item["sha256"], "checkpoint member corrupted")


def validate_recovery(result, sha, stage, source=None):
    require(result.get("status") == "passed", "recovery worker failed")
    r = result.get("report", {})
    require(
        r.get("kind") == "pod-recovery" and r.get("commit") == sha and r.get("stage") == stage,
        "recovery identity mismatch",
    )
    require(
        r.get("torch") == "2.8.0+cu128"
        and r.get("cuda") == "12.8"
        and r.get("gpu") == "NVIDIA GeForce RTX 4090",
        "unexpected recovery runtime",
    )
    require(
        json.loads(json.dumps(r.get("config")))
        == json.loads((ROOT / "configs/4stems.json").read_text()),
        "wrong model preset",
    )
    require(
        r.get("assets") == json.loads((ROOT / "runpod/training-assets.json").read_text()),
        "wrong recovery data",
    )
    require(
        all(
            isinstance(r.get(k), str) and re.fullmatch(r"[a-zA-Z0-9]+", r[k])
            for k in ("pod_id", "source_pod")
        ),
        "missing Pod identity",
    )
    require(
        r.get("cursor") == ([3, 0, 3] if stage == "export" else [8, 2, 0]), "wrong recovery cursor"
    )
    require(
        r.get("optimizer_steps") == ([3] if stage == "export" else [8])
        and r.get("weights_finite") is True,
        "optimizer/weights invalid",
    )
    expected = r.get("expected_metrics", [])
    order = [("train", i, (i - 1) // 4) for i in range(1, 5)] + [("validation", 4, 0)]
    order += [("train", i, (i - 1) // 4) for i in range(5, 9)] + [("validation", 8, 1)]
    require(
        [(x.get("kind"), x.get("step"), x.get("epoch")) for x in expected] == order,
        "incomplete reference metrics",
    )
    for x in expected:
        require(number(x.get("loss")), "nonfinite reference loss")
        if x["kind"] == "train":
            require(
                number(x.get("grad_norm")) and x.get("update_skipped") is False,
                "invalid reference gradient",
            )
    require(
        r.get("metrics") == (expected[:3] if stage == "export" else expected),
        "recovery metric history differs",
    )
    for k in ("expected_last", "expected_best"):
        hashes = r.get(k, {})
        require(
            set(hashes) == STATE_KEYS
            and all(
                isinstance(h, str) and re.fullmatch(r"[a-f0-9]{64}", h) for h in hashes.values()
            ),
            "reference state hashes missing",
        )
    if stage == "resume":
        require(
            r["pod_id"] != r["source_pod"] and r.get("inference_finite") is True,
            "same Pod or invalid inference",
        )
        require(
            r.get("actual_last") == r["expected_last"]
            and r.get("actual_best") == r["expected_best"],
            "recovered state differs",
        )
        if source is not None:
            for key in (
                "commit",
                "source_pod",
                "assets",
                "decoded",
                "expected_last",
                "expected_best",
                "expected_metrics",
            ):
                require(
                    r.get(key) == source.get(key),
                    "recovery differs from exported reference: " + key,
                )
    else:
        require(r["pod_id"] == r["source_pod"], "export Pod identity mismatch")
    require(number(r.get("seconds"), positive=True), "invalid recovery duration")
    validate_descriptor(r.get("bundle", {}), complete=stage == "resume")
    print("PASS: validated " + stage + " checkpoint and recovery evidence", flush=True)
