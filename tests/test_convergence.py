"""Reject partial training, altered continuation, selective scores and unsafe exports."""

import copy
import hashlib
import importlib
import json
import random
from pathlib import Path

import pytest
from test_finetune_evaluation import modules, report  # noqa: F401 -- shared full-song fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def convergence(report, monkeypatch):  # noqa: F811 -- shared pytest fixture
    checker = importlib.import_module("convergence_result")
    protocol = importlib.import_module("convergence_evaluation")
    r = report["report"]
    pinned = json.loads(r["provenance_json"])
    ev = copy.deepcopy(checker.evidence())
    ev["resume"]["dataset"] = r["dataset"]
    monkeypatch.setattr(checker, "evidence", lambda: ev)
    # Synthetic metric fixture tests all report checks except reproduction, tested separately.
    monkeypatch.setattr(checker, "validate_baseline", lambda summary: None)
    r.update(
        kind="convergence-pilot",
        protocol=protocol.PROTOCOL,
        complete=True,
        matched_rng=True,
        museval="0.4.1",
        initial_bn_fingerprint="a" * 64,
        storage={
            "global_volume_id": "global",
            "snapshot_sha256": protocol.SNAPSHOT_SHA256,
            "verified_audio_files": 500,
        },
        gpu_runtime={
            "requested_gpu": checker.MIG24,
            "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb",
            "device_count": 1,
            "capability": [12, 0],
            "total_memory_bytes": 24 * 1024**3,
        },
        budget_projections=[
            {"accepted": True, "projected_seconds": 2000, "remaining_seconds": 2500}
        ],
        arms={},
        curves={},
    )
    for arm, lr in protocol.ARMS.items():
        history, proofs = [], []
        for epoch in range(10):
            history.extend(
                {
                    "kind": "train",
                    "epoch": epoch,
                    "step": step,
                    "loss": 5.0,
                    "grad_norm": 2.0,
                    "update_skipped": False,
                }
                for step in range(epoch * 90 + 1, (epoch + 1) * 90 + 1)
            )
            step = (epoch + 1) * 90
            history.append({"kind": "validation", "epoch": epoch, "step": step, "loss": 4.0})
            if step not in protocol.MILESTONES:
                continue
            files = {
                n: {"bytes": 10, "sha256": hashlib.sha256(f"{arm}/{step}/{n}".encode()).hexdigest()}
                for n in (
                    "last.pt",
                    "config.json",
                    "arguments.json",
                    "metrics.jsonl",
                    "timing.json",
                )
            }
            proof = {
                "cursor": [step, epoch + 1, 0],
                "optimizer_steps": [step],
                "finite": True,
                "bn_buffers_unchanged": True,
                "bn_affine_changed": True,
                "parameters_changed": True,
                "bn_fingerprint": r["initial_bn_fingerprint"],
                "signature": {
                    k: v
                    for k, v in protocol.PROTOCOL["recipe"].items()
                    if k not in ("epochs", "workers", "optimizer", "save_every")
                },
                "timing": {"seconds": 30, "peak_allocated_bytes": 1024},
                "metrics": copy.deepcopy(history),
                "global_restore_equal": True,
                "resume_sha256": proofs[-1]["checkpoint"]["files"]["last.pt"]["sha256"]
                if proofs
                else None,
                "checkpoint": {
                    "prefix": f"convergence-runs/{r['experiment']}/{arm}/{step}",
                    "files": files,
                    "readback_verified": True,
                    "manifest_sha256": hashlib.sha256(
                        checker.encoded({"files": files})
                    ).hexdigest(),
                },
            }
            proof["signature"].update(lr=lr, train="b" * 64, validation="c" * 64)
            proofs.append(proof)
        r["arms"][arm] = proofs
    for record in r["records"]:
        for key in ("scores", "output_samples"):
            baseline = record[key]["official_converted"]
            record[key] = {v: copy.deepcopy(baseline) for v in protocol.VARIANTS}
    for label in {protocol.VARIANTS[0]} | {
        f"{a}_step{s}" for a in protocol.ARMS for s in protocol.MILESTONES
    }:
        r["curves"][label] = {
            "seconds": 10,
            "peak_allocated_bytes": 1024,
            "tracks": [
                {
                    "id": t["id"],
                    "samples": t["decoded_samples"],
                    "scores": dict.fromkeys(checker.STEMS, 1.0),
                }
                for t in r["records"]
            ],
        }
    selected = {
        t["id"].removesuffix(".stem.mp4"): t
        for t in pinned["tracks"]
        if t["id"].removesuffix(".stem.mp4") in pinned["split_track_ids"]["validation"]
    }
    r["summary"] = protocol.summarize(r["records"], selected)
    tracks = {t["id"].removesuffix(".stem.mp4"): t for t in pinned["tracks"]}
    coverage = {"epochs": 10, "crop_samples": 527360, "tracks": []}
    for i, tid in enumerate(pinned["split_track_ids"]["train"]):
        n = tracks[tid]["samples"]
        starts = [random.Random(f"17/{e}/{i}").randint(0, max(0, n - 527360)) for e in range(10)]
        end, count = 0, 0
        for start in sorted(starts):
            stop = min(n, start + 527360)
            count += max(0, stop - max(start, end))
            end = max(end, stop)
        coverage["tracks"].append(
            {"id": tid, "samples": n, "starts": starts, "covered_samples": count}
        )
    r["coverage"] = coverage
    names = {"report.json", "LISTENING.txt"} | {f"track-{i:02d}.json" for i in range(10)}
    for i, sample in enumerate(r["listening"]):
        clips = {f"sample-{i + 1}/reference-{s}.wav" for s in ("mixture", *checker.STEMS)}
        clips |= {f"sample-{i + 1}/{v}-{s}.wav" for v in protocol.VARIANTS for s in checker.STEMS}
        sample["files"] = sorted(clips)
        names |= clips
    r["bundle"]["files"] = {n: {"bytes": 1, "sha256": "d" * 64} for n in names}
    return checker, report


def test_complete_report_and_fail_closed_mutations(convergence):
    checker, result = convergence
    args = ("a" * 40, "global", "example", "41189465452-1")
    checker.validate(result, *args)
    mutations = [
        lambda r: r.update(complete=False),
        lambda r: r["arms"]["lr1e5"].pop(),
        lambda r: r["arms"]["lr1e5"][1].update(resume_sha256="0" * 64),
        lambda r: r["arms"]["lr1e5"][1]["signature"].update(lr=3e-6),
        lambda r: r["arms"]["lr1e5"][2]["cursor"].__setitem__(0, 899),
        lambda r: r["arms"]["lr1e5"][2]["metrics"][0].update(loss=float("nan")),
        lambda r: r["arms"]["lr1e5"][0].update(global_restore_equal=False),
        lambda r: r["arms"]["lr3e6"][2]["signature"].update(train="e" * 64),
        lambda r: r["curves"]["lr1e5_step450"]["tracks"].pop(),
        lambda r: r["curves"]["lr1e5_step450"]["tracks"][0]["scores"].update(bass=float("inf")),
        lambda r: r["curves"]["lr1e5_step900"]["tracks"][0]["scores"].update(bass=2.0),
        lambda r: r["coverage"]["tracks"][0]["starts"].__setitem__(0, 0),
        lambda r: r["records"].pop(),
        lambda r: r["bundle"].update(bytes=101 * 1024**2),
        lambda r: r["budget_projections"][0].update(projected_seconds=3000),
    ]
    for mutate in mutations:
        changed = copy.deepcopy(result)
        mutate(changed["report"])
        with pytest.raises(ValueError):
            checker.validate(changed, *args)


def test_official_reproduction_gate(modules):  # noqa: F811 -- shared pytest fixture
    checker = importlib.import_module("convergence_result")
    old = json.loads((ROOT / "docs/results/step90-validation-41196010747.json").read_text())
    summary = {"candidate": {"metrics": old["metrics"]}}
    checker.validate_baseline(summary)
    summary["candidate"]["metrics"]["bsseval_v4_sdr_db"]["bass"]["official_converted"][
        "median"
    ] += 0.01
    with pytest.raises(ValueError, match="reproduction"):
        checker.validate_baseline(summary)


def test_global_restore_requires_real_copy_and_rejects_corruption(modules, tmp_path, monkeypatch):  # noqa: F811
    worker = importlib.import_module("convergence")
    root = tmp_path / "global"
    monkeypatch.setattr(worker, "Path", lambda x: root if x == "/global" else Path(x))
    source = root / "test"
    source.mkdir(parents=True)
    raw = b"checkpoint test bytes"
    (source / "last.pt").write_bytes(raw)
    files = {"last.pt": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
    (source / "complete.json").write_bytes(worker.encoded({"files": files}))
    descriptor = {
        "prefix": "test",
        "files": files,
        "manifest_sha256": worker.digest(source / "complete.json"),
    }
    restored = worker.restore(descriptor, tmp_path / "restore")
    assert restored.read_bytes() == raw
    with pytest.raises(FileExistsError):
        worker.restore(descriptor, tmp_path / "restore")
    (source / "last.pt").write_bytes(b"corrupted")
    with pytest.raises((ValueError, RuntimeError)):
        worker.restore(descriptor, tmp_path / "restore2")


def test_finite_zero_step_is_accepted_without_relabeling_failed_worker(convergence):
    checker, result = convergence
    r = result["report"]
    for proofs in r["arms"].values():
        entry = next(m for m in proofs[-1]["metrics"] if m["kind"] == "train" and m["step"] == 598)
        entry.update(loss=0.0, grad_norm=0.0)
    args = ("a" * 40, "global", "example", "41189465452-1")
    checker.validate(result, *args)
    result["status"] = "failed"
    checker.validate_report(r, *args)
    with pytest.raises(ValueError, match="Worker did not report success"):
        checker.validate(result, *args)
    for value in (-1.0, float("nan"), float("inf"), True):
        entry["grad_norm"] = value
        with pytest.raises(ValueError, match="training history"):
            checker.validate_report(r, *args)
