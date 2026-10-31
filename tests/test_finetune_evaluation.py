"""Paired inference metrics, complete-selection rejection and export integrity."""

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    for folder in ("scripts", "runpod"):
        monkeypatch.syspath_prepend(str(ROOT / folder))
    return importlib.import_module("finetune_evaluation"), importlib.import_module(
        "finetune_result"
    )


def test_real_pair_identical_and_degraded(modules):
    ev, _ = modules
    score = importlib.import_module("musdb_evaluation").score_variants
    rng = np.random.default_rng(9)
    targets = {s: rng.normal(0, 0.03, (2, 44100)).astype(np.float32) for s in ev.STEMS}
    targets["mixture"] = sum(targets.values())
    clean = {s: targets[s] + rng.normal(0, 0.001, targets[s].shape) for s in ev.STEMS}
    variants = dict.fromkeys(ev.VARIANTS, clean)
    same = score(targets, variants, ev.STEMS, length_policy="strict", variant_names=ev.VARIANTS)
    same["id"] = "song"
    summary = ev.summarize_pair([same])
    for m in summary["metrics"].values():
        assert all(x["paired_delta"]["median"] == 0 for x in m.values())
    variants[ev.VARIANTS[1]] = {
        s: targets[s] + rng.normal(0, 0.03, targets[s].shape) for s in ev.STEMS
    }
    worse = score(targets, variants, ev.STEMS, length_policy="strict", variant_names=ev.VARIANTS)
    worse["id"] = "song"
    summary = ev.summarize_pair([worse])
    for m in summary["metrics"].values():
        assert all(x["paired_delta"]["median"] < -10 for x in m.values())
    variants[ev.VARIANTS[1]]["bass"] = variants[ev.VARIANTS[1]]["bass"][:, :-1]
    with pytest.raises(ValueError, match="length"):
        score(targets, variants, ev.STEMS, length_policy="strict", variant_names=ev.VARIANTS)


@pytest.fixture
def report(modules, monkeypatch):
    import hashlib

    ev, checker = modules
    evidence = json.loads((ROOT / "docs/results/persistent-pilot-41189465452.json").read_text())
    pinned = json.loads((ROOT / "docs/results/musdb18-training-provenance.json").read_text())
    provenance_json = json.dumps(pinned)
    evidence["resume"]["dataset"]["provenance_sha256"] = hashlib.sha256(
        provenance_json.encode()
    ).hexdigest()
    monkeypatch.setattr(checker, "load_evidence", lambda: evidence)
    selected = {
        t["id"].removesuffix(".stem.mp4"): t
        for t in pinned["tracks"]
        if t["id"].removesuffix(".stem.mp4") in pinned["split_track_ids"]["validation"]
    }
    records = []
    for track_id, t in sorted(selected.items()):
        scores = {
            v: {
                s: {
                    "global_stereo_si_sdr_db": 1.0,
                    "bsseval_v4_error": None,
                    "bsseval_v4": {
                        m: [1.0] * (t["samples"] // 44100) for m in ("sdr", "isr", "sir", "sar")
                    },
                }
                for s in ev.STEMS
            }
            for v in ev.VARIANTS
        }
        records.append(
            {
                "id": track_id,
                "input_sha256": {s: f["sha256"] for s, f in t["files"].items()},
                "decoded_samples": t["samples"],
                "evaluated_samples": t["samples"],
                "segment_start_sample": 0,
                "discarded_reference_tail_samples": 0,
                "output_samples": {v: dict.fromkeys(ev.STEMS, t["samples"]) for v in ev.VARIANTS},
                "scores": scores,
            }
        )
    listening = []
    names = {"report.json", "LISTENING.txt"} | {f"track-{i:02d}.json" for i in range(10)}
    for i, tid in enumerate(sorted(selected)[:2]):
        files = {f"sample-{i + 1}/reference-{s}.wav" for s in ("mixture", *ev.STEMS)}
        files |= {f"sample-{i + 1}/{v}-{s}.wav" for v in ev.VARIANTS for s in ev.STEMS}
        names |= files
        listening.append(
            {
                "id": tid,
                "samples": 8 * 44100,
                "start_sample": (selected[tid]["samples"] - 8 * 44100) // 2,
                "gain": 1.0,
                "files": sorted(files),
            }
        )
    return {
        "status": "passed",
        "report": {
            "kind": "finetune-validation",
            "commit": "a" * 40,
            "volume_id": evidence["volume"]["id"],
            "pod_id": "example",
            "experiment": "41189465452-1",
            "results_directory": "/data/evaluations/41189465452-1/example",
            "protocol": ev.PROTOCOL,
            "dataset": evidence["resume"]["dataset"],
            "provenance_json": provenance_json,
            "backup": evidence["resume"]["backup"],
            "initial_sha256": evidence["resume"]["initial_sha256"],
            "weights_unchanged": True,
            "torch": "2.8.0+cu128",
            "cuda": "12.8",
            "gpu": "NVIDIA GeForce RTX 4090",
            "packages": {"museval": "0.4.1"},
            "records": records,
            "summary": ev.summarize_pair(records),
            "seconds": 10,
            "peak_allocated_bytes": 1024,
            "listening": listening,
            "bundle": {
                "bytes": 100,
                "sha256": "b" * 64,
                "files": {n: {"bytes": 1, "sha256": "c" * 64} for n in names},
            },
        },
    }


def test_complete_report_recomputes_and_rejects_partial_or_swapped_data(modules, report):
    _, checker = modules
    args = ("a" * 40, "j3aqjpw393", "example")
    checker.validate_finetune(report, *args)
    mutations = [
        lambda r: r["records"].pop(),
        lambda r: r["records"][0].update(evaluated_samples=44100),
        lambda r: r["records"][0]["scores"]["official_converted"]["bass"]["bsseval_v4"][
            "sdr"
        ].pop(),
        lambda r: r["records"][0]["input_sha256"].update(bass="a" * 64),
        lambda r: r["summary"].update(track_count=9),
        lambda r: r.update(initial_sha256="b" * 64),
        lambda r: r["backup"]["files"]["last.pt"].update(sha256="b" * 64),
        lambda r: r["listening"][0].update(id=r["listening"][1]["id"]),
        lambda r: r["bundle"]["files"].update({"../unexpected": {"bytes": 1, "sha256": "a" * 64}}),
        lambda r: r.update(weights_unchanged=False),
        lambda r: r.update(provenance_json=r["provenance_json"] + " "),
    ]
    for mutate in mutations:
        changed = copy.deepcopy(report)
        mutate(changed["report"])
        with pytest.raises((ValueError, RuntimeError)):
            checker.validate_finetune(changed, *args)


def test_undefined_windows_are_counted_and_unpaired(modules, report):
    ev, _ = modules
    records = report["report"]["records"]
    score = records[0]["scores"][ev.VARIANTS[1]]["vocals"]
    score["bsseval_v4"]["sdr"] = [None] * len(score["bsseval_v4"]["sdr"])
    score["global_stereo_si_sdr_db"] = None
    summary = ev.summarize_pair(records)
    for m in summary["metrics"].values():
        assert m["vocals"]["paired_delta"]["finite"] == 9
        assert m["vocals"]["paired_delta"]["undefined"] == 1


def test_inventory_uses_verified_training_decode_not_local_wav_hash(modules):
    import hashlib

    ev, _ = modules
    pinned = json.loads((ROOT / "docs/results/musdb18-training-provenance.json").read_text())
    cloud = copy.deepcopy(pinned)
    tid = pinned["split_track_ids"]["validation"][0]
    track = next(t for t in cloud["tracks"] if t["id"].removesuffix(".stem.mp4") == tid)
    track["files"]["vocals"]["sha256"] = "a" * 64
    raw = json.dumps(cloud)
    expected_sha = hashlib.sha256(raw.encode()).hexdigest()
    inventory = ev.validation_inventory(raw, expected_sha, pinned)
    assert inventory[tid]["files"]["vocals"]["sha256"] == "a" * 64
    # An unverified new decode, source substitution or sample loss must still fail.
    with pytest.raises(ValueError, match="hash"):
        ev.validation_inventory(raw + " ", expected_sha, pinned)
    for field in ("source_sha256", "samples"):
        bad = copy.deepcopy(cloud)
        bad["tracks"][0][field] = "wrong"
        raw = json.dumps(bad)
        with pytest.raises(ValueError, match="source inventory"):
            ev.validation_inventory(raw, hashlib.sha256(raw.encode()).hexdigest(), pinned)


def test_export_tampering_and_duplicate_members(modules, tmp_path):
    import hashlib
    import io
    import tarfile

    _, checker = modules
    path = tmp_path / "export.tar"
    content = b'{"records": []}'
    report = {
        "records": [],
        "bundle": {
            "files": {
                "report.json": {
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            }
        },
    }

    def write(duplicate=False):
        with tarfile.open(path, "w") as tar:
            for _ in range(2 if duplicate else 1):
                member = tarfile.TarInfo("report.json")
                member.size = len(content)
                tar.addfile(member, io.BytesIO(content))
        report["bundle"].update(
            bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )

    write()
    checker.validate_export(path, report)
    write(duplicate=True)
    with pytest.raises(RuntimeError, match="inventory"):
        checker.validate_export(path, report)
    write()
    path.write_bytes(path.read_bytes()[:-1] + b"x")
    with pytest.raises(RuntimeError, match="checksum"):
        checker.validate_export(path, report)


def test_three_arm_projection_preserves_pair_direction_and_rejects_partial(modules, report):
    frozen = importlib.import_module("frozen_bn_evaluation")
    ev, _ = modules
    r = report["report"]
    records = copy.deepcopy(r["records"])
    pinned = json.loads((ROOT / "docs/results/musdb18-training-provenance.json").read_text())
    expected = ev.validation_inventory(
        r["provenance_json"], r["dataset"]["provenance_sha256"], pinned
    )
    for record in records:
        base = record["scores"]["official_converted"]
        record["scores"] = {}
        for variant, value in zip(frozen.VARIANTS, (1.0, -8.0, 2.0)):
            record["scores"][variant] = copy.deepcopy(base)
            for stem in record["scores"][variant].values():
                stem["global_stereo_si_sdr_db"] = value
                stem["bsseval_v4"]["sdr"] = [value] * len(stem["bsseval_v4"]["sdr"])
        lengths = record["output_samples"]["official_converted"]
        record["output_samples"] = {v: dict(lengths) for v in frozen.VARIANTS}
    result = frozen.summarize(records, expected)
    for pair, delta in [
        ("normal_vs_official", -9),
        ("frozen_vs_official", 1),
        ("frozen_vs_normal", 10),
    ]:
        for metric in result[pair]["metrics"].values():
            assert all(stem["paired_delta"]["median"] == delta for stem in metric.values())
    with pytest.raises(ValueError):
        frozen.summarize(records[:-1], expected)
    records[0]["output_samples"]["frozen_step90"]["bass"] -= 1
    with pytest.raises(ValueError):
        frozen.summarize(records, expected)
