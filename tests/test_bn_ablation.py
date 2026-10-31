"""Intervention isolation, causal controls, metric integrity and report rejection."""

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_finetune_evaluation import modules, report  # noqa: F401
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bn(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("bn_ablation")


def test_exact_swap_and_eval_no_mutation(bn):
    a = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    b = copy.deepcopy(a)
    with torch.no_grad():
        b[0].weight.add_(0.3)
        b[1].weight.fill_(2)
        b[1].bias.fill_(0.4)
        b[1].running_mean.fill_(0.7)
        b[1].running_var.fill_(2)
        b[1].num_batches_tracked.fill_(90)
    original = {k: v.clone() for k, v in a.state_dict().items()}
    models, proof = bn.build_models(a, b)
    for i, source, stats in ((2, b, a), (3, a, b)):
        state = models[bn.VARIANTS[i]].state_dict()
        for k, t in state.items():
            expected = stats if k in proof["bn_buffer_keys"] else source
            assert torch.equal(t, expected.state_dict()[k])
    x = torch.randn(8, 3)
    before = {v: {k: t.clone() for k, t in m.state_dict().items()} for v, m in models.items()}
    assert not torch.equal(models[bn.VARIANTS[1]](x), models[bn.VARIANTS[2]](x))
    for v, m in models.items():
        assert not m.training
        m(x)
        assert all(torch.equal(t, before[v][k]) for k, t in m.state_dict().items())
    assert all(torch.equal(t, original[k]) for k, t in a.state_dict().items())
    bad = copy.deepcopy(proof)
    bad["groups"][bn.VARIANTS[2]]["non_bn"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        bn.validate_proof(bad)
    b[1].running_var.fill_(-1)
    with pytest.raises(ValueError, match="variance"):
        bn.build_models(a, b)


def test_stats_only_intervention_restores_output(bn):
    a = nn.Sequential(nn.BatchNorm1d(3))
    b = copy.deepcopy(a)
    b[0].running_mean.fill_(2)
    b[0].running_var.fill_(3)
    models, _ = bn.build_models(a, b)
    x = torch.randn(4, 3)
    assert torch.equal(models[bn.VARIANTS[0]](x), models[bn.VARIANTS[2]](x))
    assert torch.equal(models[bn.VARIANTS[1]](x), models[bn.VARIANTS[3]](x))


def test_real_scores_identity_degradation_lengths_and_interaction(bn):
    rng = np.random.default_rng(4)
    targets = {s: rng.normal(size=(2, 4410)).astype("float32") for s in bn.STEMS}
    targets["mixture"] = sum(targets.values())
    clean = {s: targets[s] + rng.normal(0, 0.01, targets[s].shape) for s in bn.STEMS}
    noisy = {s: targets[s] + rng.normal(0, 0.5, targets[s].shape) for s in bn.STEMS}
    variants = dict(zip(bn.VARIANTS, (clean, noisy, clean, noisy)))
    r = bn.score_variants(targets, variants, bn.STEMS)
    r["id"] = "synthetic"
    s = bn.summarize_pair([r])
    for stem in bn.STEMS:
        c = s["contrasts"][stem]
        assert c["learned_params_only"]["median"] == 0
        assert c["restore_stats_effect"]["median"] > 20
        assert c["interaction"]["median"] == pytest.approx(0)
    variants[bn.VARIANTS[2]]["bass"] = clean["bass"][:, :-1]
    with pytest.raises(ValueError, match="length"):
        bn.score_variants(targets, variants, bn.STEMS)


def test_full_report_and_tamper_rejection(bn, modules, report):  # noqa: F811
    _, checker = modules
    r = copy.deepcopy(report["report"])
    r["kind"] = "bn-ablation"
    r["protocol"] = bn.PROTOCOL
    r["results_directory"] = "/data/evaluations/41189465452-1/bn-example"
    r["listening"] = []
    r["bundle"]["files"] = {
        k: v for k, v in r["bundle"]["files"].items() if not k.startswith("sample-")
    }
    previous = json.loads((ROOT / "docs/results/step90-validation-41196010747.json").read_text())
    for i, rec in enumerate(r["records"]):
        rec["scores"] = {}
        rec["output_samples"] = {
            v: dict.fromkeys(bn.STEMS, rec["decoded_samples"]) for v in bn.VARIANTS
        }
        for variant in bn.VARIANTS:
            rec["scores"][variant] = {}
            for stem in bn.STEMS:
                old = previous["metrics"]["global_stereo_si_sdr_db"][stem][
                    variant if variant in bn.VARIANTS[:2] else bn.VARIANTS[0]
                ]
                value = old["median"] if i < 9 else old["mean"] * 10 - old["median"] * 9
                rec["scores"][variant][stem] = {"global_stereo_si_sdr_db": value}
    r["summary"] = bn.summarize_pair(r["records"])
    r["bn_proof"] = {
        "bn_buffer_keys": ["bn.running_mean", "bn.running_var"],
        "groups": dict(
            zip(
                bn.VARIANTS,
                [
                    {"bn": "a" * 64, "non_bn": "b" * 64},
                    {"bn": "c" * 64, "non_bn": "d" * 64},
                    {"bn": "a" * 64, "non_bn": "d" * 64},
                    {"bn": "c" * 64, "non_bn": "b" * 64},
                ],
            )
        ),
    }
    validate = lambda q: checker.validate_finetune(
        {"status": "passed", "report": q}, q["commit"], q["volume_id"], q["pod_id"], ablation=True
    )
    validate(r)
    for mutate in (
        lambda q: q["records"].pop(),
        lambda q: q["records"][0]["scores"].pop(bn.VARIANTS[2]),
        lambda q: q["records"][0].update(evaluated_samples=2),
        lambda q: q["bn_proof"]["groups"][bn.VARIANTS[2]].update(non_bn="f" * 64),
        lambda q: q["summary"]["metrics"]["vocals"]["official_converted"].update(median=100),
    ):
        bad = copy.deepcopy(r)
        mutate(bad)
        with pytest.raises((ValueError, RuntimeError)):
            validate(bad)
    changed = copy.deepcopy(r["summary"])
    changed["metrics"]["vocals"]["official_converted"]["mean"] += 0.01
    with pytest.raises(ValueError, match="reproduce"):
        bn.validate_baselines(changed, previous)
