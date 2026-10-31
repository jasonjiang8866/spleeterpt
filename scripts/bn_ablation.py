"""Four-cell BN-buffer intervention; full-song SI-SDR, without BSSEval scoring."""

import copy
import hashlib
import math

from finetune_evaluation import PROTOCOL as PAIRED_PROTOCOL
from finetune_evaluation import STEMS
from musdb_evaluation import decode_number, encode_number, statistics, validate_outputs

VARIANTS = (
    "official_converted",
    "finetuned_step90",
    "learned_official_bn",
    "official_learned_bn",
)
PROTOCOL = {
    **{
        k: v
        for k, v in PAIRED_PROTOCOL.items()
        if k
        not in {
            "bsseval",
            "window_samples",
            "hop_samples",
            "padding",
            "aggregation",
            "listening",
        }
    },
    "aggregation": "one global stereo SI-SDR per full song; median songs per stem",
    "bsseval": "not computed in this diagnostic",
    "listening": "none; metric-only ablation",
    "intervention": "swap only BN running_mean, running_var and num_batches_tracked; affine stays with parameters",
}
COMPARISONS = {
    "tuned_minus_official": (1, 0),
    "learned_params_only": (2, 0),
    "learned_stats_only": (3, 0),
    "restore_stats_effect": (2, 1),
    "learned_params_at_learned_stats": (1, 3),
}


def build_models(official, learned):
    import torch
    from torch import nn

    a, b = official.state_dict(), learned.state_dict()
    if a.keys() != b.keys() or any(a[k].shape != b[k].shape or a[k].dtype != b[k].dtype for k in a):
        raise ValueError("Incompatible model states")
    keys = set()
    for name, m in official.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            other = dict(learned.named_modules()).get(name)
            if (
                type(m) is not type(other)
                or not m.track_running_stats
                or not other.track_running_stats
            ):
                raise ValueError("Incompatible BN modules")
            if (m.eps, m.momentum, m.affine) != (other.eps, other.momentum, other.affine):
                raise ValueError("Incompatible BN configuration")
            keys.update(
                f"{name}.{k}" if name else k
                for k in ("running_mean", "running_var", "num_batches_tracked")
            )
    if not keys or not keys <= a.keys():
        raise ValueError("Missing BN buffers")
    if any(not torch.isfinite(t).all() for state in (a, b) for t in state.values()):
        raise ValueError("Nonfinite model state")
    if any((state[k] < 0).any() for state in (a, b) for k in keys if k.endswith("running_var")):
        raise ValueError("Negative BN variance")
    models = dict(zip(VARIANTS[:2], (official, learned)))
    for name, params, stats, template in (
        (VARIANTS[2], b, a, learned),
        (VARIANTS[3], a, b, official),
    ):
        m = copy.deepcopy(template)
        m.load_state_dict({k: stats[k] if k in keys else params[k] for k in a}, strict=True)
        models[name] = m

    def fingerprint(state, subset):
        h = hashlib.sha256()
        for k in sorted(subset):
            t = state[k].detach().cpu().contiguous()
            h.update(f"{k}:{t.dtype}:{tuple(t.shape)}:".encode())
            h.update(t.numpy().tobytes())
        return h.hexdigest()

    proof = {"bn_buffer_keys": sorted(keys), "groups": {}}
    for name, m in models.items():
        m.eval()
        proof["groups"][name] = {
            "bn": fingerprint(m.state_dict(), keys),
            "non_bn": fingerprint(m.state_dict(), set(a) - keys),
        }
    validate_proof(proof)
    return models, proof


def validate_proof(proof):
    import re

    keys, groups = proof["bn_buffer_keys"], proof["groups"]
    if not keys or keys != sorted(set(keys)) or set(groups) != set(VARIANTS):
        raise ValueError("Invalid BN proof inventory")
    if any(not k.endswith((".running_mean", ".running_var", ".num_batches_tracked")) for k in keys):
        raise ValueError("Unexpected BN buffer")
    for g in groups.values():
        if set(g) != {"bn", "non_bn"} or any(
            not re.fullmatch(r"[a-f0-9]{64}", v) for v in g.values()
        ):
            raise ValueError("Invalid group hash")
    a, b, c, d = (groups[v] for v in VARIANTS)
    if not (
        c == {"bn": a["bn"], "non_bn": b["non_bn"]} and d == {"bn": b["bn"], "non_bn": a["non_bn"]}
    ):
        raise ValueError("Hybrid provenance mismatch")


def score_variants(targets, variants, instruments, **kwargs):
    from evaluate_preview import si_sdr

    n, lengths = validate_outputs(targets, variants, instruments, 1024, "strict", VARIANTS)
    return {
        "evaluated_samples": n,
        "discarded_reference_tail_samples": 0,
        "output_samples": lengths,
        "scores": {
            v: {
                s: {"global_stereo_si_sdr_db": encode_number(si_sdr(targets[s], stems[s]))}
                for s in STEMS
            }
            for v, stems in variants.items()
        },
    }


def summarize_pair(records):
    result = {"track_count": len(records), "metrics": {}, "contrasts": {}}
    for stem in STEMS:
        values = {
            v: [decode_number(r["scores"][v][stem]["global_stereo_si_sdr_db"]) for r in records]
            for v in VARIANTS
        }
        result["metrics"][stem] = {
            v: statistics([encode_number(x) for x in xs]) for v, xs in values.items()
        }
        pairs = {}
        for name, (i, j) in COMPARISONS.items():
            xs = [
                x - y if math.isfinite(x) and math.isfinite(y) else None
                for x, y in zip(values[VARIANTS[i]], values[VARIANTS[j]])
            ]
            pairs[name] = {
                **statistics(xs),
                "better": sum(x is not None and x > 0 for x in xs),
                "worse": sum(x is not None and x < 0 for x in xs),
                "equal": sum(x == 0 for x in xs),
                "per_track": dict(zip([r["id"] for r in records], xs)),
            }
        # Interaction = (learned params effect at learned stats) - (at official stats).
        xs = [
            b - c - d + a if all(math.isfinite(x) for x in (a, b, c, d)) else None
            for a, b, c, d in zip(*(values[v] for v in VARIANTS))
        ]
        pairs["interaction"] = statistics(xs)
        result["contrasts"][stem] = pairs
    return result


def validate_records(records, expected):
    if len(expected) != 10 or [r["id"] for r in records] != sorted(expected):
        raise ValueError("Validation selection mismatch")
    for r in records:
        spec = expected[r["id"]]
        n = spec["samples"]
        if (
            r["input_sha256"] != {s: x["sha256"] for s, x in spec["files"].items()}
            or r["decoded_samples"] != n
            or r["evaluated_samples"] != n
            or r["segment_start_sample"] != 0
            or r["discarded_reference_tail_samples"] != 0
            or r["output_samples"] != {v: dict.fromkeys(STEMS, n) for v in VARIANTS}
            or set(r["scores"]) != set(VARIANTS)
        ):
            raise ValueError("Ablation identity or length mismatch")
        for scores in r["scores"].values():
            if set(scores) != set(STEMS):
                raise ValueError("Missing stem")
            for x in scores.values():
                if set(x) != {"global_stereo_si_sdr_db"}:
                    raise ValueError("Unexpected metric")
                decode_number(x["global_stereo_si_sdr_db"])


def validate_baselines(summary, previous, *, tolerance=1e-5):
    if tolerance not in (1e-5, 1e-3):
        raise ValueError("Unapproved baseline tolerance")
    for stem in STEMS:
        for variant in VARIANTS[:2]:
            actual = summary["metrics"][stem][variant]
            old = previous["metrics"]["global_stereo_si_sdr_db"][stem][variant]
            if any(
                actual[k] != old[k]
                for k in ("total", "finite", "undefined", "positive_infinity", "negative_infinity")
            ):
                raise ValueError("Baseline coverage changed")
            for k in ("median", "mean"):
                if not math.isclose(
                    decode_number(actual[k]), decode_number(old[k]), abs_tol=tolerance, rel_tol=0
                ):
                    raise ValueError("Baseline SI-SDR does not reproduce prior run")
