"""Predeclared matched 90-step BN training comparison on one GPU."""

from finetune_evaluation import PROTOCOL as BASE_PROTOCOL
from finetune_evaluation import summarize_pair, validate_records

VARIANTS = ("official_converted", "normal_step90", "frozen_step90")
PAIRS = {
    "normal_vs_official": (VARIANTS[0], VARIANTS[1]),
    "frozen_vs_official": (VARIANTS[0], VARIANTS[2]),
    "frozen_vs_normal": (VARIANTS[1], VARIANTS[2]),
}
PROTOCOL = {
    **BASE_PROTOCOL,
    "variants": list(VARIANTS),
    "training": "two independent 90-step runs from identical official initialization on the same MIG",
    "intervention": "freeze BN running buffers; train affine parameters; preserve training dropout",
    "recipe": {
        "epochs": 1,
        "batch_size": 1,
        "crops_per_track": 1,
        "validation_crops": 1,
        "workers": 0,
        "lr": 1e-5,
        "seed": 17,
        "grad_clip": 5.0,
        "optimizer": "Adam eps=1e-8",
        "save_every": 10,
        "deterministic": True,
    },
    "baseline_tolerance_db": 0.001,
}


def projected(records, baseline, candidate):
    return [
        {
            **r,
            **{
                key: {"official_converted": r[key][baseline], "finetuned_step90": r[key][candidate]}
                for key in ("scores", "output_samples")
            },
        }
        for r in records
    ]


def summarize(records, expected=None):
    result = {}
    if any(set(r["scores"]) != set(VARIANTS) for r in records):
        raise ValueError("Incorrect matched trial variants")
    for name, pair in PAIRS.items():
        subset = projected(records, *pair)
        if expected is not None:
            validate_records(subset, expected)
        result[name] = summarize_pair(subset)
    return result
