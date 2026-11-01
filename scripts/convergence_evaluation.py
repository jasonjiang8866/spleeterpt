"""Predeclared longer-training screen; no claim of convergence from step count."""

from benchmark_metrics import summarize as summarize_pair
from finetune_evaluation import PROTOCOL as BASE_PROTOCOL
from finetune_evaluation import STEMS, validate_records

ARMS = {"lr1e5": 1e-5, "lr3e6": 3e-6}
MILESTONES = (90, 450, 900)
VARIANTS = ("official_converted", *(f"{a}_step900" for a in ARMS))
SNAPSHOT_SHA256 = "172c50e5310b22ceab3d4813a4ba1146c682db682c8aebc4c03ad235de376964"
PROTOCOL = {
    **BASE_PROTOCOL,
    "variants": list(VARIANTS),
    "training": "two independent official-initialized frozen-BN arms; longer-training screen",
    "objective": "sum of per-stem mean L1 magnitude error; original Spleeter port objective",
    "milestones": list(MILESTONES),
    "learning_rates": ARMS,
    "recipe": {
        "epochs": 10,
        "batch_size": 1,
        "crops_per_track": 1,
        "validation_crops": 4,
        "workers": 0,
        "seed": 17,
        "grad_clip": 5.0,
        "optimizer": "Adam eps=1e-8",
        "save_every": 90,
        "deterministic": True,
        "freeze_bn_stats": True,
        "precision": "fp32",
    },
    "selection": "fixed step900 endpoints; validation loss does not select or promote weights",
    "curves": "global stereo SI-SDR on all ten full songs at 0/90/450/900; BSSEval only 0/900",
    "resume": "fresh process at each milestone; preceding full checkpoint read back from Global",
    "baseline_tolerance_db": 0.001,
    "convergence_established": False,
}


def summarize(records, expected):
    validate_records(records, expected, variants=VARIANTS)
    result = {}
    for candidate in VARIANTS[1:]:
        pair = (VARIANTS[0], candidate)
        subset = [
            {**r, **{k: {v: r[k][v] for v in pair} for k in ("scores", "output_samples")}}
            for r in records
        ]
        result[candidate] = summarize_pair(subset, variants=pair, stems=STEMS)
    return result
