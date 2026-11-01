"""Separate native two-source protocol; never relabel historical four-source scores."""

import json

from benchmark_metrics import PROTOCOL as FOUR_STEM_PROTOCOL
from benchmark_metrics import RATE, REPEATS  # noqa: F401 -- shared public profile constants
from benchmark_metrics import summarize as common_summary
from roformer_kim import RECIPE, SOURCE_FILES, WEIGHT_SHA256

STEMS = ("vocals", "accompaniment")
VARIANTS = ("official_2stems", "melband_kim")
WEIGHTS = {
    "official_2stems": "2032ba2a17765f80e02051ecd0b7a5dd1552a077221275615227c2652f873532",
    "melband_kim": WEIGHT_SHA256,
}
DEPENDENCIES = {
    "numpy": "2.0.2",
    "scipy": "1.15.3",
    "soundfile": "0.13.1",
    "einops": "0.8.1",
    "rotary-embedding-torch": "0.8.9",
    "beartype": "0.18.5",
    "librosa": "0.11.0",
    "numba": "0.61.2",
    "pyyaml": "6.0.2",
    "museval": "0.4.1",
}
PROTOCOL = {
    **{
        k: v
        for k, v in FOUR_STEM_PROTOCOL.items()
        if k not in ("htdemucs", "baseline_tolerance_db")
    },
    "task": "native vocal/accompaniment; independent two-source screen",
    "stems": list(STEMS),
    "variants": list(VARIANTS),
    "spleeter": "native 2 stems, 512 frames, 1024 bins, zero high-band extension",
    "roformer": RECIPE,
    "roformer_source_files": SOURCE_FILES,
    "dependencies": DEPENDENCIES,
    "reference_accompaniment": "float32 drums + bass + other, in that order; not mixture minus vocals",
    "predicted_accompaniment": {
        "official_2stems": "native predicted mask",
        "melband_kim": "mixture minus predicted vocals",
    },
    "memory": "CUDA reset peaks including resident model; CPU /proc/self/status VmHWM after exec",
    "baseline": "new two-source baseline; authenticated official weights and prior native-2 TF/PT parity, no four-source score gate",
    "load_timing": "includes model-specific dynamic source imports and checksum validation, cache uncontrolled",
}
PROTOCOL = json.loads(json.dumps(PROTOCOL))


def summarize(records):
    return common_summary(records, variants=VARIANTS, stems=STEMS)
