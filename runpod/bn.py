"""Explicit entrypoint for the metric-only four-cell BN intervention."""

import argparse
import json
import os
from pathlib import Path

from finetune import run
from finetune_result import validate_finetune

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = run(ablation=True)
    a.output.write_text(json.dumps(r, indent=2, allow_nan=False))
    validate_finetune(
        {"status": "passed", "report": r},
        os.environ["SPT_COMMIT"],
        os.environ["SPT_VOLUME_ID"],
        os.environ["RUNPOD_POD_ID"],
        ablation=True,
        expected_gpu=os.environ["SPT_GPU_ID"],
    )
