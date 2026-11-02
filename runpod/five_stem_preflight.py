"""Read-only five-stem resource inventory; never creates compute or storage."""

import json
from pathlib import Path

from benchmark_preflight import observe
from control import IMAGE, Runpod, identity


def main():
    identity()
    resources = observe(Runpod())
    resources["pending"] = [
        "Explicit project-specific paid-run budget approval",
        "Official five-stem CPU diagnosis and frozen acceptance protocol",
        "Mounted input/checkpoint hashes and image runtime verification",
        "GPU worker, deadline, independent result validation and recovery check",
    ]
    report = {
        "profile": "five-stem-validation",
        "resources": resources,
        "configured_image": IMAGE,
        "image_status": "configured tag only; not a fresh pull or digest verification",
        "paid_compute_started": False,
    }
    output = Path("runpod-results")
    output.mkdir(exist_ok=True)
    (output / "five-stem-preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 -- never expose provider credentials
        print(f"ERROR: {type(error).__name__}; read-only inventory incomplete")
        raise SystemExit(1) from None
