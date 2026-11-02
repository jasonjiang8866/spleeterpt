"""Read-only inventory; quotes do not authorize or provision a GPU."""

import json
import re
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from benchmark_preflight import ReadOnly
from control import Runpod, identity
from gpu_profile import ADA, MIG24


def main():
    identity()
    api = ReadOnly(Runpod())
    quotes = []
    for gpu in (
        MIG24,
        ADA,
        "NVIDIA GeForce RTX 3090",
        "NVIDIA GeForce RTX 5090",
        "NVIDIA RTX A6000",
    ):
        item = api.call(
            "/catalog/gpus/"
            + urllib.parse.quote(gpu, safe="")
            + "?include=AVAILABILITY&product=POD&cloud=SECURE&count=1&minCudaVersion=12.8"
        )
        quotes.append(
            {
                "gpu": gpu,
                "price": item.get("price", {}).get("secure"),
                "availability": item.get("availability")
                if item.get("availability") in {"HIGH", "MEDIUM", "LOW", "NONE"}
                else "UNKNOWN",
                "catalog_region_count": len(item.get("dataCenters", [])),
                "availability_labels": sorted(
                    {str(d.get("availability"))[:40] for d in item.get("dataCenters", [])}
                ),
                "available_regions": sorted(
                    d["id"]
                    for d in item.get("dataCenters", [])
                    if isinstance(d.get("id"), str)
                    and re.fullmatch(r"[A-Z]{2}-[A-Z0-9]+-[0-9]+", d["id"])
                    and d.get("availability") in {"HIGH", "MEDIUM", "LOW"}
                ),
            }
        )
    report = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "account_pod_count": len(api.pods()),
        "quotes": quotes,
    }
    out = Path("runpod-results")
    out.mkdir(exist_ok=True)
    (out / "five-stem-capacity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 -- provider data may contain credentials
        print(f"ERROR: {type(error).__name__}; read-only capacity inventory incomplete")
        raise SystemExit(1) from None
