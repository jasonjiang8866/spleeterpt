"""One named, bounded-use global volume and read-only compute placement."""

import math
import re
import urllib.parse

NAME = "spt-1436489970-step90-inputs-v1"
SOURCE_ID = "j3aqjpw393"
CPU_RATE_CAP = 0.20


def valid_id(value):
    # REST v2 specifies an opaque string, not the `gsb_` prefix used in examples.
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value)


def ensure_global(api, *, create=False):
    matches, seen, path = [], set(), "/global-volumes"
    while True:
        page = api.call(path)
        matches.extend(v for v in page["globalVolumes"] if v.get("name") == NAME)
        if page["pagination"]["hasNextPage"] is False:
            break
        cursor = page["pagination"].get("nextCursor")
        if not isinstance(cursor, str) or not cursor or cursor in seen:
            raise RuntimeError("Invalid global-volume pagination")
        seen.add(cursor)
        path = "/global-volumes?" + urllib.parse.urlencode({"cursor": cursor})
    if len(matches) > 1:
        raise RuntimeError("Duplicate global volumes; refusing ambiguous selection")
    if not matches:
        if not create:
            raise RuntimeError("Run global-export first")
        # No automatic POST retry: a lost response is recovered by name next time.
        matches = [api.call("/global-volumes", "POST", {"name": NAME})]
    volume = matches[0]
    if volume.get("name") != NAME or not valid_id(volume.get("id")):
        raise RuntimeError("Global volume identity mismatch")
    return volume


def cpu_placement(api, region):
    catalog = api.call("/catalog/cpus?include=AVAILABILITY&product=POD&vcpuCount=2")
    choices = []
    for cpu in catalog["cpus"]:
        rate = cpu["price"]["securePerVcpu"] * 2
        if (
            isinstance(rate, (int, float))
            and math.isfinite(rate)
            and 0 < rate <= CPU_RATE_CAP
            and cpu["vcpu"]["min"] <= 2 <= cpu["vcpu"]["max"]
            and any(
                d.get("id") == region and d.get("availability") in {"HIGH", "MEDIUM", "LOW"}
                for d in cpu.get("dataCenters", [])
            )
        ):
            choices.append((rate, cpu["id"]))
    if not choices:
        raise RuntimeError("No CPU capacity within $0.20/hour at the source volume")
    rate, cpu_id = min(choices)
    print(f"CPU copy placement: {region}; catalog rate=${rate}/hour", flush=True)
    return {"id": cpu_id, "vcpuCount": 2}


def mig_regions(api, gpu_id):
    gpu = api.call(
        "/catalog/gpus/"
        + urllib.parse.quote(gpu_id, safe="")
        + "?include=AVAILABILITY&product=POD&cloud=SECURE&count=1&minCudaVersion=12.8"
    )
    rate = gpu.get("price", {}).get("secure")
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 < rate <= 0.69:
        raise RuntimeError("MIG catalog price missing or above $0.69/hour")
    regions = sorted(
        d["id"]
        for d in gpu.get("dataCenters", [])
        if d.get("availability") in {"HIGH", "MEDIUM", "LOW"}
        and isinstance(d.get("id"), str)
        and re.fullmatch(r"[A-Z]{2}-[A-Z0-9]+-[0-9]+", d["id"])
    )
    if not regions:
        raise RuntimeError("No available MIG region; no Pod created")
    print("Global-volume MIG placement: " + ",".join(regions), flush=True)
    return regions
