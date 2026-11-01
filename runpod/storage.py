"""One bounded, explicitly named network volume; never deletes persistent storage."""

import re
import urllib.parse

NAME = "spt-1436489970-musdb18-v1"
SIZE = 80


def valid_volume(v):
    if (
        v.get("name") != NAME
        or v.get("size") != SIZE
        or v.get("type") != "STANDARD"
        or not re.fullmatch(r"[a-zA-Z0-9]+", v.get("id", ""))
        or not re.fullmatch(r"[A-Z0-9-]+", v.get("dataCenter", ""))
    ):
        raise RuntimeError("Persistent volume identity, size or tier mismatch")
    return v


def volumes(api):
    result, seen, path = [], set(), "/network-volumes"
    while True:
        page = api.call(path)
        result.extend(page["networkVolumes"])
        if page["pagination"]["hasNextPage"] is False:
            return result
        cursor = page["pagination"].get("nextCursor")
        if not isinstance(cursor, str) or not cursor or cursor in seen:
            raise RuntimeError("Invalid volume pagination")
        seen.add(cursor)
        path = "/network-volumes?" + urllib.parse.urlencode({"cursor": cursor})


def ensure_volume(api, *, create=False):
    from control import GPU

    matches = [v for v in volumes(api) if v.get("name") == NAME]
    if len(matches) > 1:
        raise RuntimeError("Duplicate persistent volumes; refusing ambiguous selection")
    if matches:
        return valid_volume(matches[0])
    if not create:
        raise RuntimeError("Run persistent-prepare first")
    gpu = api.call(
        "/catalog/gpus/"
        + urllib.parse.quote(GPU, safe="")
        + "?include=AVAILABILITY&product=POD&cloud=SECURE&count=1&minCudaVersion=12.8"
    )
    dcs = api.call("/catalog/datacenters?networkVolumeTypes=STANDARD")["dataCenters"]
    supported = {d["id"] for d in dcs if "STANDARD" in d.get("networkVolumeTypes", [])}
    choices = [
        d
        for d in gpu.get("dataCenters", [])
        if d["id"] in supported and d.get("availability") in {"HIGH", "MEDIUM", "LOW"}
    ]
    choices.sort(key=lambda d: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}[d["availability"]], d["id"]))
    if not choices:
        raise RuntimeError("No compatible standard-volume datacenter has a 4090")
    # Exactly one POST. A lost response is recovered by name on a later invocation.
    return valid_volume(
        api.call(
            "/network-volumes",
            "POST",
            {"name": NAME, "size": SIZE, "type": "STANDARD", "dataCenter": choices[0]["id"]},
        )
    )


def safe_mounts(pod):
    """Only the explicitly declared network mount may survive deleting this Pod."""
    mounts = {k: v for k, v in (pod.get("mounts") or {}).items() if v}
    env = pod.get("env", {})
    vid = env.get("SPT_VOLUME_ID", "")
    if env.get("SPT_GLOBAL_VOLUME_ID"):
        from global_storage import NAME as GLOBAL_NAME
        from global_storage import SOURCE_ID, valid_id

        gid = env["SPT_GLOBAL_VOLUME_ID"]
        if env.get("SPT_GLOBAL_VOLUME_NAME") != GLOBAL_NAME or not valid_id(gid):
            return False
        expected = {"global": [{"volumeId": gid, "path": "/global"}]}
        if env.get("SPT_TASK") in {"global-export", "global-training-export"}:
            if vid != SOURCE_ID or env.get("SPT_VOLUME_NAME") != NAME:
                return False
            expected["network"] = [{"volumeId": SOURCE_ID, "path": "/data"}]
        elif env.get("SPT_TASK") not in {"bn", "frozen", "benchmark"} or vid != gid:
            return False
        return mounts == expected
    return (
        env.get("SPT_VOLUME_NAME") == NAME
        and bool(re.fullmatch(r"[a-zA-Z0-9]+", vid))
        and mounts == {"network": [{"volumeId": vid, "path": "/data"}]}
    )
