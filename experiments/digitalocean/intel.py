"""Create one price-checked Intel VM, or reconcile that run's temporary VM."""

import ipaddress
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SIZE = "s-4vcpu-8gb-intel"
RATE_LIMIT = 0.16
STATE = Path(os.environ["RUNNER_TEMP"]) / "spt-do"
OUT = Path("do-results")


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def api(path, method="GET", data=None):
    token = os.environ.get("DIGITALOCEAN_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Repository secret DIGITALOCEAN_ACCESS_TOKEN is unavailable")
    req = urllib.request.Request(
        "https://api.digitalocean.com/v2/" + path,
        method=method,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        data=json.dumps(data).encode() if data is not None else None,
    )
    with urllib.request.urlopen(req, timeout=25) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def listing(resource):
    page = 1
    while True:
        result = api(f"{resource}?per_page=200&page={page}")
        yield from result[resource]
        if not result.get("links", {}).get("pages", {}).get("next"):
            return
        page += 1


def cleanup():
    marker = STATE / "name"
    if not marker.exists():
        return
    name = marker.read_text()
    id_file = STATE / "id"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        matches = [d for d in listing("droplets") if d["name"] == name]
        if not matches:
            # An acknowledged create has a provider ID. Verify its absence directly.
            if id_file.exists():
                try:
                    obj = api("droplets/" + id_file.read_text())["droplet"]
                except urllib.error.HTTPError as exc:
                    if exc.code != 404:
                        raise
                else:
                    if obj["name"] != name:
                        raise RuntimeError("Droplet ownership mismatch")
                    matches = [obj]
            elif time.time() - marker.stat().st_mtime < 60:
                time.sleep(5)
                continue
        if not matches:
            save(OUT / "cleanup.json", {"absent": True, "end_unix": time.time()})
            return
        for d in matches:
            if id_file.exists() and str(d["id"]) != id_file.read_text():
                raise RuntimeError("Unexpected duplicate Droplet")
            try:
                api("droplets/" + str(d["id"]), "DELETE")
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
        time.sleep(5)
    raise RuntimeError("Droplet cleanup not confirmed")


def quote():
    OUT.mkdir(exist_ok=True)
    sizes = list(listing("sizes"))
    rows = [
        {
            key: s.get(key)
            for key in ("slug", "available", "price_hourly", "memory", "vcpus", "disk", "regions")
        }
        for s in sizes
        if s["slug"].endswith("-intel") and (s["slug"] == SIZE or 8192 <= s["memory"] <= 32768)
    ]
    save(OUT / "intel-catalog.json", rows)
    return sizes


def run():
    STATE.mkdir(mode=0o700)
    sizes = quote()
    size = next((s for s in sizes if s["slug"] == SIZE and s["available"]), None)
    if not size or not 0 < size["price_hourly"] <= RATE_LIMIT or size["memory"] < 8192:
        raise RuntimeError("Requested Intel size unavailable or over rate cap")
    regions = [
        r["slug"] for r in listing("regions") if r["available"] and r["slug"] in size["regions"]
    ]
    if not regions:
        raise RuntimeError("No region offers the requested size")
    region = "sgp1" if "sgp1" in regions else min(regions)
    audit = {"slug": SIZE, "hourly_usd": size["price_hourly"], "region": region, "regions": regions}
    save(OUT / "quote.json", audit)
    if any(d["name"].startswith("spt-pr50-intel-") for d in listing("droplets")):
        raise RuntimeError("An earlier experiment Droplet still exists")
    # Only disposable SSH credentials reach the VM. Keep the account token on the runner.
    for key in ("client", "host"):
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", STATE / key], check=True
        )
    user = {
        "name": "spt",
        "shell": "/bin/bash",
        "sudo": "ALL=(ALL) NOPASSWD:ALL",
        "ssh_authorized_keys": [(STATE / "client.pub").read_text().strip()],
    }
    cfg = {
        "users": [user],
        "ssh_pwauth": False,
        "ssh_keys": {
            "ed25519_private": (STATE / "host").read_text(),
            "ed25519_public": (STATE / "host.pub").read_text().strip(),
        },
    }
    name = "spt-pr50-intel-" + os.environ["GITHUB_RUN_ID"] + "-" + secrets.token_hex(5)
    (STATE / "name").write_text(name)
    audit["start_unix"] = time.time()
    save(OUT / "quote.json", audit)
    vm = api(
        "droplets",
        "POST",
        {
            "name": name,
            "size": SIZE,
            "region": region,
            "image": "ubuntu-24-04-x64",
            "backups": False,
            "user_data": "#cloud-config\n" + json.dumps(cfg),
        },
    )["droplet"]
    vm_id = str(vm["id"])
    (STATE / "id").write_text(vm_id)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        d = api("droplets/" + vm_id)["droplet"]
        if d["name"] != name or d["size_slug"] != SIZE or d["region"]["slug"] != region:
            raise RuntimeError("Unexpected assigned Droplet")
        public = [n["ip_address"] for n in d["networks"]["v4"] if n["type"] == "public"]
        if public and d["status"] == "active":
            break
        time.sleep(5)
    else:
        raise TimeoutError("Droplet did not become active")
    ip = str(ipaddress.IPv4Address(public[0]))
    hosts = STATE / "known_hosts"
    hosts.write_text(ip + " " + (STATE / "host.pub").read_text())
    ssh = [
        "ssh",
        "-i",
        str(STATE / "client"),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={hosts}",
        "spt@" + ip,
    ]
    while time.monotonic() < deadline:
        probe = subprocess.run(
            ssh + ["sed '/^$/q' /proc/cpuinfo"], capture_output=True, text=True, check=False
        )
        if probe.returncode == 0:
            break
        time.sleep(5)
    else:
        raise TimeoutError("Authenticated SSH unavailable")
    cpu = dict(line.split(":", 1) for line in probe.stdout.splitlines() if ":" in line)
    cpu = {k.strip(): v.strip() for k, v in cpu.items()}
    save(OUT / "cpu.json", cpu)
    print("CPU:", cpu.get("vendor_id"), cpu.get("model name"), flush=True)
    if cpu.get("vendor_id") != "GenuineIntel":
        raise RuntimeError("Intel vendor check failed")
    with (STATE.parent / "trios.zip").open("rb") as archive:
        subprocess.run(ssh + ["cat > trios.zip"], stdin=archive, check=True)
    with Path(__file__).with_name("replay.sh").open() as script:
        replay = subprocess.run(ssh + ["bash -s"], stdin=script, check=False)
    report = subprocess.run(ssh + ["cat native.json"], capture_output=True, check=False)
    if report.returncode == 0:
        save(OUT / "native.json", json.loads(report.stdout))
    replay.check_returncode()
    report.check_returncode()


if __name__ == "__main__":
    {"quote": quote, "run": run, "cleanup": cleanup}[sys.argv[1]]()
