"""Check all installed requirements, allowing only the probed upstream audio upper bound."""

import importlib.metadata
import json

from packaging.requirements import Requirement

COMMIT = "e976d93ecc3865e5757426930257e200846a520a"


def check():
    source = json.loads(importlib.metadata.distribution("demucs").read_text("direct_url.json"))
    if source.get("vcs_info", {}).get("commit_id") != COMMIT:
        raise ValueError("Unpinned Demucs source")
    for package in ("torch", "torchaudio"):
        if importlib.metadata.version(package) != "2.8.0+cu128":
            raise ValueError("Pinned CUDA package missing")
    exceptions = []
    for dist in importlib.metadata.distributions():
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate():
                continue
            installed = importlib.metadata.version(req.name)
            if installed in req.specifier:
                continue
            if (
                dist.metadata["Name"].lower() == "demucs"
                and req.name.lower() == "torchaudio"
                and str(req.specifier) == "<2.1,>=0.8"
                and installed == "2.8.0+cu128"
            ):
                exceptions.append(raw)
            else:
                raise ValueError(
                    "Unapproved dependency conflict: " + dist.metadata["Name"] + " / " + req.name
                )
    if len(exceptions) != 1:
        raise ValueError("Expected exactly the documented HTDemucs Torchaudio exception")
    print(
        "PASS: requirements validated; only pinned/probed Demucs torchaudio upper-bound exception"
    )


if __name__ == "__main__":
    check()
