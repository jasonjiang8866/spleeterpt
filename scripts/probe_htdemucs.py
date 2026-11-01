"""Verify one official checkpoint and CPU inference before any paid GPU work."""

import hashlib
import importlib.metadata
import json
import platform
import urllib.request
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"
COMMIT = "e976d93ecc3865e5757426930257e200846a520a"
SHA256 = "8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4"


def main():
    import torch
    from demucs.apply import apply_model
    from demucs.states import load_model

    source = json.loads(importlib.metadata.distribution("demucs").read_text("direct_url.json"))
    if source.get("vcs_info", {}).get("commit_id") != COMMIT:
        raise ValueError("Demucs installed source does not match pinned commit")
    out = Path("runpod-results")
    out.mkdir(exist_ok=True)
    checkpoint = out / "955717e8-8726e21a.th"
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(URL, timeout=60) as response, checkpoint.open("wb") as target:
        while chunk := response.read(1024**2):
            size += len(chunk)
            if size > 256 * 1024**2:
                raise ValueError("Checkpoint exceeds 256 MiB limit")
            digest.update(chunk)
            target.write(chunk)
    sha = digest.hexdigest()
    if sha != SHA256:
        raise ValueError("Upstream checkpoint checksum mismatch")
    # This single hash-checked official checkpoint serializes its Python class.
    # Keep explicit pickle loading local; do not change global Torch load policy.
    package = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = load_model(package, strict=True).float().eval()
    if model.sources != ["drums", "bass", "other", "vocals"] or model.samplerate != 44100:
        raise ValueError("Unexpected checkpoint targets or sample rate")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    audio = torch.randn(1, 2, 44100) * 0.01
    with torch.inference_mode():
        output = apply_model(
            model,
            audio,
            shifts=0,
            split=True,
            overlap=0.25,
            segment=float(model.segment),
            device="cpu",
            num_workers=0,
        )
    if output.shape != (1, 4, 2, 44100) or not torch.isfinite(output).all():
        raise ValueError("Invalid real-model CPU output")
    report = {
        "source_commit": COMMIT,
        "python": platform.python_version(),
        "checkpoint_url": URL,
        "checkpoint_sha256": sha,
        "checkpoint_bytes": size,
        "sources": model.sources,
        "sample_rate": model.samplerate,
        "segment_seconds": float(model.segment),
        "parameters": sum(p.numel() for p in model.parameters()),
        "versions": {
            n: importlib.metadata.version(n)
            for n in (
                "torch",
                "torchaudio",
                "demucs",
                "dora-search",
                "einops",
                "julius",
                "openunmix",
            )
        },
        "cpu_output_shape": list(output.shape),
        "finite": True,
        "scope": "synthetic CPU compatibility only; no music quality or GPU result",
    }
    (out / "htdemucs-probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    checkpoint.unlink()  # Public reports only; no weight redistribution in artifacts.


if __name__ == "__main__":
    main()
