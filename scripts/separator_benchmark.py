"""One model per process. No scoring, TensorFlow, or second model on the timed GPU."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import resource
import time
from pathlib import Path

from benchmark_metrics import RATE, REPEATS, STEMS, VARIANTS
from probe_htdemucs import COMMIT, SHA256


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def demucs_separate(model, waveform, *, device="cuda"):
    import torch
    from demucs.apply import apply_model

    # Match upstream demucs/separate.py: normalize the mixture using mono reference,
    # then reverse that transformation separately on each returned source.
    ref = waveform.mean(0)
    mean, std = ref.mean(), ref.std()
    if not torch.isfinite(std) or std <= 0:
        raise ValueError("Degenerate mixture normalization")
    normalized = (waveform - mean) / std
    with torch.inference_mode():
        output = apply_model(
            model,
            normalized[None],
            shifts=0,
            split=True,
            overlap=0.25,
            segment=7.8,
            device=device,
            num_workers=0,
        )[0]
        output = (output * std + mean).cpu()
    return {s: output[model.sources.index(s)] for s in STEMS}


def run(args):
    from importlib import import_module

    profile = import_module("vocal_metrics" if args.profile == "vocal" else "benchmark_metrics")
    protocol, stems = profile.PROTOCOL, profile.STEMS
    if args.model not in profile.VARIANTS:
        raise ValueError("Model does not belong to selected benchmark profile")
    start = time.perf_counter()
    import numpy as np
    import soundfile as sf
    import torch

    imported = time.perf_counter() - start
    if torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8":
        raise ValueError("Pinned CUDA runtime required")
    torch.set_num_threads(2)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    runtime = {
        "requested_gpu": os.environ["SPT_GPU_ID"],
        "name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    spec = json.loads(args.manifest.read_text())
    if spec.get("profile", "four-stem") != args.profile:
        raise ValueError("Manifest/profile mismatch")
    weight_sha = digest(args.weights)
    if weight_sha != spec["weight_sha256"][args.model]:
        raise ValueError("Weight checksum mismatch before load")
    tick = time.perf_counter()
    if args.model in ("official_converted", "official_2stems"):
        from spleeterpt.audio import separate
        from spleeterpt.checkpoint import load_model
        from spleeterpt.config import ModelConfig

        imported += time.perf_counter() - tick
        tick = time.perf_counter()
        model = load_model(args.weights, "cuda")
        if model.config != ModelConfig.preset(2 if args.profile == "vocal" else 4):
            raise ValueError("Unexpected Spleeter configuration")
        infer = separate
    elif args.model == "melband_kim":
        from roformer_kim import load_model as load_roformer
        from roformer_kim import separate as separate_roformer

        imported += time.perf_counter() - tick
        tick = time.perf_counter()
        model = load_roformer(args.weights.parent, device="cuda")
        from functools import partial

        infer = partial(separate_roformer, device="cuda")
    else:
        from demucs.states import load_model

        imported += time.perf_counter() - tick
        tick = time.perf_counter()
        source = json.loads(importlib.metadata.distribution("demucs").read_text("direct_url.json"))
        if source.get("vcs_info", {}).get("commit_id") != COMMIT or weight_sha != SHA256:
            raise ValueError("Demucs source identity mismatch")
        package = torch.load(args.weights, map_location="cpu", weights_only=False)
        model = load_model(package, strict=True).float().eval().to("cuda")
        del package
        if (
            model.sources != ["drums", "bass", "other", "vocals"]
            or model.samplerate != RATE
            or float(model.segment) != 7.8
        ):
            raise ValueError("Unexpected HTDemucs architecture")
        infer = demucs_separate
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - tick

    def state_hash():
        h = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    state_before = state_hash()
    args.output.mkdir(parents=True, exist_ok=False)

    def read(track):
        audio, rate = sf.read(
            args.data / track["name"] / "mixture.wav", dtype="float32", always_2d=True
        )
        if rate != RATE or audio.shape != (track["samples"], 2) or not np.isfinite(audio).all():
            raise ValueError("Input shape/rate/finiteness mismatch")
        return torch.from_numpy(audio.T.copy())

    def checked_infer(waveform):
        outputs = infer(model, waveform)
        if set(outputs) != set(stems) or any(
            a.shape != waveform.shape or not torch.isfinite(a).all() for a in outputs.values()
        ):
            raise ValueError("Output shape/stem/finiteness mismatch")
        return outputs

    waveform = read(spec["tracks"][0])[:, : 8 * RATE].contiguous()
    tick = time.perf_counter()
    outputs = checked_infer(waveform)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - tick
    del waveform, outputs
    tracks = []
    # --smoke exercises the first full song (and exports), not an arbitrary synthetic input.
    selected = spec["tracks"][:1] if args.smoke else spec["tracks"]
    for index, track in enumerate(selected):
        path = args.data / track["name"] / "mixture.wav"
        if digest(path) != track["mixture_sha256"]:
            raise ValueError("Mixture changed after staging")
        repeats = []
        output_dir = args.output / f"track-{index:02d}"
        output_dir.mkdir()
        for repeat in range(1 if args.smoke else REPEATS):
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            # Each repeat starts without a previous waveform/output or cached allocator blocks.
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            waveform = read(track)
            decoded = time.perf_counter()
            outputs = checked_infer(waveform)
            torch.cuda.synchronize()
            inferred = time.perf_counter()
            for stem, audio in outputs.items():
                sf.write(
                    output_dir / f"{stem}{'-repeat' if repeat else ''}.wav",
                    audio.numpy().T,
                    RATE,
                    subtype="FLOAT",
                )
            exported = time.perf_counter()
            repeats.append(
                {
                    "decode_seconds": decoded - tick,
                    "inference_seconds": inferred - decoded,
                    "export_seconds": exported - inferred,
                    "end_to_end_seconds": exported - tick,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
            del waveform, outputs
            if repeat:
                for stem in stems:
                    (output_dir / f"{stem}-repeat.wav").unlink()
        tracks.append(
            {
                "id": track["id"],
                "samples": track["samples"],
                "repeats": repeats,
                "files": {
                    s: {
                        "sha256": digest(output_dir / f"{s}.wav"),
                        "bytes": (output_dir / f"{s}.wav").stat().st_size,
                    }
                    for s in stems
                },
            }
        )
        print(f"Progress: benchmark {args.model} songs {index + 1}/{len(selected)}", flush=True)
    if digest(args.weights) != weight_sha:
        raise ValueError("Weight file changed during inference")
    if state_hash() != state_before:
        raise ValueError("Model state changed during inference")
    report = {
        "model": args.model,
        "demucs_source_commit": COMMIT if args.model == "htdemucs" else None,
        "smoke": args.smoke,
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_runtime": runtime,
        "weight_sha256": weight_sha,
        "import_seconds": imported,
        "load_seconds": load_seconds,
        "state_unchanged": True,
        "warmup_seconds": warmup_seconds,
        "tracks": tracks,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "seconds": time.perf_counter() - start,
    }
    if args.profile == "vocal":
        from roformer_kim import peak_rss_bytes, recipe_sha256

        report.update(
            profile="vocal",
            rss_source="/proc/self/status VmHWM",
            process_peak_rss_bytes=peak_rss_bytes(),
            roformer_recipe_sha256=recipe_sha256() if args.model == "melband_kim" else None,
        )
    (args.output / "measurements.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=(*VARIANTS, "official_2stems", "melband_kim"), required=True)
    p.add_argument("--profile", choices=("four-stem", "vocal"), default="four-stem")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())
