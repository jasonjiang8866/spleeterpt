"""Pinned Kim Mel-Band RoFormer assets and explicit FP32 inference recipe."""

import hashlib
import importlib
import json
import sys
import urllib.request
from pathlib import Path

MSST_COMMIT = "9cc76c03ccd4da5ddf3477e93d386a1648a77732"
WEIGHT_REVISION = "ac9b0614ab3cd7f77219e18ba494dfd93956c348"
WEIGHT_SHA256 = "87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e"
WEIGHT_BYTES = 913106900
SOURCE_FILES = {
    "models/bs_roformer/mel_band_roformer.py": (
        22708,
        "a28fce62020faad6461176660e012cf5799ae8a26c76fe4b2bc636f5b44b9fc5",
    ),
    "models/bs_roformer/attend.py": (
        4410,
        "c7abbc40a3fd20ff7f6001fa9f8ee9ad5df6452272712a5e719b8f8468bf2223",
    ),
    "configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml": (
        1721,
        "f63f38eb1e6e40a7db0dade714a5ae257555dd8748f4e774eae8679275a81926",
    ),
    "LICENSE": (1081, "3282dc057695ef5b9a64909a7092ca40b2c292c232580fc6ace6e5d665cc0207"),
}
RATE = 44100
CHUNK = 352800
OVERLAP = 2
MODEL_CONFIG = {
    "dim": 384,
    "depth": 6,
    "stereo": True,
    "num_stems": 1,
    "time_transformer_depth": 1,
    "freq_transformer_depth": 1,
    "num_bands": 60,
    "dim_head": 64,
    "heads": 8,
    "attn_dropout": 0,
    "ff_dropout": 0,
    "flash_attn": False,
    "dim_freqs_in": 1025,
    "sample_rate": RATE,
    "stft_n_fft": 2048,
    "stft_hop_length": 441,
    "stft_win_length": 2048,
    "stft_normalized": False,
    "mask_estimator_depth": 2,
    "multi_stft_resolution_loss_weight": 1.0,
    "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
    "multi_stft_hop_size": 147,
    "multi_stft_normalized": False,
    # Original Kim implementation does not zero the DC bin. Pin new MSST defaults.
    "zero_dc": False,
    "match_input_audio_length": False,
    "linear_transformer_depth": 0,
    "mlp_expansion_factor": 4,
    "use_torch_checkpoint": False,
    "skip_connection": False,
    "use_pope": False,
}
RECIPE = {
    "msst_commit": MSST_COMMIT,
    "weight_revision": WEIGHT_REVISION,
    "weight_sha256": WEIGHT_SHA256,
    "model": MODEL_CONFIG,
    "chunk_size": CHUNK,
    "num_overlap": OVERLAP,
    "batch_size": 1,
    "precision": "fp32",
    "attention": "explicit eager einsum/softmax; no SDPA/autocast",
    "window": "author linear fade C//10, reflect borders, constant/reflect final pad",
    "normalization": "none",
    "tta": False,
    "accompaniment": "input mixture minus predicted vocals",
    "nonfinite": "reject; never nan_to_num",
}
RECIPE = json.loads(json.dumps(RECIPE))  # canonical JSON types for independent validation


def recipe_sha256():
    return hashlib.sha256(
        json.dumps(RECIPE, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def checked_download(url, path, size, sha):
    if path.exists():
        if path.stat().st_size != size or digest(path) != sha:
            raise ValueError("Cached asset checksum mismatch: " + path.name)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            count = 0
            while block := response.read(4 * 1024**2):
                count += len(block)
                if count > size:
                    raise ValueError("Asset exceeds pinned byte count")
                stream.write(block)
        if temporary.stat().st_size != size or digest(temporary) != sha:
            raise ValueError("Downloaded asset checksum mismatch: " + path.name)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def prepare(folder):
    for name, (size, sha) in SOURCE_FILES.items():
        checked_download(
            f"https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/{MSST_COMMIT}/{name}",
            folder / "msst" / name,
            size,
            sha,
        )
    return checked_download(
        f"https://huggingface.co/KimberleyJSN/melbandroformer/resolve/{WEIGHT_REVISION}/MelBandRoformer.ckpt",
        folder / "MelBandRoformer.ckpt",
        WEIGHT_BYTES,
        WEIGHT_SHA256,
    )


def load_model(folder, device="cpu"):
    import torch

    weight = prepare(folder)
    source = (folder / "msst").resolve()
    # Dedicated process required: never reuse an unrelated already-imported models package.
    if any(k == "models" or k.startswith("models.") for k in sys.modules):
        raise RuntimeError("RoFormer loader requires a fresh models import namespace")
    sys.path.insert(0, str(source))
    module = importlib.import_module("models.bs_roformer.mel_band_roformer")
    if Path(module.__file__).resolve() != source / "models/bs_roformer/mel_band_roformer.py":
        raise RuntimeError("Unexpected model import origin")
    model = module.MelBandRoformer(**MODEL_CONFIG)
    state = torch.load(weight, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    return model.float().eval().to(device)


def separate(model, waveform, *, device="cpu", chunk_size=CHUNK, num_overlap=OVERLAP):
    """Author overlap-add recipe, FP32 and finite/length checks made explicit."""
    import torch
    import torch.nn.functional as F

    if (
        waveform.ndim != 2
        or waveform.shape[0] != 2
        or waveform.shape[1] < 1
        or waveform.dtype != torch.float32
        or not torch.isfinite(waveform).all()
    ):
        raise ValueError("Expected finite stereo FP32 waveform")
    if chunk_size < 10 or num_overlap < 1 or chunk_size % num_overlap:
        raise ValueError("Invalid chunk/overlap geometry")
    count = waveform.shape[-1]
    step, fade = chunk_size // num_overlap, chunk_size // 10
    border = chunk_size - step
    padded = count > 2 * border and border > 0
    mix = waveform.to(device)
    if padded:
        mix = F.pad(mix, (border, border), mode="reflect")
    window = torch.ones(chunk_size, device=device, dtype=torch.float32)
    window[:fade] *= torch.linspace(0, 1, fade, device=device)
    window[-fade:] *= torch.linspace(1, 0, fade, device=device)
    result = torch.zeros_like(mix)
    counter = torch.zeros_like(mix)
    with torch.inference_mode():
        for offset in range(0, mix.shape[-1], step):
            part = mix[:, offset : offset + chunk_size]
            length = part.shape[-1]
            if length < chunk_size:
                mode = "reflect" if length > chunk_size // 2 + 1 else "constant"
                part = F.pad(part, (0, chunk_size - length), mode=mode)
            predicted = model(part[None])[0]
            if predicted.shape != part.shape or not torch.isfinite(predicted).all():
                raise ValueError("RoFormer chunk output length/finiteness mismatch")
            weights = window.clone()
            if offset == 0:
                weights[:fade] = 1
            elif offset + chunk_size >= mix.shape[-1]:
                weights[-fade:] = 1
            result[:, offset : offset + length] += predicted[:, :length] * weights[:length]
            counter[:, offset : offset + length] += weights[:length]
        if not (counter > 0).all():
            raise ValueError("Uncovered overlap-add samples")
        vocals = result / counter
        if padded:
            vocals = vocals[:, border:-border]
        vocals = vocals.cpu()
    if vocals.shape != waveform.shape or not torch.isfinite(vocals).all():
        raise ValueError("Invalid reconstructed vocals")
    return {"vocals": vocals, "accompaniment": waveform.cpu() - vocals}


def peak_rss_bytes():
    """Linux mm-specific high-water mark, reset by exec unlike ru_maxrss."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB" or int(fields[1]) <= 0:
                break
            return int(fields[1]) * 1024
    raise RuntimeError("Linux process-specific VmHWM unavailable")
