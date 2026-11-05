"""Fingerprint two preregistered TRIOS recordings without running any separator."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf

ARCHIVE_MD5 = "0922b785187d655bd6f2d83265af2098"
PIECES = {"brahms": ("violin", "horn"), "schubert": ("violin", "cello")}
START, DURATION = 5, 12


def sha(data):
    return hashlib.sha256(data).hexdigest()


def register(archive, piece, other):
    targets = {role: f"TRIOS Dataset/{piece}/{role}.wav" for role in ("mix", "piano", *other)}
    metadata, clips = {}, {}
    geometry = None
    for role, member in targets.items():
        data = archive.read(member)
        signal, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        shape = (rate, *signal.shape)
        if geometry is None:
            geometry = shape
        if shape != geometry or rate != 44100 or signal.shape[1] not in (1, 2):
            raise ValueError(f"{member}: inconsistent audio geometry {shape} vs {geometry}")
        clip = signal[START * rate : (START + DURATION) * rate]
        if len(clip) != DURATION * rate or not np.isfinite(signal).all():
            raise ValueError(f"{member}: invalid samples")
        clips[role] = clip.astype(np.float64)
        metadata[role] = {
            "member": member,
            "sha256": sha(data),
            "bytes": len(data),
            "sample_rate": rate,
            "channels": signal.shape[1],
            "frames": len(signal),
            "clip_pcm_f32le_interleaved_sha256": sha(clip.astype("<f4").tobytes()),
        }
    residual = clips["mix"] - clips["piano"] - clips[other[0]] - clips[other[1]]
    return {
        "recording_id": "trios-" + piece,
        "start_seconds": START,
        "duration_seconds": DURATION,
        "other": list(other),
        "assets": metadata,
        "sum_max_error": float(abs(residual).max()),
        "sum_rms_error": float(np.sqrt(np.mean(residual**2))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = args.archive.read_bytes()
    if hashlib.md5(data).hexdigest() != ARCHIVE_MD5:
        raise ValueError("Zenodo archive checksum mismatch")
    with ZipFile(io.BytesIO(data)) as archive:
        recordings = [register(archive, piece, other) for piece, other in PIECES.items()]
    result = {
        "archive_sha256": sha(data),
        "archive_bytes": len(data),
        "source": "https://doi.org/10.5281/zenodo.6797837",
        "license": "CC-BY-NC-SA-2.0-UK",
        "recordings": recordings,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if any(r["sum_max_error"] > 4 / 32768 for r in recordings):
        raise ValueError("published mix sum exceeds the four PCM16-step peak limit")


if __name__ == "__main__":
    main()
