"""Verify the official full MUSDB18 archive and prepare only its 100-song training pool."""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

ARCHIVE_BYTES = 4684228845
ARCHIVE_MD5 = "af06762477334799bfc5abf237648207"
SOURCE = "https://zenodo.org/records/1117372"
STEMS = ("mixture", "drums", "bass", "other", "vocals")


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def inventory(names):
    songs = {"train": [], "test": []}
    for name in names:
        p = PurePosixPath(name)
        if p.suffix != ".mp4":
            continue
        if len(p.parts) != 2 or p.parts[0] not in songs or not p.name.endswith(".stem.mp4"):
            raise ValueError("Unexpected archive audio path")
        songs[p.parts[0]].append(name)
    if len(songs["train"]) != 100 or len(songs["test"]) != 50 or len(set(names)) != len(names):
        raise ValueError("Expected 100 train and 50 test tracks without duplicates")
    return {k: sorted(v) for k, v in songs.items()}


def prepare(archive, output):
    import soundfile as sf

    from spleeterpt.data import make_manifests

    archive = Path(archive).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required")
    if archive.stat().st_size != ARCHIVE_BYTES or file_hash(archive, "md5") != ARCHIVE_MD5:
        raise ValueError("Not the checksum-verified official complete MUSDB18 archive")
    if shutil.disk_usage(output.parent).free < 45 * 1024**3:
        raise RuntimeError("At least 45 GiB free space is required for float WAV preparation")
    with zipfile.ZipFile(archive) as z:
        songs = inventory(z.namelist())
        output.mkdir()
        rows = []
        for index, name in enumerate(songs["train"]):
            track = output / "train-pool" / Path(name).name.removesuffix(".stem.mp4")
            track.mkdir(parents=True)
            source = track / "source.stem.mp4"
            with z.open(name) as src, source.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            record = {"id": name, "source_sha256": file_hash(source), "files": {}}
            shapes = set()
            for stream, stem in enumerate(STEMS):
                dest = track / f"{stem}.wav"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        str(source),
                        "-map",
                        f"0:a:{stream}",
                        "-c:a",
                        "pcm_f32le",
                        str(dest),
                    ],
                    check=True,
                )
                info = sf.info(dest)
                if info.samplerate != 44100 or info.channels != 2 or info.frames < 527360:
                    raise ValueError("Expected full-length aligned 44.1kHz stereo source")
                shapes.add(info.frames)
                record["files"][stem] = {"sha256": file_hash(dest), "bytes": dest.stat().st_size}
            if len(shapes) != 1:
                raise ValueError("Unaligned decoded sources")
            record["samples"] = shapes.pop()
            rows.append(record)
            source.unlink()
            print(f"Prepared training-pool track {index + 1}/100", flush=True)
        manifests = make_manifests(
            output / "train-pool", output / "manifests", validation_tracks=10, seed=17
        )
        provenance = {
            "dataset": "MUSDB18 AAC full training pool",
            "source": SOURCE,
            "archive_bytes": ARCHIVE_BYTES,
            "archive_md5": ARCHIVE_MD5,
            "archive_sha256": file_hash(archive),
            "full_archive_verified": True,
            "training_pool_tracks": 100,
            "train_tracks": 90,
            "validation_tracks": 10,
            "split": "project-specific track split, Python random seed 17; not the standard 86/14 split",
            "test_tracks_in_archive": songs["test"],
            "test_audio_extracted": False,
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            "manifests": {p.name: file_hash(p) for p in manifests},
            "tracks": rows,
        }
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(
            "PASS: complete 100-track training pool, 90/10 split, test audio untouched", flush=True
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    prepare(args.archive, args.output)
