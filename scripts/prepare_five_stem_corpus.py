"""Prepare explicitly selected consistency clips; never silently open a holdout."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import soundfile as sf
from register_trios import PIECES, register

from spleeterpt.audio import read_audio
from spleeterpt.pretrained import digest, verified_download


def validate(manifest, *, require_piano_coverage=False):
    records = manifest["records"]
    ids, hashes = set(), set()
    for item in records:
        if item["recording_id"] in ids or item["sha256"] in hashes:
            raise ValueError("recording reuse / calibration-holdout leakage")
        ids.add(item["recording_id"])
        hashes.add(item["sha256"])
        if item["split"] not in {"calibration", "holdout"}:
            raise ValueError("unknown split")
        if item["split"] == "holdout" and item["inference_observed"]:
            raise ValueError("observed recording cannot be a fresh holdout")
        if item["ground_truth_stems"]:
            if (
                item["split"] != "holdout"
                or item["category"] != "mixed_piano"
                or not item["piano_confirmed"]
                or "trios" not in item
            ):
                raise ValueError("missing registered mixed-piano reference tracks")
            registered = item["trios"]
            if (
                registered["recording_id"] != item["recording_id"]
                or registered["assets"]["mix"]["sha256"] != item["sha256"]
                or set(registered["prepared_pcm"]) != {"mix", "piano", "other"}
                or registered["sum_max_error"] > 4 / 32768
            ):
                raise ValueError("unverified reference identity or additive residual")
    piano = [r for r in records if r["piano_confirmed"]]
    limits = manifest["release_coverage"]
    complete = (
        len(piano) >= limits["minimum_independent_piano_recordings"]
        and sum(r["split"] == "holdout" for r in piano)
        >= limits["minimum_holdout_piano_recordings"]
        and any(
            r["split"] == "holdout" and r["category"] == "mixed_piano" and r["ground_truth_stems"]
            for r in piano
        )
    )
    if require_piano_coverage and not complete:
        raise ValueError("independent piano coverage incomplete")
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--split", choices=["calibration", "holdout"], required=True)
    parser.add_argument("--require-piano-coverage", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).with_name("five_stem_corpus.json")
    manifest = json.loads(path.read_text())
    complete = validate(manifest, require_piano_coverage=args.require_piano_coverage)
    args.work.mkdir(parents=True, exist_ok=True)
    report = {
        "manifest_sha256": digest(path),
        "split": args.split,
        "piano_coverage_complete": complete,
        "release_eligible": False,
        "clips": [],
    }
    for item in manifest["records"]:
        if item["split"] != args.split:
            continue
        if "trios" in item:
            archive = manifest["trios_archive"]
            source = verified_download(
                archive["url"],
                args.work / "trios.zip",
                sha256=archive["sha256"],
                size=archive["bytes"],
            )
            piece = item["recording_id"].removeprefix("trios-")
            dest = args.work / item["recording_id"]
            with ZipFile(source) as zf:
                measured = register(zf, piece, PIECES[piece], dest)
            if measured != item["trios"]:
                raise ValueError(f"TRIOS reference differs from registration: {piece}")
            report["clips"].append(
                {
                    "recording_id": item["recording_id"],
                    "source_sha256": item["sha256"],
                    "path": str(dest / "mix.wav"),
                    "sha256": digest(dest / "mix.wav"),
                    "start_seconds": measured["start_seconds"],
                    "duration_seconds": measured["duration_seconds"],
                    "reference_paths": {k: str(dest / f"{k}.wav") for k in ("piano", "other")},
                    "pcm_sha256": measured["prepared_pcm"],
                }
            )
            continue
        source = verified_download(
            item["url"],
            args.work / (item["recording_id"] + ".ogg"),
            sha256=item["sha256"],
            size=item["bytes"],
        )
        waveform = read_audio(source, 44100)
        for i, clip in enumerate(item["clips"]):
            start = round(clip["start_seconds"] * 44100)
            count = round(clip["duration_seconds"] * 44100)
            samples = waveform[:, start : start + count]
            if samples.shape != (2, count):
                raise ValueError("source shorter than registered clip")
            dest = args.work / f"{item['recording_id']}-{i}.wav"
            sf.write(dest, samples.numpy().T, 44100, subtype="FLOAT")
            report["clips"].append(
                {
                    "recording_id": item["recording_id"],
                    "source_sha256": item["sha256"],
                    "path": str(dest),
                    "sha256": digest(dest),
                    **clip,
                }
            )
    (args.work / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
