"""Dataset, metrics and integrity helpers; no TensorFlow/model imports or downloads."""

import hashlib
import json
from pathlib import Path

import numpy as np

VARIANTS = ("tensorflow_current", "spleeterpt", "legacy_unmodified", "legacy_corrected")
SOURCES = ("mixture", "drums", "bass", "other", "vocals")
BSS_METRICS = ("sdr", "isr", "sir", "sar")
RATE = 44100


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def discover(root, layout, subsets, track_ids=None, limit=None, seed=None):
    """Canonical relative IDs, optionally seeded sampling, always sorted for execution."""
    root = Path(root)
    found = {}
    for subset in sorted(set(subsets)):
        if subset not in {"train", "test"}:
            raise ValueError("subsets must be train or test")
        folder = root / subset
        if not folder.is_dir():
            raise ValueError(f"missing subset directory: {folder}")
        candidates = folder.glob("*.stem.mp4") if layout == "stems" else folder.iterdir()
        for path in sorted(candidates):
            if layout == "wav" and not path.is_dir():
                continue
            name = path.name.removesuffix(".stem.mp4") if layout == "stems" else path.name
            identifier = f"{subset}/{name}"
            files = [path] if layout == "stems" else [path / f"{s}.wav" for s in SOURCES]
            if any(not file.is_file() for file in files):
                raise ValueError(f"incomplete track: {identifier}")
            if identifier in found:
                raise ValueError(f"duplicate track ID: {identifier}")
            found[identifier] = files
    ids = sorted(found)
    if track_ids is not None:
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("duplicate requested track IDs")
        missing = set(track_ids) - set(found)
        if missing:
            raise ValueError(f"unknown requested track IDs: {sorted(missing)}")
        ids = sorted(track_ids)
    if seed is not None and limit is None:
        raise ValueError("seed requires limit")
    if limit is not None:
        if not 1 <= limit <= len(ids):
            raise ValueError("limit must be positive and no greater than available tracks")
        if seed is not None:
            ids = sorted(np.random.default_rng(seed).choice(ids, limit, replace=False).tolist())
        else:
            ids = ids[:limit]
    if not ids:
        raise ValueError("no tracks selected")
    return [{"id": key, "files": [str(p.relative_to(root)) for p in found[key]]} for key in ids]


def decode_track(root, record, layout):
    """Use the official MUSDB stream mapping; WAV inputs must already be 44.1k stereo."""
    import soundfile as sf

    if layout == "stems":
        # Share the preview decoder without importing TensorFlow.
        from evaluate_preview import decode_stream

        streams = [decode_stream(Path(root) / record["files"][0], i).numpy() for i in range(5)]
    else:
        streams = []
        for filename in record["files"]:
            audio, rate = sf.read(Path(root) / filename, dtype="float32", always_2d=True)
            if rate != RATE:
                raise ValueError(f"expected {RATE} Hz, got {rate}: {filename}")
            streams.append(audio.T)
    lengths = []
    for audio in streams:
        if audio.ndim != 2 or audio.shape[0] != 2 or not audio.shape[1]:
            raise ValueError("expected nonempty stereo sources")
        if not np.isfinite(audio).all():
            raise ValueError("nonfinite source audio")
        lengths.append(audio.shape[1])
    if len(set(lengths)) != 1:
        raise ValueError(f"source lengths differ (no implicit reference crop): {lengths}")
    targets = dict(zip(SOURCES, streams))
    targets["accompaniment"] = targets["drums"] + targets["bass"] + targets["other"]
    return targets


def encode_number(value):
    if value is None or np.isnan(value):
        return None
    if np.isposinf(value):
        return "+inf"
    if np.isneginf(value):
        return "-inf"
    return float(value)


def decode_number(value):
    if value is None:
        return np.nan
    if value == "+inf":
        return np.inf
    if value == "-inf":
        return -np.inf
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"invalid encoded score: {value!r}")
    return float(value)


def statistics(values, *, counts_only=False):
    values = np.array([decode_number(value) for value in values])
    valid = values[~np.isnan(values)]
    with np.errstate(invalid="ignore"):
        median = encode_number(np.median(valid)) if len(valid) else None
        mean = encode_number(np.mean(valid)) if len(valid) else None
    result = {
        "total": len(values),
        "finite": int(np.isfinite(values).sum()),
        "undefined": int(np.isnan(values).sum()),
        "positive_infinity": int(np.isposinf(values).sum()),
        "negative_infinity": int(np.isneginf(values).sum()),
        "median": median,
        "mean": mean,
    }
    if counts_only:
        result.pop("median")
        result.pop("mean")
    return result


def validate_outputs(
    targets, variants, instruments, hop_length, length_policy, variant_names=VARIANTS
):
    if set(variants) != set(variant_names):
        raise ValueError("all requested variants are required")
    length = targets["mixture"].shape[1]
    lengths = {}
    for variant in variant_names:
        if set(variants[variant]) != set(instruments):
            raise ValueError(f"incorrect target set: {variant}")
        lengths[variant] = {}
        for name in instruments:
            audio = np.asarray(variants[variant][name])
            if audio.ndim != 2 or audio.shape[0] != 2 or not np.isfinite(audio).all():
                raise ValueError(f"invalid output: {variant}/{name}")
            size = audio.shape[1]
            # Permit only the known original legacy fractional-hop tail loss.
            allowed_loss = (
                hop_length - 1
                if (variant == "legacy_unmodified" and length_policy == "common-crop")
                else 0
            )
            if not 0 <= length - size <= allowed_loss or size == 0:
                raise ValueError(f"unexpected output length: {variant}/{name}: {size}/{length}")
            lengths[variant][name] = size
    return min(n for values in lengths.values() for n in values.values()), lengths


def score_variants(
    targets,
    variants,
    instruments,
    hop_length=1024,
    length_policy="common-crop",
    variant_names=VARIANTS,
):
    import museval
    from evaluate_preview import si_sdr

    common, lengths = validate_outputs(
        targets, variants, instruments, hop_length, length_policy, variant_names
    )
    reference = np.stack([targets[name][:, :common].T for name in instruments]).astype(np.float64)
    scores = {}
    for variant in variant_names:
        estimate = np.stack([variants[variant][name][:, :common].T for name in instruments])
        # museval's MUSDB convention: 1-second windows/hops, fixed whole-track
        # 512-tap filters (v4), source-image metrics, no source permutation.
        metric_error = None
        try:
            metrics = museval.evaluate(
                reference,
                estimate.astype(np.float64),
                win=RATE,
                hop=RATE,
                mode="v4",
                padding=False,
            )
        except ValueError as exc:
            # BSSEval cannot fit its system if an entire source/estimate is silent.
            # Preserve the selected song with explicit undefined scores/reason.
            if "non-silent" not in str(exc):
                raise
            metric_error = str(exc)
            metrics = [
                np.full((len(instruments), max(1, common // RATE)), np.nan) for _ in BSS_METRICS
            ]
        scores[variant] = {}
        for index, name in enumerate(instruments):
            scores[variant][name] = {
                "global_stereo_si_sdr_db": si_sdr(
                    targets[name][:, :common], variants[variant][name][:, :common]
                ),
                "bsseval_v4_error": metric_error,
                "bsseval_v4": {
                    metric: [encode_number(x) for x in values[index]]
                    for metric, values in zip(BSS_METRICS, metrics)
                },
            }
    return {
        "evaluated_samples": common,
        "discarded_reference_tail_samples": targets["mixture"].shape[1] - common,
        "output_samples": lengths,
        "scores": scores,
    }


def track_score(record, variant, name, metric):
    score = record["scores"][variant][name]
    if metric == "global_stereo_si_sdr_db":
        return score[metric]
    return statistics(score["bsseval_v4"]["sdr"])["median"]


def summarize(records, instruments):
    aggregates, paired = {}, {}
    for metric in ("bsseval_v4_sdr_db", "global_stereo_si_sdr_db"):
        aggregates[metric] = {
            variant: {
                name: statistics([track_score(r, variant, name, metric) for r in records])
                for name in instruments
            }
            for variant in VARIANTS
        }
        paired[metric] = {}
        for candidate, baseline in (
            ("spleeterpt", "tensorflow_current"),
            ("spleeterpt", "legacy_unmodified"),
            ("legacy_corrected", "legacy_unmodified"),
            ("legacy_corrected", "tensorflow_current"),
        ):
            comparison = {}
            for name in instruments:
                deltas = []
                for record in records:
                    a = decode_number(track_score(record, candidate, name, metric))
                    b = decode_number(track_score(record, baseline, name, metric))
                    deltas.append(float(a - b) if np.isfinite(a) and np.isfinite(b) else None)
                stats = statistics(deltas)
                finite = [x for x in deltas if x is not None]
                stats.update(
                    {
                        "candidate_better": sum(x > 0 for x in finite),
                        "candidate_worse": sum(x < 0 for x in finite),
                        "equal": sum(x == 0 for x in finite),
                        "max_absolute_delta_db": max(map(abs, finite), default=None),
                        "per_track_delta_db": dict(zip([r["id"] for r in records], deltas)),
                    }
                )
                comparison[name] = stats
            paired[metric][f"{candidate}_minus_{baseline}"] = comparison
    frame_coverage = {
        variant: {
            name: {
                metric: statistics(
                    [
                        value
                        for record in records
                        for value in record["scores"][variant][name]["bsseval_v4"][metric]
                    ],
                    counts_only=True,
                )
                for metric in BSS_METRICS
            }
            for name in instruments
        }
        for variant in VARIANTS
    }
    return {
        "track_count": len(records),
        "aggregates": aggregates,
        "paired_differences": paired,
        "frame_coverage": frame_coverage,
    }


def validate_run(output, require_complete=True):
    """Recompute statistics from every expected, hashed per-track result; fail closed."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    status = json.loads((output / "status.json").read_text())
    if require_complete and status["state"] != "complete":
        raise ValueError("run is not complete")
    if status["manifest_sha256"] != sha256(output / "manifest.json"):
        raise ValueError("manifest hash mismatch")
    instruments = manifest["instruments"]
    if instruments not in (["vocals", "accompaniment"], ["vocals", "drums", "bass", "other"]):
        raise ValueError("invalid MUSDB target set")
    selection = manifest["selection"]
    ids = [r["id"] for r in selection]
    if not ids or ids != sorted(set(ids)):
        raise ValueError("selection must be nonempty, unique and sorted")
    expected = {f"{index:04d}.json" for index in range(len(ids))}
    if {p.name for p in (output / "tracks").glob("*.json")} != expected:
        raise ValueError("missing or unexpected per-track results")
    if set(status["record_sha256"]) != expected:
        raise ValueError("incomplete result hash inventory")
    records = []
    for index, selected in enumerate(selection):
        filename = f"{index:04d}.json"
        path = output / "tracks" / filename
        if sha256(path) != status["record_sha256"][filename]:
            raise ValueError(f"result hash mismatch: {filename}")
        record = json.loads(path.read_text())
        if record["id"] != selected["id"] or record["input_sha256"] != selected["sha256"]:
            raise ValueError("result identity or input fingerprint mismatch")
        if record["manifest_sha256"] != status["manifest_sha256"]:
            raise ValueError("result belongs to another run")
        size = record["evaluated_samples"]
        if not isinstance(size, int) or size <= 0:
            raise ValueError("invalid evaluated length")
        start = round(manifest["start_seconds"] * RATE)
        requested = manifest["duration_seconds"]
        count = record["decoded_samples"] - start if requested is None else round(requested * RATE)
        if record["segment_start_sample"] != start or record["segment_samples"] != count:
            raise ValueError("segment does not match requested interval")
        if record["decoded_samples"] < start + record["segment_samples"] or (
            record["segment_samples"] - size != record["discarded_reference_tail_samples"]
        ):
            raise ValueError("invalid segment/tail accounting")
        if set(record["scores"]) != set(VARIANTS) or set(record["output_samples"]) != set(VARIANTS):
            raise ValueError("missing variants")
        for variant in VARIANTS:
            if set(record["scores"][variant]) != set(instruments) or (
                set(record["output_samples"][variant]) != set(instruments)
            ):
                raise ValueError("missing targets")
            for name in instruments:
                length = record["output_samples"][variant][name]
                loss = record["segment_samples"] - length
                allowed = (
                    manifest["model_config"]["hop_length"] - 1
                    if (
                        variant == "legacy_unmodified"
                        and manifest["length_policy"] == "common-crop"
                    )
                    else 0
                )
                if not 0 <= loss <= allowed or length < size:
                    raise ValueError("invalid output length")
                score = record["scores"][variant][name]
                decode_number(score["global_stereo_si_sdr_db"])
                if set(score["bsseval_v4"]) != set(BSS_METRICS):
                    raise ValueError("missing BSSEval metrics")
                for values in score["bsseval_v4"].values():
                    if len(values) != max(1, size // RATE):
                        raise ValueError("incorrect BSSEval frame count")
                    for value in values:
                        decode_number(value)
        if size != min(n for values in record["output_samples"].values() for n in values.values()):
            raise ValueError("evaluated interval differs from common output length")
        records.append(record)
    summary = json.loads((output / "summary.json").read_text())
    if summary != summarize(records, instruments):
        raise ValueError("summary does not match complete per-track results")
    if status.get("summary_sha256") != sha256(output / "summary.json"):
        raise ValueError("summary hash mismatch")
    return summary
