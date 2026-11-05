"""Score previously exported candidate and TF piano/other estimates against TRIOS."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROLES = ("piano", "other")


def centered(samples):
    x = samples.astype(np.float64).ravel()
    return x - x.mean()


def score(reference, estimate):
    x, y = centered(reference), centered(estimate)
    energy = x @ x
    if np.sqrt(np.mean(x**2)) <= 1e-5:
        return None
    target = (y @ x / energy) * x
    residual = y - target
    a, b = target @ target, residual @ residual
    if a == 0 or b == 0:
        return None
    return float(10 * np.log10(a / b))


def leakage(reference, distractor, estimate):
    x, z, y = map(centered, (reference, distractor, estimate))
    if x @ x == 0 or z @ z == 0:
        return {"status": "undefined"}
    residual = z - (z @ x / (x @ x)) * x
    power = residual @ residual
    if power <= np.finfo(np.float64).eps * (z @ z):
        return {"status": "undefined"}
    ratio = (y @ residual) ** 2 / (power * (z @ z))
    if ratio == 0:
        return {"status": "zero", "power_ratio": 0.0}
    return {"status": "finite", "power_ratio": float(ratio), "db": float(10 * np.log10(ratio))}


def assess(inputs, tf, candidate):
    mix = inputs["mix"]
    result = {}
    for role, counterpart in (("piano", "other"), ("other", "piano")):
        ref = inputs[role]
        scores = {
            key: score(ref, est)
            for key, est in (("mix", mix), ("tf", tf[role]), ("candidate", candidate[role]))
        }
        complete = all(s is not None for s in scores.values())
        result[role] = {
            "si_sdr_db": scores,
            "estimate_pcm_sha256": {
                source: hashlib.sha256(data[role].astype("<f4").tobytes()).hexdigest()
                for source, data in (("tf", tf), ("candidate", candidate))
            },
            "candidate_minus_tf_db": scores["candidate"] - scores["tf"] if complete else None,
            "candidate_minus_mix_db": scores["candidate"] - scores["mix"] if complete else None,
            "leakage": {
                k: leakage(ref, inputs[counterpart], e[role])
                for k, e in (("candidate", candidate), ("tf", tf))
            },
            "passed": complete
            and scores["candidate"] > scores["mix"]
            and scores["candidate"] >= scores["tf"] - 0.1,
        }
    return result


def read(root, name, role):
    path = root / name / (role + ".wav")
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 44100 or wave.shape != (12 * rate, 2) or not np.isfinite(wave).all():
        raise ValueError(f"invalid audio geometry or samples: {path}")
    return wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", required=True, type=Path, help="prepared holdout corpus directory"
    )
    parser.add_argument("--tf", required=True, type=Path, help="root/<recording_id>/<role>.wav")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("five_stem_corpus.json").read_text())
    results = {}
    for item in manifest["records"]:
        if not item.get("trios"):
            continue
        name = item["recording_id"]

        inputs = {k: read(args.inputs, name, k) for k in ("mix", *ROLES)}
        for role, wave in inputs.items():
            if (
                hashlib.sha256(wave.astype("<f4").tobytes()).hexdigest()
                != item["trios"]["prepared_pcm"][role]
            ):
                raise ValueError(f"reference PCM differs from frozen registration: {name}/{role}")
        tf = {k: read(args.tf, name, k) for k in ROLES}
        candidate = {k: read(args.candidate, name, k) for k in ROLES}
        results[name] = assess(inputs, tf, candidate)
    if set(results) != {"trios-brahms", "trios-schubert"}:
        raise ValueError("expected both registered recordings")
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    if not all(row["passed"] for recording in results.values() for row in recording.values()):
        raise SystemExit("piano quality gate failed")


if __name__ == "__main__":
    main()
