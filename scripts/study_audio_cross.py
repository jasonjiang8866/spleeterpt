"""CPU magnitude x inverse-STFT study; fixed Torch encoder and official TF decoder.

The primary 2x2 holds TF masks and TF carrier STFT fixed. Secondary bridge cells
cross post-logit arithmetic and carrier STFT independently, closing the gap to
the existing native-audio/TF-decoder control. No runtime implementation changes.
"""

import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
import torch
from diagnose_numerics import metrics, nchw, nhwc
from parity_metrics import ATOL, RTOL, agreement
from study_network_halves import graph_boundary, native_with_tf_decoder
from verify_audio_parity import OfficialGraph, audio_cases

from spleeterpt.audio import istft, ratio_masks, read_audio, separate, stft
from spleeterpt.checkpoint import load_model, read_checkpoint
from spleeterpt.pretrained import digest

BACKENDS = ("tf", "pt")
MUSIC_CASES = ("music_original", "music_multiwindow")


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def key(frontend, postlogit, carrier, inverse):
    return f"magnitude_{frontend}__postlogit_{postlogit}__carrier_{carrier}__inverse_{inverse}"


def waveform_metrics(reference, value):
    if reference.shape != value.shape:
        raise ValueError("waveform shape mismatch")
    row = agreement(reference, value)
    row["output_sha256"] = array_hash(value)
    error = np.abs(value.astype(np.float64) - reference)
    row["peak_channel_sample"] = list(map(int, np.unravel_index(error.argmax(), error.shape)))
    return row


def factorial_contrasts(reference, y00, y10, y01, y11, axes=("frontend", "inverse")):
    """Signed samplewise contrasts, never differences of nonlinear failure counts.

    10 changes only magnitude, 01 only inverse STFT. Ratios use the original TF
    reference's unchanged tolerance as a scale, not as a new acceptance gate.
    """
    values = (reference, y00, y10, y01, y11)
    if any(v.shape != reference.shape or not np.isfinite(v).all() for v in values):
        raise ValueError("contrasts require matching finite waveforms")
    r, a, b, c, d = (v.astype(np.float64) for v in values)
    limit = ATOL + RTOL * np.abs(r)
    left, right = axes
    effects = {
        f"{left}_at_tf_{right}": b - a,
        f"{left}_at_pt_{right}": d - c,
        f"{right}_at_tf_{left}": c - a,
        f"{right}_at_pt_{left}": d - b,
        "interaction": (d - c) - (b - a),
    }
    return {
        name: {
            "max_absolute_delta": float(np.max(np.abs(delta))),
            "rms_delta": float(np.sqrt(np.mean(delta**2))),
            "max_delta_over_reference_tolerance": float(np.max(np.abs(delta) / limit)),
        }
        for name, delta in effects.items()
    }


def native_magnitude(spectrum, config):
    windows = []
    for start in range(0, spectrum.shape[-2], config.segment_frames):
        chunk = spectrum[:, start : start + config.segment_frames]
        value = torch.nn.functional.pad(
            chunk.abs()[..., : config.frequency_bins],
            (0, 0, 0, config.segment_frames - chunk.shape[-2]),
        )
        windows.append(nhwc(value.unsqueeze(0)))
    return np.concatenate(windows)


def fixed_decoder_logits(oracle, model, magnitude):
    # Exactly one model window and stem per TF session call, as in the existing
    # native_with_tf_decoder control. No Torch decoder execution in this arm.
    windows = []
    for window in magnitude:
        stems = []
        for index, net in enumerate(model.unets.values()):
            raw, value = [], nchw(window[None])
            for i, conv in enumerate(net.encoder):
                value = conv(value)
                raw.append(nhwc(value))
                if i < 5:
                    value = net.encoder_activation(net.encoder_norm[i](value))
            cut, output = graph_boundary(oracle.graph, index)
            feed = {oracle.builder.spectrogram_feature: window[None], **dict(zip(cut, raw))}
            stems.append(oracle.session.run(output, feed))
        windows.append(np.stack(stems, axis=-1))
    return np.concatenate(windows)


def native_masked_spectra(logits, magnitude, spectrum, config):
    """Replay native post-logit arithmetic and its per-window tensor layouts."""
    result = torch.empty((len(config.instruments), *spectrum.shape), dtype=spectrum.dtype)
    for i, start in enumerate(range(0, spectrum.shape[-2], config.segment_frames)):
        chunk = spectrum[:, start : start + config.segment_frames]
        value = torch.from_numpy(logits[i : i + 1]).permute(0, 4, 3, 1, 2).contiguous()
        masks = value.softmax(1) if config.mask_activation == "softmax" else value.sigmoid()
        estimates = masks * nchw(magnitude[i : i + 1]).unsqueeze(1)
        masks = ratio_masks(estimates, config)[0, :, :, : chunk.shape[-2]]
        result[:, :, start : start + chunk.shape[-2]] = masks * chunk.unsqueeze(0)
    return result


def tf_layout(values, stems):
    return torch.from_numpy(np.stack([values[s].transpose(2, 0, 1) for s in stems]))


def reconstruct(oracle, config, waveform, masked, backend):
    if backend == "pt":
        return dict(zip(config.instruments, istft(masked, config, waveform.shape[-1]).numpy()))
    feed = {oracle.waveform: waveform.numpy().T}
    for i, stem in enumerate(config.instruments):
        feed[oracle.builder.masked_stfts[stem]] = masked[i].numpy().transpose(1, 2, 0)
    return {s: v.T for s, v in oracle.session.run(oracle.outputs, feed).items()}


def run_case(oracle, model, waveform):
    config, stems = model.config, model.config.instruments
    feed = {oracle.waveform: waveform.numpy().T}
    feature, carrier = oracle.builder.spectrogram_feature, oracle.builder.stft_feature
    stacked = oracle.graph.get_tensor_by_name("pretrained/stack:0")
    tf_magnitude, tf_spectrum = oracle.session.run([feature, carrier], feed)
    pt_spectrum = stft(waveform, config, inference=True)
    pt_magnitude = native_magnitude(pt_spectrum, config)
    spectra = {"tf": torch.from_numpy(tf_spectrum.transpose(2, 0, 1).copy()), "pt": pt_spectrum}
    magnitudes = {"tf": tf_magnitude, "pt": pt_magnitude}
    if tf_magnitude.shape != pt_magnitude.shape or spectra["tf"].shape != pt_spectrum.shape:
        raise ValueError("frontend window geometry mismatch")
    reference = oracle.separate(waveform)
    native = {s: v.numpy() for s, v in separate(model, waveform).items()}
    existing_hybrid = {
        s: v.numpy() for s, v in native_with_tf_decoder(oracle, model, waveform).items()
    }
    controls, waves, rows, logits_by_front, bridge_waves = {}, {}, {}, {}, {}

    # Feeding the official masked spectra must reproduce the untouched oracle.
    official_masked = tf_layout(oracle.session.run(oracle.builder.masked_stfts, feed), stems)
    refeed = reconstruct(oracle, config, waveform, official_masked, "tf")
    controls["official_masked_refeed_exact"] = all(
        np.array_equal(reference[s], refeed[s]) for s in stems
    )
    for front in BACKENDS:
        magnitude = magnitudes[front]
        logits = fixed_decoder_logits(oracle, model, magnitude)
        logits_by_front[front] = logits
        for post, phase in itertools.product(BACKENDS, repeat=2):
            spectrum = spectra[phase]
            if post == "tf":
                cell_feed = {
                    **feed,
                    feature: magnitude,
                    stacked: logits,
                    carrier: spectrum.numpy().transpose(1, 2, 0),
                }
                masked = tf_layout(
                    oracle.session.run(oracle.builder.masked_stfts, cell_feed), stems
                )
            else:
                masked = native_masked_spectra(logits, magnitude, spectrum, config)
            for inverse in BACKENDS:
                cell = key(front, post, phase, inverse)
                value = reconstruct(oracle, config, waveform, masked, inverse)
                rows[cell] = {s: waveform_metrics(reference[s], value[s]) for s in stems}
                if post == phase == "tf":
                    waves[front, inverse] = value
                if inverse == "pt":
                    bridge_waves[front, post, phase] = value
                if front == post == phase == inverse == "pt":
                    controls["native_hybrid_replay_exact"] = all(
                        np.array_equal(value[s], existing_hybrid[s]) for s in stems
                    )
                    controls["native_hybrid_replay"] = {
                        s: waveform_metrics(existing_hybrid[s], value[s]) for s in stems
                    }
                # Feeding masked tensors must not change the surrounding TF graph.
                if post == "tf" and inverse == "tf":
                    direct = oracle.session.run(oracle.outputs, cell_feed)
                    controls[f"tf_post_refeed_{front}_{phase}_exact"] = all(
                        np.array_equal(value[s], direct[s].T) for s in stems
                    )
    return {
        "decoded_pcm_sha256": array_hash(waveform.numpy()),
        "samples": waveform.shape[-1],
        "frames": pt_spectrum.shape[-2],
        "windows": len(tf_magnitude),
        "last_window_frames": (pt_spectrum.shape[-2] - 1) % config.segment_frames + 1,
        "boundaries": {
            "magnitude": metrics(tf_magnitude, pt_magnitude),
            "carrier_stft": metrics(tf_spectrum, pt_spectrum.numpy().transpose(1, 2, 0)),
            "fixed_decoder_logits": metrics(logits_by_front["tf"], logits_by_front["pt"]),
            "magnitude_hashes": {k: array_hash(v) for k, v in magnitudes.items()},
            "logit_hashes": {k: array_hash(v) for k, v in logits_by_front.items()},
        },
        "controls": controls,
        "attribution_valid": all(v for k, v in controls.items() if k.endswith("_exact")),
        "native_pipeline": {s: waveform_metrics(reference[s], native[s]) for s in stems},
        "existing_native_tf_decoder": {
            s: waveform_metrics(reference[s], existing_hybrid[s]) for s in stems
        },
        "cells": rows,
        "primary_contrasts": {
            s: factorial_contrasts(
                reference[s],
                waves["tf", "tf"][s],
                waves["pt", "tf"][s],
                waves["tf", "pt"][s],
                waves["pt", "pt"][s],
            )
            for s in stems
        },
        "bridge_contrasts_at_pt_inverse": {
            front: {
                s: factorial_contrasts(
                    reference[s],
                    bridge_waves[front, "tf", "tf"][s],
                    bridge_waves[front, "pt", "tf"][s],
                    bridge_waves[front, "tf", "pt"][s],
                    bridge_waves[front, "pt", "pt"][s],
                    axes=("postlogit", "carrier"),
                )
                for s in stems
            }
            for front in BACKENDS
        },
    }


def magnitude_spectra(oracle, config, waveform, official_spectrum):
    """Cross the analysis window and FFT backend with the native frame geometry."""
    n_fft, hop = config.n_fft, config.hop_length
    torch_window = torch.hann_window(n_fft, dtype=waveform.dtype)
    with oracle.graph.as_default():
        tf_window = tf.signal.hann_window(n_fft, dtype=tf.float32)
        padded_tf = tf.pad(tf.transpose(oracle.waveform), [[0, 0], [n_fft, 0]])
        tf_spectra = {
            backend: tf.signal.stft(
                padded_tf,
                n_fft,
                hop,
                window_fn=lambda _length, dtype, coefficients=window: coefficients,
                pad_end=True,
            )
            for backend, window in (
                ("tf", tf_window),
                ("pt", tf.constant(torch_window.numpy())),
            )
        }
    tf_values, tf_replayed = oracle.session.run(
        [tf_window, tf_spectra], {oracle.waveform: waveform.numpy().T}
    )
    if not np.array_equal(tf_replayed["tf"].transpose(1, 2, 0), official_spectrum):
        raise ValueError("magnitude study did not reproduce official TF STFT")
    frame_count = official_spectrum.shape[0]
    needed = n_fft + (frame_count - 1) * hop
    padded_pt = torch.nn.functional.pad(waveform, (n_fft, needed - n_fft - waveform.shape[-1]))
    windows = {"tf": torch.from_numpy(tf_values), "pt": torch_window}
    spectra = {("tf", backend): torch.from_numpy(spec) for backend, spec in tf_replayed.items()}
    for backend, window in windows.items():
        spectra["pt", backend] = torch.stft(
            padded_pt, n_fft, hop, window=window, center=False, return_complex=True
        ).transpose(-2, -1)
    native = stft(waveform, config, inference=True)
    if not torch.equal(spectra["pt", "pt"], native) or any(
        spec.shape != native.shape for spec in spectra.values()
    ):
        raise ValueError("magnitude study did not reproduce native STFT or frame geometry")
    return spectra, windows


def run_magnitude_case(oracle, model, waveform):
    config, stems = model.config, model.config.instruments
    feed = {oracle.waveform: waveform.numpy().T}
    feature, carrier = oracle.builder.spectrogram_feature, oracle.builder.stft_feature
    stacked = oracle.graph.get_tensor_by_name("pretrained/stack:0")
    official_magnitude, official_spectrum = oracle.session.run([feature, carrier], feed)
    reference = oracle.separate(waveform)
    spectra, windows = magnitude_spectra(oracle, config, waveform, official_spectrum)
    waves, cells = {}, {}
    for (fft, hann), spectrum in spectra.items():
        magnitudes = {
            "tf": oracle.session.run(feature, {carrier: spectrum.numpy().transpose(1, 2, 0)}),
            "pt": native_magnitude(spectrum, config),
        }
        if (fft, hann) == ("tf", "tf") and not np.array_equal(magnitudes["tf"], official_magnitude):
            raise ValueError("TF STFT refeed did not reproduce the official magnitude")
        for absolute, magnitude in magnitudes.items():
            logits = fixed_decoder_logits(oracle, model, magnitude)
            # Preserve the official carrier STFT, masks and reconstruction.
            outputs = oracle.session.run(
                oracle.outputs, {**feed, feature: magnitude, stacked: logits}
            )
            value = {s: outputs[s].T for s in stems}
            waves[fft, hann, absolute] = value
            cells[f"fft_{fft}__hann_{hann}__abs_{absolute}"] = {
                "magnitude": metrics(official_magnitude, magnitude),
                "magnitude_sha256": array_hash(magnitude),
                "logits_sha256": array_hash(logits),
                "waveforms": {s: waveform_metrics(reference[s], value[s]) for s in stems},
            }
    return {
        "decoded_pcm_sha256": array_hash(waveform.numpy()),
        "samples": waveform.shape[-1],
        "frames": official_spectrum.shape[0],
        "windows": len(official_magnitude),
        "last_window_frames": (official_spectrum.shape[0] - 1) % config.segment_frames + 1,
        "hann": metrics(windows["tf"].numpy(), windows["pt"].numpy()),
        "spectra": {
            f"fft_{fft}__hann_{hann}": {
                "sha256": array_hash(spec.numpy()),
                **metrics(official_spectrum, spec.numpy().transpose(1, 2, 0)),
            }
            for (fft, hann), spec in spectra.items()
        },
        "cells": cells,
        "fft_hann_contrasts_by_abs": {
            absolute: {
                s: factorial_contrasts(
                    reference[s],
                    waves["tf", "tf", absolute][s],
                    waves["pt", "tf", absolute][s],
                    waves["tf", "pt", absolute][s],
                    waves["pt", "pt", absolute][s],
                    axes=("fft", "hann"),
                )
                for s in stems
            }
            for absolute in BACKENDS
        },
        "abs_contrasts": {
            f"fft_{fft}__hann_{hann}": {
                s: float(
                    np.max(
                        np.abs(
                            waves[fft, hann, "tf"][s].astype(np.float64) - waves[fft, hann, "pt"][s]
                        )
                    )
                )
                for s in stems
            }
            for fft, hann in spectra
        },
        "attribution_valid": True,
    }


def environment():
    cpu = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": next(
            (s.split(":", 1)[1].strip() for s in cpu.splitlines() if s.startswith("model name")),
            None,
        ),
        "cpu_flags": next(
            (s.split(":", 1)[1].strip() for s in cpu.splitlines() if s.startswith("flags")), None
        ),
        "torch": torch.__version__,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "soundfile": sf.__version__,
        "libsndfile": sf.__libsndfile_version__,
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_build": torch.__config__.show(),
        "tf_build": tf.sysconfig.get_build_info(),
        "torch_mkldnn": torch.backends.mkldnn.enabled,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "environment_variables": {
            k: os.getenv(k)
            for k in (
                "TF_ENABLE_ONEDNN_OPTS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "TF_NUM_INTRAOP_THREADS",
                "TF_NUM_INTEROP_THREADS",
                "ATEN_CPU_CAPABILITY",
            )
        },
        "tf_session_threads": {"intra": 2, "inter": 2},
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replicate", required=True)
    parser.add_argument(
        "--magnitude-breakdown",
        action="store_true",
        help="cross FFT, Hann coefficients and complex magnitude on fixed downstream TF audio",
    )
    args = parser.parse_args()
    if os.getenv("TF_ENABLE_ONEDNN_OPTS") != "0":
        raise ValueError("set TF_ENABLE_ONEDNN_OPTS=0 before starting Python")
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    pins = json.loads((root / "scripts/five_stem_source.json").read_text())
    history = json.loads(
        (root / "docs/results/five-stem-20261102/5stems-strict-final.json").read_text()
    )
    checkpoint, source = args.weights / "5stems.pt", args.upstream_root / "audio_example.mp3"
    upstream = subprocess.check_output(
        ["git", "-C", args.upstream_root, "rev-parse", "HEAD"], text=True
    ).strip()
    if (upstream, digest(source), digest(checkpoint)) != (
        pins["upstream_commit"],
        pins["audio_sha256"],
        pins["converted_sha256"],
    ):
        raise ValueError("calibration provenance mismatch")
    payload = read_checkpoint(checkpoint)
    tf_hashes = {name: digest(args.weights / "5stems" / name) for name in payload["source_sha256"]}
    if tf_hashes != payload["source_sha256"]:
        raise ValueError("TF checkpoint differs from converted weights")
    tf_hashes["model.meta"] = digest(args.weights / "5stems/model.meta")
    if tf_hashes["model.meta"] != history["tf_checkpoint_files"][".meta"]:
        raise ValueError("TF graph differs from historical reference")
    model = load_model(checkpoint).eval()
    waveforms = audio_cases(read_audio(source, model.config.sample_rate), model.config, True)
    for name in MUSIC_CASES:
        if (
            array_hash(waveforms[name].numpy())
            != history["case_provenance"][name]["decoded_pcm_sha256"]
        ):
            raise ValueError(f"{name}: calibration PCM mismatch")
    # Include new diagnostic sources even before committing a local preparation.
    sources = sorted(
        {
            str(p.relative_to(root))
            for folder in ("src", "scripts")
            for p in (root / folder).rglob("*")
            if p.suffix in (".py", ".json")
        }
        | {"pyproject.toml", "uv.lock"}
    )
    code_hashes = {
        p: digest(root / p)
        for p in sources
        if (p.startswith(("src/", "scripts/")) and p.endswith((".py", ".json")))
        or p in ("pyproject.toml", "uv.lock")
    }
    report = {
        "schema_version": 1,
        "replicate": args.replicate,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "working_tree_modified": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root)
        ),
        "code_sha256": code_hashes,
        "upstream_commit": upstream,
        "checkpoint_sha256": digest(checkpoint),
        "tf_checkpoint_sha256": tf_hashes,
        "audio_sha256": digest(source),
        "config": model.config.to_dict(),
        "thresholds": {"atol": ATOL, "rtol": RTOL, "changed": False},
        "environment": environment(),
        "fixed_network": "Torch encoder / official TensorFlow decoder; one window per call",
        "primary_factors": "magnitude frontend x inverse STFT; TF postlogit and TF carrier fixed",
        "postlogit_scope": "softmax, magnitude multiplication, ratio masks, extension, complex multiplication",
        "bridge_factors": "postlogit arithmetic and carrier STFT crossed independently",
        "holdouts_used": False,
        "release_eligible": False,
        "cases": {},
    }
    oracle = OfficialGraph(
        args.weights / "5stems/model", model.config.instruments, args.upstream_root
    )
    study_case = run_case
    if args.magnitude_breakdown:
        report["primary_factors"] = "FFT backend x Hann coefficients x complex absolute value"
        del report["bridge_factors"]
        study_case = run_magnitude_case
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name in MUSIC_CASES:
            report["cases"][name] = study_case(oracle, model, waveforms[name])
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(
                name,
                {
                    k: sum(r["strict_failed_samples"] for r in rows.get("waveforms", rows).values())
                    for k, rows in report["cases"][name]["cells"].items()
                },
                flush=True,
            )
    finally:
        oracle.close()
    if not all(case["attribution_valid"] for case in report["cases"].values()):
        raise SystemExit("FAIL: replay controls differ; attribution invalid; inspect report")
    print(
        "Diagnostic completed; strict numerical failures remain failures, not release acceptance."
    )


if __name__ == "__main__":
    main()
