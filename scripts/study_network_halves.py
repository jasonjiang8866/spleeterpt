"""CPU encoder/decoder swaps on fixed TF magnitudes with official TF synthesis."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from diagnose_numerics import metrics, nchw, nhwc
from parity_metrics import ATOL, RTOL, agreement
from verify_audio_parity import OfficialGraph, audio_cases

from spleeterpt.audio import read_audio, separate
from spleeterpt.checkpoint import load_model
from spleeterpt.convert import layer_name
from spleeterpt.pretrained import digest


def torch_decode(net, raw):
    # The six raw conv outputs are five skips and the unactivated bottleneck.
    x = raw[-1]
    for i, (conv, norm) in enumerate(zip(net.decoder, net.decoder_norm)):
        x = norm(net.decoder_activation(conv(x)))
        if i < 5:
            x = torch.cat((raw[4 - i], x), dim=1)
    return net.output(x)


def torch_boundary(net, magnitude, tf_boundary):
    logits, from_tf, raw_windows = [], [], []
    for window, value in enumerate(magnitude):
        # Match native separate(): one Torch window per model call.
        with ExitStack() as stack:
            raw = []
            for conv in net.encoder:
                handle = conv.register_forward_hook(
                    lambda _, __, output, raw=raw: raw.append(output)
                )
                stack.callback(handle.remove)
            normal = net(nchw(value[None]))
        if not torch.equal(normal, torch_decode(net, raw)):
            raise ValueError("Torch decoder replay changed its own logits")
        raw_windows.append([nhwc(t) for t in raw])
        exact = [nchw(t[window : window + 1]) for t in tf_boundary]
        from_tf.append(nhwc(torch_decode(net, exact)))
        logits.append(nhwc(normal))
    boundary = [np.concatenate(stage) for stage in zip(*raw_windows)]
    return np.concatenate(logits), np.concatenate(from_tf), boundary


def graph_boundary(graph, index):
    def conv_tensor(i):
        name = layer_name("conv2d", index * 7 + i)
        return graph.get_tensor_by_name(f"pretrained/{name}/BiasAdd:0")

    return [conv_tensor(i) for i in range(6)], conv_tensor(6)


def native_with_tf_decoder(oracle, model, waveform):
    # Install on each stem only for this diagnostic call. All surrounding audio,
    # mask, and softmax operations run through the normal Torch entry point.
    with ExitStack() as stack:
        for index, net in enumerate(model.unets.values()):
            cut, logits = graph_boundary(oracle.graph, index)
            raw = []
            for layer in net.encoder:
                handle = layer.register_forward_hook(
                    lambda _, __, output, raw=raw: raw.append(nhwc(output))
                )
                stack.callback(handle.remove)

            def decode(_, inputs, native, raw=raw, cut=cut, logits=logits):
                if len(raw) != len(cut):
                    raise ValueError("incomplete native encoder state")
                feeds = dict(zip(cut, raw))
                raw.clear()
                feeds[oracle.builder.spectrogram_feature] = nhwc(inputs[0])
                value = oracle.session.run(logits, feeds)
                return torch.empty_like(native).copy_(nchw(value))

            stack.callback(net.register_forward_hook(decode).remove)
        return separate(model, waveform)


def crossed_logits(oracle, model, magnitude):
    modes = {key: [] for key in ("tf_tf", "tf_refeed", "pt_pt", "tf_pt", "pt_tf")}
    graph = oracle.graph
    feed = {oracle.builder.spectrogram_feature: magnitude}
    for index, net in enumerate(model.unets.values()):
        cut, out = graph_boundary(graph, index)
        original = oracle.session.run(out, feed)
        tf_raw = oracle.session.run(cut, feed)
        native, hybrid, pt_raw = torch_boundary(net, magnitude, tf_raw)
        modes["pt_pt"].append(native)
        modes["tf_pt"].append(hybrid)
        modes["tf_tf"].append(original)
        for name, values in (("tf_refeed", tf_raw), ("pt_tf", pt_raw)):
            modes[name].append(oracle.session.run(out, {**feed, **dict(zip(cut, values))}))
    return {k: np.stack(v, axis=-1) for k, v in modes.items()}


def detailed(ref, result, baseline):
    if ref.shape != result.shape or baseline.shape != ref.shape:
        raise ValueError("waveform shape mismatch")
    row = agreement(ref, result)
    absolute = np.abs(result.astype(np.float64) - ref)
    limit = ATOL + RTOL * np.abs(ref.astype(np.float64))
    current = absolute > limit
    previous = np.abs(baseline.astype(np.float64) - ref) > limit
    row["fixed_vs_pt_pt"] = int(np.sum(previous & ~current))
    row["new_vs_pt_pt"] = int(np.sum(current & ~previous))
    return row


def run_case(oracle, model, waveform):
    feed = {oracle.waveform: waveform.numpy().T}
    feature = oracle.builder.spectrogram_feature
    stacked = oracle.graph.get_tensor_by_name("pretrained/stack:0")
    magnitude, reference_logits = oracle.session.run([feature, stacked], feed)
    expected = oracle.separate(waveform)
    modes = crossed_logits(oracle, model, magnitude)
    results = {}
    for mode, logits in modes.items():
        result = oracle.session.run(oracle.outputs, {**feed, stacked: logits})
        results[mode] = {name: v.T for name, v in result.items()}
    results["native_pipeline"] = {k: v.numpy() for k, v in separate(model, waveform).items()}
    return {
        "windows": len(magnitude),
        "logits": {m: metrics(reference_logits, v) for m, v in modes.items()},
        "waveforms": {
            m: {
                name: detailed(expected[name], audio[name], results["pt_pt"][name])
                for name in expected
            }
            for m, audio in results.items()
        },
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    scripts = Path(__file__).resolve().parent
    pins = json.loads((scripts / "five_stem_source.json").read_text())
    history = json.loads(
        (scripts.parent / "docs/results/five-stem-20261102/5stems-strict-final.json").read_text()
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
    model = load_model(checkpoint)
    audio = read_audio(source, model.config.sample_rate)
    oracle = OfficialGraph(
        args.weights / "5stems/model", model.config.instruments, args.upstream_root
    )
    report = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "upstream": upstream,
        "audio_sha256": pins["audio_sha256"],
        "checkpoint_sha256": pins["converted_sha256"],
        "environment": {
            "tf": tf.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tf_onednn": os.getenv("TF_ENABLE_ONEDNN_OPTS"),
            "torch_mkldnn": torch.backends.mkldnn.enabled,
            "threads": torch.get_num_threads(),
        },
        "thresholds": {"atol": ATOL, "rtol": RTOL},
        "cases": {},
        "native_decoder_gate": {},
        "release_eligible": False,
    }
    try:
        for name, wave in audio_cases(audio, model.config, True).items():
            pcm = hashlib.sha256(wave.numpy().tobytes()).hexdigest()
            if pcm != history["case_provenance"][name]["decoded_pcm_sha256"]:
                raise ValueError(f"{name}: PCM mismatch")
            reference = oracle.separate(wave)
            hybrid = native_with_tf_decoder(oracle, model, wave)
            rows = {stem: agreement(ref, hybrid[stem].numpy()) for stem, ref in reference.items()}
            report["native_decoder_gate"][name] = {"pcm_sha256": pcm, "stems": rows}
            print(
                name,
                "native_tf_decoder",
                {s: r["strict_failed_samples"] for s, r in rows.items()},
                flush=True,
            )
            if name not in ("music_original", "music_multiwindow"):
                continue
            case = run_case(oracle, model, wave)
            case["decoded_pcm_sha256"] = pcm
            report["cases"][name] = case
            for mode, rows in case["waveforms"].items():
                print(
                    name, mode, {s: r["strict_failed_samples"] for s, r in rows.items()}, flush=True
                )
    finally:
        oracle.close()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for case in report["cases"].values():
        for mode in ("tf_tf", "tf_refeed"):
            if any(
                not row["allclose_atol_1e-5_rtol_1e-4"] for row in case["waveforms"][mode].values()
            ):
                raise SystemExit(f"{mode}: control drift invalidates attribution; inspect artifact")


if __name__ == "__main__":
    main()
