"""Compare two FP32 inference BatchNorm orders against the frozen TF graph on CPU."""

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
from parity_metrics import agreement
from verify_audio_parity import OfficialGraph, audio_cases

from spleeterpt.audio import read_audio, separate
from spleeterpt.checkpoint import load_model
from spleeterpt.convert import layer_name
from spleeterpt.pretrained import digest

ORDERS = ("native", "center_first", "affine_first")


def ordered_bn(layer, x, order):
    scale = layer.weight * torch.rsqrt(layer.running_var + layer.eps)
    mean = layer.running_mean[None, :, None, None]
    bias = layer.bias[None, :, None, None]
    scale = scale[None, :, None, None]
    if order == "center_first":
        return (x - mean) * scale + bias
    if order == "affine_first":
        return x * scale + (bias - mean * scale)
    raise ValueError(order)


def first_layer_replay(model, oracle, waveform):
    feature = oracle.builder.spectrogram_feature
    magnitude = oracle.session.run(feature, {oracle.waveform: waveform.numpy().T})[:1]
    results = {}
    for index, (stem, net) in enumerate(model.unets.items()):
        conv = layer_name("conv2d", index * 7)
        bn = layer_name("batch_normalization", index * 12)
        value, expected = oracle.session.run(
            [f"pretrained/{conv}/BiasAdd:0", f"pretrained/{bn}/cond/Merge:0"],
            {feature: magnitude},
        )
        layer = net.encoder_norm[0]
        x = nchw(value)
        results[stem] = {"native": metrics(expected, nhwc(layer(x)))}
        for order in ORDERS[1:]:
            results[stem][order] = metrics(expected, nhwc(ordered_bn(layer, x, order)))
    return results


def decoder_controls(model, oracle):
    # Preselect decoder.1 from the archived same-input relative-L2 ranking.
    controls = []
    for i, net in enumerate(model.unets.values()):
        base = "pretrained/" + layer_name("conv2d_transpose", i * 6 + 1)
        output = oracle.graph.get_tensor_by_name(base + "/BiasAdd:0")
        source = oracle.graph.get_operation_by_name(base + "/conv2d_transpose").inputs[2]
        controls.append((net.decoder[1], source, output))
    return controls


def separate_with_order(model, waveform, order, oracle, controls):
    with ExitStack() as stack:
        if order == "tf_decoder1":
            for layer, source, target in controls:

                def substitute(module, inputs, output, source=source, target=target):
                    feature = oracle.builder.spectrogram_feature
                    config = model.config
                    shape = (inputs[0].shape[0], config.segment_frames, config.frequency_bins, 2)
                    values = oracle.session.run(
                        target, {source: nhwc(inputs[0]), feature: np.zeros(shape, np.float32)}
                    )
                    # Preserve the original layout seen by downstream Torch operators.
                    return torch.empty_like(output).copy_(nchw(values))

                stack.callback(layer.register_forward_hook(substitute).remove)
        elif order in ORDERS[1:]:
            for layer in model.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    handle = layer.register_forward_hook(
                        lambda module, args, output: ordered_bn(module, args[0], order)
                    )
                    stack.callback(handle.remove)
        return separate(model, waveform)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
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
    checkpoint = args.weights / "5stems.pt"
    rev = subprocess.check_output(
        ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if (digest(checkpoint), digest(args.input), rev) != (
        pins["converted_sha256"],
        pins["audio_sha256"],
        pins["upstream_commit"],
    ):
        raise ValueError("reference provenance mismatch")
    model = load_model(checkpoint)
    audio = read_audio(args.input, model.config.sample_rate)
    stems = model.config.instruments
    oracle = OfficialGraph(args.weights / "5stems/model", stems, args.upstream_root)
    orders = (*ORDERS, "tf_decoder1")
    controls = decoder_controls(model, oracle)
    report = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "upstream_commit": rev,
        "input_sha256": pins["audio_sha256"],
        "weights_sha256": pins["converted_sha256"],
        "environment": {
            "torch": torch.__version__,
            "tf": tf.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "onednn": os.getenv("TF_ENABLE_ONEDNN_OPTS"),
            "mkldnn": torch.backends.mkldnn.enabled,
            "threads": torch.get_num_threads(),
        },
        "cases": {},
        "failed_outputs": {order: [] for order in orders},
        "release_eligible": False,
    }
    try:
        report["shared_input_bn1"] = first_layer_replay(model, oracle, audio)
        for name, waveform in audio_cases(audio, model.config, True).items():
            pcm = hashlib.sha256(waveform.numpy().tobytes()).hexdigest()
            if pcm != history["case_provenance"][name]["decoded_pcm_sha256"]:
                raise ValueError(f"decoded input mismatch: {name}")
            reference = oracle.separate(waveform)
            case = {"pcm_sha256": pcm}
            report["cases"][name] = case
            for order in orders:
                actual = separate_with_order(model, waveform, order, oracle, controls)
                case[order] = {}
                for stem in stems:
                    ref, got = reference[stem], actual[stem].numpy()
                    result = agreement(ref, got)
                    error = abs(ref.astype(np.float64) - got)
                    result["peak_channel_sample"] = list(
                        map(int, np.unravel_index(error.argmax(), error.shape))
                    )
                    case[order][stem] = result
                    if not result["allclose_atol_1e-5_rtol_1e-4"]:
                        report["failed_outputs"][order].append([name, stem])
    finally:
        oracle.close()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Failed outputs by order:", json.dumps(report["failed_outputs"]))
    for order in orders:
        rows = [r for case in report["cases"].values() for r in case[order].values()]
        print(
            order,
            len(rows),
            "outputs;",
            sum(r["strict_failed_samples"] for r in rows),
            "failing samples; peak",
            max(r["max_absolute_error"] for r in rows),
        )


if __name__ == "__main__":
    main()
