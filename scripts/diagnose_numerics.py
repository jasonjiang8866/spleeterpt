"""CPU stage interventions against the restored official five-stem graph.

No thresholds are relaxed. This is numerical agreement, not a quality benchmark.
Set TF_ENABLE_ONEDNN_OPTS before starting Python to compare TF CPU backends.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from verify_audio_parity import OfficialGraph

from spleeterpt.audio import istft, ratio_masks, read_audio, stft
from spleeterpt.checkpoint import load_model
from spleeterpt.convert import layer_name


def metrics(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    delta = (
        candidate.astype(np.complex128 if np.iscomplexobj(reference) else np.float64) - reference
    )
    absolute = np.abs(delta)
    allowed = 1e-5 + 1e-4 * np.abs(reference)
    power = float(np.mean(np.abs(reference.astype(delta.dtype)) ** 2))
    mse = float(np.mean(absolute**2))
    return {
        "max_absolute_error": float(absolute.max()),
        "rmse": float(np.sqrt(mse)),
        "relative_l2_error": float(np.sqrt(mse / power)) if power else None,
        "strict_failures": int(np.count_nonzero(absolute > allowed)),
        "elements": reference.size,
        "max_tolerance_ratio": float(np.max(absolute / allowed)),
    }


def nchw(value):
    return torch.from_numpy(np.ascontiguousarray(value.transpose(0, 3, 1, 2)))


def nhwc(value):
    return value.detach().numpy().transpose(0, 2, 3, 1)


def trace_stem(oracle, net, index, magnitude):
    """Accumulated drift and each operator replayed with its exact TF input.

    Convolution/BN outputs are the actual imported graph tensors, including TF's
    fused inference BN. Feeding a tensor cuts its upstream graph dependencies.
    A fresh per-stem trace bounds memory; no output tensor is used as its input.
    """
    graph = oracle.graph

    def tensor(name):
        return graph.get_tensor_by_name("pretrained/" + name + ":0")

    stages = []
    for i, layer in enumerate(net.encoder):
        base = layer_name("conv2d", index * 7 + i)
        stages.append(
            (
                f"encoder.{i}.conv",
                layer,
                tensor(base + "/BiasAdd"),
                tensor(base + "/Conv2D").op.inputs[0],
            )
        )
        if i < 5:
            base_bn = layer_name("batch_normalization", index * 12 + i)
            norm = tensor(base_bn + "/cond/Merge")
            stages.append((f"encoder.{i}.bn", net.encoder_norm[i], norm, tensor(base + "/BiasAdd")))
            # The next convolution consumes the actual activated BN output.
            next_base = layer_name("conv2d", index * 7 + i + 1)
            act = tensor(next_base + "/Conv2D").op.inputs[0]
            stages.append((f"encoder.{i}.activation", net.encoder_activation, act, norm))
    for i, layer in enumerate(net.decoder):
        base = layer_name("conv2d_transpose", index * 6 + i)
        raw = tensor(base + "/BiasAdd")
        base_bn = layer_name("batch_normalization", index * 12 + 6 + i)
        norm = tensor(base_bn + "/cond/Merge")
        fused = tensor(base_bn + "/cond/FusedBatchNorm_1")
        act = fused.op.inputs[0]
        while act.op.type in ("Switch", "Identity"):
            act = act.op.inputs[0]
        stages.extend(
            [
                (f"decoder.{i}.conv", layer, raw, tensor(base + "/conv2d_transpose").op.inputs[2]),
                (f"decoder.{i}.activation", net.decoder_activation, act, raw),
                (f"decoder.{i}.bn", net.decoder_norm[i], norm, act),
            ]
        )
    stages.append(
        ("logits", net.output, tensor(layer_name("conv2d", index * 7 + 6) + "/BiasAdd"), norm)
    )
    refs = oracle.session.run(
        {name: (out, inp) for name, _, out, inp in stages},
        {oracle.builder.spectrogram_feature: magnitude},
    )
    # Activations are shared modules; capture in call order, not by module name.
    captured = {}
    handles = []
    calls = {}
    for name, module, _, _ in stages:
        calls.setdefault(module, []).append(name)
    for module, names in calls.items():
        pending = iter(names)

        def hook(module, args, output, pending=pending):
            captured[next(pending)] = nhwc(output).copy()

        handles.append(module.register_forward_hook(hook))
    try:
        net(nchw(magnitude))
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for name, module, _, _ in stages:
        ref, inp = refs[name]
        replay = nhwc(module(nchw(inp)))
        result[name] = {
            "accumulated": metrics(ref, captured[name]),
            "same_input": metrics(ref, replay),
        }
    normal = oracle.session.run(stages[-1][2], {oracle.builder.spectrogram_feature: magnitude})
    return result, metrics(normal, refs["logits"][0])


def native_softmax(logits):
    """Use the runtime model's contiguous N,S,C,T,F layout and stem axis."""
    value = torch.from_numpy(logits).permute(0, 4, 3, 1, 2).contiguous()
    return value.softmax(1).permute(0, 3, 4, 2, 1).numpy()


def tf64_logits(model, magnitude):
    """Independent FP64 operator reconstruction; not the archived FP32 oracle.

    TF FusedBatchNorm does not support float64, so use its mathematical inference
    formula. Shared FP32 checkpoint values are promoted without changing weights.
    """

    def constant(value):
        return tf.constant(value.detach().numpy(), dtype=tf.float64)

    def conv(x, layer):
        kernel = tf.transpose(constant(layer.weight), [2, 3, 1, 0])
        return tf.nn.bias_add(
            tf.nn.conv2d(
                x,
                kernel,
                strides=[1, *layer.stride, 1],
                padding="SAME",
                dilations=[1, *layer.dilation, 1],
            ),
            constant(layer.bias),
        )

    def bn(x, layer):
        return tf.nn.batch_normalization(
            x,
            constant(layer.running_mean),
            constant(layer.running_var),
            constant(layer.bias),
            constant(layer.weight),
            1e-3,
        )

    def activate(x, name):
        if name == "leaky_relu":
            return tf.nn.leaky_relu(x, alpha=0.2)
        return {"elu": tf.nn.elu, "relu": tf.nn.relu}[name](x)

    outputs = []
    for net in model.unets.values():
        x = tf.constant(magnitude, dtype=tf.float64)
        skips = []
        for i, layer in enumerate(net.encoder):
            x = conv(x, layer)
            skips.append(x)
            if i < 5:
                x = activate(bn(x, net.encoder_norm[i]), model.config.encoder_activation)
        for i, layer in enumerate(net.decoder):
            kernel = tf.transpose(constant(layer.weight), [2, 3, 1, 0])
            x = tf.nn.conv2d_transpose(
                x,
                kernel,
                [x.shape[0], x.shape[1] * 2, x.shape[2] * 2, layer.out_channels],
                [1, 2, 2, 1],
                "SAME",
            )
            x = bn(
                activate(tf.nn.bias_add(x, constant(layer.bias)), model.config.decoder_activation),
                net.decoder_norm[i],
            )
            if i < 5:
                x = tf.concat([skips[4 - i], x], -1)
        outputs.append(conv(x, net.output).numpy())
    return np.stack(outputs, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--tf-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-float64", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_model(args.checkpoint).eval()
    config = model.config
    if config.mask_activation != "softmax":
        parser.error("this diagnostic targets cross-stem softmax checkpoints")
    waveform = read_audio(args.input, config.sample_rate)
    oracle = OfficialGraph(args.tf_checkpoint, config.instruments, args.upstream_root)
    report = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "decoded_pcm_sha256": hashlib.sha256(waveform.numpy().tobytes()).hexdigest(),
        "decoded_samples": waveform.shape[-1],
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "meta_sha256": hashlib.sha256(
            Path(str(args.tf_checkpoint) + ".meta").read_bytes()
        ).hexdigest(),
        "source_checkpoint": torch.load(args.checkpoint, weights_only=True).get("source_sha256"),
        "environment": {
            "torch": torch.__version__,
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "machine": platform.machine(),
            "tf_onednn": os.environ.get("TF_ENABLE_ONEDNN_OPTS", "default"),
            "threads": 2,
            "device": "cpu",
        },
        "note": "First network window; interventions isolate numerical stages, not separation quality. Layer strict_failures apply the waveform tolerance only as a drift indicator, not as a layer correctness criterion.",
    }
    try:
        feed = {oracle.waveform: waveform.numpy().T}
        feature = oracle.builder.spectrogram_feature
        spectrum, magnitude = oracle.session.run([oracle.builder.stft_feature, feature], feed)
        # Single fixed 512-frame window for all network interventions.
        magnitude = magnitude[:1]
        tspectrum = stft(waveform, config, inference=True).numpy().transpose(1, 2, 0)
        tmagnitude = np.zeros_like(magnitude)
        frames = min(config.segment_frames, spectrum.shape[0])
        tmagnitude[0, :frames] = np.abs(tspectrum[:frames, : config.frequency_bins])
        report["frontend"] = {
            "stft": metrics(spectrum, tspectrum),
            "magnitude": metrics(magnitude, tmagnitude),
        }
        logits_tensor = oracle.graph.get_tensor_by_name("pretrained/stack:0")
        softmax_tensor = oracle.graph.get_tensor_by_name("pretrained/softmax/Softmax:0")
        logits, masks = oracle.session.run([logits_tensor, softmax_tensor], {feature: magnitude})
        report["logit_range"] = [float(logits.min()), float(logits.max())]
        report["layers"] = {}
        report["trace_fetch_invariance"] = {}
        with torch.inference_mode():
            for index, (name, net) in enumerate(model.unets.items()):
                report["layers"][name], report["trace_fetch_invariance"][name] = trace_stem(
                    oracle, net, index, magnitude
                )
                print("traced", name, flush=True)

            def run_logits(value):
                return np.stack([nhwc(net(nchw(value))) for net in model.unets.values()], -1)

            torch_logits = run_logits(magnitude)
            shared = torch.from_numpy(logits)
            report["network"] = {
                "same_magnitude_logits": metrics(logits, torch_logits),
                "same_magnitude_softmax": metrics(masks, native_softmax(torch_logits)),
                "same_tf_logits_softmax": metrics(masks, native_softmax(logits)),
                "same_tf_logits_last_axis_softmax": metrics(masks, shared.softmax(-1).numpy()),
                "softmax_fp64_on_tf_logits": metrics(masks, shared.double().softmax(-1).numpy()),
            }
            # Isolate logit drift while keeping exactly the reference softmax kernel.
            swapped_masks = oracle.session.run(softmax_tensor, {logits_tensor: torch_logits})
            report["network"]["torch_logits_tf_softmax"] = metrics(masks, swapped_masks)
            coordinate = np.unravel_index(np.argmax(np.abs(swapped_masks - masks)), masks.shape)[
                :-1
            ]
            report["worst_softmax_bin"] = {
                "coordinate_batch_time_frequency_channel": [int(v) for v in coordinate],
                "tf_logits": logits[coordinate].tolist(),
                "torch_logits": torch_logits[coordinate].tolist(),
                "tf_masks": masks[coordinate].tolist(),
                "torch_logits_tf_masks": swapped_masks[coordinate].tolist(),
            }
            alternatives = {}
            for label, enabled, channels_last in [
                ("no_mkldnn", False, False),
                ("channels_last", True, True),
            ]:
                with torch.backends.mkldnn.flags(enabled=enabled):
                    if channels_last:
                        model.to(memory_format=torch.channels_last)
                    alt = run_logits(magnitude)
                    alternatives[label] = alt
                    report["network"][label] = {
                        "logits": metrics(logits, alt),
                        "softmax": metrics(masks, native_softmax(alt)),
                    }
            model.to(memory_format=torch.contiguous_format)
            if not args.skip_float64:
                model.double()
                double_logits = run_logits(magnitude.astype(np.float64))
                ref64 = tf64_logits(model, magnitude)
                report["precision"] = {
                    "tf64_vs_torch64": metrics(ref64, double_logits),
                    "tf64_vs_tf32": metrics(ref64, logits),
                    "tf64_vs_torch32": metrics(ref64, torch_logits),
                    "tf32_vs_torch64": metrics(logits, double_logits),
                    "fp64_softmax_cross_framework": metrics(
                        tf.nn.softmax(ref64, -1).numpy(),
                        torch.from_numpy(double_logits).softmax(-1).numpy(),
                    ),
                }
                model.float()
            # Fix exact TF network estimates in BOTH complete audio backends.
            estimates = oracle.session.run(oracle.builder.model_outputs, feed)
            fixed = torch.from_numpy(
                np.stack([estimates[f"{n}_spectrogram"] for n in config.instruments], 1).transpose(
                    0, 1, 4, 2, 3
                )
            )
            fixed_masks = ratio_masks(fixed, config)
            tf_masks = oracle.session.run(
                oracle.builder.masks,
                {**feed, **{oracle.builder.model_outputs[k]: v for k, v in estimates.items()}},
            )
            report["backend_same_estimates"] = {}
            expected_audio = oracle.separate(waveform)
            # For a one-window clip, change only one stage in the official audio graph.
            # The mixture phase, ratio-mask and synthesis remain the TF reference.
            if spectrum.shape[0] <= config.segment_frames:
                interventions = {
                    "torch_magnitude_tf_network": {feature: tmagnitude},
                    "torch_logits_tf_softmax": {logits_tensor: torch_logits},
                    "same_logits_torch_softmax": {softmax_tensor: native_softmax(logits)},
                }
                for label, alt in alternatives.items():
                    interventions[label + "_logits_tf_softmax"] = {logits_tensor: alt}
                if not args.skip_float64:
                    interventions["torch64_logits_tf32_softmax"] = {
                        logits_tensor: double_logits.astype(np.float32)
                    }
                report["waveform_interventions"] = {}
                for label, cut in interventions.items():
                    changed = oracle.session.run(oracle.outputs, {**feed, **cut})
                    report["waveform_interventions"][label] = {
                        n: metrics(expected_audio[n], changed[n].T) for n in config.instruments
                    }
            for i, name in enumerate(config.instruments):
                mask = fixed_masks[:, i].permute(0, 2, 3, 1).reshape(-1, config.n_fft // 2 + 1, 2)
                mask = mask[: spectrum.shape[0]]
                synth = istft(
                    torch.from_numpy((mask.numpy() * spectrum).transpose(2, 0, 1)),
                    config,
                    waveform.shape[-1],
                ).numpy()
                synth_torch_spectrum = istft(
                    torch.from_numpy((mask.numpy() * tspectrum).transpose(2, 0, 1)),
                    config,
                    waveform.shape[-1],
                ).numpy()
                # Independently cut at the exact TF masked complex STFT.
                masked = oracle.session.run(oracle.builder.masked_stfts[name], feed)
                inverse = istft(
                    torch.from_numpy(masked.transpose(2, 0, 1)), config, waveform.shape[-1]
                ).numpy()
                report["backend_same_estimates"][name] = {
                    "ratio_mask": metrics(tf_masks[name], mask.numpy()),
                    "waveform_torch_spectrum": metrics(expected_audio[name], synth_torch_spectrum),
                    "waveform_tf_spectrum": metrics(expected_audio[name], synth),
                    "same_masked_stft_istft": metrics(expected_audio[name], inverse),
                }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("wrote", args.output, flush=True)
    finally:
        oracle.close()


if __name__ == "__main__":
    main()
