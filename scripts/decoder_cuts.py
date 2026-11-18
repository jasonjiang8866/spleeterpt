"""Cross decoder prefixes on frozen Torch skips and the official TF magnitude."""

import contextlib
import copy

import numpy as np
import torch
from diagnose_numerics import nchw, nhwc
from study_bn_order import decoder_controls, separate_with_tf_convolutions
from study_network_halves import graph_boundary, torch_decode

from spleeterpt.audio import separate
from spleeterpt.convert import layer_name


def torch_suffix(net, raw, k, normalized):
    """Continue after the kth normalization, including its unconsumed skip."""
    x = normalized
    for i in range(k, len(net.decoder)):
        x = torch.cat((raw[5 - i], x), dim=1)
        x = net.decoder_norm[i](net.decoder_activation(net.decoder[i](x)))
    return net.output(x)


def decoder_logits(oracle, model, magnitude, directory):
    """Each of six blocks ends after transposed convolution, activation and BN."""
    all_modes = {}
    feature = oracle.builder.spectrogram_feature
    for w, window in enumerate(magnitude):
        for index, net in enumerate(model.unets.values()):
            x = nchw(window[None])
            raw, pt_stages = [], []
            with contextlib.ExitStack() as hooks:
                for layers, target in ((net.encoder, raw), (net.decoder_norm, pt_stages)):
                    for layer in layers:
                        hook = layer.register_forward_hook(
                            lambda m, a, out, t=target: t.append(out)
                        )
                        hooks.callback(hook.remove)
                pt_logits = net(x)
            if not torch.equal(torch_decode(net, raw), pt_logits):
                raise ValueError("native decoder endpoint replay failed")

            skips, end = graph_boundary(oracle.graph, index)
            stages = [
                oracle.graph.get_tensor_by_name(
                    "pretrained/"
                    + layer_name("batch_normalization", index * 12 + 6 + i)
                    + "/cond/Merge:0"
                )
                for i in range(6)
            ]
            feed = {feature: window[None], **dict(zip(skips, map(nhwc, raw)))}
            tf_logits = oracle.session.run(end, feed)
            tf_stages = oracle.session.run(stages, feed)
            modes = {"torch": nhwc(pt_logits), "tf": tf_logits}
            for k, (node, tf_value, pt_value) in enumerate(zip(stages, tf_stages, pt_stages), 1):
                if not torch.equal(torch_suffix(net, raw, k, pt_value), pt_logits):
                    raise ValueError(f"Torch decoder block {k} replay failed")
                if not np.array_equal(oracle.session.run(end, {**feed, node: tf_value}), tf_logits):
                    raise ValueError(f"TF decoder block {k} replay failed")
                modes[f"tf_prefix_{k}"] = nhwc(torch_suffix(net, raw, k, nchw(tf_value)))
                modes[f"torch_prefix_{k}"] = oracle.session.run(end, {**feed, node: nhwc(pt_value)})
            for mode, value in modes.items():
                if mode not in all_modes:
                    path = directory / f"{mode}.npy"
                    shape = (len(magnitude), *value.shape[1:], len(model.unets))
                    all_modes[mode] = np.lib.format.open_memmap(
                        path, mode="w+", dtype=value.dtype, shape=shape
                    )
                all_modes[mode][w, ..., index] = value[0]
    return {mode: directory / f"{mode}.npy" for mode in all_modes}


def adjacent_waveform(before, after, reference, atol, rtol):
    """Measure the same output samples on either side of a single block switch."""
    left, right, ref = (np.asarray(x, dtype=np.float64) for x in (before, after, reference))
    tolerance = atol + rtol * np.abs(ref)
    prev = np.abs(left - ref) > tolerance
    curr = np.abs(right - ref) > tolerance
    return {
        "fixed": int(np.count_nonzero(prev & ~curr)),
        "introduced": int(np.count_nonzero(curr & ~prev)),
        "max_waveform_change": float(np.max(np.abs(right - left))),
        "max_change_over_tolerance": float(np.max(np.abs(right - left) / tolerance)),
    }


def native_block_six(model, oracle, waveform):
    """Replace only the preactivation transposed convolution in the final block."""
    controls = decoder_controls(model, oracle, block=5)
    results = {"native": separate(model, waveform)}
    results["native_tf_conv6"] = separate_with_tf_convolutions(model, waveform, oracle, controls)
    with contextlib.ExitStack() as stack:
        for layer, _, _ in controls:
            widened = copy.deepcopy(layer).double()
            handle = layer.register_forward_hook(
                lambda module, args, out, conv=widened: torch.empty_like(out).copy_(
                    conv(args[0].double())
                )
            )
            stack.callback(handle.remove)
        results["native_wide_conv6"] = separate(model, waveform)
    return results
