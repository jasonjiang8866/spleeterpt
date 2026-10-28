"""Optional numerical oracle; TensorFlow is not a runtime dependency.

Run with tensorflow-cpu installed. This checks operators/topology using identical
weights, not official pretrained checkpoint conversion or training convergence.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch

from spleeterpt.audio import istft, stft
from spleeterpt.model import Spleeter

tf = pytest.importorskip("tensorflow")


def array(tensor):
    return tensor.detach().cpu().numpy()


def reference_unet(x, net, config):
    def conv(value, layer):
        kernel = array(layer.weight).transpose(2, 3, 1, 0)
        result = tf.nn.conv2d(
            value,
            kernel,
            strides=[1, *layer.stride, 1],
            padding="SAME",
            dilations=[1, *layer.dilation, 1],
        )
        return tf.nn.bias_add(result, array(layer.bias))

    def bn(value, layer):
        return tf.nn.batch_normalization(
            value,
            array(layer.running_mean),
            array(layer.running_var),
            array(layer.bias),
            array(layer.weight),
            variance_epsilon=1e-3,
        )

    def activate(value, name):
        if name == "leaky_relu":
            return tf.nn.leaky_relu(value, alpha=0.2)
        return {"relu": tf.nn.relu, "elu": tf.nn.elu}[name](value)

    skips = []
    for i in range(6):
        x = conv(x, net.encoder[i])
        skips.append(x)
        if i < 5:
            x = activate(bn(x, net.encoder_norm[i]), config.encoder_activation)
    for i in range(6):
        layer = net.decoder[i]
        kernel = array(layer.weight).transpose(2, 3, 1, 0)
        shape = [x.shape[0], x.shape[1] * 2, x.shape[2] * 2, layer.out_channels]
        x = tf.nn.conv2d_transpose(
            x, kernel, output_shape=shape, strides=[1, 2, 2, 1], padding="SAME"
        )
        x = tf.nn.bias_add(x, array(layer.bias))
        x = bn(activate(x, config.decoder_activation), net.decoder_norm[i])
        if i < 5:
            x = tf.concat((skips[4 - i], x), axis=-1)
    return conv(x, net.output)


@pytest.mark.parametrize(
    "mask,activation", [("sigmoid", "leaky_relu"), ("sigmoid", "elu"), ("softmax", "elu")]
)
def test_model_inference_matches_tensorflow(tiny_config, mask, activation):
    config = replace(
        tiny_config,
        mask_activation=mask,
        encoder_activation=activation,
        decoder_activation="relu" if activation == "leaky_relu" else "elu",
    )
    torch.manual_seed(321)
    model = Spleeter(config).eval()
    # Nontrivial running stats ensure BN mapping is checked as well as convolution.
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.running_mean.uniform_(-0.1, 0.1)
                layer.running_var.uniform_(0.5, 1.5)
        x = torch.rand(1, 2, 64, 64)
        actual = array(model(x)).transpose(0, 1, 3, 4, 2)
    tx = tf.constant(array(x).transpose(0, 2, 3, 1))
    logits = tf.stack([reference_unet(tx, net, config) for net in model.unets.values()], axis=1)
    weights = tf.nn.softmax(logits, axis=1) if mask == "softmax" else tf.sigmoid(logits)
    expected = weights * tx[:, None]
    np.testing.assert_allclose(actual, expected.numpy(), atol=3e-5, rtol=3e-4)


def test_audio_matches_tensorflow(tiny_config):
    torch.manual_seed(18)
    audio = torch.randn(2, 2451)
    padded = tf.pad(tf.constant(array(audio)), [[0, 0], [tiny_config.n_fft, 0]])
    expected = tf.signal.stft(padded, tiny_config.n_fft, tiny_config.hop_length, pad_end=True)
    actual = stft(audio, tiny_config, inference=True)
    np.testing.assert_allclose(array(actual), expected.numpy(), atol=5e-6, rtol=1e-4)
    inverse = tf.signal.inverse_stft(expected, tiny_config.n_fft, tiny_config.hop_length) * (2 / 3)
    expected_audio = inverse[:, tiny_config.n_fft : tiny_config.n_fft + audio.shape[-1]]
    np.testing.assert_allclose(
        array(istft(actual, tiny_config, audio.shape[-1])),
        expected_audio.numpy(),
        atol=1e-6,
        rtol=1e-4,
    )
