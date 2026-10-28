"""Spleeter U-Nets in NCHW, with spatial axes (time, frequency).

Adapted from Deezer Spleeter (MIT); see THIRD_PARTY_NOTICES.md.
Raw convolution outputs are used as skips and as the bottleneck, as upstream.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


def activation(name: str):
    return {"leaky_relu": lambda: nn.LeakyReLU(0.2), "relu": nn.ReLU, "elu": nn.ELU}[name]()


class SameConv2d(nn.Conv2d):
    """TensorFlow SAME, including asymmetric padding for stride two."""

    def forward(self, x):
        pads = []
        for size, kernel, stride, dilation in reversed(
            list(zip(x.shape[-2:], self.kernel_size, self.stride, self.dilation))
        ):
            out = (size + stride - 1) // stride
            total = max((out - 1) * stride + dilation * (kernel - 1) + 1 - size, 0)
            pads.extend([total // 2, total - total // 2])
        return super().forward(F.pad(x, pads))


class SameConvTranspose2d(nn.ConvTranspose2d):
    """TF SAME transpose for kernel=5, stride=2: crop one left, two right.

    Symmetric padding=2/output_padding=1 has the right shape but wrong alignment.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=5, stride=2)

    def forward(self, x):
        height, width = x.shape[-2:]
        return super().forward(x)[..., 1 : 1 + 2 * height, 1 : 1 + 2 * width]


def batch_norm(channels):
    # Keras momentum=.99 is the old-stat weight; PyTorch uses the new-stat weight.
    return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)


class UNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        widths = config.filters
        self.encoder = nn.ModuleList(
            [SameConv2d(a, b, kernel_size=5, stride=2) for a, b in zip((2,) + widths[:-1], widths)]
        )
        # Upstream computes batch6/activation6, but they are disconnected from output.
        self.encoder_norm = nn.ModuleList([batch_norm(n) for n in widths[:-1]])
        decoder_in = (widths[-1],) + tuple(2 * n for n in reversed(widths[:-1]))
        decoder_out = tuple(reversed(widths[:-1])) + (1,)
        self.decoder = nn.ModuleList(
            [SameConvTranspose2d(a, b) for a, b in zip(decoder_in, decoder_out)]
        )
        self.decoder_norm = nn.ModuleList([batch_norm(n) for n in decoder_out])
        self.encoder_activation = activation(config.encoder_activation)
        self.decoder_activation = activation(config.decoder_activation)
        self.dropout = nn.Dropout(0.5)
        self.output = SameConv2d(1, 2, kernel_size=4, dilation=2)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                # Matches He-uniform scaling, not TensorFlow's random sequence.
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        skips = []
        for i, conv in enumerate(self.encoder):
            x = conv(x)
            skips.append(x)
            if i < 5:
                x = self.encoder_activation(self.encoder_norm[i](x))
        for i, (conv, norm) in enumerate(zip(self.decoder, self.decoder_norm)):
            x = norm(self.decoder_activation(conv(x)))
            if i < 3:
                x = self.dropout(x)
            if i < 5:
                x = torch.cat((skips[4 - i], x), dim=1)
        return self.output(x)


class Spleeter(nn.Module):
    """Input [batch, 2, time, frequency]; output [batch, stems, 2, time, frequency]."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.unets = nn.ModuleDict({name: UNet(config) for name in config.instruments})

    def forward(self, magnitude):
        if magnitude.ndim != 4 or magnitude.shape[1] != 2:
            raise ValueError("expected [batch, stereo=2, time, frequency]")
        if magnitude.shape[-2:] != (self.config.segment_frames, self.config.frequency_bins):
            raise ValueError("spectrogram dimensions must match model config")
        logits = torch.stack([net(magnitude) for net in self.unets.values()], dim=1)
        masks = (
            logits.softmax(dim=1) if self.config.mask_activation == "softmax" else logits.sigmoid()
        )
        return masks * magnitude.unsqueeze(1)
