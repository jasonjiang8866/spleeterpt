# Model architecture diagram

[Open the scalable diagram](assets/model-architecture.svg).

The README figure documents the implemented standard presets. Its top panel uses
four stems as an example; the U-Net detail applies to all three presets. It is a
documentation illustration of this port, not a claim of a newly invented network.

## Read the diagram

- **Audio path:** 44.1 kHz stereo enters a periodic-Hann STFT with FFT 4096 and hop
  1024. Inference uses leading/tail padding and `center=False`. Network windows
  contain 512 time frames and the first 1024 frequency bins; the final partial
  window is padded, then trimmed. The full STFT remains resident in memory.
- **Independent networks:** one U-Net per stem, each receiving the same stereo
  mixture magnitude. There is no shared encoder between stems. Spatial dimensions
  halve through six 5×5 stride-two convolutions and double through six transpose
  convolutions with TensorFlow SAME alignment.
- **Skip semantics:** E1–E5 branch from raw convolution outputs, before BN and
  activation, into D5–D1 respectively. E6 is a raw bottleneck. D1–D3 apply ordinary
  element-wise dropout at probability 0.5 after activation and BN, before channel
  concatenation. D6 has no skip and produces one channel; the 4×4 dilation-two head
  produces two stereo logits.
- **Network output:** logits are stacked along the stem axis. Two/four stems use
  sigmoid; five stems use softmax across stems. Multiplying the resulting masks
  by the input magnitude produces the magnitude estimates returned by `Spleeter`.
- **Inference output:** `ratio_masks` raises those estimates to the configured
  exponent (2 by default), normalizes across stems with epsilon stabilization,
  and extends the frequency axis. Masks multiply the original complex mixture
  spectrum; inverse FFT and Hann overlap-add restore the original sample count.
  Default zero extension drops the unmodeled high band above about 11 kHz.
- **Training:** summed per-stem mean absolute magnitude error supervises the
  network estimates directly, before inference ratio-mask reconstruction. The
  optional frozen-BN policy uses stored moments in the forward pass and leaves
  affine parameters and dropout trainable/active.

The figure omits batch dimensions. Decoder spatial shapes are shown before
concatenation; the channel count after concatenation is explicitly labeled.

## Source of truth

| Diagram element | Implementation |
|---|---|
| Channels, dimensions and preset activations | [`config.py`](../src/spleeterpt/config.py), [`configs/`](../configs) |
| U-Nets, raw skips, dropout and logits | [`model.py`](../src/spleeterpt/model.py) |
| STFT, ratio masks and reconstruction | [`audio.py`](../src/spleeterpt/audio.py) |
| Magnitude loss and frozen BN behavior | [`train.py`](../src/spleeterpt/train.py) |

## Regenerate

The SVG is self-contained, has a fixed light background for GitHub light/dark
themes, and includes a title and text description for accessibility. Its generator
reads the preset JSON geometry and rejects inconsistent channel schedules.
Operations and connections are deliberately laid out by hand; review them against
the source files above whenever the implementation changes.

Use a separate documentation environment; Matplotlib is not a model dependency:

```bash
python -m venv /tmp/spleeterpt-diagram
/tmp/spleeterpt-diagram/bin/python -m pip install matplotlib==3.10.8
/tmp/spleeterpt-diagram/bin/python scripts/draw_architecture.py
# Optional PNG for reviewing labels and layout:
/tmp/spleeterpt-diagram/bin/python scripts/draw_architecture.py --preview /tmp/model-architecture.png
```

The generator writes `docs/assets/model-architecture.svg`. Commit the generator
and regenerated SVG together. SVG metadata omits the generation date so repeated
runs in the same environment produce identical bytes.
