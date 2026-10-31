"""Render the README architecture from the standard preset geometry.

Documentation-only dependency: matplotlib==3.10.8. Run from any directory:
    python scripts/draw_architecture.py
    python scripts/draw_architecture.py --preview /tmp/model-architecture.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
INK = "#17263C"
MUTED = "#52647A"
BLUE = "#3368C6"
TEAL = "#147D80"
PURPLE = "#7652B3"
ORANGE = "#AF6424"
BORDER = "#D9E2ED"
BG = "#F4F7FB"


def draw(output, preview=None):
    presets = {n: json.loads((ROOT / f"configs/{n}stems.json").read_text()) for n in (2, 4, 5)}
    config = presets[4]
    for preset in presets.values():
        for key in ("filters", "segment_frames", "frequency_bins", "n_fft", "hop_length"):
            if preset[key] != config[key]:
                raise ValueError("Preset geometry differs; revise the architecture layout")
    if config["filters"] != [16, 32, 64, 128, 256, 512]:
        raise ValueError("Revise the layout for a changed channel schedule")
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "spt"}
    )
    fig = plt.figure(figsize=(20, 22.5), dpi=72, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 1440), ylim=(1620, 0))
    ax.set_axis_off()

    def text(x, y, value, size=20, color=INK, weight="normal", align="left", **kwargs):
        return ax.text(
            x,
            y,
            value,
            fontsize=size,
            color=color,
            weight=weight,
            ha=align,
            va="center",
            **kwargs,
        )

    def box(x, y, w, h, fill="white", edge=BORDER, radius=14):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.4,
            )
        )

    def arrow(points, color=MUTED, dashed=False, width=2):
        xs, ys = zip(*points)
        if len(points) > 2:
            ax.plot(xs[:-1], ys[:-1], color=color, lw=width, ls="--" if dashed else "-")
        ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=width,
                color=color,
                linestyle="--" if dashed else "-",
                shrinkA=0,
                shrinkB=1,
            )
        )

    text(56, 42, "SPLEETERPT / MODEL ARCHITECTURE", 17, BLUE, "bold")
    text(56, 88, "From stereo mixture to separate stems", 38, weight="bold")
    text(56, 126, "Trainable PyTorch port · standard 2 / 4 / 5-stem presets", 21, MUTED)

    box(40, 158, 1360, 302)
    text(66, 190, "01  AUDIO → SPECTRUM → ESTIMATES → WAVEFORMS", 18, weight="bold")
    xs = (68, 390, 712, 1034)
    for x, color in zip(xs, (MUTED, BLUE, PURPLE, TEAL)):
        box(x, 254, 290, 146, "#FAFCFF")
        ax.plot([x + 18, x + 68], [254, 254], color=color, lw=4)
    text(88, 283, "Stereo mixture", 23, weight="bold")
    text(88, 318, f"{config['sample_rate'] / 1000:g} kHz · 2 channels", 19, MUTED)
    for i, amplitude in enumerate((7, 16, 10, 24, 14, 28, 18, 10, 20, 9, 15, 6)):
        ax.plot(
            [90 + i * 19] * 2,
            [365 - amplitude / 2, 365 + amplitude / 2],
            color=MUTED,
            lw=4,
            solid_capstyle="round",
        )
    text(410, 283, "STFT + windowing", 22, weight="bold")
    text(410, 318, f"FFT {config['n_fft']} · hop {config['hop_length']}", 19, MUTED)
    text(410, 350, "Periodic Hann · center=False", 17, MUTED)
    text(410, 380, "2 × 512 × 1024 magnitudes", 18, BLUE)
    text(732, 283, "Independent U-Nets", 22, weight="bold")
    text(732, 316, "One network per stem", 19, MUTED)
    for i, name in enumerate(config["instruments"]):
        x, y = 732 + (i % 2) * 132, 336 + (i // 2) * 31
        box(x, y, 120, 25, "#F0EBF9", "#F0EBF9", 6)
        text(x + 60, y + 12, name, 16, PURPLE, align="center")
    text(1054, 283, "Reconstruction", 23, weight="bold")
    text(1054, 318, "Squared-magnitude ratio masks", 16.5, TEAL)
    text(1054, 350, "Mask × complex mixture STFT", 17, MUTED)
    text(1054, 380, "Hann overlap-add → stems", 17, MUTED)
    for x in xs[:-1]:
        arrow([(x + 290, 326), (x + 322, 326)])
    arrow([(535, 254), (535, 222), (1179, 222), (1179, 254)], TEAL, True)
    text(
        857,
        222,
        "Complex mixture X · phase retained",
        17,
        TEAL,
        align="center",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 5},
    )
    text(
        66,
        430,
        "Network estimates: stack stem logits → mask activation → multiply by |X|",
        19,
        MUTED,
    )

    box(40, 482, 1360, 830)
    text(66, 515, "02  INSIDE EACH U-NET", 18, weight="bold")
    text(1374, 515, "Shapes: C × T × F · batch dimension omitted", 17, MUTED, align="right")
    text(86, 557, "ENCODER ↓", 19, BLUE, "bold")
    text(86, 583, "Conv → BN → activation", 17, MUTED)
    text(986, 557, "DECODER ↑", 19, TEAL, "bold")
    text(986, 583, "Upconv → activation → BN", 17, MUTED)
    text(716, 557, "To stem mask activation", 18, PURPLE, align="center")
    box(86, 612, 330, 70, "#F1F5FB")
    text(106, 636, "Network input", 21, weight="bold")
    text(106, 664, "2 × 512 × 1024", 19, MUTED)
    box(486, 612, 450, 70, "#F2EDF9", "#D8CBEA")
    text(711, 636, "4 × 4 conv · dilation 2 · 2 channels", 19, PURPLE, "bold", "center")
    text(711, 664, "Stereo logits: 2 × 512 × 1024", 19, PURPLE, align="center")
    arrow([(986, 647), (936, 647)], PURPLE)
    arrow([(711, 612), (711, 578)], PURPLE)
    box(986, 612, 330, 70, "#ECF7F6", "#B7DCDA")
    text(1006, 636, "D6 · upconv to 1 channel", 20, TEAL, "bold")
    text(1006, 664, "1 × 512 × 1024 · no skip", 18, MUTED)

    t, f = config["segment_frames"], config["frequency_bins"]
    for i, channels in enumerate(config["filters"][:-1]):
        y = 712 + 100 * i
        t, f = t // 2, f // 2
        box(86, y, 330, 70, "#EEF3FD", "#C1D2EE")
        text(106, y + 24, f"E{i + 1} · conv to {channels} channels", 20, BLUE, "bold")
        text(106, y + 52, f"{channels} × {t} × {f}", 19, MUTED)
        arrow([(251, y - 30), (251, y)], BLUE)
        box(986, y, 330, 70, "#ECF7F6", "#B7DCDA")
        decoder = 5 - i
        text(1006, y + 24, f"D{decoder} · upconv {channels}", 20, TEAL, "bold")
        text(1006, y + 52, f"{t} × {f} · concat → {channels * 2} channels", 17.5, MUTED)
        if decoder <= 3:
            box(1204, y + 10, 96, 26, "#FFF0DF", "#FFF0DF", 6)
            text(1252, y + 23, "drop 0.5", 15.5, ORANGE, align="center")
        arrow([(1151, y), (1151, y - 30)], TEAL)
        arrow([(416, y + 45), (986, y + 45)], BLUE, width=1.8)
        text(
            701,
            y + 45,
            f"Raw E{i + 1} skip → concatenate",
            18,
            BLUE,
            align="center",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 5},
        )
    box(516, 1212, 400, 70, "#F2EDF9", "#D8CBEA")
    text(716, 1236, "E6 · raw bottleneck", 21, PURPLE, "bold", "center")
    text(716, 1264, "512 × 8 × 16 · no BN / activation", 18, PURPLE, align="center")
    arrow([(251, 1182), (251, 1247), (516, 1247)], BLUE)
    arrow([(916, 1247), (1151, 1247), (1151, 1182)], TEAL)

    box(40, 1334, 1360, 232)
    text(66, 1367, "03  PRESETS & TRAINING", 18, weight="bold")
    for i, n in enumerate((2, 4, 5)):
        x = 66 + i * 445
        cfg = presets[n]
        text(x, 1408, f"{n} STEMS", 21, (BLUE, TEAL, PURPLE)[i], "bold")
        enc = {"leaky_relu": "LeakyReLU (0.2)", "elu": "ELU"}[cfg["encoder_activation"]]
        dec = {"relu": "ReLU", "elu": "ELU"}[cfg["decoder_activation"]]
        text(x, 1442, f"Encoder {enc} · decoder {dec}", 17, MUTED)
        mask = "Softmax across stems" if n == 5 else "Independent sigmoid masks"
        text(x, 1472, mask, 18, MUTED)
    ax.plot([66, 1374], [1500, 1500], color=BORDER, lw=1)
    text(
        66,
        1532,
        "Training: sum of stem-wise magnitude L1 losses · optional frozen BN statistics",
        20,
    )
    text(
        56,
        1593,
        "5 × 5 / stride 2 down- and upconvs · TF SAME alignment · skips precede encoder BN / activation",
        17,
        MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output, format="svg", metadata={"Date": None, "Title": "spleeterpt model architecture"}
    )
    svg = output.read_text().replace("'DejaVu Sans'", "'DejaVu Sans', Arial, sans-serif")
    start = svg.index("<svg ")
    end = svg.index(">", start)
    svg = svg[:end] + ' role="img" aria-labelledby="arch-title arch-desc"' + svg[end:]
    end = svg.index(">", start)
    description = (
        '<title id="arch-title">spleeterpt model architecture</title>'
        '<desc id="arch-desc">Stereo STFT magnitudes feed independent six-level U-Nets, '
        "one per stem. Raw encoder convolutions provide five skip concatenations. "
        "A raw bottleneck feeds six upconvolutions; the first three use dropout. "
        "Two-channel logits become magnitude estimates, then squared-magnitude ratio "
        "masks applied to the complex mixture STFT before Hann overlap-add synthesis. "
        "Two/four-stem presets use sigmoid; five stems use softmax across stems.</desc>"
    )
    output.write_text(svg[: end + 1] + description + svg[end + 1 :])
    if preview:
        fig.savefig(preview, dpi=90, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/model-architecture.svg")
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    draw(args.output, args.preview)
