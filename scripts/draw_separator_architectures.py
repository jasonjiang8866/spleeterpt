"""Draw the curated architecture comparison used in the README.

Documentation-only dependency: matplotlib==3.10.8. Sources and scope:
docs/source-separation-landscape.md. This is a conceptual map, not introspection.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
INK = "#17263C"
MUTED = "#52647A"
BORDER = "#D9E2ED"
BG = "#F4F7FB"
BLUE = "#3368C6"
TEAL = "#147D80"
PURPLE = "#7652B3"
ORANGE = "#AF6424"
XS = (68, 410, 752, 1094)
WIDTH = 278


def draw(output, preview=None):
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "spt-landscape"}
    )
    fig = plt.figure(figsize=(20, 26), dpi=72, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 1440), ylim=(1872, 0))
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

    def box(x, y, w, h, fill="white", edge=BORDER, radius=12):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.3,
            )
        )

    def arrow(start, end, color=MUTED):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=2,
                color=color,
                shrinkA=1,
                shrinkB=2,
            )
        )

    def card(y, height, number, title, kind, color, note):
        box(40, y, 1360, height)
        box(62, y + 18, 46, 38, color, color, 8)
        text(85, y + 37, number, 18, "white", "bold", "center")
        text(124, y + 38, title, 26, weight="bold")
        text(1372, y + 38, kind, 18, color, "bold", "right")
        text(68, y + height - 22, note, 18, MUTED)

    def node(x, y, title, detail, color, height=90):
        box(x, y, WIDTH, height, "#FAFCFF")
        ax.plot([x + 17, x + 58], [y, y], color=color, lw=3.5)
        text(x + WIDTH / 2, y + height / 2 - 15, title, 21, weight="bold", align="center")
        text(x + WIDTH / 2, y + height / 2 + 20, detail, 18, MUTED, align="center")

    def row(y, number, title, kind, color, stages, note):
        card(y, 210, number, title, kind, color, note)
        for x, (heading, detail) in zip(XS, stages):
            node(x, y + 72, heading, detail, color)
        for x in XS[:-1]:
            arrow((x + WIDTH + 5, y + 117), (x + 337, y + 117), color)

    text(56, 40, "SPLEETERPT / SOURCE-SEPARATION LANDSCAPE", 17, BLUE, "bold")
    text(56, 86, "Seven architectures. Different ways to separate.", 36, weight="bold")
    text(56, 126, "Read left → right: representation, modeling and reconstruction", 22, MUTED)
    text(
        56,
        158,
        "Conceptual inference paths · not a ranking · sources [R1–R12] in the companion guide",
        18,
        MUTED,
    )

    row(
        190,
        "01",
        "Spleeter / this port",
        "MAGNITUDE CNN",
        BLUE,
        [
            ("Magnitude STFT", "Stereo mixture |X|"),
            ("Independent U-Nets", "Six levels · one per stem"),
            ("Ratio masks", "From predicted magnitudes"),
            ("Mask × mixture X", "iSTFT / overlap-add → stems"),
        ],
        "Per-stem encoder–decoder skips · mixture phase retained · upstream Wiener option omitted here [R1 / local code]",
    )
    row(
        418,
        "02",
        "Open-Unmix",
        "MAGNITUDE RNN",
        TEAL,
        [
            ("Magnitude STFT", "Dense frequency projection"),
            ("3-layer BiLSTM", "One network per target"),
            ("Magnitude estimates", "Multichannel Wiener filter"),
            ("Complex estimates", "iSTFT → stems"),
        ],
        "Recurrent time context with both past and future frames · source estimates are jointly filtered [R2]",
    )

    y = 646
    card(
        y,
        270,
        "03",
        "HTDemucs v4",
        "HYBRID CNN + ATTENTION",
        ORANGE,
        "Two routes from the same mixture · each encoder has decoder skips (condensed) · spectral and temporal outputs sum [R3–R4]",
    )
    for yy, title, detail in (
        (y + 72, "STFT encoder", "Complex channels"),
        (y + 156, "Waveform encoder", "Time-domain features"),
    ):
        node(XS[0], yy, title, detail, ORANGE, 68)
    box(XS[1], y + 72, WIDTH, 152, "#FFF7EF", "#E9CEAE")
    text(XS[1] + WIDTH / 2, y + 105, "Cross-domain", 23, ORANGE, "bold", "center")
    text(XS[1] + WIDTH / 2, y + 135, "Transformer", 23, ORANGE, "bold", "center")
    text(XS[1] + WIDTH / 2, y + 174, "Self-attention +", 19, MUTED, align="center")
    text(XS[1] + WIDTH / 2, y + 200, "cross-attention", 19, MUTED, align="center")
    node(XS[2], y + 72, "Spectral decoder", "Complex spectrum → iSTFT", ORANGE, 68)
    node(XS[2], y + 156, "Waveform decoder", "Waveform estimates", ORANGE, 68)
    box(XS[3], y + 72, WIDTH, 152, "#FFF7EF", "#E9CEAE")
    text(XS[3] + WIDTH / 2, y + 117, "+", 43, ORANGE, "bold", "center")
    text(XS[3] + WIDTH / 2, y + 165, "Sum waveforms", 22, weight="bold", align="center")
    text(XS[3] + WIDTH / 2, y + 197, "→ output stems", 19, MUTED, align="center")
    for yy in (y + 106, y + 190):
        for x in XS[:-1]:
            arrow((x + WIDTH + 5, yy), (x + 337, yy), ORANGE)

    row(
        934,
        "04",
        "MDX23C / TFC-TDF v3",
        "COMPLEX-SPECTRAL CNN",
        BLUE,
        [
            ("Complex STFT", "Channel-wise subbands"),
            ("TFC-TDF U-Net", "2D conv + frequency MLP"),
            ("Complex output head", "Unpack subband channels"),
            ("Complex spectra", "iSTFT → stems"),
        ],
        "Residual conv/dense blocks + U-Net skips · direct complex output · TDF is a fully connected block, not attention [R5]",
    )
    row(
        1162,
        "05",
        "BS-RoFormer",
        "BAND ATTENTION",
        PURPLE,
        [
            ("Complex STFT", "Disjoint band projection"),
            ("RoPE attention", "Alternate time / band axes"),
            ("Complex band masks", "Concatenate frequency bins"),
            ("Mask × mixture X", "iSTFT → target audio"),
        ],
        "Non-overlapping bands · long-range time and cross-band interactions · target count depends on the checkpoint [R6 / R8]",
    )
    row(
        1390,
        "06",
        "Mel-Band RoFormer",
        "OVERLAPPING BAND ATTENTION",
        PURPLE,
        [
            ("Complex STFT", "Overlapping mel-band groups"),
            ("RoPE attention", "Alternate time / band axes"),
            ("Complex band masks", "Average overlapping bins"),
            ("Mask × mixture X", "iSTFT → target audio"),
        ],
        "Mel spacing defines bin membership · real / imaginary features retained · not a mel-spectrogram vocoder [R7–R8]",
    )
    row(
        1618,
        "07",
        "SCNet",
        "SPARSE COMPRESSION + RNN",
        TEAL,
        [
            ("Complex STFT", "Unequal band compression"),
            ("Dual-path RNN", "Feature FFT / iFFT exchange"),
            ("Sparse upsampling", "Encoder skip fusion"),
            ("Complex spectra", "iSTFT → stems"),
        ],
        "Higher bands are compressed more strongly · lower-band detail receives more computation [R10 / R12]",
    )
    text(
        56,
        1853,
        "Original redrawing · block counts / sizes vary by configuration · reviewed 2026-11-01",
        17,
        MUTED,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        format="svg",
        metadata={"Date": None, "Title": "Music source-separation architecture comparison"},
    )
    svg = output.read_text().replace("'DejaVu Sans'", "'DejaVu Sans', Arial, sans-serif")
    start = svg.index("<svg ")
    end = svg.index(">", start)
    svg = svg[:end] + ' role="img" aria-labelledby="comparison-title comparison-desc"' + svg[end:]
    end = svg.index(">", start)
    description = (
        '<title id="comparison-title">Seven music source-separation architectures</title>'
        '<desc id="comparison-desc">Spleeter uses per-stem magnitude U-Nets and '
        "mixture-phase ratio masks in this port. Open-Unmix uses per-target BiLSTMs "
        "and multichannel Wiener filtering. HTDemucs joins spectral and waveform "
        "paths with a cross-domain Transformer and sums decoded waveforms. "
        "MDX23C uses channel-wise subbands and a TFC-TDF U-Net to estimate complex "
        "spectra. BS-RoFormer predicts complex masks from disjoint bands with time "
        "and band attention. Mel-Band RoFormer uses overlapping bands and averages "
        "overlapping masks. SCNet combines unequal band compression, dual-path "
        "recurrent processing and a sparse decoder. These are simplified inference "
        "paths, not a quality ranking. Sources are in the companion guide.</desc>"
    )
    svg = svg[: end + 1] + description + svg[end + 1 :]
    output.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n")
    if preview:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=90, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/assets/separator-architectures.svg"
    )
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    draw(args.output, args.preview)
