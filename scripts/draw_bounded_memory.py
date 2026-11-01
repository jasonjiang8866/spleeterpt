"""Render measured CUDA memory from the committed RunPod report.

Documentation-only dependency: matplotlib==3.10.8. From any directory:
    python scripts/draw_bounded_memory.py
    python scripts/draw_bounded_memory.py --preview /tmp/bounded-memory.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/results/bounded-mig-41308828220.json"
INK, MUTED = "#17263C", "#52647A"
ORANGE, TEAL = "#B46028", "#087F82"


def draw(preview=None):
    report = json.loads(REPORT.read_text())["report"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "svg.hashsalt": "spleeterpt-bounded-memory",
        }
    )
    fig = plt.figure(figsize=(14, 8.4), facecolor="#F4F7FB")
    fig.text(0.055, 0.935, "Longer audio. Bounded memory.", fontsize=28, weight="bold")
    fig.text(
        0.055,
        0.89,
        "Official converted Spleeter weights · RTX PRO 6000 Blackwell MIG 1g.24gb",
        fontsize=13,
        color=MUTED,
    )
    reductions = []
    for panel, stems in enumerate((2, 4)):
        cases = report["benchmarks"][f"{stems}stems-synthetic"]["cases"]
        x = [c["seconds"] for c in cases]
        series = {
            mode: [
                next(m["cuda_peak_allocated_mib"] for m in c["measurements"] if m["mode"] == mode)
                for c in cases
            ]
            for mode in ("legacy", "bounded")
        }
        reductions.append(100 * (1 - series["bounded"][-1] / series["legacy"][-1]))
        ax = fig.add_axes((0.085 + panel * 0.475, 0.29, 0.39, 0.40), facecolor="white")
        ax.set_title(f"{stems}-stem", loc="left", fontsize=19, weight="bold", pad=16)
        for mode, color, marker in (("legacy", ORANGE, "o"), ("bounded", TEAL, "s")):
            ax.plot(
                x, series[mode], color=color, marker=marker, markersize=6, linewidth=2.5, zorder=3
            )
            value = series[mode][-1]
            ax.annotate(
                f"{value:,.0f} MiB",
                (x[-1], value),
                xytext=(-9, 13),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=13,
                weight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
            )
        ax.set(xlim=(0, 640), ylim=(0, 16000), xlabel="Audio duration (seconds)")
        ax.set_xticks((0, 60, 300, 600))
        ax.set_yticks((0, 4000, 8000, 12000, 16000))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.set_ylabel("Peak allocated CUDA memory (MiB)")
        ax.grid(axis="y", color="#E1E7EE", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#D9E2ED")
        ax.tick_params(length=0, pad=7)
    fig.text(
        0.055,
        0.815,
        f"~{min(reductions):.1f}% less peak CUDA memory at 10 minutes",
        fontsize=21,
        color=TEAL,
        weight="bold",
    )
    fig.legend(
        handles=[
            Line2D(
                [], [], color=ORANGE, marker="o", linewidth=2.5, label="Original spleeterpt path"
            ),
            Line2D([], [], color=TEAL, marker="s", linewidth=2.5, label="Bounded-memory path"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.05, 0.782),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.055,
        0.185,
        "Same bounded peak from 60 to 600 seconds.",
        fontsize=17,
        weight="bold",
        color=TEAL,
    )
    fig.text(
        0.055,
        0.14,
        "Measured synthetic inputs: 1, 12, 60 and 600 s. Lines connect measured points; "
        "both panels use the same linear axes.",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.103,
        "PyTorch allocated memory, not total device usage. Baseline: this port, "
        "not upstream TensorFlow. No production-acceptance claim.",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.066,
        "Source: RunPod / Actions 41308828220 · docs/results/bounded-mig-41308828220.json",
        fontsize=11,
        color=MUTED,
    )
    fig.savefig(ROOT / "docs/assets/bounded-memory.svg", metadata={"Date": None})
    if preview:
        fig.savefig(preview, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path)
    draw(parser.parse_args().preview)
