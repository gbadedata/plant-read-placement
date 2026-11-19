"""Draw the README's results figure from the frozen test metrics, in a light and a dark version.

Reads results/test/metrics.json (written by the single preregistered test run) and writes
docs/figures/results-light.png and docs/figures/results-dark.png. The colours are a validated
categorical palette (colour-blind separation checked for adjacent slots in both modes); marker shapes
repeat the identity for print and colour-blind readers.

Usage:  python scripts/make_readme_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "test" / "metrics.json"
OUT = ROOT / "docs" / "figures"

# method id, label, categorical slot (fixed order), marker
METHODS = [
    ("minimap2", "minimap2, full genomes", 0, "o"),
    ("kmer_head", "k-mers + head", 1, "s"),
    ("plantcad_head", "PlantCaduceus + head", 2, "D"),
    ("plantcad_ft2_head", "PlantCaduceus, adapted + head", 3, "^"),
    ("minimap2_train", "minimap2, training regions only", 4, "v"),
]
SIZES = ["1", "10", "50"]

THEMES = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
    },
}


def style_axes(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=t["muted"], labelsize=8.5, length=0, pad=6)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def rounded_bar(ax, x, w, h, color, radius_px=8):
    """Bar with a rounded data end (about 4 CSS px at 200 dpi) and a square end on the baseline.
    The radius is converted from pixels to data units on each axis so the corners stay circular."""
    if h <= 0:
        return
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    px_x, px_y = bbox.width / (x1 - x0), bbox.height / (y1 - y0)
    rx = min(radius_px / px_x, w / 2)
    ry = rx * px_x / px_y
    if h < 2 * ry:  # too short for round corners: plain bar
        ax.add_patch(Rectangle((x, 0), w, h, linewidth=0, facecolor=color))
        return
    ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle=f"round,pad=0,rounding_size={rx}",
                                mutation_aspect=px_x / px_y, linewidth=0, facecolor=color))
    ax.add_patch(Rectangle((x, 0), w, ry, linewidth=0, facecolor=color))


def draw(theme: str, m: dict) -> Path:
    t = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.22})
    fig.patch.set_facecolor(t["surface"])

    # Panel A: balanced accuracy against reads per specimen, with species-level 95% intervals
    style_axes(a, t)
    xs = range(len(SIZES))
    for i, (key, label, slot, marker) in enumerate(METHODS):
        c = t["series"][slot]
        dx = (i - 2) * 0.06  # dodge so the intervals do not sit on top of each other
        ys = [m[key][n]["balanced_accuracy"] for n in SIZES]
        ci = [m[key][n]["balanced_accuracy_ci95"] for n in SIZES]
        px = [x + dx for x in xs]
        for x, (lo, hi) in zip(px, ci, strict=True):
            a.plot([x, x], [lo, hi], color=c, linewidth=1.0, alpha=0.75, solid_capstyle="butt")
        a.plot(px, ys, color=c, linewidth=1.8, marker=marker, markersize=6.5, label=label,
               markeredgecolor=t["surface"], markeredgewidth=1.2, zorder=3)
    a.axhline(1 / 3, color=t["muted"], linewidth=0.9, zorder=1)
    a.text(len(SIZES) - 1 + 0.5, 1 / 3 - 0.015, "chance (1/3)", color=t["muted"], fontsize=8, va="top", ha="right")
    a.set_xticks(list(xs), [f"{n} read" if n == "1" else f"{n} reads" for n in SIZES])
    a.set_xlim(-0.35, len(SIZES) - 1 + 0.55)
    a.set_ylim(0, 1.0)
    a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a.set_title("Balanced accuracy by reads per specimen", loc="left", color=t["ink"], fontsize=10.5,
                fontweight="bold", pad=22)
    a.text(0, 1.035, "Test split, held-out species; bars are species-level 95% intervals",
           transform=a.transAxes, color=t["ink2"], fontsize=8.5)

    # Panel B: recognition against the relative call, 50 reads per specimen
    style_axes(b, t)
    groups = [("Known specimens:\nnearest species right", "known_species_top1"),
              ("Unseen relatives:\ncalled relative, right genus", "recall_relative")]
    w, gap = 0.09, 0.012
    labelled = {"kmer_head", "plantcad_head"}
    b.set_xlim(-0.6, 1.6)
    b.set_ylim(0, 1.0)
    for g, (_, metric) in enumerate(groups):
        n = len(METHODS)
        start = g - (n * w + (n - 1) * gap) / 2
        for i, (key, _, slot, _) in enumerate(METHODS):
            v = m[key]["50"][metric]
            x = start + i * (w + gap)
            rounded_bar(b, x, w, v, t["series"][slot])
            if key in labelled:
                b.text(x + w / 2, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=t["ink"],
                       fontweight="bold" if key == "plantcad_head" else "normal")
    b.set_xticks([0, 1], [gname for gname, _ in groups])
    b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    b.set_title("Recognises species, misses relatives", loc="left", color=t["ink"], fontsize=10.5,
                fontweight="bold", pad=22)
    b.text(0, 1.035, "Test split, 50 reads per specimen",
           transform=b.transAxes, color=t["ink2"], fontsize=8.5)

    handles, labels = a.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.5,
                     bbox_to_anchor=(0.5, -0.01), handlelength=2.2, columnspacing=1.6)
    for text in leg.get_texts():
        text.set_color(t["ink2"])
    fig.subplots_adjust(left=0.05, right=0.985, top=0.83, bottom=0.2)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"results-{theme}.png"
    fig.savefig(path, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    return path


def main() -> None:
    m = json.loads(METRICS.read_text())["methods"]
    for theme in THEMES:
        print(draw(theme, m))


if __name__ == "__main__":
    main()
