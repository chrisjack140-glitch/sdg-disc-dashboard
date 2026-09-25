"""
DISC radar — shared geometry, and a print image for the Leadership Roadmap
===========================================================================
The dashboard draws the DISC radar interactively with Plotly (app.py). The
Roadmap needs the same chart as a picture inside the Word document, drawn
on the server — and the server has no browser for Plotly's image export.
So the print version is drawn here with matplotlib, reproducing the
dashboard's light-mode Graph Shift Radar as it was laid out for the
reference booklet: a -8..8 grid, the eight behaviour symbols around it, and
one person's Stress and Mirror graphs overlaid.

The axis order, symbols and graph colours come from utils/radar_geometry.py,
which app.py reads too, so the two versions cannot drift apart.
"""
import io
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # no display on the server
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib import font_manager        # noqa: E402
from matplotlib.lines import Line2D        # noqa: E402
import numpy as np                         # noqa: E402
from PIL import Image                      # noqa: E402

from utils.radar_geometry import (          # noqa: E402
    GRAPH_OVERLAY_STYLE, RADAR_AXIS_ICONS, RADAR_CATEGORIES, RADIAL_MAX,
    RADIAL_MIN, radar_values)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ICON_DIR = ASSETS / "radar_icons"

# The dashboard's typeface (see assets/fonts/README.txt); DejaVu Sans, which
# ships with matplotlib, if the files are missing.
for _ttf in (ASSETS / "fonts").glob("Inter-*.ttf"):
    font_manager.fontManager.addfont(str(_ttf))
_FONT = "Inter" if any((ASSETS / "fonts").glob("Inter-*.ttf")) else "DejaVu Sans"

# ── Print image ────────────────────────────────────────────────────────────
# Light-theme colours from app.py's THEME, and the booklet's navy for the
# legend.
_TEXT, _MUTED, _GRID, _NAVY = "#1a1814", "#6b6458", "#d4cfc4", "#1B2A44"
_MARKERS = {"circle": "o", "diamond": "D", "square": "s"}

# Layout in chart units, where the outer ring has radius 1. These are the
# positions tuned for the reference booklet's RADAR SHIFTS page: symbols
# pulled in close to the ring so the ring can be as large as the page
# width allows, and the bottom symbol dropped clear of the "SC" label.
_X_RANGE = (-1.72, 1.72)
_Y_RANGE = (-1.64, 1.52)
_PULL_IN = 0.86
_ICON_SIZE = 0.24
_PX = 0.72                      # the dashboard's 1px at print size, in pt


def _xy(k: int, rho: float):
    """Chart position of axis k (clockwise from the top) at radius rho."""
    angle = math.radians(90 - 45 * k)
    return rho * math.cos(angle), rho * math.sin(angle)


def _rho(score: float) -> float:
    return (score - RADIAL_MIN) / (RADIAL_MAX - RADIAL_MIN)


def _icon(stem: str, dpi: int):
    """A behaviour symbol at print resolution.

    The icon files are 51px; scaled up inside the plot they break into
    noise, so they are resized smoothly first to the size they print at
    (~0.5in).
    """
    path = ICON_DIR / "light" / f"{stem}.png"
    if not path.exists():
        return None
    px = int(0.5 * dpi)
    img = Image.open(path).convert("RGBA").resize((px, px), Image.LANCZOS)
    return np.asarray(img)


def graph_shift_png(profile: dict, graphs=("mirror", "stress"),
                    width_in: float = 7.25, height_in: float = 7.24,
                    dpi: int = 300) -> bytes:
    """One person's graphs overlaid on the DISC radar, as a PNG.

    Drawn stress last (on top), as on the dashboard. Missing graphs are
    skipped, so a profile with only a Mirror graph still renders.
    """
    with plt.rc_context({"font.family": _FONT}):
        return _draw(profile, graphs, width_in, height_in, dpi)


def _draw(profile, graphs, width_in, height_in, dpi) -> bytes:
    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    fig.patch.set_facecolor("white")
    # plot area below a legend strip, as the dashboard's margins (8/56px)
    top = 1 - 0.56 / height_in
    ax = fig.add_axes([0.08 / width_in, 0.08 / height_in,
                       1 - 0.16 / width_in, top - 0.08 / height_in])
    ax.set_xlim(*_X_RANGE)
    ax.set_ylim(*_Y_RANGE)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # grid: rings at each tick, spokes to every axis
    for tick in (-4, 0, 4, 8):
        ring = plt.Circle((0, 0), _rho(tick), fill=False, color=_GRID,
                          linewidth=_PX, zorder=1)
        ax.add_patch(ring)
    for k in range(8):
        ax.plot([0, _xy(k, 1)[0]], [0, _xy(k, 1)[1]], color=_GRID,
                linewidth=_PX, zorder=1)

    # radial scale, laid between the IS and I spokes so it clears both
    angle = math.radians(22.5)
    for tick in (-8, -4, 0, 4, 8):
        r = _rho(tick)
        ax.text(r * math.cos(angle), r * math.sin(angle) + 0.015, str(tick),
                color=_MUTED, fontsize=9 * _PX, ha="center", va="bottom",
                zorder=3)

    # axis codes just outside the ring
    for k, cat in enumerate(RADAR_CATEGORIES):
        x, y = _xy(k, 1.075)
        ax.text(x, y, cat, color=_TEXT, fontsize=11 * _PX, ha="center",
                va="center", zorder=3)

    # behaviour symbols and their labels
    for k, (_cat, stem, label) in enumerate(RADAR_AXIS_ICONS):
        radius = (1.6 if k % 2 else 1.48) * _PULL_IN
        x, y = _xy(k, radius)
        icon_y, label_y = y + 0.09, y - 0.05
        if k == 4:                          # bottom symbol: clear of "SC"
            icon_y, label_y = icon_y - 0.14, label_y - 0.14
        img = _icon(stem, dpi)
        if img is not None:
            h = _ICON_SIZE / 2
            ax.imshow(img, extent=(x - h, x + h, icon_y - h, icon_y + h),
                      zorder=4, interpolation="antialiased")
        ax.text(x, label_y, label.replace("<br>", "\n"), color=_TEXT,
                fontsize=13 * _PX, fontweight="bold", ha="center", va="top",
                linespacing=1.15, zorder=4)

    # the graphs
    handles = []
    for g in ("public", "mirror", "stress"):          # stress drawn on top
        if g not in graphs or g not in (profile.get("graphs") or {}):
            continue
        st = GRAPH_OVERLAY_STYLE[g]
        vals = radar_values(profile["graphs"][g])
        pts = [_xy(k, _rho(max(RADIAL_MIN, min(RADIAL_MAX, v))))
               for k, v in enumerate(vals)]
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        ax.fill(xs, ys, color=st["color"], alpha=0.08, linewidth=0, zorder=5)
        ax.plot(xs, ys, color=st["color"], linewidth=2.5 * _PX, zorder=6,
                solid_joinstyle="round")
        marker = _MARKERS[st["symbol"]]
        size = 7 * _PX * (0.8 if marker == "D" else 1)
        ax.plot(xs[:-1], ys[:-1], linestyle="none", marker=marker,
                markersize=size, color=st["color"], zorder=7)
        handles.append(Line2D([0], [0], color=st["color"],
                              linewidth=2.5 * _PX, marker=marker,
                              markersize=size, label=st["label"]))

    if handles:
        fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                   frameon=False, fontsize=14 * _PX, labelcolor=_NAVY,
                   bbox_to_anchor=(0.5, 1 - 0.08 / height_in),
                   handlelength=2.2, columnspacing=1.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    return buf.getvalue()
