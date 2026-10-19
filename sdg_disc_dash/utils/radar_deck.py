"""
Radar slides — a PowerPoint deck that builds the DISC radar person by person
=============================================================================
Comparisons tab → Radar Slides. Slide 1 shows the first person's graph with
their name in the key; slide 2 adds the second person's graph and name; and
so on, so a deck for N people has N slides and the last one overlays all of
them. The layout follows the reference deck SDG presents from ("NWF Dec 9th
R graphs.pptx"): a white 10.67 x 6 in slide, the -8..8 radar in the middle,
the eight behaviour symbols at the reference's positions, and the key at
the bottom right in the reference's colours, marker shapes and fonts.

The radar is drawn with native PowerPoint shapes (rings, spokes, one line
per person, marker shapes) rather than pasted in as a picture, so it stays
sharp when projected and can still be edited in PowerPoint. Everything
sits at the same place on every slide, so clicking through the deck only
ever adds a line and a name.
"""
import io
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Emu, Pt

from utils.radar_geometry import (RADAR_AXIS_ICONS, RADAR_CATEGORIES,
                                  RADIAL_MAX, RADIAL_MIN, radar_values)

ICON_DIR = Path(__file__).resolve().parent.parent / "assets" / "radar_icons"

DECK_GRAPHS = {"mirror": "Mirror", "public": "Public", "stress": "Stress"}

# ── Reference deck geometry (EMU; 914400 per inch) ─────────────────────────
SLIDE_W, SLIDE_H = 9753600, 5486400

# Ring centre and outer (score 8) radius. The reference's chart is a Plotly
# screenshot that drifts a little from slide to slide; this is its average.
CX, CY, RING_R = 4870000, 2620000, 1640000

# Behaviour symbols: icon box and label centre, as placed in the reference.
_SYMBOLS = {
    "DI": ((4640132, 0, 476250, 476250),        (4877552, 557173)),
    "I":  ((6495679, 600801, 590550, 504825),   (6783257, 1168984)),
    "IS": ((7087448, 2366086, 495300, 485775),  (7333536, 3059811)),
    "S":  ((6505708, 4111295, 581025, 495300),  (6784629, 4679461)),
    "SC": ((4610062, 4554350, 542925, 542925),  (4876381, 5175190)),
    "C":  ((2664238, 4091235, 523875, 523875),  (2920251, 4759738)),
    "CD": ((2192817, 2366077, 466725, 485775),  (2419264, 2980868)),
    "D":  ((2664247, 620883, 514350, 485775),   (2921689, 1169013)),
}

# ── Reference deck styling ─────────────────────────────────────────────────
# The first five colours and marker shapes are the reference's, sampled
# from its key; the rest extend it for larger groups (a colour never
# repeats within 15 people, and colour + marker never within 135).
SERIES_COLORS = [
    "FF0000", "00FF00", "0000FF", "F09F1D", "BB28C9",
    "00B3C7", "8B4513", "FF69B4", "556B2F", "1B2A44",
    "7F7F7F", "E6C700", "800000", "2E8B57", "6A0DAD",
]
SERIES_MARKERS = [
    MSO_SHAPE.OVAL, MSO_SHAPE.DIAMOND, MSO_SHAPE.RECTANGLE,
    MSO_SHAPE.ISOSCELES_TRIANGLE, MSO_SHAPE.FLOWCHART_MERGE,  # ▲ ▼
    MSO_SHAPE.REGULAR_PENTAGON, MSO_SHAPE.HEXAGON, MSO_SHAPE.STAR_5_POINT,
    MSO_SHAPE.CROSS,
]
_MARKER = 64000                   # marker box (~5pt); squares a little less
_LINE_PT = 1.5

_GRID, _RING, _AXIS_TEXT = "E6E9EF", "C9D3E3", "8C8C8C"
_LABEL_FONT, _LABEL_COLOR, _LABEL_PT = "Nunito", "0E2121", 10.5
_KEY_FONT, _KEY_COLOR, _KEY_PT = "Roboto Condensed", "2B2B35", 9.75

# Key: swatch + name rows, growing downward from the reference's first row.
_KEY_X, _KEY_TOP, _KEY_ROW, _SWATCH = 8558000, 4100000, 170000, 128600
_KEY_BOTTOM = SLIDE_H - 180000
_KEY_GAP = 20000                  # swatch → name
_RIGHT_MARGIN = 110000
# Average name width per character, in em. Roboto Condensed Bold measures
# ~0.44; the slack keeps long names on the slide if PowerPoint has to
# substitute a wider font.
_CHAR_EM = 0.5


def _rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def _plain(shape):
    """Drop the theme style python-pptx gives new shapes (it can add a
    shadow or theme-coloured outline), so only what we set shows."""
    style = shape._element.find(qn("p:style"))
    if style is not None:
        shape._element.remove(style)
    return shape


@contextmanager
def _group(shapes, name: str):
    """A named group, sized once after it is filled: python-pptx re-measures
    every member on each add, which made a 15-person deck take seconds."""
    grp = shapes.add_group_shape()
    grp.name = name
    grp.shapes._recalculate_extents = lambda: None
    yield grp.shapes
    grp._element.recalculate_extents()


def _line(shape, color: str, width_pt: float):
    shape.line.color.rgb = _rgb(color)
    shape.line.width = Pt(width_pt)


def _xy(k: int, radius: float):
    """Slide position of axis k (clockwise from the top) at radius EMU."""
    angle = math.radians(90 - 45 * k)
    return CX + radius * math.cos(angle), CY - radius * math.sin(angle)


def _radius(score: float) -> float:
    score = max(RADIAL_MIN, min(RADIAL_MAX, score))
    return RING_R * (score - RADIAL_MIN) / (RADIAL_MAX - RADIAL_MIN)


def _text(shapes, x, y, w, h, lines, font, size_pt, color, bold=False,
          align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, name=None):
    box = shapes.add_textbox(Emu(int(x)), Emu(int(y)), Emu(int(w)),
                             Emu(int(h)))
    if name:
        box.name = name
    tf = box.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.word_wrap = False
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = anchor
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = 0.9
        run = p.add_run()
        run.text = line
        run.font.name = font
        run.font.size = Pt(size_pt)
        run.font.bold = bold
        run.font.color.rgb = _rgb(color)
    return box


# ── Slide parts ────────────────────────────────────────────────────────────

def _grid(shapes):
    """Rings at -4, 0, 4 and 8, spokes, the radial scale and axis codes."""
    for tick in (-4, 0, 4, 8):
        r = _radius(tick)
        ring = _plain(shapes.add_shape(MSO_SHAPE.OVAL, Emu(int(CX - r)),
                                       Emu(int(CY - r)), Emu(int(2 * r)),
                                       Emu(int(2 * r))))
        ring.fill.background()
        _line(ring, _RING if tick == RADIAL_MAX else _GRID,
              1.0 if tick == RADIAL_MAX else 0.75)
        ring.name = f"Ring {tick}"
    for k, cat in enumerate(RADAR_CATEGORIES):
        x, y = _xy(k, RING_R)
        spoke = _plain(shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Emu(CX), Emu(CY), Emu(int(x)),
            Emu(int(y))))
        _line(spoke, _GRID, 0.75)
        spoke.name = f"Spoke {cat}"

    # radial scale up the top spoke, just left of it, as in the reference
    for tick in (-8, -4, 0, 4):
        y = CY - _radius(tick)
        _text(shapes, CX - 320000, y - 125000, 300000, 120000, [str(tick)],
              _LABEL_FONT, 7, _AXIS_TEXT, align=PP_ALIGN.RIGHT,
              anchor=MSO_ANCHOR.BOTTOM, name=f"Scale {tick}")

    for k, cat in enumerate(RADAR_CATEGORIES):
        x, y = _xy(k, RING_R + 120000)
        _text(shapes, x - 200000, y - 60000, 400000, 120000, [cat],
              _LABEL_FONT, 8, _AXIS_TEXT, name=f"Axis {cat}")


def _series(shapes, values: list, idx: int, name: str):
    """One person's closed line and its markers."""
    color = SERIES_COLORS[idx % len(SERIES_COLORS)]
    marker = SERIES_MARKERS[idx % len(SERIES_MARKERS)]
    pts = [_xy(k, _radius(v)) for k, v in enumerate(values)]

    fb = shapes.build_freeform(pts[0][0], pts[0][1], scale=1.0)
    fb.add_line_segments(pts[1:], close=True)
    path = _plain(fb.convert_to_shape())
    path.fill.background()
    _line(path, color, _LINE_PT)
    path.line._get_or_add_ln().append(OxmlElement("a:round"))  # round joins
    path.name = f"{name} line"

    size = _MARKER * (0.88 if marker == MSO_SHAPE.RECTANGLE else 1)
    for k, (x, y) in enumerate(pts):
        dot = _plain(shapes.add_shape(marker, Emu(int(x - size / 2)),
                                      Emu(int(y - size / 2)),
                                      Emu(int(size)), Emu(int(size))))
        dot.fill.solid()
        dot.fill.fore_color.rgb = _rgb(color)
        dot.line.fill.background()
        dot.name = f"{name} {RADAR_CATEGORIES[k]}"


def _symbols(shapes):
    """The eight behaviour icons and their labels."""
    for cat, stem, label in RADAR_AXIS_ICONS:
        (bx, by, bw, bh), (lx, ly) = _SYMBOLS[cat]
        path = ICON_DIR / "light" / f"{stem}.png"
        if path.exists():
            with Image.open(path) as img:
                iw, ih = img.size
            scale = min(bw / iw, bh / ih)        # fit, keep proportions
            w, h = iw * scale, ih * scale
            pic = shapes.add_picture(str(path), Emu(int(bx + (bw - w) / 2)),
                                     Emu(int(by + (bh - h) / 2)),
                                     Emu(int(w)), Emu(int(h)))
            pic.name = f"Icon {cat}"
        # first line centred where the reference centres its label; a
        # second line runs on below it, away from the icon
        lines = [s.strip() for s in label.split("<br>")]
        _text(shapes, lx - 850000, ly - 80000, 1700000, 175000 * len(lines),
              lines, _LABEL_FONT, _LABEL_PT, _LABEL_COLOR, bold=True,
              anchor=MSO_ANCHOR.TOP, name=f"Label {cat}")


def _key_layout(names: list) -> dict:
    """Where the key goes, from the whole deck's names, so every slide puts
    each row in the same place. Matches the reference for up to seven short
    names; longer names move it left, then shrink it; more rows move it up,
    then tighten them."""
    n = len(names)
    row, swatch, size = _KEY_ROW, _SWATCH, _KEY_PT
    top = _KEY_TOP
    if top + n * row > _KEY_BOTTOM:
        top = max(250000, _KEY_BOTTOM - n * row)
        if top + n * row > _KEY_BOTTOM:
            row = (_KEY_BOTTOM - top) / n
            swatch = min(_SWATCH, row * 0.75)
            size = max(6.0, _KEY_PT * row / _KEY_ROW)

    # keep clear of the "Keep the Peace" symbol, or of "Build Relationships"
    # too once the key reaches that high
    min_x = 7850000 if top < 3250000 else 7400000
    longest = max((len(s) for s in names), default=0)

    def text_w(pt):
        return longest * _CHAR_EM * pt * 12700

    x = _KEY_X
    room = SLIDE_W - _RIGHT_MARGIN - (x + swatch + _KEY_GAP)
    if text_w(size) > room:
        x = max(min_x, SLIDE_W - _RIGHT_MARGIN - text_w(size) - swatch
                - _KEY_GAP)
        room = SLIDE_W - _RIGHT_MARGIN - (x + swatch + _KEY_GAP)
        if text_w(size) > room:
            size = max(6.0, size * room / text_w(size))
    return dict(x=x, top=top, row=row, swatch=swatch, size=size,
                text_w=room)


def _key(shapes, names: list, layout: dict):
    """Swatch + name per person, in slide order."""
    row, sw = layout["row"], layout["swatch"]
    for i, name in enumerate(names):
        y = layout["top"] + i * row
        box = _plain(shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                      Emu(int(layout["x"])),
                                      Emu(int(y + (row - sw) / 2)),
                                      Emu(int(sw)), Emu(int(sw))))
        box.fill.solid()
        box.fill.fore_color.rgb = _rgb(SERIES_COLORS[i % len(SERIES_COLORS)])
        box.line.fill.background()
        box.name = f"Key swatch {i + 1}"
        _text(shapes, layout["x"] + sw + _KEY_GAP, y, layout["text_w"], row,
              [name], _KEY_FONT, layout["size"], _KEY_COLOR, bold=True,
              align=PP_ALIGN.LEFT, name=f"Key name {i + 1}")


# ── Deck ───────────────────────────────────────────────────────────────────

def build_radar_deck(profiles: list, graph: str = "mirror",
                     names: list = None) -> bytes:
    """A .pptx with one slide per person, each adding that person's graph.

    ``profiles`` are in slide order; ``graph`` is "mirror", "public" or
    "stress"; ``names`` (same order) replaces the participant names in the
    key, e.g. for anonymised decks.
    """
    if graph not in DECK_GRAPHS:
        raise ValueError(f"Unknown graph {graph!r}")
    if not profiles:
        raise ValueError("No people selected")
    names = list(names) if names else [p["participant_name"]
                                       for p in profiles]
    values = [radar_values(p["graphs"][graph]) for p in profiles]
    label = DECK_GRAPHS[graph]

    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(SLIDE_W), Emu(SLIDE_H)
    blank = prs.slide_layouts[6]
    key = _key_layout(names)

    for k in range(1, len(profiles) + 1):
        slide = prs.slides.add_slide(blank)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = _rgb("FFFFFF")

        with _group(slide.shapes, "DISC radar") as radar:
            _grid(radar)
            for i in range(k):
                _series(radar, values[i], i, names[i])
        with _group(slide.shapes, "Behaviour symbols") as symbols:
            _symbols(symbols)
        with _group(slide.shapes, "Key") as legend:
            _key(legend, names[:k], key)

        slide.notes_slide.notes_text_frame.text = (
            f"{label} graph, slide {k} of {len(profiles)}: "
            + ", ".join(names[:k]))

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cp = prs.core_properties
    cp.title = f"DISC Radar Slides ({label})"
    cp.author = cp.last_modified_by = "SDG DISC Dashboard"
    cp.created = cp.modified = now
    cp.revision = 1

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()
