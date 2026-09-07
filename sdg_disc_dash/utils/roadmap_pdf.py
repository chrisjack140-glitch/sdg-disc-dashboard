"""
Leadership Roadmap — PDF renderer (ReportLab)
==============================================
Walks a RoadmapDocument's block list and emits PDF bytes. Colors come
from roadmap_content_model.PALETTE so PDF and DOCX output stay in sync.

Fonts: ReportLab ships Helvetica (Arial-equivalent) but not Georgia.
If the Windows Georgia TTFs are present they are registered for a
pixel-accurate match; otherwise headers fall back to Times-Roman.
"""
import io
import os

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Spacer,
    Paragraph as RLParagraph, PageBreak as RLPageBreak,
)

from utils.roadmap_content_model import (
    PALETTE, DISC_COLORS, PageBreak, HeaderBand, Paragraph, BulletList,
    CalloutBox, DataTable, BlankWorksheetTable, ShadedGroup, Divider,
    ScoreStrip, BarChart, RoadmapDocument,
)

PAGE_W, PAGE_H = LETTER
MARGIN = 0.55 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

_FONTS_REGISTERED = False
_HEADER_FONT = "Times-Roman"
_HEADER_FONT_BOLD = "Times-Bold"
_BODY_FONT = "Helvetica"
_BODY_FONT_BOLD = "Helvetica-Bold"
_BODY_FONT_ITALIC = "Helvetica-Oblique"


def _register_fonts():
    """Register Georgia from Windows fonts if available; fall back to
    Times. Idempotent."""
    global _FONTS_REGISTERED, _HEADER_FONT, _HEADER_FONT_BOLD
    if _FONTS_REGISTERED:
        return
    georgia = r"C:\Windows\Fonts\georgia.ttf"
    georgia_b = r"C:\Windows\Fonts\georgiab.ttf"
    try:
        if os.path.exists(georgia) and os.path.exists(georgia_b):
            pdfmetrics.registerFont(TTFont("Georgia", georgia))
            pdfmetrics.registerFont(TTFont("Georgia-Bold", georgia_b))
            _HEADER_FONT = "Georgia"
            _HEADER_FONT_BOLD = "Georgia-Bold"
    except Exception:
        pass  # keep Times fallback
    _FONTS_REGISTERED = True


def _font_for(block_font: str, bold: bool, italic: bool = False) -> str:
    if block_font == "Georgia":
        return _HEADER_FONT_BOLD if bold else _HEADER_FONT
    if italic:
        return _BODY_FONT_ITALIC
    return _BODY_FONT_BOLD if bold else _BODY_FONT


def _pstyle(name, font, size, color, leading_mult=1.35, align=None) -> ParagraphStyle:
    st = ParagraphStyle(
        name=name, fontName=font, fontSize=size,
        leading=size * leading_mult, textColor=HexColor(color),
    )
    if align is not None:
        st.alignment = align
    return st


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))


def _cell_para(text, size=9.5, color=PALETTE["body_text"], bold=False):
    font = _BODY_FONT_BOLD if bold else _BODY_FONT
    return RLParagraph(_esc(text),
                       _pstyle("cell", font, size, color, 1.25))


_BOX_PAD = 10


def _header_band(block: HeaderBand):
    """Navy band: title + subtitle left, brand tag right."""
    title_p = RLParagraph(
        _esc(block.title),
        _pstyle("hb-t", _font_for("Georgia", True), 16, block.fg, 1.2),
    )
    sub_p = RLParagraph(
        _esc(block.subtitle),
        _pstyle("hb-s", _BODY_FONT, 10, "#DDE3EA", 1.25),
    ) if block.subtitle else Spacer(0, 0)
    tag_p = RLParagraph(
        _esc(block.brand_tag).replace(" ", "<br/>"),
        _pstyle("hb-tag", _BODY_FONT_BOLD, 8, "#DDE3EA", 1.3, TA_CENTER),
    )
    inner = Table(
        [[[title_p, Spacer(0, 3), sub_p], tag_p]],
        colWidths=[CONTENT_W * 0.80 - _BOX_PAD * 2, CONTENT_W * 0.20],
    )
    inner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), HexColor(block.bg)),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), _BOX_PAD),
        ("RIGHTPADDING",  (0, 0), (-1, -1), _BOX_PAD),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    return [inner, Spacer(0, 12)]


def _shaded_group(block: ShadedGroup):
    paras = []
    for i, line in enumerate(block.lines):
        font = _font_for(line.font, line.bold, line.italic)
        paras.append(RLParagraph(
            _esc(line.text),
            _pstyle(f"sg{i}", font, line.size, line.color, 1.3, TA_CENTER),
        ))
        paras.append(Spacer(0, 6))
    tbl = Table([[paras]], colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), HexColor(block.bg)),
        ("LEFTPADDING",   (0, 0), (-1, -1), 20),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
        ("TOPPADDING",    (0, 0), (-1, -1), 26),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 26),
    ]))
    return [Spacer(0, 60), tbl, Spacer(0, 4)]


def _divider(block: Divider):
    tbl = Table([[""]], colWidths=[CONTENT_W], rowHeights=[block.height])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor(block.color)),
    ]))
    return [tbl, Spacer(0, 14)]


def _paragraph(block: Paragraph):
    font = _font_for(block.font, block.bold, block.italic)
    # Content-model sizes are docx-ish; scale down slightly for print body
    size = max(8.5, block.size * 0.72) if block.size <= 16 else block.size * 0.72
    return [RLParagraph(_esc(block.text),
                        _pstyle("p", font, size, block.color)),
            Spacer(0, 8)]


def _bullets(block: BulletList):
    out = []
    size = max(8.5, block.size * 0.72)
    for item in block.items:
        out.append(RLParagraph(
            f"•&nbsp;&nbsp;{_esc(item)}",
            _pstyle("b", _BODY_FONT, size, block.color, 1.3),
        ))
        out.append(Spacer(0, 4))
    out.append(Spacer(0, 4))
    return out


def _callout(block: CalloutBox):
    paras = []
    if block.heading:
        paras.append(RLParagraph(
            _esc(block.heading),
            _pstyle("co-h", _BODY_FONT_BOLD, 10, PALETTE["navy_header"]),
        ))
        paras.append(Spacer(0, 4))
    paras.append(RLParagraph(
        _esc(block.body),
        _pstyle("co-b", _BODY_FONT, 9.5, PALETTE["body_text"], 1.3),
    ))
    tbl = Table([[paras]], colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), HexColor(block.tint)),
        ("LEFTPADDING",   (0, 0), (-1, -1), _BOX_PAD),
        ("RIGHTPADDING",  (0, 0), (-1, -1), _BOX_PAD),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return [Spacer(0, 2), tbl, Spacer(0, 10)]


def _data_table(block: DataTable):
    n_cols = (len(block.header_row) if block.header_row
              else max(len(r.cells) for r in block.rows))
    widths = (block.col_widths if block.col_widths
              else [1.0 / n_cols] * n_cols)
    col_w = [w * CONTENT_W for w in widths]

    data, style_cmds, r_idx = [], [], 0
    if block.header_row:
        data.append([_cell_para(c, bold=True, color=PALETTE["white"])
                     for c in block.header_row])
        style_cmds.append(("BACKGROUND", (0, 0), (-1, 0),
                           HexColor(PALETTE["slate_header"])))
        r_idx = 1

    for row in block.rows:
        cells = list(row.cells) + [""] * (n_cols - len(row.cells))
        data.append([_cell_para(c) for c in cells])
        if row.fill:
            style_cmds.append(("BACKGROUND", (0, r_idx), (-1, r_idx),
                               HexColor(row.fill)))
        else:
            # Template pattern: label column light, value columns white
            style_cmds.append(("BACKGROUND", (0, r_idx), (0, r_idx),
                               HexColor(PALETTE["row_label_fill"])))
            style_cmds.append(("BACKGROUND", (1, r_idx), (-1, r_idx),
                               HexColor(PALETTE["row_value_fill"])))
        r_idx += 1

    tbl = Table(data, colWidths=col_w, repeatRows=1 if block.header_row else 0)
    tbl.setStyle(TableStyle(style_cmds + [
        ("GRID",          (0, 0), (-1, -1), 0.75, HexColor(PALETTE["table_grid"])),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return [tbl, Spacer(0, 10)]


def _worksheet_table(block: BlankWorksheetTable):
    n_cols = max(len(block.header_row), 1 + block.n_blank_cols)
    if n_cols == 1:
        widths = [CONTENT_W]
    else:
        widths = [CONTENT_W * 0.42] + \
                 [CONTENT_W * 0.58 / (n_cols - 1)] * (n_cols - 1)

    data, style_cmds = [], []
    data.append([_cell_para(c, bold=True, color=PALETTE["white"])
                 for c in block.header_row] +
                [_cell_para("")] * (n_cols - len(block.header_row)))
    style_cmds.append(("BACKGROUND", (0, 0), (-1, 0),
                       HexColor(PALETTE["slate_header"])))

    blank_row_h = 24
    row_heights = [None]
    for i, prompt in enumerate(block.prompts, start=1):
        data.append([_cell_para(prompt)] + [_cell_para("")] * (n_cols - 1))
        style_cmds.append(("BACKGROUND", (0, i), (0, i),
                           HexColor(PALETTE["row_label_fill"])))
        row_heights.append(blank_row_h if prompt else 18)

    tbl = Table(data, colWidths=widths, rowHeights=row_heights)
    tbl.setStyle(TableStyle(style_cmds + [
        ("GRID",          (0, 0), (-1, -1), 0.75, HexColor(PALETTE["table_grid"])),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return [tbl, Spacer(0, 10)]


def _score_strip(block):
    """DISC D/I/S/C band: colored letter cells over a cream score row."""
    def centered(text, size, color):
        return RLParagraph(_esc(text),
                           _pstyle("strip", _BODY_FONT_BOLD, size, color,
                                   1.25, TA_CENTER))

    letters, values, fills = [], [], []
    for letter, score in block.scores:
        color = DISC_COLORS.get(letter, PALETTE["slate_header"])
        fills.append(color)
        letters.append(centered(letter, 11, PALETTE["white"]))
        values.append(centered(f"{score:+.2f}", 9, color))
    n = len(block.scores) or 1
    tbl = Table([letters, values], colWidths=[CONTENT_W / n] * n)
    style = [
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND",    (0, 1), (-1, 1), HexColor(PALETTE["score_cell_fill"])),
    ]
    for i, fill in enumerate(fills):
        style.append(("BACKGROUND", (i, 0), (i, 0), HexColor(fill)))
    tbl.setStyle(TableStyle(style))

    out = []
    if block.caption:
        out.append(RLParagraph(
            block.caption.upper(),
            ParagraphStyle("stripcap", fontName=_font_for("Arial", True),
                           fontSize=7.5, textColor=HexColor(PALETTE["gold_accent"]),
                           spaceBefore=8, spaceAfter=4, leading=10)))
    out.extend([tbl, Spacer(1, 10)])
    return out


def _bar_chart(block):
    """Horizontal score bars — label, filled track, value."""
    label_w = CONTENT_W * 0.32
    value_w = CONTENT_W * 0.10
    track_w = CONTENT_W - label_w - value_w
    rows, style = [], [
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (0, 0), (0, -1), "RIGHT"),
        ("ALIGN",         (2, 0), (2, -1), "LEFT"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (1, 0), (1, -1), 6),
        ("RIGHTPADDING",  (1, 0), (1, -1), 6),
    ]
    for label, score in block.rows:
        fill = max(0.02, min(1.0, score / block.scale))
        bar = Table([[""], ], colWidths=[track_w - 12], rowHeights=[7])
        bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), HexColor(block.track)),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        filled = Table([[""], ], colWidths=[(track_w - 12) * fill],
                       rowHeights=[7])
        filled.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), HexColor(block.color)),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        gauge = Table([[filled, ""]],
                      colWidths=[(track_w - 12) * fill,
                                 (track_w - 12) * (1 - fill)],
                      rowHeights=[7])
        gauge.setStyle(TableStyle([
            ("BACKGROUND", (1, 0), (1, 0), HexColor(block.track)),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        rows.append([
            _cell_para(label, size=9.5, color="#444444"),
            gauge,
            _cell_para(str(score), size=9.5, bold=True, color=block.color),
        ])
    tbl = Table(rows, colWidths=[label_w, track_w, value_w])
    tbl.setStyle(TableStyle(style))

    out = []
    if block.caption:
        out.append(RLParagraph(
            block.caption.upper(),
            ParagraphStyle("barcap", fontName=_font_for("Arial", True),
                           fontSize=7.5, textColor=HexColor(block.color),
                           spaceBefore=8, spaceAfter=5, leading=10)))
    out.extend([tbl, Spacer(1, 10)])
    return out


_RENDERERS = {
    HeaderBand:          _header_band,
    ScoreStrip:          _score_strip,
    BarChart:            _bar_chart,
    ShadedGroup:         _shaded_group,
    Divider:             _divider,
    Paragraph:           _paragraph,
    BulletList:          _bullets,
    CalloutBox:          _callout,
    DataTable:           _data_table,
    BlankWorksheetTable: _worksheet_table,
}


def render_pdf(doc: RoadmapDocument) -> bytes:
    _register_fonts()
    buf = io.BytesIO()
    pdf = SimpleDocTemplate(
        buf, pagesize=LETTER,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title=f"{doc.person_name} — Leadership Roadmap",
        author="The Strategic Design Group",
    )
    flowables = []
    for block in doc.blocks:
        if isinstance(block, PageBreak):
            flowables.append(RLPageBreak())
            continue
        renderer = _RENDERERS.get(type(block))
        if renderer:
            flowables.extend(renderer(block))
    pdf.build(flowables)
    return buf.getvalue()
