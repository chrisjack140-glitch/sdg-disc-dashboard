"""
Leadership Roadmap — DOCX renderer (python-docx)
=================================================
Walks the same RoadmapDocument block list as utils/roadmap_pdf.py and
emits .docx bytes. Fonts are referenced by name ("Georgia"/"Arial") —
Word resolves them client-side, so no font files are needed server-side.
Cell shading uses raw OOXML (w:shd) since python-docx has no shading API.
"""
import io

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Inches

from utils.roadmap_content_model import (
    PALETTE, PageBreak, HeaderBand, Paragraph, BulletList, CalloutBox,
    DataTable, BlankWorksheetTable, ShadedGroup, Divider, RoadmapDocument,
)

_CONTENT_W_IN = 7.4   # usable width with 0.55" margins on Letter


def _rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color.lstrip("#").upper())


def _shade_cell(cell, hex_color: str):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_color.lstrip("#").upper())
    cell._tc.get_or_add_tcPr().append(shd)


def _set_cell_margins(cell, top=100, start=110, bottom=100, end=110):
    tc_pr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for tag, val in (("top", top), ("start", start),
                     ("bottom", bottom), ("end", end)):
        el = OxmlElement(f"w:{tag}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tc_pr.append(mar)


def _hide_table_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "1")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "FFFFFF")
        borders.append(el)
    tbl_pr.append(borders)


def _light_table_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "D8DEE6")
        borders.append(el)
    tbl_pr.append(borders)


def _add_run(paragraph, text, font="Arial", size=10, color=PALETTE["body_text"],
             bold=False, italic=False):
    run = paragraph.add_run(text)
    run.font.name = font
    run.font.size = Pt(size)
    run.font.color.rgb = _rgb(color)
    run.font.bold = bold
    run.font.italic = italic
    return run


def _cell_text(cell, text, font="Arial", size=9.5, color=PALETTE["body_text"],
               bold=False, italic=False, align=None):
    p = cell.paragraphs[0]
    if align is not None:
        p.alignment = align
    _add_run(p, text, font=font, size=size, color=color, bold=bold,
             italic=italic)


def _one_cell_table(document, fill: str):
    tbl = document.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.columns[0].width = Inches(_CONTENT_W_IN)
    cell = tbl.cell(0, 0)
    cell.width = Inches(_CONTENT_W_IN)
    _shade_cell(cell, fill)
    _set_cell_margins(cell, 160, 180, 160, 180)
    _hide_table_borders(tbl)
    return tbl, cell


# ─────────────────────────────────────────
# Block renderers
# ─────────────────────────────────────────
def _r_header_band(document, block: HeaderBand):
    tbl = document.add_table(rows=1, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _hide_table_borders(tbl)
    left, right = tbl.cell(0, 0), tbl.cell(0, 1)
    left.width = Inches(_CONTENT_W_IN * 0.8)
    right.width = Inches(_CONTENT_W_IN * 0.2)
    for c in (left, right):
        _shade_cell(c, block.bg)
        _set_cell_margins(c, 140, 160, 140, 160)

    p = left.paragraphs[0]
    _add_run(p, block.title, font="Georgia", size=15, bold=True,
             color=block.fg)
    if block.subtitle:
        p2 = left.add_paragraph()
        _add_run(p2, block.subtitle, size=9.5, color="#DDE3EA")

    rp = right.paragraphs[0]
    rp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, word in enumerate(block.brand_tag.split()):
        if i:
            rp.add_run().add_break()
        _add_run(rp, word, size=8, bold=True, color="#DDE3EA")
    document.add_paragraph()


def _r_shaded_group(document, block: ShadedGroup):
    document.add_paragraph()
    document.add_paragraph()
    _, cell = _one_cell_table(document, block.bg)
    _set_cell_margins(cell, 350, 280, 350, 280)
    first = True
    for line in block.lines:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_run(p, line.text, font=line.font, size=line.size,
                 color=line.color, bold=line.bold, italic=line.italic)


def _r_divider(document, block: Divider):
    tbl = document.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _hide_table_borders(tbl)
    cell = tbl.cell(0, 0)
    cell.width = Inches(_CONTENT_W_IN)
    _shade_cell(cell, block.color)
    tr_pr = tbl.rows[0]._tr.get_or_add_trPr()
    tr_h = OxmlElement("w:trHeight")
    tr_h.set(qn("w:val"), "180")
    tr_pr.append(tr_h)
    document.add_paragraph()


def _r_paragraph(document, block: Paragraph):
    p = document.add_paragraph()
    _add_run(p, block.text, font=block.font, size=block.size * 0.72,
             color=block.color, bold=block.bold, italic=block.italic)


def _r_bullets(document, block: BulletList):
    for item in block.items:
        p = document.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        _add_run(p, f"-  {item}", size=block.size * 0.72,
                 color=block.color)


def _r_callout(document, block: CalloutBox):
    _, cell = _one_cell_table(document, block.tint)
    first = True
    if block.heading:
        p = cell.paragraphs[0]
        first = False
        _add_run(p, block.heading, size=10, bold=True,
                 color=PALETTE["navy_header"])
    p = cell.paragraphs[0] if first else cell.add_paragraph()
    _add_run(p, block.body, size=9.5, color=PALETTE["body_text"])
    document.add_paragraph()


def _r_data_table(document, block: DataTable):
    n_cols = (len(block.header_row) if block.header_row
              else max(len(r.cells) for r in block.rows))
    widths = (block.col_widths if block.col_widths
              else [1.0 / n_cols] * n_cols)
    n_rows = len(block.rows) + (1 if block.header_row else 0)
    tbl = document.add_table(rows=n_rows, cols=n_cols)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _light_table_borders(tbl)

    r = 0
    if block.header_row:
        for c, text in enumerate(block.header_row):
            cell = tbl.cell(0, c)
            cell.width = Inches(_CONTENT_W_IN * widths[c])
            _shade_cell(cell, PALETTE["slate_header"])
            _set_cell_margins(cell)
            _cell_text(cell, text, size=9, bold=True,
                       color=PALETTE["white"])
        r = 1

    for row in block.rows:
        cells = list(row.cells) + [""] * (n_cols - len(row.cells))
        for c, text in enumerate(cells):
            cell = tbl.cell(r, c)
            cell.width = Inches(_CONTENT_W_IN * widths[c])
            fill = row.fill or (PALETTE["row_label_fill"] if c == 0
                                else PALETTE["row_value_fill"])
            _shade_cell(cell, fill)
            _set_cell_margins(cell)
            if text:
                _cell_text(cell, text, size=9)
        r += 1
    document.add_paragraph()


def _r_worksheet_table(document, block: BlankWorksheetTable):
    n_cols = max(len(block.header_row), 1 + block.n_blank_cols)
    n_rows = 1 + len(block.prompts)
    tbl = document.add_table(rows=n_rows, cols=n_cols)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _light_table_borders(tbl)

    if n_cols == 1:
        widths = [1.0]
    else:
        widths = [0.42] + [0.58 / (n_cols - 1)] * (n_cols - 1)

    header = list(block.header_row) + [""] * (n_cols - len(block.header_row))
    for c, text in enumerate(header):
        cell = tbl.cell(0, c)
        cell.width = Inches(_CONTENT_W_IN * widths[c])
        _shade_cell(cell, PALETTE["slate_header"])
        _set_cell_margins(cell)
        if text:
            _cell_text(cell, text, size=9, bold=True,
                       color=PALETTE["white"])

    for r, prompt in enumerate(block.prompts, start=1):
        tr_pr = tbl.rows[r]._tr.get_or_add_trPr()
        tr_h = OxmlElement("w:trHeight")
        tr_h.set(qn("w:val"), "420")
        tr_pr.append(tr_h)
        for c in range(n_cols):
            cell = tbl.cell(r, c)
            cell.width = Inches(_CONTENT_W_IN * widths[c])
            _shade_cell(cell, PALETTE["row_label_fill"] if c == 0
                        else PALETTE["row_value_fill"])
            _set_cell_margins(cell)
            if c == 0 and prompt:
                _cell_text(cell, prompt, size=9)
    document.add_paragraph()


_RENDERERS = {
    HeaderBand:          _r_header_band,
    ShadedGroup:         _r_shaded_group,
    Divider:             _r_divider,
    Paragraph:           _r_paragraph,
    BulletList:          _r_bullets,
    CalloutBox:          _r_callout,
    DataTable:           _r_data_table,
    BlankWorksheetTable: _r_worksheet_table,
}


def render_docx(doc: RoadmapDocument) -> bytes:
    document = Document()

    # Letter page with 0.55" margins to match the PDF renderer
    for section in document.sections:
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.left_margin = Inches(0.55)
        section.right_margin = Inches(0.55)
        section.top_margin = Inches(0.55)
        section.bottom_margin = Inches(0.55)

    # Default document font
    style = document.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(10)
    style.font.color.rgb = _rgb(PALETTE["body_text"])

    for block in doc.blocks:
        if isinstance(block, PageBreak):
            document.add_page_break()
            continue
        renderer = _RENDERERS.get(type(block))
        if renderer:
            renderer(document, block)

    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()
