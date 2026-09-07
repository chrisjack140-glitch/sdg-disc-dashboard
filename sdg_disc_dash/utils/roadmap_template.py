"""
Leadership Roadmap — template filler
=====================================
Produces a booklet by filling utils/roadmap_template.docx rather than
rebuilding the layout in code. The template is the hand-finished reference
booklet with one participant's data swapped for {{TOKENS}} (see
tools/build_roadmap_template.py), so output is identical to the reference
in every respect except the person's own content.

Two things cannot be expressed as a token because their row count varies
per person — the Core EQ-i Strengths and Key Development Areas bar charts.
Those tables are marked in the template and rebuilt here by cloning their
first row, which keeps their styling without hard-coding it.
"""
import copy
import io
import re
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.table import Table
from docx.text.paragraph import Paragraph

from utils.eqi_benchmarks import split_subscales

TEMPLATE_PATH = Path(__file__).parent / "roadmap_template.docx"

BAR_SCALE = 140          # bar fill is score/140 of the track, as in the source
_TOKEN_RE = re.compile(r"\{\{[A-Z0-9_:]+\}\}")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


# ─────────────────────────────────────────
# Low-level docx helpers
# ─────────────────────────────────────────
def _iter_paragraphs(container):
    """Every paragraph in the document, including inside nested tables."""
    if hasattr(container, "element"):
        body = container.element.body
    else:
        body = container
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, container)
        elif child.tag == qn("w:tbl"):
            yield from _iter_table_paragraphs(Table(child, container))


def _iter_table_paragraphs(table):
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                yield p
            for nested in cell.tables:
                yield from _iter_table_paragraphs(nested)


def _set_paragraph_text(paragraph, text):
    """Replace a paragraph's text, keeping the first run's formatting.

    ``**bold**`` in the text becomes a bold run, so generated prose can
    carry the same inline emphasis the reference booklet uses.
    """
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(text)
        return
    template_run = runs[0]
    base = copy.deepcopy(template_run._r.find(qn("w:rPr")))
    for run in runs[1:]:
        run._r.getparent().remove(run._r)

    parts = _BOLD_RE.split(text)          # odd indices were inside ** **
    template_run.text = parts[0]
    anchor = template_run._r
    for i, part in enumerate(parts[1:], start=1):
        if not part:
            continue
        new_r = copy.deepcopy(template_run._r)
        for t in new_r.findall(qn("w:t")):
            new_r.remove(t)
        rPr = new_r.find(qn("w:rPr"))
        if rPr is not None:
            new_r.remove(rPr)
        if base is not None:
            rPr = copy.deepcopy(base)
            new_r.insert(0, rPr)
        else:
            rPr = None
        if i % 2 == 1:                     # this part was wrapped in ** **
            if rPr is None:
                rPr = copy.deepcopy(base) if base is not None else None
            if rPr is not None and rPr.find(qn("w:b")) is None:
                rPr.append(docx.oxml.OxmlElement("w:b"))
        t = docx.oxml.OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = part
        new_r.append(t)
        anchor.addnext(new_r)
        anchor = new_r


def _paragraph_token(paragraph):
    text = "".join(r.text or "" for r in paragraph.runs).strip()
    match = _TOKEN_RE.fullmatch(text)
    return match.group(0) if match else None


# ─────────────────────────────────────────
# Bar chart rebuild
# ─────────────────────────────────────────
def _bar_track_width(row):
    """Total width of the gauge in a prototype bar row, in DXA."""
    gauge = row.cells[1]._tc.find(qn("w:tbl"))
    if gauge is None:
        return None
    grid = gauge.find(qn("w:tblGrid"))
    return sum(int(g.get(qn("w:w"))) for g in grid.findall(qn("w:gridCol")))


def _fill_bar_row(row, label, score, track):
    """Point one cloned bar row at a different subscale."""
    _set_paragraph_text(row.cells[0].paragraphs[0], label)
    _set_paragraph_text(row.cells[2].paragraphs[0], str(score))

    gauge = row.cells[1]._tc.find(qn("w:tbl"))
    if gauge is None or track is None:
        return
    fill = max(1, min(track - 1, int(track * score / BAR_SCALE)))
    grid = gauge.find(qn("w:tblGrid")).findall(qn("w:gridCol"))
    grid[0].set(qn("w:w"), str(fill))
    grid[1].set(qn("w:w"), str(track - fill))
    cells = gauge.find(qn("w:tr")).findall(qn("w:tc"))
    for tc, width in zip(cells, (fill, track - fill)):
        tc.find(qn("w:tcPr")).find(qn("w:tcW")).set(qn("w:w"), str(width))


def _clear_break_before(paragraph_el):
    pPr = paragraph_el.find(qn("w:pPr"))
    if pPr is not None:
        for brk in pPr.findall(qn("w:pageBreakBefore")):
            pPr.remove(brk)


def _drop_caption_above(table_el, caption_prefix):
    """Remove a chart's caption (and its page break) when the chart is gone.

    A person can legitimately clear every subscale, or none, so one of the
    two EQ charts may be empty. Left alone its caption would sit on an
    otherwise blank page.
    """
    node = table_el.getprevious()
    while node is not None:
        if node.tag == qn("w:tbl"):
            return
        text = " ".join("".join(t.text or "" for t in node.iter(qn("w:t"))).split())
        if text.upper().startswith(caption_prefix.upper()):
            # the note above the caption carries the page break; without a
            # chart to introduce, it should not start a page of its own
            # the break sits on the disclaimer that opens this page; with
            # no chart to introduce, nothing here should start a page
            previous = node.getprevious()
            while previous is not None and previous.tag == qn("w:p"):
                _clear_break_before(previous)
                previous = previous.getprevious()
            node.getparent().remove(node)
            return
        if text:                      # ran into other copy — leave it alone
            return
        node = node.getprevious()


# A bar row is ~0.545in at the reference spacing, and the strengths chart
# has ~7.5in of page beneath its banner and intro — so about 13 bars fit
# before spilling onto a second page. Past that the rows are tightened
# instead, which is invisible on a normal-length chart because it only
# applies when one would otherwise overflow.
TIGHTEN_ABOVE = 12


def _tighten_bar_row(row):
    """Compress one bar row so a long chart still fits on its page.

    Most of the row's height is a spare empty paragraph in the value
    cell — a full line each — plus generous cell margins. Removing the
    first and halving the second takes the pitch from ~0.55in to ~0.31in
    without touching the bar itself.
    """
    for cell in row.cells:
        paragraphs = cell.paragraphs
        if len(paragraphs) > 1:
            for p in paragraphs[1:]:
                if not p.text.strip() and p._p.find(qn("w:tbl")) is None:
                    p._p.getparent().remove(p._p)
        tcPr = cell._tc.find(qn("w:tcPr"))
        if tcPr is None:
            continue
        mar = tcPr.find(qn("w:tcMar"))
        if mar is None:
            continue
        for tag in ("w:top", "w:bottom"):
            el = mar.find(qn(tag))
            if el is not None:
                el.set(qn("w:w"), str(max(10, int(el.get(qn("w:w")) or 0) // 3)))


def _rebuild_bar_table(table, entries, caption_prefix=None):
    """Re-point a marked bar table at this person's subscales.

    The first row is the prototype: it is cloned for each entry so the
    chart inherits the reference booklet's styling rather than restating
    it here. An empty list removes the table and its caption.
    """
    prototype = copy.deepcopy(table.rows[0]._tr)
    track = _bar_track_width(table.rows[0])

    for tr in list(table._tbl.tr_lst):
        table._tbl.remove(tr)

    if not entries:
        table_el = table._tbl
        if caption_prefix:
            _drop_caption_above(table_el, caption_prefix)
        table_el.getparent().remove(table_el)
        return

    tighten = len(entries) > TIGHTEN_ABOVE
    for label, score in entries:
        tr = copy.deepcopy(prototype)
        table._tbl.append(tr)
        row = table.rows[-1]
        _fill_bar_row(row, label, score, track)
        if tighten:
            _tighten_bar_row(row)


def _rebuild_text_table(table, entries, header_label=None):
    """Rebuild a plain text table, cloning a body row per entry.

    Used for the EQ Dimension table, whose row count follows the number
    of strengths. The header row is kept as-is and the first body row is
    the prototype.
    """
    rows = table._tbl.tr_lst
    if len(rows) < 2:
        return
    header = copy.deepcopy(rows[0])
    prototype = copy.deepcopy(rows[1])

    for tr in list(table._tbl.tr_lst):
        table._tbl.remove(tr)
    table._tbl.append(header)
    if header_label is not None:
        # the rebuild marker lives in the header's first cell
        _set_paragraph_text(table.rows[0].cells[0].paragraphs[0], header_label)

    if not entries:
        table._tbl.getparent().remove(table._tbl)
        return

    for cells in entries:
        tr = copy.deepcopy(prototype)
        table._tbl.append(tr)
        row = table.rows[-1]
        for i, text in enumerate(cells):
            if i < len(row.cells):
                _set_paragraph_text(row.cells[i].paragraphs[0], str(text))


# ─────────────────────────────────────────
# Public API
# ─────────────────────────────────────────
def render_from_template(values: dict, eqi_scores: dict,
                         dimension_rows=None,
                         template_path: Path = TEMPLATE_PATH) -> bytes:
    """Fill the reference booklet with one participant's content.

    ``values`` maps token names (without braces) to replacement text and
    ``eqi_scores`` drives the two bar charts. ``dimension_rows`` supplies
    the EQ Dimension table as (label, score, meaning) triples, since its
    row count follows the number of strengths. Tokens with no value are
    left blank rather than printed, so a partial profile degrades quietly.
    """
    document = docx.Document(str(template_path))

    strengths, development = split_subscales(eqi_scores or {})
    # (rows, the caption to remove with the chart if it is empty)
    bar_tables = {
        "{{TABLE:EQ_STRENGTH_BARS}}":
            ([(l, s) for l, s, _ in strengths], "CORE EQ-I STRENGTHS"),
        "{{TABLE:EQ_DEVELOPMENT_BARS}}":
            ([(l, s) for l, s, _ in development],
             "LEADERSHIP CAPACITIES TO STRENGTHEN"),
    }
    # (rows, header label the marker replaced)
    text_tables = {"{{TABLE:EQ_DIMENSIONS}}":
                   (list(dimension_rows or []), "EQ Dimension")}

    # rebuilt tables first — doing so invalidates paragraph references
    for child in list(document.element.body.iterchildren()):
        if child.tag != qn("w:tbl"):
            continue
        table = Table(child, document)
        if not table.rows:
            continue
        cell_text = " ".join(table.rows[0].cells[0].text.split())
        # match on containment, not equality: a stray label left beside the
        # marker must not silently skip the rebuild and leave the reference
        # participant's rows in someone else's booklet
        marker = next((m for m in (*bar_tables, *text_tables)
                       if m in cell_text), None)
        if marker in bar_tables:
            rows, caption = bar_tables[marker]
            _rebuild_bar_table(table, rows, caption_prefix=caption)
        elif marker in text_tables:
            rows, label = text_tables[marker]
            _rebuild_text_table(table, rows, header_label=label)

    # then every remaining token
    for paragraph in _iter_paragraphs(document):
        token = _paragraph_token(paragraph)
        if token:
            key = token.strip("{}")
            _set_paragraph_text(paragraph, values.get(key, ""))
            continue
        # tokens embedded mid-sentence
        full = "".join(r.text or "" for r in paragraph.runs)
        if "{{" not in full:
            continue
        replaced = _TOKEN_RE.sub(
            lambda m: values.get(m.group(0).strip("{}"), ""), full)
        if replaced != full:
            _set_paragraph_text(paragraph, replaced)

    _collapse_empty_runs(document)

    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def _collapse_empty_runs(document, keep=1):
    """Drop stray whitespace left behind by filling.

    Blanked-out tokens and removed charts leave empty paragraphs, and a
    long enough run of them prints as a blank page. Consecutive page
    breaks are collapsed for the same reason.
    """
    body = document.element.body

    def is_break(el):
        return el.tag == qn("w:p") and el.find(
            ".//" + qn("w:br") + '[@{%s}type="page"]'
            % "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        ) is not None

    def is_empty(el):
        return (el.tag == qn("w:p")
                and not "".join(t.text or "" for t in el.iter(qn("w:t"))).strip()
                and not is_break(el))

    run = []
    for child in list(body.iterchildren()) + [None]:
        if child is not None and is_empty(child):
            run.append(child)
            continue
        for extra in run[keep:]:
            body.remove(extra)
        run = []

    # a page break immediately followed by another produces an empty page
    previous_break = False
    for child in list(body.iterchildren()):
        if child.tag == qn("w:p") and is_break(child):
            if previous_break:
                body.remove(child)
                continue
            previous_break = True
        elif child.tag in (qn("w:p"), qn("w:tbl")):
            previous_break = False
