"""
Leadership Roadmap — template filler
=====================================
Produces a booklet by filling utils/roadmap_template.docx rather than
rebuilding the layout in code. The template is the hand-finished reference
booklet with one participant's data swapped for {{TOKENS}} (see
tools/build_roadmap_template.py), so output is identical to the reference
in every respect except the person's own content.

Some content cannot be expressed as a text token:
- the Core EQ-i Strengths and Leadership Capacities bar charts and the EQ
  Dimension table, whose row count varies per person — marked in the
  template and rebuilt here by cloning their first row, which keeps their
  styling without hard-coding it;
- the Leadership Derailment Risk chart — four fixed rows whose bars and
  scores are set here, cranberry under 100 and gold at 100 or above;
- the RADAR SHIFTS picture, whose image is swapped for the person's own.

Every booklet keeps the reference's page layout, so the table of contents'
stored page numbers are always right: pages the reference left to the flow
of the text get real page breaks, and the variable-length charts and the EQ
Dimension table are sized to fit their pages (see Pagination below).
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
    carry the same inline emphasis the reference booklet uses. A bold run
    left empty in the template (see tools/build_roadmap_template.py) gives
    the exact formatting for those words — e.g. the lead "DISC" a size up.
    """
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(text)
        return
    template_run = runs[0]
    base = copy.deepcopy(template_run._r.find(qn("w:rPr")))
    bold_model = None
    for r in runs[1:]:
        rpr = r._r.find(qn("w:rPr"))
        style = rpr.find(qn("w:rStyle")) if rpr is not None else None
        if not (r.text or "") and rpr is not None and (
                rpr.find(qn("w:b")) is not None
                or (style is not None and style.get(qn("w:val")) == "Strong")):
            bold_model = copy.deepcopy(rpr)
            break
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
        if i % 2 == 1 and bold_model is not None:
            if rPr is not None:
                new_r.remove(rPr)
            rPr = copy.deepcopy(bold_model)
            new_r.insert(0, rPr)
        elif i % 2 == 1:                   # this part was wrapped in ** **
            if rPr is None:
                rPr = docx.oxml.OxmlElement("w:rPr")
                new_r.insert(0, rPr)
            if rPr.find(qn("w:b")) is None:
                rPr.append(docx.oxml.OxmlElement("w:b"))
        t = docx.oxml.OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = part
        new_r.append(t)
        anchor.addnext(new_r)
        anchor = new_r


def _content_paragraph(cell):
    """The paragraph in a cell that carries the formatting.

    Cells in the reference booklet routinely open with an empty,
    unformatted paragraph before the one holding the styled text. Writing
    into `paragraphs[0]` therefore drops the font, size and colour — which
    is how the EQ Dimension header came out small and black instead of
    white Georgia bold.
    """
    paragraphs = cell.paragraphs
    if not paragraphs:
        return cell.add_paragraph()
    for p in paragraphs:
        if p.text.strip():
            return p
    for p in paragraphs:
        runs = p.runs
        if runs and runs[0]._r.find(qn("w:rPr")) is not None:
            return p
    return paragraphs[0]


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


# Leadership Derailment Risk: the four subscales the EQ-i report flags for
# derailment under pressure. Below the mid-range of 100 the bar is
# cranberry; at or above it, gold.
DERAILERS_CAPTION = "LEADERSHIP DERAILMENT RISK"
DERAILER_KEYS = {
    "Impulse Control": "impulse_control",
    "Stress Tolerance": "stress_tolerance",
    "Problem Solving": "problem_solving",
    "Independence": "independence",
}
DERAILER_LOW, DERAILER_OK = "9B1B30", "C59B2D"


def _fill_derailers(document, eqi_scores):
    """Point the derailers chart at this person's four scores.

    The chart's rows and labels are fixed; each row's bar length, bar
    colour and score colour are set from the score. Without EQ-i scores the
    chart and its caption are removed rather than printed empty.
    """
    body = document.element.body
    caption = next((el for el in body.iterchildren()
                    if el.tag == qn("w:p") and " ".join("".join(
                        t.text or "" for t in el.iter(qn("w:t"))).split()
                    ).upper() == DERAILERS_CAPTION), None)
    if caption is None:
        return
    table_el = caption.getnext()
    while table_el is not None and table_el.tag != qn("w:tbl"):
        table_el = table_el.getnext()
    if table_el is None:
        return
    table = Table(table_el, document)
    scores = {label: (eqi_scores or {}).get(key)
              for label, key in DERAILER_KEYS.items()}
    if not any(isinstance(v, (int, float)) for v in scores.values()):
        body.remove(table_el)
        body.remove(caption)
        return
    for row in table.rows:
        label = " ".join(row.cells[0].text.split())
        score = scores.get(label)
        if not isinstance(score, (int, float)):
            continue
        score = int(score)
        colour = DERAILER_LOW if score < ABOVE_AVERAGE else DERAILER_OK
        _fill_bar_row(row, label, score, _bar_track_width(row),
                      fill_color=colour)
        for c in row.cells[2]._tc.iter(qn("w:color")):
            c.set(qn("w:val"), colour)


def _swap_images(document, images, name=""):
    """Replace marked pictures with this person's images.

    A picture is marked by its alt text ("{{IMAGE:RADAR}}"); its image part
    is swapped in place, so size and placement stay the template's. The alt
    text is then set to a description of the new image.
    """
    descriptions = {
        "RADAR": f"Radar chart overlaying {name}'s Stress and Mirror DISC "
                 "graphs" if name else "Radar chart of Stress and Mirror "
                 "DISC graphs",
    }
    wp = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
    a = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    for doc_pr in document.element.body.iter(wp + "docPr"):
        match = re.fullmatch(r"\{\{IMAGE:([A-Z_]+)\}\}", doc_pr.get("descr") or "")
        if not match:
            continue
        key = match.group(1)
        blip = next(doc_pr.getparent().iter(a + "blip"), None)
        if key in (images or {}) and blip is not None:
            part = document.part.related_parts[blip.get(qn("r:embed"))]
            part._blob = images[key]
        doc_pr.set("descr", descriptions.get(key, ""))


# In the Leadership Capacities chart a score can sit below its leadership
# bar while still being at or above the population average of 100. Those
# read as range to extend rather than shortfalls, so their bar and score
# are drawn in the strengths green. The subscale stays where it is — only the bar colour
# changes.
ABOVE_AVERAGE = 100
ABOVE_AVERAGE_FILL = "1A6B4A"     # the Core EQ-i Strengths green


def _fill_bar_row(row, label, score, track, fill_color=None):
    """Point one cloned bar row at a different subscale."""
    _set_paragraph_text(_content_paragraph(row.cells[0]), label)
    _set_paragraph_text(_content_paragraph(row.cells[2]), str(score))

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

    if fill_color:
        # only the filled portion — the grey remainder stays as it is
        shd = cells[0].find(qn("w:tcPr")).find(qn("w:shd"))
        if shd is not None:
            shd.set(qn("w:fill"), fill_color)


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


# How many bars each chart holds at the reference spacing before it would
# run off its page; past that its rows are tightened, which is invisible on
# a normal-length chart because it only applies when one would otherwise
# overflow. Strength bars are ~43pt apart and start ~3.8in down page 16
# under the banner and note; capacity bars are ~36pt apart and share page
# 17 with the note above them and two paragraphs below. When there is no
# strengths chart the capacities chart opens its page and has room for all
# fifteen.
STRENGTH_BARS_FIT = 10
DEVELOPMENT_BARS_FIT = 11
DEVELOPMENT_BARS_FIT_ALONE = 15


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


def _rebuild_bar_table(table, entries, caption_prefix=None,
                       recolour_above_average=False, fit=STRENGTH_BARS_FIT):
    """Re-point a marked bar table at this person's subscales.

    The first row is the prototype: it is cloned for each entry so the
    chart inherits the reference booklet's styling rather than restating
    it here. An empty list removes the table and its caption. More than
    ``fit`` entries tightens the rows so the chart stays on its page.
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

    tighten = len(entries) > fit
    for label, score in entries:
        tr = copy.deepcopy(prototype)
        table._tbl.append(tr)
        row = table.rows[-1]
        colour = (ABOVE_AVERAGE_FILL
                  if recolour_above_average and score >= ABOVE_AVERAGE
                  else None)
        _fill_bar_row(row, label, score, track, fill_color=colour)
        if colour:                          # the score number matches
            for c in row.cells[2]._tc.iter(qn("w:color")):
                c.set(qn("w:val"), colour)
        if tighten:
            _tighten_bar_row(row)


# The EQ Dimension table and the EQ Practice box share page 19. At the
# reference spacing its header is ~63pt and each strength row ~63pt (a
# spare empty paragraph above and below the text), so four strengths and a
# long development list fill the page. Past that the strength rows lose
# their spare paragraphs, go to single line spacing, and share out the room
# that is left in proportion to their text; if even that is too tall the
# row text drops to 10pt. The development row keeps the reference layout;
# its height is estimated from its text (one line per score, or the
# meaning column's wrapped sentences, whichever is longer).
DIMENSION_TABLE_BUDGET = 575      # pt for the table, leaving the EQ Practice box room
DIMENSION_HEADER = 70             # pt
DIMENSION_HEADER_TIGHT = 36       # pt, one line of heading plus margins
DIMENSION_ROW = 63                # pt, reference strength row
DIMENSION_LINE = 14               # pt per line at the reference spacing
DIMENSION_COMPACT_LINE = 14.2     # pt per line at single spacing, 12pt
DIMENSION_SMALL_LINE = 12.5       # pt per line at single spacing, 10pt
DIMENSION_CELL_MARGIN = 7         # pt of top + bottom cell margin, which
                                  # Word adds on top of the row height
DIMENSION_ROW_PAD = 9             # pt of margin and border in a row
# Wrapping is estimated on the short side of the true width, so a row is
# never given less height than its text takes.
DIMENSION_LABEL_CHARS = 17        # characters per line, EQ Dimension (bold)
DIMENSION_SCORE_CHARS = 22        # characters per line, Score / Theme
DIMENSION_MEANING_CHARS = 44      # characters per line, Leadership Meaning


def _lines(text, width):
    """Wrapped line count for text in a column ``width`` characters wide."""
    count = 0
    for part in str(text).split("\n"):
        count += max(1, -(-len(part) // width))
    return count


def _dimension_layout(entries):
    """How to fit the EQ Dimension table on its page.

    Returns None when the reference spacing already fits, else a dict:
    ``rows`` (a height in points for each strength row, in order), ``font``
    (the row text size, None to keep the template's), and ``header`` /
    ``closing`` (heights for the header and Development Areas rows, None
    to leave them as they are — only set when nothing else is enough).
    """
    strengths = [e for e in entries if e and e[0] != "Development Areas"]
    if not strengths:
        return None
    closing = next((e for e in entries if e and e[0] == "Development Areas"),
                   None)

    def closing_height(line, scale, pad):
        _label, scores, meaning = (list(closing) + ["", ""])[:3]
        return line * max(_lines(scores, int(DIMENSION_SCORE_CHARS * scale)),
                          _lines(meaning, int(DIMENSION_MEANING_CHARS * scale))
                          ) + pad

    def needed(line, scale):
        return [line * max(_lines(e[0], int(DIMENSION_LABEL_CHARS * scale)),
                           _lines(e[2] if len(e) > 2 else "",
                                  int(DIMENSION_MEANING_CHARS * scale)))
                + DIMENSION_ROW_PAD for e in strengths]

    # the Development Areas row is at least a reference row tall
    closing_now = (max(DIMENSION_ROW, closing_height(DIMENSION_LINE, 1.0, 20))
                   if closing else 0)
    room = DIMENSION_TABLE_BUDGET - DIMENSION_HEADER - closing_now
    if DIMENSION_ROW * len(strengths) <= room:
        return None

    layout = {"font": None, "header": None, "closing": None}
    heights = needed(DIMENSION_COMPACT_LINE, 1.0)
    if sum(heights) > room:
        layout["font"] = 10
        heights = needed(DIMENSION_SMALL_LINE, 1.2)
    if sum(heights) > room:
        # last resort: tighten the header and the Development Areas row too
        layout["header"] = DIMENSION_HEADER_TIGHT
        if closing:
            layout["closing"] = closing_height(DIMENSION_SMALL_LINE, 1.2,
                                               DIMENSION_ROW_PAD)
        room = (DIMENSION_TABLE_BUDGET - DIMENSION_HEADER_TIGHT
                - (layout["closing"] or 0))
    spare = max(0.0, room - sum(heights)) / len(strengths)
    layout["rows"] = [min(DIMENSION_ROW, h + spare) for h in heights]
    return layout


def _compact_text_row(row, height_pt, font_pt=None):
    """Drop a row's spare empty paragraphs, single-space its text and set
    its height, so a long table still fits its page. The text stays
    centred in the shorter row."""
    for cell in row.cells:
        for p in list(cell.paragraphs):
            if len(cell.paragraphs) > 1 and not p.text.strip() \
                    and p._p.find(qn("w:tbl")) is None:
                p._p.getparent().remove(p._p)
        for p in cell.paragraphs:
            fmt = p.paragraph_format
            fmt.space_before = Pt(0)
            fmt.space_after = Pt(0)
            fmt.line_spacing = 1.0
            if font_pt:
                for run in p.runs:
                    run.font.size = Pt(font_pt)
    trPr = row._tr.get_or_add_trPr()
    height = trPr.find(qn("w:trHeight"))
    if height is None:
        height = docx.oxml.OxmlElement("w:trHeight")
        trPr.append(height)
    inner = max(1.0, height_pt - DIMENSION_CELL_MARGIN)
    height.set(qn("w:val"), str(int(inner * 20)))


def _rebuild_text_table(table, entries, header_label=None):
    """Rebuild a plain text table, cloning a body row per entry.

    Used for the EQ Dimension table, whose row count follows the number
    of strengths. The header row is kept as-is and the first body row is
    the prototype; a closing "Development Areas" row uses the template's
    last row, which is laid out for a long score list. Strength rows are
    compacted when the table would otherwise run off its page.
    """
    rows = table._tbl.tr_lst
    if len(rows) < 2:
        return
    header = copy.deepcopy(rows[0])
    prototype = copy.deepcopy(rows[1])
    closing = copy.deepcopy(rows[-1]) if len(rows) > 2 else prototype

    for tr in list(table._tbl.tr_lst):
        table._tbl.remove(tr)
    table._tbl.append(header)
    if header_label is not None:
        # the rebuild marker lives in the header's first cell
        _set_paragraph_text(_content_paragraph(table.rows[0].cells[0]),
                            header_label)

    if not entries:
        table._tbl.getparent().remove(table._tbl)
        return

    layout = _dimension_layout(entries) or {}
    heights = iter(layout.get("rows") or [])
    if layout.get("header"):
        _compact_text_row(table.rows[0], layout["header"])
    for cells in entries:
        is_closing = bool(cells) and cells[0] == "Development Areas"
        tr = copy.deepcopy(closing if is_closing else prototype)
        table._tbl.append(tr)
        row = table.rows[-1]
        for i, text in enumerate(cells):
            if i < len(row.cells):
                _set_paragraph_text(_content_paragraph(row.cells[i]), str(text))
        if is_closing and layout.get("closing"):
            _compact_text_row(row, layout["closing"], layout["font"])
        elif layout and not is_closing:
            _compact_text_row(row, next(heights), layout["font"])


# ─────────────────────────────────────────
# Pagination
# ─────────────────────────────────────────
# The reference booklet opens several pages by letting a run of empty
# paragraphs push the next page's content down: the two EQ-i chart pages,
# the Flywheel grid, "Make the Signature Observable" and the 30-60-90 grid.
# That only lands right for the reference participant's chart lengths and
# wording. A longer chart pushes the run onto a page of its own, a shorter
# one pulls the next page's content up beside it, and every later page
# moves, so the table of contents points at the wrong pages. Each of those
# pages gets a real page break at the element that opens it in the
# reference, and the spacer run in front of it is dropped. With the charts
# and the EQ Dimension table sized to their pages, every booklet has the
# reference's page count and its contents page is right as stored.
_KEEP_TAGS = ("w:drawing", "w:pict", "w:br", "w:sectPr", "w:bookmarkStart",
              "w:fldChar")


def _text_of(el):
    return " ".join("".join(t.text or "" for t in el.iter(qn("w:t"))).split())


def _is_spacer(el):
    """An empty body paragraph whose only job is to take up space."""
    if el is None or el.tag != qn("w:p") or _text_of(el):
        return False
    return not any(True for tag in _KEEP_TAGS for _ in el.iter(qn(tag)))


def _is_break_only(el):
    """An empty body paragraph holding nothing but a page break. When the
    page above is full it lands on the next page and breaks again, leaving
    a blank page; a page break before the next page's content cannot."""
    if el is None or el.tag != qn("w:p") or _text_of(el):
        return False
    breaks = [br for br in el.iter(qn("w:br")) if br.get(qn("w:type")) == "page"]
    others = [tag for tag in _KEEP_TAGS if tag != "w:br"
              for _ in el.iter(qn(tag))]
    return bool(breaks) and not others


def _drop_spacers_before(el, keep=0):
    """Remove the run of spacer paragraphs directly above ``el``, except
    the last ``keep`` (which the reference shows at the top of the page).
    A page-break paragraph in the run goes too, since the caller puts a
    page break at ``el``. Returns the kept paragraphs in document order."""
    run = []
    prev = el.getprevious()
    while _is_spacer(prev) or _is_break_only(prev):
        run.append(prev)
        prev = prev.getprevious()
    run.reverse()
    # only true spacers after the last break can open the new page
    tail = len(run)
    while tail and not _is_break_only(run[tail - 1]):
        tail -= 1
    keep = min(keep, len(run) - tail)
    kept = run[len(run) - keep:] if keep else []
    for p in run[:len(run) - len(kept)]:
        p.getparent().remove(p)
    return kept


def _open_page(document, el, keep=0):
    """Start a page at ``el`` (or at the ``keep`` spacers above it)."""
    kept = _drop_spacers_before(el, keep)
    first = kept[0] if kept else el
    Paragraph(first, document).paragraph_format.page_break_before = True


def _open_page_at_table(document, table_el):
    """Start a page with a floating table.

    A floating table sits at a fixed offset on the page where it falls in
    the text, so the break goes on an empty paragraph just in front of it;
    the table's own position on that page is unchanged.
    """
    opener = docx.oxml.OxmlElement("w:p")
    table_el.addprevious(opener)
    _drop_spacers_before(opener)
    Paragraph(opener, document).paragraph_format.page_break_before = True


def _body_paragraph(document, prefix):
    prefix = prefix.upper()
    for el in document.element.body.iterchildren(qn("w:p")):
        if _text_of(el).upper().startswith(prefix):
            return el
    return None


def _body_table(document, *headers):
    for el in document.element.body.iterchildren(qn("w:tbl")):
        table = Table(el, document)
        if not table.rows:
            continue
        cells = [" ".join(c.text.split()) for c in table.rows[0].cells]
        if all(h in cells for h in headers):
            return el
    return None


def _paginate(document, has_strengths, has_development):
    """Give each flow-dependent page of the reference a real page break."""
    # Page 17: the note on the two kinds of score and the capacities chart.
    # With no strengths chart, page 16 has room for the note, and the
    # capacities chart opens page 17 instead.
    note = _body_paragraph(document, "Two types of scores appear on this page")
    caption = _body_paragraph(document, "LEADERSHIP CAPACITIES TO STRENGTHEN")
    if note is not None:
        if not has_strengths and has_development and caption is not None:
            _drop_spacers_before(note)
            _open_page(document, caption, keep=1)
        else:
            _open_page(document, note)

    # Page 18: the derailment risk explanation, two blank lines down
    derailment = _body_paragraph(document, "If scores appear in this section")
    if derailment is not None:
        _open_page(document, derailment, keep=2)

    # Page 19: the EQ Dimension table (in place of a page-break paragraph)
    dimensions = _body_table(document, "EQ Dimension", "Leadership Meaning")
    if dimensions is not None:
        _open_page_at_table(document, dimensions)

    # Page 25: the Flywheel quadrant grid
    grid = _body_table(document, "Flywheel Quadrant", "Development Stretch")
    if grid is not None:
        _open_page_at_table(document, grid)

    # Page 26: derailment indicators and the Flywheel, two blank lines down
    # (in place of the reference's page-break paragraph, which a full page
    # 25 pushed onto a blank page of its own)
    flywheel_risk = _body_paragraph(
        document, "When leadership derailment indicators are present")
    if flywheel_risk is not None:
        _open_page(document, flywheel_risk, keep=2)

    # Page 32: "Make the Signature Observable", three blank lines down
    observable = _body_paragraph(document, "Make the Signature Observable")
    if observable is not None:
        _open_page(document, observable, keep=3)

    # Page 38: the 30-60-90 grid
    plan = _body_table(document, "Timeline", "Primary Coaching Focus")
    if plan is not None:
        _open_page_at_table(document, plan)


# ─────────────────────────────────────────
# Public API
# ─────────────────────────────────────────
def render_from_template(values: dict, eqi_scores: dict,
                         dimension_rows=None, images: dict = None,
                         template_path: Path = TEMPLATE_PATH) -> bytes:
    """Fill the reference booklet with one participant's content.

    ``values`` maps token names (without braces) to replacement text and
    ``eqi_scores`` drives the two bar charts. ``dimension_rows`` supplies
    the EQ Dimension table as (label, score, meaning) triples, since its
    row count follows the number of strengths. ``images`` maps picture
    markers to PNG bytes ({"RADAR": ...}, from build_template_images). Tokens
    with no value are left blank rather than printed, so a partial profile
    degrades quietly.
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
            development = marker == "{{TABLE:EQ_DEVELOPMENT_BARS}}"
            if not development:
                fit = STRENGTH_BARS_FIT
            elif strengths:
                fit = DEVELOPMENT_BARS_FIT
            else:
                fit = DEVELOPMENT_BARS_FIT_ALONE
            _rebuild_bar_table(table, rows, caption_prefix=caption,
                               recolour_above_average=development, fit=fit)
        elif marker in text_tables:
            rows, label = text_tables[marker]
            _rebuild_text_table(table, rows, header_label=label)

    _fill_derailers(document, eqi_scores)
    _swap_images(document, images, values.get("NAME", ""))

    # then every remaining token
    blanked = []
    for paragraph in _iter_paragraphs(document):
        token = _paragraph_token(paragraph)
        if token:
            key = token.strip("{}")
            _set_paragraph_text(paragraph, values.get(key, ""))
            if not values.get(key):
                blanked.append(paragraph._p)
            continue
        # tokens embedded mid-sentence
        full = "".join(r.text or "" for r in paragraph.runs)
        if "{{" not in full:
            continue
        replaced = _TOKEN_RE.sub(
            lambda m: values.get(m.group(0).strip("{}"), ""), full)
        if replaced != full:
            _set_paragraph_text(paragraph, replaced)

    _drop_blanked(document, blanked)
    _paginate(document, has_strengths=bool(strengths),
              has_development=bool(development))

    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def _drop_blanked(document, blanked):
    """Remove body paragraphs that a blank token left empty.

    The template's own empty paragraphs are layout (spacers, the cover's
    page break, divider pages) and stay as they are; only a paragraph that
    held a token with no value for this person — e.g. the EQ-i sentences
    for someone without an EQ-i report — is taken out, so it cannot leave
    a gap. Paragraphs inside table cells stay, since a cell needs one.
    """
    body = document.element.body
    for p in blanked:
        if p.getparent() is body and not "".join(
                t.text or "" for t in p.iter(qn("w:t"))).strip():
            body.remove(p)
