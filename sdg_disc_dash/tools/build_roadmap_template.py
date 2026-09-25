"""
Author the Leadership Roadmap template from a finished booklet
===============================================================
Takes a hand-finished Roadmap .docx and writes utils/roadmap_template.docx:
the same document with one participant's variable content swapped for
{{TOKENS}}. Everything else — every banner, divider page, table style,
gridline, colour, margin and blank line — is carried through untouched, so
generated booklets match the reference apart from the person's own data.

The reference is Courtney Stanford's V4 booklet ("Color change and
alignment"), approved as the base for every Roadmap. It is already laid out
as the final document, so this only tokenizes; earlier references (Ariel
REV12) also needed restyling and re-paginating here, which lives in git
history.

Run this again whenever the reference booklet is revised:

    python tools/build_roadmap_template.py "path/to/NEW_REFERENCE.docx"

The runtime filler is utils/roadmap_template.py and the token values come
from build_template_values() in utils/roadmap_generator.py; the token names
below are the contract between the three.
"""
import re
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE = Path(r"C:\Users\Gaming pc\OneDrive\Documents"
                      r"\Courtney_Stanford_Leadership_Roadmap V4 Color "
                      r"change and alignment .docx")
TEMPLATE_OUT = ROOT / "utils" / "roadmap_template.docx"

_TOKEN_ONLY = re.compile(r"\{\{[A-Z0-9_:]+\}\}")

# ── Whole-paragraph swaps: the reference sentence -> token ────────────────
# Matched on the start of the paragraph's text with whitespace collapsed.
PARAGRAPH_TOKENS = [
    ("DISC identifies you as a",                           "{{SUMMARY_DISC}}"),
    ("Your EQ-i profile adds an important leadership",     "{{SUMMARY_EQI}}"),
    ("The EQ development opportunity is to make that",     "{{SUMMARY_EQI_DEV}}"),
    ("The Flywheel analysis identifies",                   "{{SUMMARY_FLYWHEEL}}"),
    ("DISC helps you understand how your leadership",      "{{DISC_CONNECTION_INTRO}}"),
    ("Your development focus is to",                       "{{ROADMAP_FOCUS}}"),
    ("The development risk is that",                       "{{FLY_DEV_RISK}}"),
    ("You demonstrate visible, repeatable",                "{{SUCCESS_INDICATOR}}"),
    ("Your profile shows exceptional",                     "{{EQI_STRENGTH_NOTE}}"),
    ("The opportunity is to translate your internal",      "{{EQI_DEV_NOTE}}"),
    ("EQ-i helps you translate strong internal",           "{{EQI_INTRO}}"),
    ("For you, the Flywheel highlights",                   "{{FLYWHEEL_FOR_YOU}}"),
]

# ── Exact-string swaps anywhere in the body (tables included) ─────────────
# Longest first so overlapping values can't clip each other.
VALUE_TOKENS = [
    ("Courtney Stanford",                    "{{NAME}}"),
    # The client the reference booklet was written for; the dashboard's
    # Organization field fills the token.
    ("Northwest Florida Health Network",     "{{ORG}}"),
    ("D 0.69 | I -8.00 | S 2.14 | C 7.08",   "{{MIRROR_LINE}}"),
    ("D 2.55 | I -6.98 | S 8.00 | C 3.71",   "{{STRESS_LINE}}"),
    ("CSD Contemplator",                     "{{STYLE_LABEL}}"),
]

# ── Score strips: the D/I/S/C value cells, in document order ──────────────
STRIP_TOKENS = [
    ["{{M_D}}", "{{M_I}}", "{{M_S}}", "{{M_C}}"],
    ["{{P_D}}", "{{P_I}}", "{{P_S}}", "{{P_C}}"],
]

# ── Bar charts rebuilt at fill time, found by the caption above them ──────
# The marker goes in the chart's first cell; the filler clears it.
CAPTION_MARKERS = {
    "CORE EQ-I STRENGTHS":                 "{{TABLE:EQ_STRENGTH_BARS}}",
    "LEADERSHIP CAPACITIES TO STRENGTHEN": "{{TABLE:EQ_DEVELOPMENT_BARS}}",
}
HEADER_MARKERS = {"EQ Dimension": "{{TABLE:EQ_DIMENSIONS}}"}
# Fixed four rows (Impulse Control, Stress Tolerance, Problem Solving,
# Independence); the filler finds it by this caption and fills the scores.
DERAILERS_CAPTION = "LEADERSHIP DERAILMENT RISK"

# ── Cell-level tokens: {table id: {(row, col): token or (tokens...)}} ─────
# A table is identified by its header cells. A single token replaces the
# cell's LAST non-empty paragraph (a bold heading above it survives); a
# tuple fills the cell's non-empty paragraphs in order.
CELL_TOKENS = {
    ("Roadmap Element",): {
        (1, 1): "{{SNAP_PATTERN}}",      (1, 2): "{{SNAP_PATTERN_MEANS}}",
                                         (2, 2): "{{SNAP_MIRROR_MEANS}}",
                                         (3, 2): "{{SNAP_STRESS_MEANS}}",
        (4, 1): "{{SNAP_EQ_STRENGTHS}}", (4, 2): "{{SNAP_EQ_STRENGTHS_MEANS}}",
        (5, 1): "{{SNAP_EQ_DEV}}",       (5, 2): "{{SNAP_EQ_DEV_MEANS}}",
        (6, 1): "{{SNAP_FLY_PRIMARY}}",  (6, 2): "{{SNAP_FLY_PRIMARY_MEANS}}",
        (7, 1): "{{SNAP_FLY_SECOND}}",   (7, 2): "{{SNAP_FLY_SECOND_MEANS}}",
        (8, 1): "{{SNAP_RISK}}",         (8, 2): "{{SNAP_RISK_MEANS}}",
        (9, 1): "{{SNAP_COACHING}}",     (9, 2): "{{SNAP_COACHING_MEANS}}",
    },
    ("Profile Element",): {
        (3, 1): "{{DISC_NATURAL_STRENGTHS}}",
        (4, 1): "{{DISC_STAFF_EXPERIENCE}}",
        (5, 1): "{{DISC_BLIND_SPOT}}",
        (6, 1): "{{DISC_PRESSURE_RISK}}",
        (7, 1): "{{DISC_COACHING_ADJUSTMENT}}",
    },
    ("Flywheel Quadrant", "Current Strength"): {
        (r, c): tok
        for r, stem in enumerate(("DIRECTION", "CULTURE", "LEARNING",
                                  "EXECUTION"), start=1)
        for c, tok in ((1, f"{{{{FLY_{stem}_NOW}}}}"),
                       (2, (f"{{{{FLY_{stem}_FROM}}}}",
                            f"{{{{FLY_{stem}_NEXT}}}}")))
    },
    ("Signature Element",): {
        (r, c): "{{SIG_%s_%d}}" % ("EL" if c == 0 else "OBS", r)
        for r in range(1, 6) for c in (0, 1)
    },
    ("Timeline",): {
        (1, 1): "{{PLAN_30_FOCUS}}", (1, 2): "{{PLAN_30_DISC}}",
        (1, 3): "{{PLAN_30_EQ}}",    (1, 4): "{{PLAN_30_FLY}}",
        (2, 1): "{{PLAN_60_FOCUS}}", (2, 2): "{{PLAN_60_DISC}}",
        (2, 3): "{{PLAN_60_EQ}}",    (2, 4): "{{PLAN_60_FLY}}",
        (3, 1): "{{PLAN_90_FOCUS}}", (3, 2): "{{PLAN_90_DISC}}",
        (3, 3): "{{PLAN_90_EQ}}",    (3, 4): "{{PLAN_90_FLY}}",
    },
}

# ── Single-cell boxes: a bold heading over body paragraphs. Matched on the
# heading; the tokens fill the last paragraphs after it, in order.
BOX_TOKENS = [
    ("Primary Strength",          ("{{FLY_PRIMARY_LABEL}}", "{{FLY_PRIMARY_BOX}}")),
    ("Secondary Strength",        ("{{FLY_SECONDARY_LABEL}}", "{{FLY_SECONDARY_BOX}}")),
    ("Coaching Goal",             ("{{FLY_COACHING_GOAL}}",)),
    ("Your Leadership Signature", ("{{LEADERSHIP_SIGNATURE}}",)),
    ("Leadership Signature",      ("{{LEADERSHIP_SIGNATURE}}",)),
]

# The radar on the RADAR SHIFTS page: found by the picture's name, marked
# in its alt text for the filler, and its image swapped for an empty grid.
RADAR_PICTURE_NAME = "Radar Shifts"
RADAR_MARKER = "{{IMAGE:RADAR}}"


def blocks(document):
    out = []
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            out.append(Paragraph(child, document))
        elif child.tag == qn("w:tbl"):
            out.append(Table(child, document))
    return out


def _is_bold(run):
    """Bold directly, or through Word's "Strong" character style."""
    rpr = run._r.find(qn("w:rPr"))
    if rpr is None:
        return False
    b = rpr.find(qn("w:b"))
    style = rpr.find(qn("w:rStyle"))
    return ((b is not None and b.get(qn("w:val")) not in ("0", "false"))
            or (style is not None and style.get(qn("w:val")) == "Strong"))


def set_text(paragraph, text):
    """Replace a paragraph's text, keeping its body formatting.

    The formatting comes from the run carrying the most text, not the first
    run: summary paragraphs open with a bold lead word ("DISC identifies
    you..."), and taking that run's formatting would print the whole
    generated paragraph in bold. That lead word's own formatting (the
    booklet sets "DISC" a size up) is kept too, as an empty run after the
    text; the filler uses it for the generated text's **bold** words.
    """
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(text)
        return
    keep = max(runs, key=lambda r: len(r.text or ""))
    bold = (next((r for r in runs if _is_bold(r) and (r.text or "").strip()),
                 None) if not _is_bold(keep) else None)
    keep.text = text
    for run in runs:
        if run._r is not keep._r and (bold is None or run._r is not bold._r):
            run._r.getparent().remove(run._r)
    if bold is not None:
        bold.text = ""
        keep._r.addnext(bold._r)


def replace_in_paragraph(paragraph, old, new):
    """Swap a substring that may be split across runs."""
    full = "".join(r.text or "" for r in paragraph.runs)
    if old not in full:
        return False
    set_text(paragraph, full.replace(old, new))
    return True


def iter_paragraphs(document):
    for block in blocks(document):
        if isinstance(block, Paragraph):
            yield block
        else:
            yield from _table_paragraphs(block)


def _table_paragraphs(table):
    for row in table.rows:
        for cell in row.cells:
            yield from cell.paragraphs
            for nested in cell.tables:
                yield from _table_paragraphs(nested)


def _norm(text):
    return " ".join((text or "").split())


def _table_after_caption(document, caption):
    """The first table following the paragraph whose text is `caption`."""
    found = False
    for block in blocks(document):
        if isinstance(block, Paragraph) and _norm(block.text).upper() == caption:
            found = True
        elif found and isinstance(block, Table):
            return block
    return None


def mark_table(table, marker):
    """Put the rebuild marker in the first cell's styled paragraph."""
    cell = table.rows[0].cells[0]
    target = next((p for p in cell.paragraphs if p.text.strip()),
                  cell.paragraphs[0])
    for extra in cell.paragraphs:
        if extra._p is not target._p:
            extra._p.getparent().remove(extra._p)
    set_text(target, marker)


def scrub_template(document):
    """Strip the reference participant's data out of the template.

    Rebuilt tables keep only the row the filler clones, blanked; the
    derailers chart keeps its four labelled rows with the scores cleared.
    Without this the template would still carry the reference person's
    real EQ-i scores. Document properties are cleared for the same reason.
    """
    scrubbed = 0
    for block in blocks(document):
        if not isinstance(block, Table) or not block.rows:
            continue
        marker = _norm(block.rows[0].cells[0].text)
        if marker not in (*CAPTION_MARKERS.values(), *HEADER_MARKERS.values()):
            continue
        keep = 2 if marker in HEADER_MARKERS.values() else 1
        rows = block._tbl.tr_lst
        # the EQ Dimension table's closing "Development Areas" row is laid
        # out differently (left-aligned score list), so it is kept as a
        # second prototype
        spare = rows[-1] if keep == 2 and len(rows) > keep else None
        for tr in list(rows[keep:]):
            if tr is not spare:
                block._tbl.remove(tr)
                scrubbed += 1
        for row_index, row in enumerate(list(block.rows)[keep - 1:],
                                        start=keep - 1):
            for index, cell in enumerate(row.cells):
                if index == 0 and row_index == 0:
                    continue          # the marker the filler looks for
                # The closing Development Areas row holds one paragraph per
                # cell: the reference splits its score list over several
                # (broken by hand), which would come back as blank lines.
                # Other rows keep their empty paragraphs; they are spacing.
                paras = cell.paragraphs
                first = next((q for q in paras if q.text.strip()), None)
                if (keep == 2 and row._tr is spare and first is not None
                        and cell._tc.find(qn("w:tbl")) is None):
                    for q in paras:
                        if q._p is not first._p:
                            q._p.getparent().remove(q._p)
                scrubbed += _placehold(cell)

    derailers = _table_after_caption(document, DERAILERS_CAPTION)
    for row in derailers.rows:
        scrubbed += _placehold(row.cells[-1])

    props = document.core_properties
    props.author = "SDG Leadership Roadmap generator"
    props.last_modified_by = ""
    props.title = "Leadership Roadmap template"
    props.subject = ""
    props.comments = ""
    props.category = ""
    props.keywords = ""
    return scrubbed


def _placehold(cell):
    """Swap a prototype cell's value for {{CELL}}.

    The placeholder sits in the paragraph that held the value, so the
    filler writes there rather than into one of the empty spacing
    paragraphs around it (which is what shifted values off-centre when the
    cell was simply blanked). Any row left unfilled prints it blank.
    """
    done = 0
    for p in cell.paragraphs:
        if p.text.strip():
            set_text(p, "" if done else "{{CELL}}")
            done += 1
    return done


def mark_radar(document):
    """Mark the RADAR SHIFTS picture and blank it to an empty radar grid."""
    from utils.radar_image import graph_shift_png

    for doc_pr in document.element.body.iter(
            "{http://schemas.openxmlformats.org/drawingml/2006/"
            "wordprocessingDrawing}docPr"):
        if doc_pr.get("name") != RADAR_PICTURE_NAME:
            continue
        doc_pr.set("descr", RADAR_MARKER)
        inline = doc_pr.getparent()
        blip = next(inline.iter(
            "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"))
        rid = blip.get(qn("r:embed"))
        part = document.part.related_parts[rid]
        part._blob = graph_shift_png({"graphs": {}})
        return True
    return False


def refresh_fields_on_open(document):
    """Have Word refresh the table of contents when a booklet is opened.

    Its page numbers are PAGEREF fields holding the reference booklet's
    pages; a person with more or fewer EQ-i rows can shift the later
    sections, so Word recomputes them on open (it asks first).
    """
    settings = document.settings.element
    if settings.find(qn("w:updateFields")) is not None:
        return False
    el = docx.oxml.OxmlElement("w:updateFields")
    el.set(qn("w:val"), "true")
    # schema order: updateFields precedes these
    for tag in ("w:hdrShapeDefaults", "w:footnotePr", "w:endnotePr",
                "w:compat", "w:docVars", "w:rsids"):
        follower = settings.find(qn(tag))
        if follower is not None:
            follower.addprevious(el)
            return True
    settings.append(el)
    return True


def main(source: Path):
    document = docx.Document(str(source))
    counts = {}

    # 1. whole-paragraph prose
    for prefix, token in PARAGRAPH_TOKENS:
        hit = 0
        for p in iter_paragraphs(document):
            if _norm(p.text).startswith(prefix):
                set_text(p, token)
                hit += 1
        counts[token] = hit

    # 2. exact values, anywhere
    for old, token in VALUE_TOKENS:
        hit = sum(replace_in_paragraph(p, old, token)
                  for p in iter_paragraphs(document))
        counts[token] = counts.get(token, 0) + hit

    # 3. the two DISC score strips, in document order
    strips = [b for b in blocks(document)
              if isinstance(b, Table) and len(b.rows) == 2
              and [c.text.strip() for c in b.rows[0].cells][:4] == list("DISC")]
    for strip, tokens in zip(strips, STRIP_TOKENS):
        for cell, token in zip(strip.rows[1].cells, tokens):
            set_text(cell.paragraphs[0], token)
        counts["(score strips)"] = counts.get("(score strips)", 0) + 1

    # 4. heading-over-body boxes (one-row tables; every cell considered, as
    # the Flywheel strength pair sits in one row of two cells)
    for heading, tokens in BOX_TOKENS:
        for block in blocks(document):
            if not isinstance(block, Table) or len(block.rows) != 1:
                continue
            for cell in block.rows[0].cells:
                paras = [p for p in cell.paragraphs if p.text.strip()]
                if (len(paras) < 1 + len(tokens)
                        or not paras[0].text.strip().startswith(heading)
                        or _TOKEN_ONLY.fullmatch(_norm(paras[-1].text))):
                    continue
                for p, token in zip(paras[-len(tokens):], tokens):
                    set_text(p, token)
                    counts[token] = counts.get(token, 0) + 1

    # 5. fixed-shape tables: only the value cells carry tokens
    for keys, cells in CELL_TOKENS.items():
        for block in blocks(document):
            if not isinstance(block, Table) or not block.rows:
                continue
            header = [_norm(c.text) for c in block.rows[0].cells]
            if not all(k in header for k in keys):
                continue
            for (r, c), token in cells.items():
                if r >= len(block.rows) or c >= len(block.columns):
                    continue
                cell = block.cell(r, c)
                if _TOKEN_ONLY.fullmatch(_norm(cell.text)):
                    continue            # a value token already covers it
                paras = [p for p in cell.paragraphs if p.text.strip()]
                if isinstance(token, tuple):
                    assert len(paras) == len(token), (token, cell.text)
                    for p, t in zip(paras, token):
                        set_text(p, t)
                        counts[t] = counts.get(t, 0) + 1
                else:
                    set_text(paras[-1] if paras else cell.paragraphs[0], token)
                    counts[token] = counts.get(token, 0) + 1
            break

    # 6. mark the tables the filler rebuilds row-by-row
    for caption, marker in CAPTION_MARKERS.items():
        table = _table_after_caption(document, caption)
        if table is not None:
            mark_table(table, marker)
        counts[marker] = int(table is not None)
    for block in blocks(document):
        if isinstance(block, Table) and block.rows:
            marker = HEADER_MARKERS.get(_norm(block.rows[0].cells[0].text))
            if marker:
                mark_table(block, marker)
                counts[marker] = counts.get(marker, 0) + 1
    counts["(derailers chart)"] = int(
        _table_after_caption(document, DERAILERS_CAPTION) is not None)

    # 7. the radar picture, table-of-contents refresh, and scrubbing
    counts[RADAR_MARKER] = int(mark_radar(document))
    counts["(TOC refreshes on open)"] = int(refresh_fields_on_open(document))
    counts["(reference data scrubbed)"] = scrub_template(document)

    TEMPLATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(TEMPLATE_OUT))

    print(f"source   : {source}")
    print(f"template : {TEMPLATE_OUT}")
    print("\ntokens placed:")
    for token, n in counts.items():
        flag = "" if n else "   <-- NOT FOUND"
        print(f"  {n:>2}x  {token}{flag}")
    missing = [t for t, n in counts.items() if not n]
    if missing:
        print(f"\n{len(missing)} token(s) unmatched — the reference booklet "
              f"may have been reworded.")
    return 0 if not missing else 1


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    sys.exit(main(src))
