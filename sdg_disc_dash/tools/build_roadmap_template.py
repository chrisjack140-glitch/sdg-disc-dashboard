"""
Author the Leadership Roadmap template from a finished booklet
===============================================================
Takes a hand-finished Roadmap .docx (the current reference is REV12) and
writes utils/roadmap_template.docx, which is the same document with one
participant's variable content swapped for {{TOKENS}}. Everything else —
every banner, table style, gridline, margin and blank line — is carried
through untouched, so generated booklets are byte-identical to the
reference apart from the person's own data.

Run this again whenever the reference booklet is revised:

    python tools/build_roadmap_template.py "path/to/NEW_REFERENCE.docx"

The runtime filler is utils/roadmap_template.py; the token names below are
the contract between the two.
"""
import copy
import re
import shutil
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path(r"C:\Users\Gaming pc\Downloads\Ariel REV12.docx")
TEMPLATE_OUT = ROOT / "utils" / "roadmap_template.docx"

_TOKEN_ONLY = re.compile(r"\{\{[A-Z0-9_:]+\}\}")

# ── Whole-paragraph swaps: the reference sentence -> token ────────────────
# Matched on the paragraph's full text with whitespace collapsed, so a
# reference edit that only changes spacing still matches.
PARAGRAPH_TOKENS = [
    ("DISC identifies you as a CS Precisionist",        "{{SUMMARY_DISC}}"),
    ("Your EQ-i profile adds an important leadership",  "{{SUMMARY_EQI}}"),
    ("The EQ development opportunity is to make that",  "{{SUMMARY_EQI_DEV}}"),
    ("The Flywheel analysis identifies Execution",      "{{SUMMARY_FLYWHEEL}}"),
    ("DISC helps you understand how staff experiences", "{{DISC_INTRO}}"),
    ("Flywheel alignment shows how your DISC and EQ-i", "{{FLYWHEEL_INTRO}}"),
    ("EQ-i helps you translate strong internal",        "{{EQI_INTRO}}"),
    ("Your profile shows exceptional",                  "{{EQI_STRENGTH_NOTE}}"),
    ("The growth opportunity is to express that",       "{{EQI_DEV_NOTE}}"),
]

# ── Corrections applied to the reference booklet on the way in ───────────
# Kept here rather than fixed by hand so they survive a re-import of a
# revised reference that still carries them.
TYPO_FIXES = [
    ("LEADDERSHIPROADMAP INTERGRATION SUMMARY",
     "LEADERSHIP ROADMAP INTEGRATION SUMMARY"),
    ("LEADDERSHIPROADMAP INTERGRATION",
     "LEADERSHIP ROADMAP INTEGRATION"),
    ("Behavior pattern and Pressure Shift",
     "Behavior Pattern and Pressure Shift"),
]

# ── Exact-string swaps anywhere in the body (tables included) ─────────────
# Longest first so overlapping values can't clip each other.
VALUE_TOKENS = [
    ("Ariel Alston",                       "{{NAME}}"),
    # The client the reference booklet was written for. Both spellings
    # appear in it; the dashboard's Organization field fills the token.
    ("Northwest Florida Health Network",   "{{ORG}}"),
    ("Southern Region",                    "{{ORG}}"),
    ("D -2.43 | I -4.38 | S 3.44 | C 7.08", "{{MIRROR_LINE}}"),
    ("D -2.56 | I -5.19 | S 8.00 | C 5.08", "{{STRESS_LINE}}"),
    ("CS – Precisionist",                  "{{STYLE_LABEL}}"),
    ("CS - Precisionist",                  "{{STYLE_LABEL}}"),
    ("CS Precisionist",                    "{{STYLE_LABEL}}"),
]

# ── Score strips: the D/I/S/C value cells, in document order ──────────────
STRIP_TOKENS = [
    ["{{M_D}}", "{{M_I}}", "{{M_S}}", "{{M_C}}"],
    ["{{P_D}}", "{{P_I}}", "{{P_S}}", "{{P_C}}"],
]

# ── Tables rebuilt row-by-row at runtime, marked so the filler finds them.
# The marker goes in the first cell; the filler clears it.
REBUILD_MARKERS = {
    "Emotional Self-Awareness": "{{TABLE:EQ_STRENGTH_BARS}}",
    "Emotional Expression":     "{{TABLE:EQ_DEVELOPMENT_BARS}}",
    "EQ Dimension":             "{{TABLE:EQ_DIMENSIONS}}",
}

# ── Cell-level tokens: {table id: {(row, col): token}} ────────────────────
# The row labels in these tables are fixed; only the value cells vary. A
# table is identified by its header cells so the spec survives the blocks
# moving around. Only the cell's LAST paragraph is replaced, which leaves
# a bold heading above the body untouched.
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
        (1, 1): "{{FLY_DIRECTION_NOW}}",  (1, 2): "{{FLY_DIRECTION_NEXT}}",
        (2, 1): "{{FLY_CULTURE_NOW}}",    (2, 2): "{{FLY_CULTURE_NEXT}}",
        (3, 1): "{{FLY_LEARNING_NOW}}",   (3, 2): "{{FLY_LEARNING_NEXT}}",
        (4, 1): "{{FLY_EXECUTION_NOW}}",  (4, 2): "{{FLY_EXECUTION_NEXT}}",
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

# ── Single-cell boxes: a bold heading over a body paragraph. Matched on
# the heading, and only the body is tokenized.
BOX_TOKENS = [
    ("Primary Strength",            "{{FLY_PRIMARY_BOX}}"),
    ("Secondary Strength",          "{{FLY_SECONDARY_BOX}}"),
    ("Coaching Goal",               "{{FLY_COACHING_GOAL}}"),
    ("Your Leadership Signature",   "{{LEADERSHIP_SIGNATURE}}"),
    ("Leadership Signature",        "{{LEADERSHIP_SIGNATURE}}"),
]


# Pagination in the reference booklet is done with long runs of empty
# paragraphs — up to 84 in a row — rather than page breaks. That is fine for
# a hand-finished document about one person, but it makes generated ones
# fragile: a chart with three more rows pushes every later page down, and a
# run that overshoots leaves a blank page behind. These thresholds convert
# that padding into explicit breaks so pagination no longer depends on how
# much any one person's content happens to fill.
PAGE_PAD_MIN = 10        # a run this long was padding to the next page
GAP_TRIM = {6: 2, 3: 1}  # shorter runs are in-page spacing; trim them down


def normalize_pagination(document):
    """Replace empty-paragraph padding with real page breaks."""
    body = document.element.body
    children = list(body.iterchildren())

    runs, current = [], []
    for child in children:
        if child.tag == qn("w:p") and not "".join(
                t.text or "" for t in child.iter(qn("w:t"))).strip():
            current.append(child)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)

    breaks = trimmed = 0
    for run in runs:
        n = len(run)
        if n >= PAGE_PAD_MIN:
            keep = run[0]
            _make_page_break(keep)
            for p in run[1:]:
                body.remove(p)
            breaks += 1
        else:
            keep_n = next((v for k, v in sorted(GAP_TRIM.items(), reverse=True)
                           if n >= k), n)
            for p in run[keep_n:]:
                body.remove(p)
            if keep_n != n:
                trimmed += 1
    return breaks, trimmed


def _make_page_break(paragraph_el):
    """Turn an empty paragraph into a hard page break."""
    for r in paragraph_el.findall(qn("w:r")):
        paragraph_el.remove(r)
    run = docx.oxml.OxmlElement("w:r")
    br = docx.oxml.OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    run.append(br)
    paragraph_el.append(run)


# ── Development-page copy ────────────────────────────────────────────────
# The section is framed as expanding leadership range rather than naming
# deficits, so the caption is renamed and carries the same gold as the
# strengths chart instead of the red that read as a warning.
DEV_CAPTION_OLD = "KEY DEVELOPMENT AREAS"
DEV_CAPTION_NEW = "LEADERSHIP CAPACITIES TO STRENGTHEN"
STRENGTH_GOLD = "B59824"

DEV_DISCLAIMER = (
    "Your EQ-i Profile highlights opportunities to expand your leadership "
    "range. These scores do not define your effectiveness; they identify "
    "where greater intention and practice can strengthen your response to "
    "different people, pressures and situations. Your existing leadership "
    "assets provide the foundation for that growth."
)
DEV_CLOSING = (
    "The Goal is not to lead from a score. The goal is to expand the "
    "leadership you can access when the moment requires it."
)


# The development bars and their score numbers, recoloured from the red
# that read as a warning. #1A5276 is the blue already used for the C factor
# in the DISC score strips, so it stays inside the document's palette.
DEV_BAR_OLD = "C0392B"
DEV_BAR_NEW = "1A5276"


def scrub_template(document):
    """Strip the reference participant's data out of the template.

    The rebuilt tables keep only the row the filler clones, and that row's
    values are blanked. Without this the template still carries the
    reference person's real EQ-i scores in rows that are never rendered —
    invisible in output, but present in a file that gets committed.
    Document properties are cleared for the same reason.
    """
    scrubbed = 0
    for block in blocks(document):
        if not isinstance(block, Table) or not block.rows:
            continue
        marker = " ".join(block.rows[0].cells[0].text.split())
        if marker not in ("{{TABLE:EQ_STRENGTH_BARS}}",
                          "{{TABLE:EQ_DEVELOPMENT_BARS}}",
                          "{{TABLE:EQ_DIMENSIONS}}"):
            continue
        keep = 2 if marker == "{{TABLE:EQ_DIMENSIONS}}" else 1
        for tr in list(block._tbl.tr_lst[keep:]):
            block._tbl.remove(tr)
            scrubbed += 1
        for row in block.rows[keep - 1:]:
            for index, cell in enumerate(row.cells):
                if index == 0 and row is block.rows[0]:
                    continue          # the marker the filler looks for
                for p in cell.paragraphs:
                    if p.text.strip():
                        set_text(p, "")
                        scrubbed += 1

    props = document.core_properties
    props.author = "SDG Leadership Roadmap generator"
    props.last_modified_by = ""
    props.title = "Leadership Roadmap template"
    props.subject = ""
    props.comments = ""
    props.category = ""
    props.keywords = ""
    return scrubbed


def recolour_development_bars(document):
    """Repaint the development chart's bars and score numbers.

    The chart is rebuilt at fill time by cloning its first row, so changing
    the prototype here changes every bar for every person.
    """
    changed = 0
    for block in blocks(document):
        if not isinstance(block, Table) or not block.rows:
            continue
        marker = " ".join(block.rows[0].cells[0].text.split())
        if marker != "{{TABLE:EQ_DEVELOPMENT_BARS}}":
            continue
        for shd in block._tbl.iter(qn("w:shd")):
            if (shd.get(qn("w:fill")) or "").upper() == DEV_BAR_OLD:
                shd.set(qn("w:fill"), DEV_BAR_NEW)
                changed += 1
        for col in block._tbl.iter(qn("w:color")):
            if (col.get(qn("w:val")) or "").upper() == DEV_BAR_OLD:
                col.set(qn("w:val"), DEV_BAR_NEW)
                changed += 1
    return changed


def pull_up_boxes(document, headings):
    """Drop the page break that strands a callout box on its own page.

    normalize_pagination turns long padding runs into breaks, which can
    leave a box like "Workshop Practice" alone overleaf from the section it
    belongs to. Removing the break lets it flow back up.
    """
    pulled = 0
    for block in blocks(document):
        if not isinstance(block, Table) or not block.rows:
            continue
        text = " ".join(block.rows[0].cells[0].text.split())
        if not any(text.startswith(h) for h in headings):
            continue
        node = block._tbl.getprevious()
        while node is not None and node.tag == qn("w:p"):
            body = "".join(t.text or "" for t in node.iter(qn("w:t"))).strip()
            is_break = node.find(".//" + qn("w:br")) is not None and \
                'w:type="page"' in node.xml
            if is_break:
                node.getparent().remove(node)
                pulled += 1
                break
            if body:
                break
            node = node.getprevious()
    return pulled


def restyle_development_page(document):
    """Rename the development caption, recolour it, and add its framing copy.

    The disclaimer goes at the top of the page — above the note that
    introduces the chart — and the closing line after the sentence that
    follows the bars, so both survive the chart growing or shrinking.
    """
    done = {"caption": 0, "disclaimer": 0, "closing": 0}

    for p in iter_paragraphs(document):
        text = " ".join(p.text.split())
        if text.upper() != DEV_CAPTION_OLD:
            continue
        # python-docx builds fresh Run wrappers on every `.runs` call, so
        # capture the list once — an identity test against p.runs[0] inside
        # the loop is never true and would blank the caption.
        runs = p.runs
        for index, run in enumerate(runs):
            run.text = DEV_CAPTION_NEW if index == 0 else ""
            rPr = run._r.find(qn("w:rPr"))
            if rPr is None:
                rPr = docx.oxml.OxmlElement("w:rPr")
                run._r.insert(0, rPr)
            for col in rPr.findall(qn("w:color")):
                rPr.remove(col)
            col = docx.oxml.OxmlElement("w:color")
            col.set(qn("w:val"), STRENGTH_GOLD)
            rPr.append(col)
        done["caption"] += 1

        # disclaimer: first paragraph on the page, i.e. above the note that
        # carries the page break
        node = p._p.getprevious()
        anchor = p._p
        while node is not None and node.tag == qn("w:p"):
            body_text = "".join(t.text or "" for t in node.iter(qn("w:t"))).strip()
            if body_text:
                anchor = node
                if node.find(qn("w:pPr")) is not None and node.find(
                        qn("w:pPr")).find(qn("w:pageBreakBefore")) is not None:
                    break
            node = node.getprevious()
        new_p = copy.deepcopy(anchor)
        _set_plain_text(new_p, DEV_DISCLAIMER)
        anchor.addprevious(new_p)
        done["disclaimer"] += 1
        break

    # closing line goes after the sentence that follows the bars
    for p in iter_paragraphs(document):
        if " ".join(p.text.split()).startswith("{{EQI_INTRO}}"):
            new_p = copy.deepcopy(p._p)
            _set_plain_text(new_p, DEV_CLOSING)
            p._p.addnext(new_p)
            done["closing"] += 1
            break
    return done


def _set_plain_text(paragraph_el, text):
    runs = paragraph_el.findall(qn("w:r"))
    for extra in runs[1:]:
        paragraph_el.remove(extra)
    if not runs:
        return
    for t in runs[0].findall(qn("w:t")):
        runs[0].remove(t)
    t = docx.oxml.OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    runs[0].append(t)


def _move_break(source_el, target_el):
    """Carry a pageBreakBefore from one paragraph to another."""
    src = source_el.find(qn("w:pPr"))
    if src is None or src.find(qn("w:pageBreakBefore")) is None:
        return
    src.remove(src.find(qn("w:pageBreakBefore")))
    dst = target_el.find(qn("w:pPr"))
    if dst is None:
        dst = docx.oxml.OxmlElement("w:pPr")
        target_el.insert(0, dst)
    if dst.find(qn("w:pageBreakBefore")) is None:
        dst.insert(0, docx.oxml.OxmlElement("w:pageBreakBefore"))


def tighten_cover(document):
    """Keep the cover on one page.

    Its panel row is sized to fill the page, and a trailing empty row plus
    the paragraph after it tip the table just past the bottom margin —
    which is where the blank page 2 came from. Dropping the spare row and
    making the following paragraph a hard break pins the cover to page 1.
    """
    first = next(iter(document.element.body.iterchildren()), None)
    if first is None or first.tag != qn("w:tbl"):
        return False
    table = Table(first, document)

    changed = False
    for tr in list(table._tbl.tr_lst[1:]):
        if not "".join(t.text or "" for t in tr.iter(qn("w:t"))).strip():
            table._tbl.remove(tr)
            changed = True

    nxt = first.getnext()
    if nxt is not None and nxt.tag == qn("w:p"):
        _make_page_break(nxt)
        changed = True
    return changed


def force_break_before(document, text_prefix):
    """Start the paragraph beginning with `text_prefix` on a new page."""
    for p in iter_paragraphs(document):
        if " ".join(p.text.split()).upper().startswith(text_prefix.upper()):
            pPr = p._p.get_or_add_pPr()
            if pPr.find(qn("w:pageBreakBefore")) is None:
                pPr.insert(0, docx.oxml.OxmlElement("w:pageBreakBefore"))
            return True
    return False


def blocks(document):
    out = []
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            out.append(Paragraph(child, document))
        elif child.tag == qn("w:tbl"):
            out.append(Table(child, document))
    return out


def set_text(paragraph, text):
    """Replace a paragraph's text, keeping the first run's formatting."""
    runs = paragraph.runs
    if not runs:
        paragraph.add_run(text)
        return
    runs[0].text = text
    for run in runs[1:]:
        run._r.getparent().remove(run._r)


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
            for row in block.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        yield p
                    for nested in cell.tables:
                        for r in nested.rows:
                            for c in r.cells:
                                yield from c.paragraphs


def main(source: Path):
    document = docx.Document(str(source))
    counts = {}

    # 0. corrections to the reference booklet's own copy
    fixed = 0
    for wrong, right in TYPO_FIXES:
        fixed += sum(replace_in_paragraph(p, wrong, right)
                     for p in iter_paragraphs(document))
    counts["(typo fixes)"] = fixed

    # 1. whole-paragraph prose
    for prefix, token in PARAGRAPH_TOKENS:
        hit = 0
        for p in iter_paragraphs(document):
            if " ".join(p.text.split()).startswith(prefix):
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
        counts["strip"] = counts.get("strip", 0) + 1

    # 4. heading-over-body boxes — tokenize the body, keep the bold heading.
    # These are single-row tables; the Flywheel strengths pair sits in one
    # row of two cells, so every cell of a one-row table is considered.
    for heading, token in BOX_TOKENS:
        if counts.get(token):
            continue
        done = False
        for block in blocks(document):
            if not isinstance(block, Table) or len(block.rows) != 1:
                continue
            for cell in block.rows[0].cells:
                paras = [p for p in cell.paragraphs if p.text.strip()]
                if (len(paras) < 2
                        or not paras[0].text.strip().startswith(heading)):
                    continue
                set_text(paras[-1], token)
                counts[token] = counts.get(token, 0) + 1
                done = True
                break
            if done:
                break

    # 5. fixed-shape tables: only the value cells carry tokens
    for keys, cells in CELL_TOKENS.items():
        for block in blocks(document):
            if not isinstance(block, Table) or not block.rows:
                continue
            header = [" ".join(c.text.split()) for c in block.rows[0].cells]
            if not all(k in header for k in keys):
                continue
            for (r, c), token in cells.items():
                if r >= len(block.rows) or c >= len(block.columns):
                    continue
                cell = block.cell(r, c)
                # Skip only when a value token already covers the whole
                # cell; a token sitting inside a longer sentence means the
                # sentence itself is person-specific and needs its own.
                if _TOKEN_ONLY.fullmatch(" ".join(cell.text.split())):
                    continue
                paras = [p for p in cell.paragraphs if p.text.strip()]
                set_text(paras[-1] if paras else cell.paragraphs[0], token)
                counts[token] = counts.get(token, 0) + 1
            break

    # 6. mark the tables the filler rebuilds row-by-row
    for block in blocks(document):
        if not isinstance(block, Table) or not block.rows:
            continue
        first = " ".join(block.rows[0].cells[0].text.split())
        marker = REBUILD_MARKERS.get(first)
        if marker and len(block.columns) == 3:
            # clear the whole cell first: a header split across two
            # paragraphs would otherwise leave its label beside the marker,
            # and the filler's lookup would miss the table entirely
            cell = block.rows[0].cells[0]
            for extra in cell.paragraphs[1:]:
                extra._p.getparent().remove(extra._p)
            set_text(cell.paragraphs[0], marker)
            counts[marker] = counts.get(marker, 0) + 1

    # 7. make pagination explicit, and keep the two EQ charts apart
    dev = restyle_development_page(document)
    counts["(dev caption renamed + gold)"] = dev["caption"]
    counts["(dev disclaimer added)"] = dev["disclaimer"]
    counts["(dev closing line added)"] = dev["closing"]
    counts["(cover pinned to page 1)"] = int(tighten_cover(document))
    breaks, trimmed = normalize_pagination(document)
    counts["(page breaks from padding)"] = breaks
    counts["(gaps trimmed)"] = trimmed
    # The break goes on the note that introduces the development chart, not
    # on the caption, so the two travel together instead of the note being
    # stranded at the foot of the strengths page.
    counts["(dev chart on its own page)"] = int(
        force_break_before(document, "Your EQ-i Profile highlights"))

    counts["(dev bars recoloured)"] = recolour_development_bars(document)
    counts["(orphaned boxes pulled up)"] = pull_up_boxes(
        document, ("Workshop Practice", "EQ Practice", "Practice Language",
                   "Coaching Goal", "Roadmap Focus"))

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
