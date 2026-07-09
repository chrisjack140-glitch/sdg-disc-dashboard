"""
Leadership Roadmap content model
=================================
Renderer-agnostic representation of a Leadership Roadmap booklet as an
ordered list of typed blocks. Two renderers (utils/roadmap_pdf.py,
utils/roadmap_docx.py) walk the same block list so the PDF and DOCX
outputs stay in sync. No Dash / rendering-library imports here — mirrors
the plain-data style of utils/disc.py.
"""
from dataclasses import dataclass, field
from typing import List, Optional

# ─────────────────────────────────────────
# Brand palette — colors extracted directly from
# Ariel_Alston_PA_Leadership_Roadmap_Workshop_Booklet.docx. Never hardcode
# these hex values outside this dict; both renderers import PALETTE.
# ─────────────────────────────────────────
PALETTE = {
    "navy_header":     "#1B2A44",   # section header bands
    "navy_cover":       "#101827",   # cover page band
    "white":            "#FFFFFF",
    "slate_header":     "#43546B",   # table header rows
    "body_text":        "#202631",
    "muted_text":       "#777777",
    "row_label_fill":   "#F7F9FB",   # alternating table row (label column)
    "row_value_fill":   "#FFFFFF",   # alternating table row (value column)
    "gold_accent":      "#C59B2D",   # cover accent stripe
    "gold_text":        "#D7B35A",
    "tint_gold_1":      "#FFF8E8",   # Framework overview / confidentiality
    "tint_gold_2":      "#FFF7E2",   # DISC section callouts
    "tint_teal":        "#E9F5F4",   # EQ section callouts
    "tint_mauve":       "#F7ECF2",   # Leadership Signature callouts
    "tint_slate_1":     "#F6F8FA",   # worksheet intro callouts
    "tint_slate_2":     "#F8FAFC",   # neutral reflection callouts
    "tint_blue":        "#EAF2F8",   # Flywheel primary-strength box
}

FONT_HEADER = "Georgia"   # cover / section title font
FONT_BODY   = "Arial"     # body copy, tables, callouts

# Section identifiers used across boilerplate + generator + tint lookup
SECTION_IDS = ["disc", "eqi", "flywheel", "signature"]

SECTION_TINTS = {
    "disc":      PALETTE["tint_gold_2"],
    "eqi":       PALETTE["tint_teal"],
    "flywheel":  PALETTE["tint_gold_1"],
    "signature": PALETTE["tint_mauve"],
}

SECTION_TITLES = {
    "disc":      "DISC",
    "eqi":       "EQ",
    "flywheel":  "FLYWHEEL",
    "signature": "LEADERSHIP SIGNATURE",
}


# ─────────────────────────────────────────
# Block types
# ─────────────────────────────────────────
@dataclass
class PageBreak:
    """Marks the end of a page."""
    pass


@dataclass
class HeaderBand:
    """Full-width colored header band at the top of a page."""
    title: str
    subtitle: str = ""
    brand_tag: str = "SDG Leadership Framework"
    bg: str = PALETTE["navy_header"]
    fg: str = PALETTE["white"]


@dataclass
class Paragraph:
    text: str
    size: int = 14
    color: str = PALETTE["body_text"]
    bold: bool = False
    italic: bool = False
    font: str = FONT_BODY


@dataclass
class BulletList:
    items: List[str]
    size: int = 14
    color: str = PALETTE["body_text"]


@dataclass
class ShadedGroup:
    """Multiple stacked lines sharing one background fill — mirrors the
    docx pattern of several paragraphs inside a single shaded table cell
    (used for the cover page's navy banner)."""
    bg: str
    lines: List[Paragraph] = field(default_factory=list)


@dataclass
class Divider:
    """A thin full-width colored bar with no text (the cover page's gold
    accent stripe)."""
    color: str = PALETTE["gold_accent"]
    height: int = 6


@dataclass
class CalloutBox:
    """Single tinted box with an optional bold heading and a body sentence."""
    body: str
    heading: str = ""
    tint: str = PALETTE["tint_slate_2"]


@dataclass
class TableRow:
    cells: List[str]
    fill: Optional[str] = None   # override; else renderer alternates row fills


@dataclass
class DataTable:
    """General-purpose content table (header row optional)."""
    header_row: Optional[List[str]] = None
    rows: List[TableRow] = field(default_factory=list)
    col_widths: Optional[List[float]] = None  # fractions summing to 1.0


@dataclass
class BlankWorksheetTable:
    """Worksheet/notes table: a header row plus blank rows for the
    participant to fill in by hand. `prompts` becomes the first column of
    each row; remaining columns render as empty cells."""
    header_row: List[str]
    prompts: List[str]
    n_blank_cols: int = 1


@dataclass
class RoadmapDocument:
    person_name: str
    blocks: List = field(default_factory=list)

    def add(self, *new_blocks):
        self.blocks.extend(new_blocks)
        return self

    def add_page(self, *new_blocks):
        """Append blocks then a page break — used between booklet pages."""
        self.blocks.extend(new_blocks)
        self.blocks.append(PageBreak())
        return self
