"""
EQI Report Parser — Dash-compatible (bytes-based)
==================================================
Supports both EQ-i 2.0 Workplace Report and Leadership Report formats.
Reads page 1 for the participant name, then scans pages 3–6 for scores,
selecting whichever page yields the most complete subscale data.
Output keys match EQI_COMPOSITES and build_eqi_bar_chart() in app.py.
"""
import io
import re
from typing import Dict, Optional, Tuple

import pdfplumber

# Canonical subscale names → snake_case keys (must match EQI_COMPOSITES in app.py)
SUBSCALE_SNAKE: Dict[str, str] = {
    "Self-Regard":                  "self_regard",
    "Self-Actualization":           "self_actualization",
    "Emotional Self-Awareness":     "emotional_self_awareness",
    "Emotional Expression":         "emotional_expression",
    "Assertiveness":                "assertiveness",
    "Independence":                 "independence",
    "Interpersonal Relationships":  "interpersonal_relationships",
    "Empathy":                      "empathy",
    "Social Responsibility":        "social_responsibility",
    "Problem Solving":              "problem_solving",
    "Reality Testing":              "reality_testing",
    "Impulse Control":              "impulse_control",
    "Flexibility":                  "flexibility",
    "Stress Tolerance":             "stress_tolerance",
    "Optimism":                     "optimism",
}

# Overview page composite labels → display names (used by build_eqi_bar_chart)
COMPOSITE_LABELS: Dict[str, str] = {
    "self-perception composite":    "Self-Perception",
    "self-expression composite":    "Self-Expression",
    "interpersonal composite":      "Interpersonal",
    "decision making composite":    "Decision Making",
    "stress management composite":  "Stress Management",
}

# Signals present on page 1 of an EQ-i 2.0 report (Workplace or Leadership)
_EQI_SIGNALS = [
    "eq-i", "eqi", "total ei",
    "multi-health systems", "self-perception composite",
    "stress management composite", "emotional intelligence",
    "workplace report", "leadership report",
]


def is_eqi_pdf(page1_text: str) -> bool:
    """Return True if page-1 text looks like an EQ-i 2.0 report."""
    lower = page1_text.lower()
    return sum(1 for s in _EQI_SIGNALS if s in lower) >= 1


def parse_eqi_bytes(file_bytes: bytes) -> Optional[Dict]:
    """
    Parse an EQ-i 2.0 PDF (Workplace or Leadership Report) from raw bytes.
    Returns None if the file cannot be identified or parsed.

    Reads page 1 for the participant name, then scans pages 3–6 (indices 2–5)
    for scores, selecting whichever page yields the most complete subscale set.
    - Workplace Report: full scores are on page 3 (index 2)
    - Leadership Report: full scores are on page 6 (index 5)

    Returned dict structure:
      name        str   participant name
      total_ei    int   Total EI score (e.g. 107)
      composites  dict  {"Self-Perception": 114, "Self-Expression": 96, ...}
      subscales   dict  {"self_regard": 115, "assertiveness": 89, ...}
    """
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if len(pdf.pages) < 3:
                return None
            page1_text = pdf.pages[0].extract_text() or ""
            # Candidate score pages: index 2 (Workplace) through 5 (Leadership)
            candidate_texts = [
                pdf.pages[i].extract_text() or ""
                for i in range(2, min(6, len(pdf.pages)))
            ]
    except Exception:
        return None

    if not is_eqi_pdf(page1_text) and not any(
        is_eqi_pdf(t) for t in candidate_texts
    ):
        return None

    name = _extract_name(page1_text)

    # Pick the candidate page that yields the most complete subscale data
    best_total: Optional[int] = None
    best_composites: Dict[str, int] = {}
    best_subscales: Dict[str, int] = {}
    for text in candidate_texts:
        total, composites, subscales = _extract_scores(text)
        if len(subscales) > len(best_subscales):
            best_total, best_composites, best_subscales = (
                total, composites, subscales
            )
        elif len(subscales) == len(best_subscales) and best_total is None:
            best_total = total

    if best_total is None and not best_subscales:
        return None

    return {
        "name":       name,
        "total_ei":   best_total,
        "composites": best_composites,
        "subscales":  best_subscales,
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_name(text: str) -> str:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    # Leadership Report: "Leadership\nReport\n<Name>\n<Date>"
    if (len(lines) >= 3
            and lines[0].lower() == "leadership"
            and lines[1].lower() == "report"):
        return lines[2]
    for line in lines[:8]:
        if re.match(r"^[A-Z][a-z]+([\s\-'][A-Z][a-z]+){1,4}$", line):
            return line
    return lines[0] if lines else "Unknown"


def _extract_scores(
    text: str,
) -> Tuple[Optional[int], Dict[str, int], Dict[str, int]]:
    total_ei: Optional[int] = None
    composites: Dict[str, int] = {}
    subscales: Dict[str, int] = {}

    lines = [ln.strip() for ln in text.split("\n")]
    # Subscale name without a trailing score (Leadership Report format):
    # score appears alone on the very next non-empty line.
    pending_subscale: Optional[str] = None

    for line in lines:
        if not line:
            continue

        # Consume a pending subscale score (Leadership Report multi-line format)
        if pending_subscale is not None:
            m = re.match(r"^(\d{2,3})$", line)
            if m:
                if pending_subscale not in subscales:
                    subscales[pending_subscale] = int(m.group(1))
                pending_subscale = None
                continue
            # Next line wasn't a bare number — abandon the pending match
            pending_subscale = None

        # Total EI — handle both "Total EI 108" and "Total EI: 108"
        m = re.search(r"Total EI:?\s+(\d{2,3})", line, re.IGNORECASE)
        if m:
            total_ei = int(m.group(1))
            continue

        line_lower = line.lower()

        # Composite labels — match before subscales to avoid overlap
        composite_hit = False
        for label_lower, display_name in COMPOSITE_LABELS.items():
            if label_lower in line_lower:
                score = _tail_int(line)
                if score is not None:
                    composites[display_name] = score
                composite_hit = True
                break
        if composite_hit:
            continue

        # Subscales — longest key first to prevent partial matches
        for label, snake_key in sorted(
            SUBSCALE_SNAKE.items(), key=lambda x: -len(x[0])
        ):
            if label.lower() in line_lower:
                score = _tail_int(line)
                if score is not None:
                    if snake_key not in subscales:
                        subscales[snake_key] = score
                else:
                    # Score is on the next line (Leadership Report layout)
                    pending_subscale = snake_key
                break

    return total_ei, composites, subscales


def _tail_int(line: str) -> Optional[int]:
    """Return the last 2–3 digit integer at the end of a line, or None."""
    m = re.search(r"(\d{2,3})\s*$", line)
    return int(m.group(1)) if m else None
