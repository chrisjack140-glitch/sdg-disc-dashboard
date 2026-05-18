"""
EQI Report Parser — Dash-compatible (bytes-based)
==================================================
Reads ONLY pages 1 (name) and 3 (score overview) for speed.
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

# Signals present on page 1 of an EQ-i 2.0 report
_EQI_SIGNALS = [
    "eq-i", "eqi", "total ei",
    "multi-health systems", "self-perception composite",
    "stress management composite", "emotional intelligence",
    "workplace report",
]


def is_eqi_pdf(page1_text: str) -> bool:
    """Return True if page-1 text looks like an EQ-i 2.0 report."""
    lower = page1_text.lower()
    return sum(1 for s in _EQI_SIGNALS if s in lower) >= 1


def parse_eqi_bytes(file_bytes: bytes) -> Optional[Dict]:
    """
    Parse an EQ-i 2.0 Workplace Report PDF from raw bytes.
    Returns None if the file cannot be identified or parsed.
    Only reads pages 1 and 3.

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
            page3_text = pdf.pages[2].extract_text() or ""
    except Exception:
        return None

    if not is_eqi_pdf(page1_text) and not is_eqi_pdf(page3_text):
        return None

    name = _extract_name(page1_text)
    total_ei, composites, subscales = _extract_scores(page3_text)

    if total_ei is None and not subscales:
        return None

    return {
        "name":       name,
        "total_ei":   total_ei,
        "composites": composites,
        "subscales":  subscales,
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_name(text: str) -> str:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
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

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Total EI
        m = re.search(r"Total EI\s+(\d{2,3})", line, re.IGNORECASE)
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
                if score is not None and snake_key not in subscales:
                    subscales[snake_key] = score
                break

    return total_ei, composites, subscales


def _tail_int(line: str) -> Optional[int]:
    """Return the last 2–3 digit integer at the end of a line, or None."""
    m = re.search(r"(\d{2,3})\s*$", line)
    return int(m.group(1)) if m else None
