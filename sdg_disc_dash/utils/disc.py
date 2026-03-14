import io
import re
import base64
from typing import Dict, List, Optional

import pandas as pd
import pdfplumber

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
FACTORS = ["D", "I", "S", "C"]
GRAPHS  = ["public", "stress", "mirror"]

FACTOR_COLORS = {
    "D": "#f85149",
    "I": "#d29922",
    "S": "#3fb950",
    "C": "#58a6ff",
}

# ─────────────────────────────────────────
# Trait library
# ─────────────────────────────────────────
TRAITS = {
    "D": {
        "very_high":     ["high acceleration / urgency","takes initiative without waiting","comfortable with risk and imperfect info","pushes for decisions and closure","challenges obstacles directly"],
        "high":          ["proactive; drives closure","direct and confident; sets direction","decisive with reasonable data","addresses issues sooner than later","prefers autonomy and ownership"],
        "moderate_high": ["direct but measured","takes initiative when needed","comfortable owning decisions; still consults","escalates problems selectively","prefers efficiency with useful process"],
        "balanced":      ["situational assertiveness","can lead or support depending on context","flexible about pace and control"],
        "moderate_low":  ["prefers cooperation over pushing","avoids unnecessary confrontation","may defer to maintain harmony"],
        "low":           ["strong preference for shared control","discomfort with high conflict / fast pace","seeks stability over rapid change"],
        "very_low":      ["highly non-confrontational","strong avoidance of risk-taking environments","can appear passive in urgent settings"],
    },
    "I": {
        "very_high":     ["high social energy","persuades via optimism/storytelling","seeks interaction and visibility","energizes the room","talks through decisions"],
        "high":          ["outgoing; relationship-forward","encouraging and enthusiastic influence","builds quick rapport","enjoys collaboration and recognition"],
        "moderate_high": ["warm but measured","persuasive when needed","socially aware; engages strategically","balances connection with task focus"],
        "balanced":      ["flexible social presence","expressive or reserved depending on context","uses influence selectively"],
        "moderate_low":  ["reserved communication style","prefers factual vs animated persuasion","may appear serious or neutral"],
        "low":           ["strong preference for privacy/quiet","minimal outward emotional expression","communication is concise and task-focused"],
        "very_low":      ["avoids social spotlight","strong preference for written/structured communication","may be perceived as distant even when caring"],
    },
    "S": {
        "very_high":     ["strong stability-seeking","high patience; steady pace","loyal and consistent follow-through","dislikes sudden change; needs time to adjust","may resist if stability feels threatened"],
        "high":          ["reliable supporter","consistent work rhythm","prefers harmony and predictability","patient with people and process"],
        "moderate_high": ["cooperative and steady","prefers paced change","supportive while still able to pivot","holds consistency without rigidity"],
        "balanced":      ["pace adapts to situation","steady or fast-moving depending on demands","team orientation varies by context"],
        "moderate_low":  ["restless with routine","more change-tolerant","may prioritize action over consensus"],
        "low":           ["strong change orientation","low patience for prolonged processing","prefers variety and movement"],
        "very_low":      ["high urgency; constant motion","rapid pivots","little tolerance for routine"],
    },
    "C": {
        "very_high":     ["high precision; strong standards","risk-averse without sufficient information","systematic quality control","may become rigid or over-critical","strong preference for rules/structure"],
        "high":          ["thorough; careful decision-making","values accuracy, process, documentation","plans before acting","protects quality and compliance"],
        "moderate_high": ["organized and structured","uses standards as guidance","balances speed and accuracy","can move forward with good enough when needed"],
        "balanced":      ["flexible with detail","uses structure when stakes require it","can be systematic or improvisational"],
        "moderate_low":  ["prefers speed and practicality over precision","less interest in documentation","comfortable deciding with partial info"],
        "low":           ["dislikes heavy rules/constraints","minimal patience for detail work","prefers action and improvisation"],
        "very_low":      ["highly unstructured approach","high tolerance for ambiguity","may create quality/control risk without safeguards"],
    },
}

# ─────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────

def bucketize(x: float) -> str:
    if x <= -4.0: return "very_low"
    if x <= -2.0: return "low"
    if x <= -0.5: return "moderate_low"
    if x <   0.5: return "balanced"
    if x <   2.0: return "moderate_high"
    if x <   4.0: return "high"
    return "very_high"


def shift_label(delta: float) -> str:
    ad = abs(delta)
    if ad >= 3.0: return "major_shift"
    if ad >= 1.5: return "moderate_shift"
    return "minor_shift"


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    parts: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
    return "\n".join(parts)


def extract_page1_text(file_bytes: bytes) -> str:
    """Extract text from page 1 only."""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        if pdf.pages:
            try:
                return pdf.pages[0].extract_text() or ""
            except Exception:
                return ""
    return ""


def extract_name_from_text(text: str, fallback_name: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    anchor_phrases = (
        "style:",
        "maxwell disc personality indicator report",
        "disc personality indicator",
    )
    for i, line in enumerate(lines[:15]):
        lower = line.lower()
        for phrase in anchor_phrases:
            if phrase in lower and i > 0:
                candidate = lines[i - 1].strip()
                if candidate and len(candidate) <= 60 and not any(c.isdigit() for c in candidate[:3]):
                    return candidate
    return fallback_name.rsplit(".", 1)[0]


# ─────────────────────────────────────────
# Style type extraction
#
# The Maxwell DISC page 1 format is:
#   Style: Attainer DCS
#   Style: Conductor D
#   Style: Persuader DI
#
# We need ONLY the trailing 1-3 DISC letters after the keyword name.
# Strategy:
#   1. Find the "Style:" line
#   2. Strip the line label and any descriptive keyword name (non-DISC word)
#   3. Extract the final 1-3 uppercase DISC letters
# ─────────────────────────────────────────

# Matches the entire Style: line
_STYLE_LINE_RE = re.compile(
    r"Style\s*:\s*(.+)",
    re.IGNORECASE,
)

# Known Maxwell style keyword names — all non-DISC words that may appear
# between "Style:" and the DISC letters
_STYLE_KEYWORDS = {
    "attainer", "conductor", "persuader", "analyzer", "promoter",
    "supporter", "coordinator", "implementer", "relater", "developer",
    "achiever", "agent", "appraiser", "counselor", "creative",
    "inspirational", "investigator", "objective", "perfectionist",
    "practitioner", "results", "specialist", "strategist", "teacher",
    "trailblazer", "winner",
}


def extract_style_type_from_page1(page1_text: str) -> Optional[str]:
    """
    Parse the DISC letters from the Style: field on page 1.

    Example inputs and expected outputs:
      "Style: Attainer DCS"  ->  "DCS"
      "Style: Conductor D"   ->  "D"
      "Style: Persuader DI"  ->  "DI"
      "Style: DC"            ->  "DC"
    """
    m = _STYLE_LINE_RE.search(page1_text)
    if not m:
        return None

    remainder = m.group(1).strip()

    # Split on whitespace and collect tokens
    tokens = remainder.split()

    # Walk tokens right-to-left looking for 1-3 all-DISC-letter token
    disc_set = set("DISC")
    for token in reversed(tokens):
        upper = token.upper()
        # Valid if 1-3 chars, all from D/I/S/C
        if 1 <= len(upper) <= 3 and all(c in disc_set for c in upper):
            return upper

    # Fallback: scan the whole remainder for a DISC-only word
    fallback = re.search(r'\b([DISCdisc]{1,3})\b', remainder)
    if fallback:
        cleaned = "".join(c for c in fallback.group(1).upper() if c in disc_set)
        if cleaned:
            return cleaned

    return None


_SCORE_RE = re.compile(
    r"D\s*=\s*([\-0-9]+(?:\.[0-9]+)?)\s*,?\s*"
    r"I\s*=\s*([\-0-9]+(?:\.[0-9]+)?)\s*,?\s*"
    r"S\s*=\s*([\-0-9]+(?:\.[0-9]+)?)\s*,?\s*"
    r"C\s*=\s*([\-0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def extract_scores(text: str) -> Dict[str, Dict[str, float]]:
    t = re.sub(r"[ \t]+", " ", text)
    matches = list(_SCORE_RE.finditer(t))
    if len(matches) < 3:
        raise ValueError(
            f"Expected at least 3 DISC score lines; found {len(matches)}. "
            "The PDF may be scanned, password-protected, or in an unexpected format."
        )
    def _row(m):
        d, i, s, c = map(float, m.groups())
        return {"D": d, "I": i, "S": s, "C": c}
    return {
        "public": _row(matches[0]),
        "stress": _row(matches[1]),
        "mirror": _row(matches[2]),
    }


def build_profile(scores: Dict[str, Dict[str, float]],
                  anchor_graph: str = "stress",
                  style_type: Optional[str] = None) -> Dict:
    factor_profiles = {}
    for f in FACTORS:
        anchor_score = scores[anchor_graph][f]
        bucket       = bucketize(anchor_score)
        delta_ps     = scores["stress"][f] - scores["public"][f]
        delta_ms     = scores["stress"][f] - scores["mirror"][f]
        factor_profiles[f] = {
            "anchor_score":                 anchor_score,
            "bucket":                       bucket,
            "traits":                       TRAITS[f][bucket],
            "public_score":                 scores["public"][f],
            "stress_score":                 scores["stress"][f],
            "mirror_score":                 scores["mirror"][f],
            "delta_public_to_stress":       delta_ps,
            "delta_mirror_to_stress":       delta_ms,
            "shift_public_to_stress_label": shift_label(delta_ps),
            "shift_mirror_to_stress_label": shift_label(delta_ms),
        }
    ranked = sorted(FACTORS, key=lambda ltr: abs(scores[anchor_graph][ltr]), reverse=True)
    if not style_type:
        style_type = "".join(ranked[:2])
    return {
        "anchor_graph":    anchor_graph,
        "graphs":          scores,
        "factor_profiles": factor_profiles,
        "style_type":      style_type,
        "summary":         {"top_two": ranked[:2], "ranked_by_abs": ranked},
    }


def decode_upload(contents: str, filename: str) -> Dict:
    _content_type, content_string = contents.split(",", 1)
    return {"name": filename, "content": base64.b64decode(content_string)}


def process_uploaded_files(files_data: List[Dict], anchor_graph: str = "stress"):
    profiles, rows, errors = [], [], []
    for file_dict in files_data:
        try:
            raw_bytes        = file_dict["content"]
            full_text        = extract_text_from_pdf_bytes(raw_bytes)
            page1_text       = extract_page1_text(raw_bytes)
            scores           = extract_scores(full_text)
            participant_name = extract_name_from_text(full_text, file_dict["name"])
            style_type       = extract_style_type_from_page1(page1_text)
            profile          = build_profile(scores, anchor_graph=anchor_graph,
                                             style_type=style_type)
            profile["source_pdf"]       = file_dict["name"]
            profile["participant_name"] = participant_name
            profiles.append(profile)

            row = {
                "source_pdf":       file_dict["name"],
                "participant_name": participant_name,
                "style_type":       profile["style_type"],
            }
            for g in GRAPHS:
                for f in FACTORS:
                    row[f"{g}_{f}"] = profile["graphs"][g][f]
            for f in FACTORS:
                fp = profile["factor_profiles"][f]
                row[f"{f}_anchor"] = fp["anchor_score"]
                row[f"{f}_bucket"] = fp["bucket"]
            row["top1_factor"] = profile["summary"]["ranked_by_abs"][0]
            row["top2_factor"] = profile["summary"]["ranked_by_abs"][1]
            rows.append(row)

        except Exception as exc:
            errors.append({"file": file_dict["name"], "error": str(exc)})

    return profiles, pd.DataFrame(rows), errors
