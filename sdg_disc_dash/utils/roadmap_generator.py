"""
Leadership Roadmap — deterministic content generator
=====================================================
Builds a personalized RoadmapDocument (24 pages) from a participant
profile dict (utils/disc.py contract) + flywheel_disc_reference.json.

All generation is rule-based string templating — no external API calls.
Tone rules follow the SDG methodology guides: strengths-first, risks
framed as overuse of strength with "may" language, "from -> to" leadership
shifts, never raw DISC jargon ("High C") in prose.
"""
import json
from pathlib import Path

from utils.roadmap_content_model import (
    RoadmapDocument, HeaderBand, Paragraph, BulletList, CalloutBox,
    DataTable, TableRow, PALETTE, SECTION_TINTS,
)
from utils.roadmap_boilerplate import (
    get_cover_boilerplate, get_how_to_use_page, get_framework_overview_page,
    get_section_intro_page, get_worksheet_page, get_notes_page,
    get_closing_commitment_page, possessive, first_name, BRAND_TAG,
)

# ─────────────────────────────────────────
# Reference data
# ─────────────────────────────────────────
_REF_PATH = Path(__file__).parent / "flywheel_disc_reference.json"
_FLYWHEEL_REF = json.loads(_REF_PATH.read_text(encoding="utf-8"))
_STYLES_BY_CODE = {s["code"]: s for s in _FLYWHEEL_REF["styles"]}
QUADRANT_LABELS = _FLYWHEEL_REF["quadrant_labels"]
SUB_DIMENSION_LABELS = _FLYWHEEL_REF["sub_dimension_labels"]

# Maxwell style keyword names for the profile line, keyed by DISC code.
STYLE_NAMES = {s["code"]: s["name"] for s in _FLYWHEEL_REF["styles"]}

# The 15 EQ-i subscale keys (snake_case) with display names — mirrors
# SUBSCALE_SNAKE in utils/eqi_parser.py (inverted).
EQI_SUBSCALE_DISPLAY = {
    "self_regard":                 "Self-Regard",
    "self_actualization":          "Self-Actualization",
    "emotional_self_awareness":    "Emotional Self-Awareness",
    "emotional_expression":        "Emotional Expression",
    "assertiveness":               "Assertiveness",
    "independence":                "Independence",
    "interpersonal_relationships": "Interpersonal Relationships",
    "empathy":                     "Empathy",
    "social_responsibility":       "Social Responsibility",
    "problem_solving":             "Problem Solving",
    "reality_testing":             "Reality Testing",
    "impulse_control":             "Impulse Control",
    "flexibility":                 "Flexibility",
    "stress_tolerance":            "Stress Tolerance",
    "optimism":                    "Optimism",
}

# Per-subscale leadership interpretation sentences.
# {name} = first name. Strength phrasing (score is a top strength).
EQI_STRENGTH_SENTENCES = {
    "self_regard": "{name} has a strong sense of capability and can lead "
                   "with more visible authority.",
    "self_actualization": "{name} is driven by growth and purpose, which "
                          "fuels sustained leadership energy.",
    "emotional_self_awareness": "{name} can recognize what they are "
                                "feeling and what the moment requires. The "
                                "next step is to express that awareness "
                                "clearly.",
    "emotional_expression": "{name} communicates feeling and conviction "
                            "openly, which makes leadership direction easy "
                            "to read.",
    "assertiveness": "{name} has the internal confidence to state "
                     "expectations, concerns, and recommendations.",
    "independence": "{name} forms judgments independently and can hold a "
                    "position under group pressure.",
    "interpersonal_relationships": "{name} builds mutually satisfying "
                                   "relationships that create trust and "
                                   "coaching credibility.",
    "empathy": "{name} can understand what others may need; the growth "
               "step is to make connection more intentional.",
    "social_responsibility": "{name} contributes to the wider team and "
                             "models organizational citizenship.",
    "problem_solving": "{name} brings disciplined thinking and can help "
                       "supervisors work through issues with clarity.",
    "reality_testing": "{name} sees situations objectively, which grounds "
                       "coaching conversations in evidence.",
    "impulse_control": "{name} stays measured before acting, which builds "
                       "credibility in high-pressure moments.",
    "flexibility": "{name} adapts thinking and behavior readily as "
                   "conditions change.",
    "stress_tolerance": "{name} stays steady under pressure, which anchors "
                        "the team during difficult periods.",
    "optimism": "{name} maintains a resilient, positive outlook that "
                "helps the team persist through setbacks.",
}

# Development phrasing (score is a bottom-2 growth area).
EQI_DEVELOPMENT_SENTENCES = {
    "self_regard": "Strengthen confidence in personal judgment so "
                   "leadership perspective is offered without hesitation.",
    "self_actualization": "Reconnect daily work to purpose and growth to "
                          "sustain leadership energy.",
    "emotional_self_awareness": "Build the habit of naming internal "
                                "reactions before responding under "
                                "pressure.",
    "emotional_expression": "Increase visible appreciation, concern, "
                            "urgency, and confidence in ways supervisors "
                            "can see and hear.",
    "assertiveness": "State expectations, concerns, and recommendations "
                     "earlier and more directly.",
    "independence": "Practice holding a position without waiting for "
                    "consensus when the situation requires direction.",
    "interpersonal_relationships": "Build stronger relational connection "
                                   "so coaching lands with trust, not "
                                   "just accuracy.",
    "empathy": "Listen for impact, not just intent, and acknowledge what "
               "others are experiencing before problem-solving.",
    "social_responsibility": "Invest more visibly in team-level outcomes "
                             "beyond individual responsibilities.",
    "problem_solving": "Slow down under pressure to work the problem "
                       "before committing to a response.",
    "reality_testing": "Test assumptions against evidence before acting "
                       "on first impressions.",
    "impulse_control": "Pause before reacting so responses stay "
                       "proportionate under pressure.",
    "flexibility": "Increase tolerance for change and responsible risk; "
                   "adapt plans sooner when conditions shift.",
    "stress_tolerance": "Build recovery habits that keep leadership "
                        "presence steady during sustained pressure.",
    "optimism": "Frame setbacks as solvable to keep the team moving "
                "forward under difficulty.",
}

# DISC factor → leadership-language phrase bank (per the Roadmap
# methodology: never say "High C" in prose; translate to behavior).
_FACTOR_LANGUAGE = {
    "D": {
        "strengths": "decisiveness, directness, urgency, and "
                     "results-focused drive",
        "experience": "direct, confident, decisive, and focused on "
                      "outcomes",
        "blind_spot": "May move faster than team trust or alignment can "
                      "support, and may under-invest in relational "
                      "connection.",
        "pressure": "May become more forceful, more impatient, or more "
                    "likely to take over rather than coach.",
        "adjustment": "Slow down to align, listen for impact, and coach "
                      "ownership instead of driving compliance.",
    },
    "I": {
        "strengths": "relational energy, engagement, optimism, and "
                     "visible influence",
        "experience": "warm, energizing, encouraging, and "
                      "relationship-forward",
        "blind_spot": "May rely on relational energy and soften "
                      "accountability to preserve harmony.",
        "pressure": "May over-explain, become scattered, or avoid the "
                    "direct conversation that accountability requires.",
        "adjustment": "Anchor enthusiasm in clear expectations, "
                      "follow-through, and direct accountability "
                      "language.",
    },
    "S": {
        "strengths": "steadiness, loyalty, patience, and dependable "
                     "follow-through",
        "experience": "calm, supportive, consistent, and committed to the "
                      "team",
        "blind_spot": "May absorb tension rather than addressing it, and "
                      "may delay difficult conversations to preserve "
                      "harmony.",
        "pressure": "May become quieter, more accommodating, or slower to "
                    "escalate performance concerns.",
        "adjustment": "Name expectations earlier, tolerate productive "
                      "tension, and hold standards visibly.",
    },
    "C": {
        "strengths": "precision, structure, quality standards, and "
                     "disciplined analysis",
        "experience": "calm, thoughtful, prepared, fair, structured, and "
                      "committed to getting the work right",
        "blind_spot": "May let the quality of the work speak for itself "
                      "rather than making leadership perspective "
                      "visible.",
        "pressure": "May become more internal, more precise, slower to "
                    "express concern, or more likely to carry the "
                    "standard personally.",
        "adjustment": "Name expectations earlier, use concise language, "
                      "ask ownership questions, and confirm follow-up.",
    },
}

_QUADRANT_ORDER = ["direction", "culture", "learning", "execution"]

# Deterministic secondary-quadrant rule: highest-priority quadrant that is
# not the primary, using a fixed priority ranking per primary anchor.
# Rationale: pairs each anchor with its most natural complement per the
# SDG methodology (execution pairs with direction, culture with execution,
# direction with execution, learning with direction).
_SECONDARY_QUADRANT = {
    "execution": "direction",
    "culture":   "execution",
    "direction": "execution",
    "learning":  "direction",
}


# ─────────────────────────────────────────
# Lookups & selection
# ─────────────────────────────────────────
def get_flywheel_style_entry(style_type: str) -> dict:
    """Look up a DISC style code in the reference. Unknown codes fall back
    to the code's first letter, then to the balanced DISC entry."""
    code = (style_type or "").upper().strip()
    if code in _STYLES_BY_CODE:
        return _STYLES_BY_CODE[code]
    if code and code[0] in _STYLES_BY_CODE:
        return _STYLES_BY_CODE[code[0]]
    return _STYLES_BY_CODE["DISC"]


def select_eqi_top_bottom(eqi_scores: dict, top_n: int = 5,
                          bottom_n: int = 2):
    """Sort ONLY the 15 snake_case subscale keys (ignore composites and
    total_ei). Returns (top, bottom) lists of (snake_key, score)."""
    subscales = [(k, v) for k, v in eqi_scores.items()
                 if k in EQI_SUBSCALE_DISPLAY and isinstance(v, (int, float))]
    if not subscales:
        return [], []
    ranked = sorted(subscales, key=lambda kv: kv[1], reverse=True)
    return ranked[:top_n], ranked[-bottom_n:][::-1]


def _mirror_scores_line(profile: dict) -> str:
    g = profile["graphs"]["mirror"]
    return (f"D {g['D']:.2f} | I {g['I']:.2f} | "
            f"S {g['S']:.2f} | C {g['C']:.2f}")


def _style_label(profile: dict) -> str:
    code = (profile.get("style_type") or "").upper()
    name = STYLE_NAMES.get(code)
    if name is None:
        name = STYLE_NAMES.get(code[:1], "Balanced / Adaptive")
    return f"{code} {name}" if code else name


# ─────────────────────────────────────────
# DISC lens
# ─────────────────────────────────────────
def build_disc_lens_text(profile: dict) -> dict:
    """Translate the style blend into leadership language. Primary factor
    drives the core narrative; secondary factor adds a supporting clause."""
    code = (profile.get("style_type") or "").upper()
    letters = [c for c in code if c in "DISC"] or \
              profile["summary"]["top_two"]
    primary, secondary = letters[0], (letters[1] if len(letters) > 1 else None)
    p = _FACTOR_LANGUAGE[primary]

    strengths = p["strengths"]
    if secondary:
        strengths = f"{strengths}, supported by " \
                    f"{_FACTOR_LANGUAGE[secondary]['strengths']}"

    style_entry = get_flywheel_style_entry(code)
    keywords = ", ".join(style_entry["behavioral_keywords"][:6])

    return {
        "primary": primary,
        "secondary": secondary,
        "style_label": _style_label(profile),
        "keywords": keywords,
        "natural_strengths": strengths,
        "supervisor_experience": p["experience"],
        "blind_spot": p["blind_spot"],
        "pressure_risk": p["pressure"],
        "coaching_adjustment": p["adjustment"],
        "mirror_line": _mirror_scores_line(profile),
    }


# ─────────────────────────────────────────
# EQ-i lens
# ─────────────────────────────────────────
def build_eqi_lens_text(profile: dict):
    """Returns None when the profile has no usable EQ-i subscale data."""
    eqi = profile.get("eqi_scores") or {}
    top, bottom = select_eqi_top_bottom(eqi)
    if not top:
        return None
    name = first_name(profile["participant_name"])

    strengths = [
        {"key": k, "label": EQI_SUBSCALE_DISPLAY[k], "score": int(v),
         "meaning": EQI_STRENGTH_SENTENCES[k].format(name=name)}
        for k, v in top
    ]
    development = [
        {"key": k, "label": EQI_SUBSCALE_DISPLAY[k], "score": int(v),
         "meaning": EQI_DEVELOPMENT_SENTENCES[k]}
        for k, v in bottom
    ]
    strengths_inline = "; ".join(f"{s['label']} ({s['score']})"
                                 for s in strengths)
    dev_inline = "; ".join(f"{d['label']} ({d['score']})"
                           for d in development)

    narrative = (
        f"{possessive(profile['participant_name'])} EQ-i profile shows "
        f"strong capacity in {strengths_inline}. "
        f"The development opportunity sits in {dev_inline}. "
        f"The next level is to express that internal capacity in ways "
        f"supervisors can see, hear, and respond to."
    )
    return {
        "strengths": strengths,
        "development": development,
        "strengths_inline": strengths_inline,
        "dev_inline": dev_inline,
        "narrative": narrative,
        "total_ei": eqi.get("total_ei"),
    }


# ─────────────────────────────────────────
# Flywheel section
# ─────────────────────────────────────────
def build_flywheel_section(profile: dict) -> dict:
    style_entry = get_flywheel_style_entry(profile.get("style_type"))
    primary = style_entry["anchor_quadrant"]
    if primary == "varies":       # balanced DISC profile
        primary = "execution"
    secondary = _SECONDARY_QUADRANT[primary]

    def _q(qid):
        q = style_entry["quadrants"][qid]
        return {
            "id": qid,
            "label": QUADRANT_LABELS[qid],
            "strengths": q["strengths"],
            "risk": q["risk"],
            "shift": q["leadership_shift"],
            "sub_dimension": q.get("sub_dimension"),
        }

    quadrants = {qid: _q(qid) for qid in _QUADRANT_ORDER}
    primary_q, secondary_q = quadrants[primary], quadrants[secondary]

    sub = style_entry.get("anchor_sub_dimension")
    primary_label = primary_q["label"]
    if sub:
        primary_label += f" — {SUB_DIMENSION_LABELS[sub]}"

    name = first_name(profile["participant_name"])
    diagnostic = (
        f"{name} naturally strengthens {primary_q['label']}. "
        f"{primary_q['strengths'][0]} "
        f"Under pressure, this may shift: "
        f"{primary_q['risk'][0].lower()}{primary_q['risk'][1:]} "
        f"{primary_q['shift']}"
    )
    return {
        "style_entry": style_entry,
        "primary": primary,
        "secondary": secondary,
        "primary_label": primary_label,
        "primary_q": primary_q,
        "secondary_q": secondary_q,
        "quadrants": quadrants,
        "activation_focus": style_entry["flywheel_activation_focus"],
        "growth_tension": style_entry["growth_tension"],
        "diagnostic": diagnostic,
    }


# ─────────────────────────────────────────
# Leadership Signature (5-step formula)
# ─────────────────────────────────────────
def build_leadership_signature(profile: dict, disc_text: dict,
                               flywheel: dict) -> dict:
    name = first_name(profile["participant_name"])
    primary_q = flywheel["primary_q"]

    # Anchor / value / influence / growth-edge → one statement
    anchor_words = disc_text["natural_strengths"].split(", supported by")[0]
    growth_edge = flywheel["activation_focus"].rstrip(".")
    statement = (
        f"I lead with {anchor_words}. I build trust through consistent "
        f"standards and authentic communication, develop people through "
        f"honest coaching, and create momentum by pairing "
        f"{primary_q['label'].lower()} strength with {growth_edge[0].lower()}"
        f"{growth_edge[1:]}."
    )

    # Observable-behavior table rows: signature element → behavior.
    elements = [
        ("Core Strength",
         f"{name} brings {anchor_words} to how direction, coaching, and "
         "follow-up are handled."),
        ("Visible Direction",
         f"{name} states what they are seeing, why it matters, and what "
         "they recommend."),
        ("Coaching Presence",
         f"{name} asks ownership questions before solving, and confirms "
         "the follow-up."),
        ("Accountability",
         f"{name} names expectations early and holds standards while "
         "developing people."),
        ("Growth Edge",
         f"{name} is intentionally strengthening: {growth_edge}."),
    ]
    return {"statement": statement, "elements": elements}


# ─────────────────────────────────────────
# Executive summary + snapshot
# ─────────────────────────────────────────
def build_executive_summary(profile, disc_text, eqi_text, flywheel,
                            signature) -> dict:
    name = first_name(profile["participant_name"])
    poss = possessive(profile["participant_name"])

    disc_para = (
        f"{poss} Leadership Roadmap reflects an integrated DISC, EQ-i, and "
        f"Flywheel profile. DISC identifies {name} as a "
        f"{disc_text['style_label']}: {disc_text['keywords']}. In a PA "
        f"leadership role, this creates {disc_text['natural_strengths']}."
    )

    if eqi_text:
        eqi_para = (
            f"{poss} EQ-i profile adds an important leadership dimension "
            f"beyond the DISC results. {name} shows strong capacity through "
            f"{eqi_text['strengths_inline']}. The development opportunity "
            f"is {eqi_text['dev_inline']} — expressing internal capacity "
            f"in ways supervisors can see, hear, and respond to."
        )
    else:
        eqi_para = (
            f"An EQ-i 2.0 assessment is not yet available for {name}. When "
            f"EQ-i results are added, this Roadmap will connect emotional "
            f"intelligence capacity to the DISC and Flywheel findings "
            f"below."
        )

    fly_para = (
        f"The Flywheel analysis identifies {flywheel['primary_label']} as "
        f"{poss} primary strength and {flywheel['secondary_q']['label']} "
        f"as a secondary strength. {flywheel['primary_q']['strengths'][0]} "
        f"The development risk: {flywheel['primary_q']['risk']}"
    )

    focus = (
        f"{poss} development focus: {flywheel['growth_tension']}"
    )
    return {"disc_para": disc_para, "eqi_para": eqi_para,
            "fly_para": fly_para, "focus": focus}


def build_roadmap_snapshot_rows(profile, disc_text, eqi_text,
                                flywheel) -> list:
    name = first_name(profile["participant_name"])
    if eqi_text:
        eq_strength_cells = (
            eqi_text["strengths_inline"] + ".",
            f"{name} has strong internal capacity. The workshop focus is "
            "to express it visibly through leadership presence and "
            "coaching.",
        )
        eq_dev_cells = (
            eqi_text["dev_inline"] + ".",
            eqi_text["development"][0]["meaning"],
        )
    else:
        eq_strength_cells = ("Not assessed.",
                             "EQ-i results not yet available; revisit when "
                             "the assessment is complete.")
        eq_dev_cells = ("Not assessed.",
                        "EQ-i results not yet available.")

    return [
        TableRow(["Natural Leadership Pattern",
                  f"{disc_text['style_label']}: {disc_text['keywords']}.",
                  disc_text["coaching_adjustment"]]),
        TableRow(["DISC Mirror Scores",
                  disc_text["mirror_line"],
                  disc_text["blind_spot"]]),
        TableRow(["EQ Strengths", *eq_strength_cells]),
        TableRow(["EQ Development Stretch", *eq_dev_cells]),
        TableRow(["Primary Flywheel Strength",
                  flywheel["primary_label"] + ".",
                  flywheel["primary_q"]["shift"]]),
        TableRow(["Secondary Flywheel Strength",
                  flywheel["secondary_q"]["label"] + ".",
                  flywheel["secondary_q"]["shift"]]),
        TableRow(["Leadership Risk Under Pressure",
                  disc_text["pressure_risk"],
                  "Pause, name the pattern, coach ownership, and confirm "
                  "the follow-up."]),
        TableRow(["Coaching Opportunity",
                  "Move from correcting or carrying to coaching "
                  "supervisors into ownership.",
                  "Ask before solving and require the supervisor to "
                  "identify the next step."]),
    ]


# ─────────────────────────────────────────
# 30-60-90 coaching plan
# ─────────────────────────────────────────
def build_coaching_plan(profile, disc_text, eqi_text, flywheel) -> dict:
    name = first_name(profile["participant_name"])
    style = disc_text["style_label"]

    if eqi_text:
        eq_30 = ("Strengthen " +
                 " and ".join(d["label"] for d in eqi_text["development"]) +
                 " by naming appreciation, concern, urgency, and "
                 "confidence more clearly.")
        eq_60 = ("Use " + eqi_text["strengths"][0]["label"] +
                 " with Flexibility. Balance direct expectations with "
                 "curiosity, support, and coaching questions.")
    else:
        eq_30 = ("Complete the EQ-i 2.0 assessment; meanwhile, practice "
                 "naming appreciation, concern, urgency, and confidence "
                 "more clearly.")
        eq_60 = ("Balance direct expectations with curiosity, support, "
                 "and coaching questions.")

    phases = [
        TableRow(["30 Days",
                  "Visibility and Leadership Presence",
                  f"Recognize the {style} pattern and where it may limit "
                  "visible direction.",
                  eq_30,
                  f"{flywheel['secondary_q']['label']}; "
                  f"{flywheel['primary_q']['label']}."]),
        TableRow(["60 Days",
                  "Coaching Supervisors into Ownership",
                  "Move from carrying the standard personally to coaching "
                  "ownership. Ask before solving.",
                  eq_60,
                  f"{flywheel['primary_q']['label']}; Learning, Feedback "
                  "& Adaptation."]),
        TableRow(["90 Days",
                  "Sustaining Accountability Rhythm",
                  "Maintain consistency under pressure. Stay visible, "
                  "direct, and coaching-oriented when the pace "
                  "increases.",
                  "Sustain visible expression, relational trust, and "
                  "coaching presence as repeatable habits.",
                  "Full Flywheel integration across all four "
                  "quadrants."]),
    ]

    d30 = [
        TableRow(["Visible Leadership Conversation",
                  "Use one clear leadership frame each week: Here is what "
                  "I am seeing; here is why it matters; here is what I "
                  "recommend."]),
        TableRow(["EQ Practice",
                  "Identify what needs to be expressed before key "
                  "conversations: appreciation, concern, urgency, "
                  "confidence, or direction."]),
        TableRow(["Flywheel Scan",
                  "Name one place where unclear direction or low "
                  "visibility is slowing momentum."]),
        TableRow(["Supervisor Coaching Step",
                  "Complete one coaching conversation that ends with "
                  "owner, action, timeline, and follow-up."]),
    ]
    d60 = [
        TableRow(["Ownership Coaching",
                  "Use the question: What do you think the next step "
                  "should be, and what will you own before our next "
                  "check-in?"]),
        TableRow(["Transfer Standards",
                  "Identify one recurring quality or follow-through issue "
                  "and coach the supervisor to own the standard."]),
        TableRow(["EQ Practice",
                  "Ask before solving: What do you need from me right now "
                  "- direction, support, or a sounding board?"]),
        TableRow(["Flywheel Review",
                  "Map one supervisor pattern across Direction, Culture, "
                  "Learning, and Execution to identify where coaching is "
                  "needed."]),
    ]
    d90 = [
        TableRow(["Pressure Check",
                  "Ask: Am I solving this because it is mine to solve, or "
                  "because I have not coached ownership clearly enough?"]),
        TableRow(["Leadership Signature Review",
                  "Identify which part of the Leadership Signature has "
                  "become more visible and which part still needs "
                  "practice."]),
        TableRow(["Accountability Rhythm",
                  "Review three coaching conversations and confirm what "
                  "changed, what repeated, and what needs follow-up."]),
        TableRow(["Flywheel Momentum Review",
                  "Identify momentum gained, drag remaining, and the next "
                  "leadership behavior that must become consistent."]),
    ]
    success = (
        f"{name} demonstrates visible, repeatable, momentum-building "
        f"leadership. Standards remain high, and supervisors are "
        f"increasingly expected to own the work, apply feedback, and "
        f"contribute to execution rather than depending on {name} to "
        f"carry the burden."
    )
    return {"phases": phases, "d30": d30, "d60": d60, "d90": d90,
            "success": success}


# ─────────────────────────────────────────
# Page assemblers (personalized pages)
# ─────────────────────────────────────────
def _exec_summary_page(profile, summary, signature):
    return [
        HeaderBand(title="EXECUTIVE SUMMARY",
                   subtitle="My Leadership Roadmap", brand_tag=BRAND_TAG),
        Paragraph(summary["disc_para"]),
        Paragraph(summary["eqi_para"]),
        Paragraph(summary["fly_para"]),
        CalloutBox(heading="Roadmap Focus", body=summary["focus"],
                   tint=PALETTE["tint_gold_1"]),
        Paragraph("Leadership Signature", bold=True, size=15),
        CalloutBox(body=signature["statement"],
                   tint=PALETTE["tint_slate_2"]),
    ]


def _snapshot_page(profile, snapshot_rows):
    name = first_name(profile["participant_name"])
    return [
        HeaderBand(title="ROADMAP SNAPSHOT",
                   subtitle="Quick reference for workshop use",
                   brand_tag=BRAND_TAG),
        DataTable(
            header_row=["Roadmap Element", f"{name}'s Profile",
                        "What It Means for Workshop Application"],
            rows=snapshot_rows,
            col_widths=[0.22, 0.40, 0.38],
        ),
        Paragraph(
            "Workshop reminder: Use this page before each worksheet to "
            f"connect the concept to {possessive(profile['participant_name'])} "
            "individual leadership pattern.", italic=True,
        ),
    ]


def _disc_connection_page(profile, disc_text):
    poss = possessive(profile["participant_name"])
    name = first_name(profile["participant_name"])
    return [
        HeaderBand(title="DISC ROADMAP CONNECTION",
                   subtitle=f"{poss} behavior pattern and pressure shift",
                   brand_tag=BRAND_TAG),
        Paragraph(
            f"DISC helps {name} understand how leadership behavior is "
            "likely experienced by supervisors and what may shift when "
            f"pressure rises. The {disc_text['style_label']} pattern gives "
            f"{name} a strong foundation in "
            f"{disc_text['natural_strengths']}. The development "
            "opportunity is to make those strengths more visible and "
            "coach others into ownership rather than carrying the "
            "standard alone."
        ),
        DataTable(
            header_row=["Profile Element",
                        f"{name} Leadership Interpretation"],
            rows=[
                TableRow(["Primary Style", disc_text["style_label"]]),
                TableRow(["Mirror Profile", disc_text["mirror_line"]]),
                TableRow(["Natural Strengths",
                          disc_text["natural_strengths"].capitalize() + "."]),
                TableRow(["How Supervisors May Experience "
                          f"{name}",
                          disc_text["supervisor_experience"].capitalize()
                          + "."]),
                TableRow(["Potential Blind Spot", disc_text["blind_spot"]]),
                TableRow(["Pressure Risk", disc_text["pressure_risk"]]),
                TableRow(["DISC Coaching Adjustment",
                          disc_text["coaching_adjustment"]]),
            ],
            col_widths=[0.32, 0.68],
        ),
        CalloutBox(
            heading="Practice Language",
            body="Here is what I am seeing. Here is why it matters. Here "
                 "is what I believe we need to do next.",
            tint=PALETTE["tint_gold_1"],
        ),
    ]


def _eqi_connection_page(profile, eqi_text):
    name = first_name(profile["participant_name"])
    poss = possessive(profile["participant_name"])
    blocks = [
        HeaderBand(title="EQ ROADMAP CONNECTION",
                   subtitle="Leadership presence and coaching capacity",
                   brand_tag=BRAND_TAG),
    ]
    if eqi_text is None:
        blocks += [
            Paragraph(
                f"An EQ-i 2.0 assessment is not yet available for {name}. "
                "This page will be populated once the assessment is "
                "completed and paired with the DISC profile. In the "
                "meantime, use the EQ anchors below as the coaching "
                "frame for this section."
            ),
            DataTable(
                header_row=["EQ Anchor", "Leadership Meaning"],
                rows=[
                    TableRow(["Authenticity",
                              "Builds trust, credibility, transparency, "
                              "and emotional safety."]),
                    TableRow(["Coaching",
                              "Develops people through support while "
                              "translating care into performance "
                              "growth."]),
                    TableRow(["Insight",
                              "Creates vision, purpose, and strategic "
                              "clarity for others to follow."]),
                    TableRow(["Innovation",
                              "Strengthens experimentation, agility, and "
                              "adaptive response."]),
                ],
            ),
        ]
    else:
        rows = [TableRow([s["label"], str(s["score"]), s["meaning"]])
                for s in eqi_text["strengths"]]
        rows.append(TableRow([
            "Development Areas",
            "; ".join(f"{d['label']} {d['score']}"
                      for d in eqi_text["development"]),
            " ".join(d["meaning"] for d in eqi_text["development"]),
        ]))
        blocks += [
            Paragraph(
                f"EQ-i helps {name} translate internal leadership "
                f"capacity into visible leadership presence. {poss} "
                "profile shows the strengths below, with a clear growth "
                "opportunity: express that clarity and confidence in "
                "ways supervisors can see, hear, and respond to."
            ),
            DataTable(
                header_row=["EQ Dimension", "Score / Theme",
                            "Leadership Meaning"],
                rows=rows,
                col_widths=[0.26, 0.16, 0.58],
            ),
        ]
    blocks.append(CalloutBox(
        heading="EQ Practice",
        body="Before responding, ask: What am I noticing? What needs to "
             "be named? What does this person need from me - direction, "
             "support, or a sounding board?",
        tint=PALETTE["tint_teal"],
    ))
    return blocks


def _flywheel_connection_page(profile, flywheel):
    poss = possessive(profile["participant_name"])
    name = first_name(profile["participant_name"])
    q_rows = []
    for qid in ["direction", "culture", "learning", "execution"]:
        q = flywheel["quadrants"][qid]
        q_rows.append(TableRow([q["label"], q["strengths"][0], q["shift"]]))
    return [
        HeaderBand(title="FLYWHEEL ROADMAP CONNECTION",
                   subtitle=f"Where {name} creates and sustains momentum",
                   brand_tag=BRAND_TAG),
        Paragraph(
            f"Flywheel alignment shows how {poss} DISC and EQ-i patterns "
            f"translate into team momentum. {flywheel['diagnostic']}"
        ),
        DataTable(
            header_row=None,
            rows=[
                TableRow(["Primary Strength",
                          f"{flywheel['primary_label']}: " +
                          flywheel["primary_q"]["strengths"][0]],
                         fill=PALETTE["tint_blue"]),
                TableRow(["Secondary Strength",
                          f"{flywheel['secondary_q']['label']}: " +
                          flywheel["secondary_q"]["strengths"][0]],
                         fill=PALETTE["tint_gold_2"]),
            ],
        ),
        Paragraph(
            "The development risk is that strong personal contribution "
            "can become dependency if others are not coached into "
            f"ownership. {name}'s Flywheel work is to ensure strengths "
            "are transferred, not just protected."
        ),
        DataTable(
            header_row=["Flywheel Quadrant",
                        f"{name}'s Current Strength",
                        "Development Stretch"],
            rows=q_rows,
            col_widths=[0.26, 0.40, 0.34],
        ),
        CalloutBox(
            heading="Coaching Goal",
            body=flywheel["activation_focus"],
            tint=PALETTE["tint_gold_1"],
        ),
    ]


def _signature_connection_page(profile, signature):
    poss = possessive(profile["participant_name"])
    name = first_name(profile["participant_name"])
    return [
        HeaderBand(title="LEADERSHIP SIGNATURE CONNECTION",
                   subtitle=f"Making {poss} leadership impact observable",
                   brand_tag=BRAND_TAG),
        Paragraph(
            "Leadership Signature is the integration point. It translates "
            "DISC behavior, EQ capacity, and Flywheel momentum into the "
            "leadership experience supervisors and peers should "
            f"consistently have with {name}."
        ),
        CalloutBox(heading=f"{poss} Leadership Signature",
                   body=signature["statement"],
                   tint=PALETTE["tint_mauve"]),
        Paragraph("Make the Signature Observable", bold=True, size=15),
        DataTable(
            header_row=["Signature Element", "Observable Behavior"],
            rows=[TableRow([el, beh])
                  for el, beh in signature["elements"]],
            col_widths=[0.32, 0.68],
        ),
        CalloutBox(
            heading="Reflection Prompt",
            body="What part of this signature is already visible, and "
                 "what part needs to become more consistent under "
                 "pressure?",
            tint=PALETTE["tint_slate_2"],
        ),
    ]


def _coaching_plan_page_1(profile, plan):
    poss = possessive(profile["participant_name"])
    name = first_name(profile["participant_name"])
    return [
        HeaderBand(title="30-60-90 COACHING PLAN",
                   subtitle="Takeaway implementation guide",
                   brand_tag=BRAND_TAG),
        Paragraph(
            f"This plan is the post-workshop takeaway. It translates "
            f"{poss} Roadmap and workshop reflections into a 90-day "
            "implementation rhythm focused on visible leadership "
            "presence, supervisor coaching, shared ownership, and "
            "sustained Flywheel momentum."
        ),
        DataTable(
            header_row=["Timeline", "Primary Coaching Focus",
                        "DISC Component", "EQ Component",
                        "Flywheel Component"],
            rows=plan["phases"],
            col_widths=[0.11, 0.20, 0.24, 0.24, 0.21],
        ),
        Paragraph("30-Day Action Commitments", bold=True, size=15),
        DataTable(
            header_row=["Action", f"{name}'s Commitment"],
            rows=plan["d30"],
            col_widths=[0.30, 0.70],
        ),
    ]


def _coaching_plan_page_2(profile, plan):
    name = first_name(profile["participant_name"])
    return [
        HeaderBand(title="30-60-90 COACHING PLAN",
                   subtitle="60-day and 90-day implementation",
                   brand_tag=BRAND_TAG),
        Paragraph("60-Day Action Commitments", bold=True, size=15),
        DataTable(header_row=["Action", f"{name}'s Commitment"],
                  rows=plan["d60"], col_widths=[0.30, 0.70]),
        Paragraph("90-Day Action Commitments", bold=True, size=15),
        DataTable(header_row=["Action", f"{name}'s Commitment"],
                  rows=plan["d90"], col_widths=[0.30, 0.70]),
        CalloutBox(heading="90-Day Success Indicator",
                   body=plan["success"], tint=PALETTE["tint_teal"]),
    ]


# ─────────────────────────────────────────
# Top-level orchestrator
# ─────────────────────────────────────────
def _display_name(raw: str) -> str:
    """Title-case names that arrive all-lower/all-upper from PDF parsing
    (e.g. 'katrina speights', 'MAURICE BROOKS'); leave mixed-case names
    like 'LaShawn Royal-Moore' untouched."""
    raw = (raw or "").strip()
    if raw == raw.lower() or raw == raw.upper():
        return raw.title()
    return raw


def generate_roadmap_document(profile: dict) -> RoadmapDocument:
    """Assemble the full 24-page booklet for one participant profile."""
    profile = dict(profile,
                   participant_name=_display_name(profile["participant_name"]))
    person_name = profile["participant_name"]

    disc_text = build_disc_lens_text(profile)
    eqi_text  = build_eqi_lens_text(profile)          # None if no EQ-i
    flywheel  = build_flywheel_section(profile)
    signature = build_leadership_signature(profile, disc_text, flywheel)
    summary   = build_executive_summary(profile, disc_text, eqi_text,
                                        flywheel, signature)
    snapshot  = build_roadmap_snapshot_rows(profile, disc_text, eqi_text,
                                            flywheel)
    plan      = build_coaching_plan(profile, disc_text, eqi_text, flywheel)

    disc_intro_sentence = (
        f"For {first_name(person_name)}, DISC provides a language for "
        f"converting {disc_text['natural_strengths'].split(',')[0]} into "
        "visible direction, coaching, and accountability."
    )
    eqi_intro_sentence = (
        f"For {first_name(person_name)}, EQ bridges internal capacity "
        "with more visible emotional expression, relational connection, "
        "and coaching presence."
    )

    doc = RoadmapDocument(person_name=person_name)

    # Pages 1-3: static framing
    doc.add_page(*get_cover_boilerplate(person_name))
    doc.add_page(*get_how_to_use_page(person_name))
    doc.add_page(*get_framework_overview_page(person_name))

    # Pages 4-5: personalized summary + snapshot
    doc.add_page(*_exec_summary_page(profile, summary, signature))
    doc.add_page(*_snapshot_page(profile, snapshot))

    # Pages 6-9: DISC section
    doc.add_page(*get_section_intro_page("disc", person_name,
                                         disc_intro_sentence))
    doc.add_page(*_disc_connection_page(profile, disc_text))
    doc.add_page(*get_worksheet_page("disc", person_name))
    doc.add_page(*get_notes_page("disc"))

    # Pages 10-13: EQ section
    doc.add_page(*get_section_intro_page("eqi", person_name,
                                         eqi_intro_sentence))
    doc.add_page(*_eqi_connection_page(profile, eqi_text))
    doc.add_page(*get_worksheet_page("eqi", person_name))
    doc.add_page(*get_notes_page("eqi"))

    # Pages 14-17: Flywheel section
    doc.add_page(*get_section_intro_page("flywheel", person_name))
    doc.add_page(*_flywheel_connection_page(profile, flywheel))
    doc.add_page(*get_worksheet_page("flywheel", person_name))
    doc.add_page(*get_notes_page("flywheel"))

    # Pages 18-21: Leadership Signature section
    doc.add_page(*get_section_intro_page("signature", person_name))
    doc.add_page(*_signature_connection_page(profile, signature))
    doc.add_page(*get_worksheet_page("signature", person_name))
    doc.add_page(*get_notes_page("signature"))

    # Pages 22-23: 30-60-90 plan
    doc.add_page(*_coaching_plan_page_1(profile, plan))
    doc.add_page(*_coaching_plan_page_2(profile, plan))

    # Page 24: closing commitment (no trailing page break)
    doc.add(*get_closing_commitment_page(person_name))

    return doc
