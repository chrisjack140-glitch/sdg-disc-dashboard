"""
DISC–EQI Correlation Engine
============================
Generates coaching insights that link a person's primary DISC style
to their EQ-i 2.0 subscale scores.

All subscale keys are snake_case to match EQI_COMPOSITES in app.py.
"""
from typing import Dict, List, Optional

# ── Knowledge base ─────────────────────────────────────────────────────────

STYLE_NAMES = {
    "D": "Dominance",
    "I": "Influence",
    "S": "Steadiness",
    "C": "Conscientiousness",
}

STYLE_DESCS = {
    "D": "Results-oriented, decisive, direct, and competitive",
    "I": "Enthusiastic, optimistic, collaborative, and expressive",
    "S": "Patient, reliable, collaborative, and supportive",
    "C": "Analytical, systematic, quality-focused, and precise",
}

# EQI subscales most aligned with each DISC style (typical strengths)
ALIGNED: Dict[str, List[str]] = {
    "D": ["independence", "problem_solving", "self_actualization",
          "self_regard", "reality_testing"],
    "I": ["optimism", "emotional_expression", "interpersonal_relationships",
          "empathy", "social_responsibility"],
    "S": ["empathy", "interpersonal_relationships", "stress_tolerance",
          "social_responsibility", "impulse_control"],
    "C": ["reality_testing", "problem_solving", "self_regard",
          "independence", "impulse_control"],
}

# EQI subscales most commonly underdeveloped for each DISC style
GROWTH: Dict[str, List[str]] = {
    "D": ["empathy", "interpersonal_relationships", "impulse_control",
          "assertiveness", "emotional_expression"],
    "I": ["reality_testing", "impulse_control", "independence",
          "flexibility", "assertiveness"],
    "S": ["assertiveness", "flexibility", "self_actualization",
          "emotional_expression", "independence"],
    "C": ["emotional_expression", "interpersonal_relationships",
          "assertiveness", "optimism", "empathy"],
}

# 3-4 sentence connection narrative per DISC style
CONNECTIONS: Dict[str, str] = {
    "D": (
        "D-style individuals are naturally driven and decisive — traits that often produce strong "
        "Problem Solving, Independence, and Self-Actualization scores. The same directness that "
        "powers D-style effectiveness can compress the time spent reading others' emotional states, "
        "making Empathy and Interpersonal Relationships the most common underdeveloped areas. "
        "Investing in these relational EQ skills unlocks a significant multiplier: people follow "
        "leaders who both set direction and make others feel seen."
    ),
    "I": (
        "I-style individuals bring natural warmth and social energy that map strongly onto high "
        "Interpersonal Relationships, Optimism, and Emotional Expression. The growth edge typically "
        "sits at the opposite end: Reality Testing, Impulse Control, and follow-through can lag "
        "behind the pace of new ideas and social commitments. Developing these areas doesn't "
        "dampen enthusiasm — it gives that enthusiasm staying power and lasting credibility."
    ),
    "S": (
        "S-style individuals are the anchors of their teams — reliable, empathic, and deeply "
        "invested in others' wellbeing. These qualities produce strong Empathy, Interpersonal "
        "Relationships, and Stress Tolerance scores. The development edge is Assertiveness: "
        "the strong preference for harmony can make it difficult to advocate clearly for their "
        "own needs or challenge the status quo, and Flexibility often follows close behind."
    ),
    "C": (
        "C-style individuals bring intellectual rigor and precision that align closely with "
        "strong Reality Testing and Problem Solving scores. The growth edge is relational: "
        "Emotional Expression and Interpersonal Relationships are the subscales most commonly "
        "underdeveloped, as analytical precision can create emotional distance. Developing "
        "these areas makes C-style insights far more persuasive and their leadership more visible."
    ),
}

# Personalised coaching tip per DISC style per growth subscale
GROWTH_TIPS: Dict[str, Dict[str, str]] = {
    "D": {
        "empathy":
            "Pause to acknowledge how people feel before moving to solutions — "
            "even 30 seconds changes the dynamic significantly and reduces resistance.",
        "interpersonal_relationships":
            "Invest in deeper one-on-one bonds, not just transactional interactions. "
            "That relational capital pays out when you need candid feedback or loyalty.",
        "impulse_control":
            "Develop the pause between stimulus and response. "
            "Composure under pressure amplifies your authority more than speed does.",
        "assertiveness":
            "For a high-D, assertiveness development is about inviting collaboration "
            "rather than compliance — better ideas come from both, and buy-in lasts longer.",
        "emotional_expression":
            "Sharing the reasoning behind your decisions — not just the decisions — "
            "builds trust and makes your leadership more persuasive.",
    },
    "I": {
        "reality_testing":
            "Ground your ideas in objective data before sharing them publicly. "
            "Credibility compounds when enthusiasm is backed by assessment.",
        "impulse_control":
            "Build a brief structured pause before committing aloud. "
            "It preserves your creative reputation while reducing over-promising.",
        "independence":
            "Develop confidence in your own judgment without needing external validation "
            "— this accelerates your impact and deepens others' confidence in you.",
        "flexibility":
            "Bring your natural social adaptability to systems and processes as well. "
            "Flexibility around structure keeps you effective when the environment changes fast.",
        "assertiveness":
            "Your agreeableness is relational gold — learning to hold ground on important "
            "positions prevents your voice from being undervalued in high-stakes moments.",
    },
    "S": {
        "assertiveness":
            "Advocating clearly for your own ideas isn't confrontation — "
            "it's giving your team the full benefit of your actual perspective.",
        "flexibility":
            "Stretch your tolerance for ambiguity and rapid change. "
            "Your consistency can still be your anchor without limiting your range.",
        "self_actualization":
            "Apply the same investment you give others' development to your own goals — "
            "your sense of purpose and satisfaction depend on it.",
        "emotional_expression":
            "Opening up emotionally allows others to truly support you, "
            "not just receive your support — that reciprocity strengthens the team.",
        "independence":
            "Developing comfort making decisions without full consensus reduces your "
            "coordination burden and builds the decisiveness others are looking for.",
    },
    "C": {
        "emotional_expression":
            "Strategic transparency — sharing what's behind your analysis — "
            "builds the kind of trust that makes your insights land with people, not just data.",
        "interpersonal_relationships":
            "Moving from transactional to relational interactions gives you access to "
            "informal intelligence and advocacy that no analysis can replicate.",
        "assertiveness":
            "Your well-researched positions deserve to be heard directly. "
            "Reducing over-qualifying will increase your organizational influence immediately.",
        "optimism":
            "Deliberately balancing risk-focus with recognition of what's working "
            "improves your resilience and how others experience working with you.",
        "empathy":
            "Pairing your objectivity with active curiosity about how others experience "
            "a situation improves both the quality of decisions and your interpersonal reach.",
    },
}

# Snake_case → display label for rendering
SUBSCALE_DISPLAY: Dict[str, str] = {
    "self_regard":                  "Self-Regard",
    "self_actualization":           "Self-Actualization",
    "emotional_self_awareness":     "Emotional Self-Awareness",
    "emotional_expression":         "Emotional Expression",
    "assertiveness":                "Assertiveness",
    "independence":                 "Independence",
    "interpersonal_relationships":  "Interpersonal Relationships",
    "empathy":                      "Empathy",
    "social_responsibility":        "Social Responsibility",
    "problem_solving":              "Problem Solving",
    "reality_testing":              "Reality Testing",
    "impulse_control":              "Impulse Control",
    "flexibility":                  "Flexibility",
    "stress_tolerance":             "Stress Tolerance",
    "optimism":                     "Optimism",
}


# ── Public API ──────────────────────────────────────────────────────────────

def eq_level_short(score: int) -> str:
    if score >= 110: return "Strength"
    if score >= 100: return "Effective"
    if score >= 90:  return "Growth Area"
    return "Priority Growth"


def generate_insights(primary_style: str, eqi_scores: Dict) -> Dict:
    """
    Full DISC–EQI insights dict.
    primary_style: single letter "D", "I", "S", or "C"
    eqi_scores:    dict with snake_case subscale keys (from profile["eqi_scores"])

    Returns empty dict if primary_style not in knowledge base.
    """
    if primary_style not in CONNECTIONS:
        return {}

    # Filter to known subscale keys only
    subscales = {k: v for k, v in eqi_scores.items() if k in SUBSCALE_DISPLAY}
    underdeveloped = {k: v for k, v in subscales.items() if isinstance(v, (int, float)) and v < 100}

    # Priority growth: predicted growth subscales that are actually underdeveloped
    priority_growth: List[Dict] = []
    for key in GROWTH.get(primary_style, []):
        if key in underdeveloped:
            score = underdeveloped[key]
            priority_growth.append({
                "key":   key,
                "label": SUBSCALE_DISPLAY[key],
                "score": score,
                "level": eq_level_short(score),
                "tip":   GROWTH_TIPS.get(primary_style, {}).get(key, ""),
            })

    # Confirmed strengths: aligned subscales scoring >= 105
    confirmed_strengths: List[Dict] = []
    for key in ALIGNED.get(primary_style, []):
        if key in subscales and isinstance(subscales[key], (int, float)) and subscales[key] >= 105:
            confirmed_strengths.append({
                "key":   key,
                "label": SUBSCALE_DISPLAY[key],
                "score": subscales[key],
            })

    # Unexpected gaps: underdeveloped but NOT in predicted growth list
    unexpected_gaps: List[Dict] = []
    for key, score in underdeveloped.items():
        if key not in GROWTH.get(primary_style, []):
            unexpected_gaps.append({
                "key":   key,
                "label": SUBSCALE_DISPLAY[key],
                "score": score,
            })

    return {
        "primary_style":       primary_style,
        "style_name":          STYLE_NAMES.get(primary_style, primary_style),
        "style_desc":          STYLE_DESCS.get(primary_style, ""),
        "connection":          CONNECTIONS.get(primary_style, ""),
        "priority_growth":     priority_growth,
        "confirmed_strengths": confirmed_strengths,
        "unexpected_gaps":     unexpected_gaps,
    }


def generate_summary_insights(primary_style: str, eqi_scores: Dict) -> Dict:
    """
    Compact version for comparison cards.
    Returns style info, connection, top growth area, and gap count.
    """
    if not primary_style or primary_style not in CONNECTIONS:
        return {}
    full = generate_insights(primary_style, eqi_scores)
    return {
        "style_name": full.get("style_name", ""),
        "connection":  full.get("connection", ""),
        "top_growth":  full["priority_growth"][0] if full.get("priority_growth") else None,
        "n_gaps":      len(full.get("priority_growth", [])),
    }
