"""
DISC–EQI Correlation Engine
============================
Maps each DISC style to its correlated EQ-i 2.0 subscales
and provides action steps for the bottom 3 underdeveloped subscales.
"""
from typing import Dict, List, Optional

# ── Style metadata ──────────────────────────────────────────────────────────

STYLE_NAMES = {
    "D": "Dominance",
    "I": "Influence",
    "S": "Steadiness",
    "C": "Compliance",
}

STYLE_DESCS = {
    "D": "Results-oriented, decisive, direct, and competitive",
    "I": "Enthusiastic, optimistic, collaborative, and expressive",
    "S": "Patient, reliable, collaborative, and supportive",
    "C": "Analytical, systematic, quality-focused, and precise",
}

# ── DISC → EQI correlations ─────────────────────────────────────────────────
# Each entry: (snake_key, display_label, correlation_description, is_inverse)

DISC_EQI_CORRELATIONS: Dict[str, List[Dict]] = {
    "D": [
        {
            "key":     "assertiveness",
            "label":   "Assertiveness",
            "note":    "Both measure directness and confidence in expression.",
            "inverse": False,
        },
        {
            "key":     "independence",
            "label":   "Independence",
            "note":    "High D individuals resist external influence, matching the Independence subscale.",
            "inverse": False,
        },
        {
            "key":     "problem_solving",
            "label":   "Problem Solving",
            "note":    "High D leaders move fast toward solutions.",
            "inverse": False,
        },
        {
            "key":     "stress_tolerance",
            "label":   "Stress Tolerance",
            "note":    "Very high D with low Stress Tolerance creates volatility under pressure.",
            "inverse": True,
        },
    ],
    "I": [
        {
            "key":     "interpersonal_relationships",
            "label":   "Interpersonal Relationships",
            "note":    "I styles build rapport naturally.",
            "inverse": False,
        },
        {
            "key":     "empathy",
            "label":   "Empathy",
            "note":    "I styles read social cues to persuade.",
            "inverse": False,
        },
        {
            "key":     "emotional_expression",
            "label":   "Emotional Expression",
            "note":    "I styles display emotions openly.",
            "inverse": False,
        },
        {
            "key":     "optimism",
            "label":   "Optimism",
            "note":    "I style and Optimism are almost always aligned.",
            "inverse": False,
        },
    ],
    "S": [
        {
            "key":     "impulse_control",
            "label":   "Impulse Control",
            "note":    "S styles pause before acting.",
            "inverse": False,
        },
        {
            "key":     "empathy",
            "label":   "Empathy",
            "note":    "S styles listen deeply.",
            "inverse": False,
        },
        {
            "key":     "flexibility",
            "label":   "Flexibility",
            "note":    "Very high S with low Flexibility creates resistance to change.",
            "inverse": True,
        },
        {
            "key":     "stress_tolerance",
            "label":   "Stress Tolerance",
            "note":    "S styles generally absorb pressure well.",
            "inverse": False,
        },
    ],
    "C": [
        {
            "key":     "problem_solving",
            "label":   "Problem Solving",
            "note":    "C styles analyze before deciding.",
            "inverse": False,
        },
        {
            "key":     "reality_testing",
            "label":   "Reality Testing",
            "note":    "C styles verify before committing.",
            "inverse": False,
        },
        {
            "key":     "emotional_self_awareness",
            "label":   "Emotional Self-Awareness",
            "note":    "High C styles often suppress emotional expression.",
            "inverse": True,
        },
        {
            "key":     "independence",
            "label":   "Independence",
            "note":    "C styles rely on their own analysis.",
            "inverse": False,
        },
    ],
}

# ── Action steps for every subscale ────────────────────────────────────────
# 3 concrete, first-person actions per subscale

SUBSCALE_ACTIONS: Dict[str, List[str]] = {
    "self_regard": [
        "Write down three things you did well at the end of each workday — consistency builds a realistic, positive self-image.",
        "When you make a mistake, distinguish between the error and your worth: critique the decision, not yourself.",
        "Ask a trusted colleague for specific, positive feedback on a strength you underestimate.",
    ],
    "self_actualization": [
        "Identify one professionally meaningful goal this quarter and break it into weekly milestones.",
        "Block 30 minutes weekly for deliberate learning — a course, book, or new skill tied to work you find meaningful.",
        "Reflect monthly on whether your daily tasks still connect to what motivates you; adjust where they don't.",
    ],
    "emotional_self_awareness": [
        "Before important meetings, pause and name your emotional state; notice how it might colour your reactions.",
        "Keep a brief emotion log for two weeks — note the trigger, the feeling, and the effect on your behaviour.",
        "When your mood shifts unexpectedly, ask 'what just happened?' rather than suppressing or ignoring it.",
    ],
    "emotional_expression": [
        "Practice using 'I feel…' statements once per day in low-stakes conversations to build comfort.",
        "After positive interactions, briefly name what made them feel good — this trains outward emotional vocabulary.",
        "Find one trusted relationship at work where you can share genuine reactions, not just professional ones.",
    ],
    "assertiveness": [
        "When you disagree, state your position once clearly and directly before softening or qualifying it.",
        "Rehearse a brief script for saying no: 'I can't take that on right now, but here's what I can do…'",
        "In your next meeting, offer your opinion before asking others — resist the urge to read the room first.",
    ],
    "independence": [
        "Make one medium-stakes decision this week using only your own judgment — then evaluate the outcome honestly.",
        "Notice when you seek reassurance before it's necessary and delay asking for 30 minutes to see if you can resolve it alone.",
        "Build your own criteria for what 'good enough' looks like so you don't need external benchmarks to feel confident.",
    ],
    "interpersonal_relationships": [
        "Schedule one 15-minute check-in with a colleague each week with no work agenda — just connection.",
        "Follow up on something personal a colleague shared last week; remembering shows investment.",
        "In the next conflict or tension, lead with curiosity about the other person's experience before stating your own.",
    ],
    "empathy": [
        "In your next difficult conversation, summarise what the other person said before responding — check that you got it right.",
        "When someone is frustrated, resist problem-solving immediately; ask 'what would be most helpful right now?'",
        "Practice noticing nonverbal cues in meetings — posture, tone, eye contact — and adjust your approach accordingly.",
    ],
    "social_responsibility": [
        "Identify one way your work positively impacts someone else this week and name it explicitly.",
        "Volunteer for one initiative that benefits the team or organisation beyond your direct role.",
        "When making decisions, ask 'who else is affected by this?' before finalising your approach.",
    ],
    "problem_solving": [
        "Before proposing a solution, write down the problem in one sentence — if you can't, keep clarifying.",
        "When emotions run high in a problem, separate the facts from the feelings and address both deliberately.",
        "Try a structured approach: define the problem, list options, consider emotional implications, then decide.",
    ],
    "reality_testing": [
        "Before acting on a strong feeling or assumption, ask 'what is the evidence for this?' — then seek one piece of contrary evidence.",
        "When planning, identify one thing that could go wrong and build a contingency — this grounds optimism in reality.",
        "Seek one outside perspective on your next major decision before committing.",
    ],
    "impulse_control": [
        "Introduce a 24-hour rule for any non-urgent commitment or reactive response — most situations improve with a pause.",
        "When provoked, use a physical anchor (deep breath, standing up) to create space between stimulus and response.",
        "Before sending a charged email or message, read it aloud and ask 'would I say this face to face?'",
    ],
    "flexibility": [
        "Deliberately try one new approach to a routine task this week — the content doesn't matter, the stretch does.",
        "When a plan changes unexpectedly, write down one potential benefit of the new direction before reacting.",
        "Seek out one opinion that contradicts your current view on a work issue and engage with it seriously.",
    ],
    "stress_tolerance": [
        "Identify your top two stress triggers and build a specific pre-planned response for each.",
        "Build a short daily recovery routine (even 10 minutes) — stress tolerance depends on recovery, not just endurance.",
        "Reframe one current pressure point as a challenge rather than a threat — language changes physiology.",
    ],
    "optimism": [
        "End each day by naming one thing that went well, however small — this is not denial, it's balance.",
        "When facing a setback, ask 'what can I control from here?' and focus your next action there.",
        "Separate what is permanent from what is temporary in a current difficulty — most obstacles are temporary.",
    ],
}

# ── Display label lookup ────────────────────────────────────────────────────

SUBSCALE_DISPLAY: Dict[str, str] = {
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


# ── Public API ──────────────────────────────────────────────────────────────

def eq_level_short(score: int) -> str:
    if score >= 110: return "Strength"
    if score >= 100: return "Effective"
    if score >= 90:  return "Growth Area"
    return "Priority Growth"


def generate_insights(primary_style: str, eqi_scores: Dict) -> Dict:
    """
    Returns:
      correlations  — list of DISC-EQI correlation dicts for this style
      bottom_three  — 3 lowest subscale scores with action steps
    """
    if primary_style not in DISC_EQI_CORRELATIONS:
        return {}

    # Snake-case subscale scores only
    subscales = {
        k: int(v) for k, v in eqi_scores.items()
        if k in SUBSCALE_DISPLAY and isinstance(v, (int, float))
    }

    if not subscales:
        return {}

    # Bottom 3 subscales by score
    ranked = sorted(subscales.items(), key=lambda x: x[1])
    bottom_three = [
        {
            "key":     k,
            "label":   SUBSCALE_DISPLAY[k],
            "score":   v,
            "level":   eq_level_short(v),
            "actions": SUBSCALE_ACTIONS.get(k, []),
        }
        for k, v in ranked[:3]
    ]

    # Enrich correlations with the person's actual score
    correlations = []
    for corr in DISC_EQI_CORRELATIONS[primary_style]:
        entry = dict(corr)
        entry["score"] = subscales.get(corr["key"])
        correlations.append(entry)

    return {
        "primary_style": primary_style,
        "style_name":    STYLE_NAMES.get(primary_style, primary_style),
        "style_desc":    STYLE_DESCS.get(primary_style, ""),
        "correlations":  correlations,
        "bottom_three":  bottom_three,
    }


def generate_summary_insights(primary_style: str, eqi_scores: Dict) -> Dict:
    """Condensed version for comparison cards."""
    if not primary_style or primary_style not in DISC_EQI_CORRELATIONS:
        return {}
    full = generate_insights(primary_style, eqi_scores)
    if not full:
        return {}
    return {
        "style_name":   full["style_name"],
        "bottom_three": full["bottom_three"],
    }
