"""
EQ-i 2.0 leadership benchmark bars
===================================
On page 6 of every EQ-i 2.0 Leadership Report each subscale carries a gold
bar behind the score bar — the range typical of the leadership norm group.
A subscale belongs in the Roadmap's **Core EQ-i Strengths** chart when the
score reaches that bar, and in **Key Development Areas** when it falls
beneath it. Together the two charts account for all fifteen subscales.

Ranking by raw score is wrong: the bar sits in a different place for every
subscale, so two people can share a score of 108 and land on opposite sides
(Impulse Control's bar opens at 104, Social Responsibility's at 109).

The bars are a property of the instrument, not of the person. Measured off
three unrelated client reports (Ariel Alston, Sonia De Escobar, Thomas
Davis) the pixel positions were identical, so they are constants here rather
than something to re-extract per PDF. Calibration residual was under 0.1
score points, and the integer thresholds below reproduce a direct
pixel-by-pixel reading of all fifteen rows exactly.
"""

# subscale key -> score at which the gold leadership bar begins
LEADERSHIP_BAR = {
    "self_regard":                 107,
    "self_actualization":          111,
    "emotional_self_awareness":    106,
    "emotional_expression":        107,
    "assertiveness":               109,
    "independence":                111,
    "interpersonal_relationships": 107,
    "empathy":                     103,
    "social_responsibility":       109,
    "problem_solving":             109,
    "reality_testing":             109,
    "impulse_control":             104,
    "flexibility":                 106,
    "stress_tolerance":            110,
    "optimism":                    108,
}

# display labels, in the order the report lists them on page 6
SUBSCALE_LABELS = {
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


def reaches_bar(subscale: str, score) -> bool:
    """True when the score meets or exceeds its leadership bar."""
    bar = LEADERSHIP_BAR.get(subscale)
    if bar is None or score is None:
        return False
    return score >= bar


def split_subscales(eqi_scores: dict):
    """Partition the fifteen subscales against their leadership bars.

    Returns ``(strengths, development)``; each is a list of
    ``(label, score, bar)`` sorted by score, strengths descending and
    development areas ascending, which is the order the Roadmap charts
    them in. Subscales missing from ``eqi_scores`` are skipped, so a
    partial EQ-i result degrades to a shorter chart rather than raising.
    """
    strengths, development = [], []
    for key, bar in LEADERSHIP_BAR.items():
        score = (eqi_scores or {}).get(key)
        if score is None:
            continue
        row = (SUBSCALE_LABELS[key], score, bar)
        (strengths if score >= bar else development).append(row)

    strengths.sort(key=lambda r: -r[1])
    development.sort(key=lambda r: r[1])
    return strengths, development
