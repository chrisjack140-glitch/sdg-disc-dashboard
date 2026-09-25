"""
DISC radar — the definitions shared by every version of the chart
==================================================================
The dashboard's interactive radar (app.py, Plotly) and the Roadmap's print
image (utils/radar_image.py, matplotlib) both read these, so the axis order,
behaviour symbols and graph colours cannot drift apart. Kept free of heavy
imports so the dashboard can load it without pulling in matplotlib.
"""

# Axes clockwise from the top; blends sit between their two factors.
RADAR_CATEGORIES = ["DI", "I", "IS", "S", "SC", "C", "CD", "D"]

# (axis, icon file stem in assets/radar_icons/<theme>/, label)
RADAR_AXIS_ICONS = [
    ("DI", "DI_persuade_others",      "Persuade Others"),
    ("I",  "I_verbalize_communicate", "Verbalize, Communicate"),
    ("IS", "IS_build_relationships",  "Build<br>Relationships"),
    ("S",  "S_keep_the_peace",        "Keep the Peace"),
    ("SC", "SC_follow_a_process",     "Follow A Process"),
    ("C",  "C_analyze_the_problem",   "Analyze the Problem"),
    ("CD", "CD_design_a_solution",    "Design a<br>Solution"),
    ("D",  "D_take_action_now",       "Take Action Now"),
]

# One person's three graphs. Colours avoid the red/yellow/green/blue that
# already mean D/I/S/C; lines are solid, so distinct marker shapes keep them
# distinguishable in print and for colour-blind readers.
GRAPH_OVERLAY_STYLE = {
    "public": {"label": "Public", "color": "#bc8cff", "symbol": "circle"},
    "stress": {"label": "Stress", "color": "#fb7185", "symbol": "diamond"},
    "mirror": {"label": "Mirror", "color": "#22d3ee", "symbol": "square"},
}

RADIAL_MIN, RADIAL_MAX = -8, 8


def radar_values(g: dict) -> list:
    """D/I/S/C scores -> the eight radar axes (blends are pairwise means)."""
    d, i, s, cv = g["D"], g["I"], g["S"], g["C"]
    return [(d + i) / 2, i, (i + s) / 2, s, (s + cv) / 2, cv, (cv + d) / 2, d]
