import io
import json
from collections import Counter
from datetime import datetime
from typing import Optional

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from utils.disc import (
    FACTORS, GRAPHS, FACTOR_COLORS,
    decode_upload, process_uploaded_files,
)
from utils.insights import (
    generate_insights, generate_summary_insights,
    SUBSCALE_DISPLAY, STYLE_NAMES,
    SUBSCALE_DISC_MAP, COMPOSITE_SUBSCALE_ORDER,
)

# ═══════════════════════════════════════════════════════════════
# SINGLE SOURCE OF TRUTH — ALL COLOURS DEFINED HERE
# Any colour used anywhere in the app must come from this dict.
# Never hardcode a hex value outside of this block.
# ═══════════════════════════════════════════════════════════════
THEME = {
    # ── Dark mode (default) — warm near-black + SDG gold accent ──
    "dark": {
        "bg":           "#08080f",   # warm near-black page background
        "surface":      "#0f0f18",   # card / panel surface
        "surface2":     "#16161f",   # elevated inner panels
        "border":       "#26263a",   # borders and dividers
        "text":         "#e8e0d0",   # warm off-white primary text
        "muted":        "#7a7490",   # warm muted secondary text
        "accent":       "#c9a535",   # SDG gold — primary brand accent
        "green":        "#3fb950",   # positive / upward
        "red":          "#f85149",   # negative / downward / D factor
        "gold":         "#d29922",   # I factor (distinct warm gold)
        "purple":       "#bc8cff",   # EQI / Total EQ
        "cyan":         "#39d353",   # extra palette
        "shadow_sm":    "rgba(0,0,0,0.45)",
        "shadow_md":    "rgba(0,0,0,0.60)",
        "shadow_lg":    "rgba(0,0,0,0.80)",
    },
    # ── Light mode ─────────────────────────────────────────────
    "light": {
        "bg":           "#fdfcf8",   # warm white
        "surface":      "#f5f3ee",   # warm card surface
        "surface2":     "#ece9e2",   # warm elevated panels
        "border":       "#d4cfc4",   # warm border
        "text":         "#1a1814",   # warm near-black text
        "muted":        "#6b6458",   # warm muted text
        "accent":       "#a07c1a",   # SDG gold adjusted for light bg
        "green":        "#1a7f37",
        "red":          "#cf222e",
        "gold":         "#9a6700",
        "purple":       "#8250df",
        "cyan":         "#0550ae",
        "shadow_sm":    "rgba(0,0,0,0.04)",
        "shadow_md":    "rgba(0,0,0,0.08)",
        "shadow_lg":    "rgba(0,0,0,0.14)",
    },
    # ── DISC factor colours (same in both themes) ───────────────
    "disc": {
        "D": "#f85149",
        "I": "#d29922",
        "S": "#3fb950",
        "C": "#58a6ff",
    },
    # ── EQ-i composite colours (same in both themes) ────────────
    "eqi": {
        "Self-Perception":   "#f97316",
        "Self-Expression":   "#a78bfa",
        "Interpersonal":     "#34d399",
        "Decision Making":   "#38bdf8",
        "Stress Management": "#fbbf24",
    },
    # ── Radar palette — 15 unique before any repeat ─────────────
    "radar": [
        "#f85149", "#58a6ff", "#3fb950", "#d29922",
        "#bc8cff", "#39d353", "#fb7185", "#f97316",
        "#22d3ee", "#a78bfa", "#34d399", "#fbbf24",
        "#e879f9", "#38bdf8", "#4ade80",
    ],
    # ── Badge / medal colours ───────────────────────────────────
    "gold_badge":   "#d29922",
    "silver_badge": "#8b949e",
    "bronze_badge": "#c0640a",
}

# EQ-i subscale → composite mapping
EQI_COMPOSITES = {
    "Self-Perception":   ["self_regard", "self_actualization", "emotional_self_awareness"],
    "Self-Expression":   ["emotional_expression", "assertiveness", "independence"],
    "Interpersonal":     ["interpersonal_relationships", "empathy", "social_responsibility"],
    "Decision Making":   ["problem_solving", "reality_testing", "impulse_control"],
    "Stress Management": ["flexibility", "stress_tolerance", "optimism"],
}

EQI_NORM = 100   # EQ-i normative mean
EQI_SD   = 15    # EQ-i normative standard deviation


def T(key: str, theme: str = "dark") -> str:
    """Shorthand: T('accent') returns the accent colour for the given theme."""
    t = "dark" if theme not in ("dark", "light") else theme
    return THEME[t].get(key, THEME["dark"][key])


# ─────────────────────────────────────────
# App init
# ─────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,900;1,400;1,700&display=swap",
    ],
    suppress_callback_exceptions=True,
)
server = app.server


# ─────────────────────────────────────────
# Style helpers — all pull from THEME dict
# ─────────────────────────────────────────

def CARD_STYLE(theme: str = "dark") -> dict:
    return {
        "backgroundColor": T("surface", theme),
        "border":          f"1px solid {T('border', theme)}",
        "borderRadius":    "12px",
        "padding":         "20px",
        "marginBottom":    "16px",
        "boxShadow":       f"0 4px 24px {T('shadow_md', theme)}",
        "transition":      "background-color 0.3s ease, border-color 0.3s ease",
    }


def SECTION_STYLE(theme: str = "dark") -> dict:
    return {
        "backgroundColor": T("surface2", theme),
        "border":          f"1px solid {T('border', theme)}",
        "borderRadius":    "10px",
        "padding":         "14px",
        "marginBottom":    "12px",
        "transition":      "background-color 0.3s ease, border-color 0.3s ease",
    }


def DROPDOWN_STYLE(theme: str = "dark") -> dict:
    return {
        "backgroundColor": T("surface2", theme),
        "color":           T("text", theme),
        "border":          f"1px solid {T('border', theme)}",
        "borderRadius":    "8px",
    }


LABEL_STYLE = {
    "color":          THEME["dark"]["muted"],
    "fontSize":       "11px",
    "fontWeight":     "600",
    "letterSpacing":  "0.06em",
    "textTransform":  "uppercase",
    "marginBottom":   "6px",
    "display":        "block",
}


# ─────────────────────────────────────────
# Chart helpers — theme-aware
# ─────────────────────────────────────────

def _tc(theme: str) -> dict:
    """Return the theme colour dict for the given theme string."""
    return THEME["light"] if theme == "light" else THEME["dark"]


def _base_layout(title: str, height: int = 380,
                 extra: dict = None, theme: str = "dark") -> dict:
    c = _tc(theme)
    layout = dict(
        paper_bgcolor=c["surface"],
        plot_bgcolor=c["surface"],
        font=dict(color=c["text"], family="Inter, system-ui, sans-serif"),
        title=dict(
            text=title,
            font=dict(color=c["muted"], size=12, family="Inter, system-ui"),
            x=0.01, xanchor="left",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=c["muted"], size=11),
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        height=height,
        margin=dict(l=44, r=16, t=48, b=36),
        # Plotly transition — gives bars/lines an animated entrance on render
        transition={"duration": 700, "easing": "cubic-in-out"},
    )
    if extra:
        layout.update(extra)
    return layout


def _axis(title_text: str = "", show_grid: bool = True,
          fixed_range: list = None, theme: str = "dark") -> dict:
    c = _tc(theme)
    d = dict(
        title=dict(text=title_text, font=dict(color=c["muted"], size=10)),
        tickfont=dict(color=c["muted"], size=10),
        showgrid=show_grid,
        gridcolor=c["border"] if show_grid else None,
        linecolor=c["border"],
        zeroline=False,
    )
    if fixed_range:
        d["range"] = fixed_range
    return d


# ── Chart builders ─────────────────────────────────────────────

def build_anchor_comparison_chart(df: pd.DataFrame,
                                   anchor_graph: str,
                                   theme: str = "dark") -> go.Figure:
    fig = go.Figure()
    for f in FACTORS:
        fig.add_trace(go.Bar(
            x=df["participant_name"],
            y=df[f"{anchor_graph}_{f}"],
            name=f,
            marker_color=THEME["disc"][f],
            marker_line_width=0,
            opacity=0.9,
            hovertemplate=f"<b>%{{x}}</b><br>{f}: %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(**_base_layout(
        f"Anchor Score Comparison — {anchor_graph.title()}",
        height=340, theme=theme,
        extra=dict(
            barmode="group",
            xaxis=_axis(show_grid=False, theme=theme),
            yaxis=_axis("Score", theme=theme),
            bargap=0.18, bargroupgap=0.04,
        ),
    ))
    return fig


def build_heatmap(df: pd.DataFrame,
                  anchor_graph: str,
                  theme: str = "dark") -> go.Figure:
    c = _tc(theme)
    z = df[[f"{anchor_graph}_{f}" for f in FACTORS]].values
    colorscale = [
        [0.00, "#1e3a8a"], [0.30, "#3b82f6"],
        [0.50, "#f9fafb"],
        [0.70, "#ef4444"], [1.00, "#7f1d1d"],
    ]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=FACTORS, y=df["participant_name"],
        colorscale=colorscale,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>",
        xgap=1, ygap=1,
    ))
    fig.update_layout(**_base_layout(
        f"Score Heatmap — {anchor_graph.title()}",
        height=max(280, 56 * len(df)), theme=theme,
        extra=dict(
            xaxis=_axis(show_grid=False, theme=theme),
            yaxis=_axis(show_grid=False, theme=theme),
            plot_bgcolor="#000000",
            paper_bgcolor=c["surface"],
        ),
    ))
    return fig


def build_disc_type_chart(profiles: list,
                           theme: str = "dark") -> go.Figure:
    c = _tc(theme)
    type_counts = Counter(p.get("style_type", "—") for p in profiles)
    labels  = sorted(type_counts.keys(), key=lambda k: -type_counts[k])
    counts  = [type_counts[k] for k in labels]
    palette = THEME["radar"]
    bar_colors = [palette[i % len(palette)] for i in range(len(labels))]
    fig = go.Figure(go.Bar(
        x=labels, y=counts,
        marker_color=bar_colors, marker_line_width=0, opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        text=counts, textposition="outside",
        textfont=dict(color=c["muted"], size=11),
    ))
    fig.update_layout(**_base_layout(
        "DISC Type Distribution (from PDF)", height=340, theme=theme,
        extra=dict(
            xaxis=_axis("Type", show_grid=False, theme=theme),
            yaxis=_axis("Count", fixed_range=[0, max(counts) + 1.5], theme=theme),
            showlegend=False,
        ),
    ))
    return fig


def build_multi_radar_chart(selected_profiles: list,
                             graph_name: str,
                             theme: str = "dark") -> go.Figure:
    categories = ["DI", "I", "IS", "S", "SC", "C", "CD", "D"]
    c = _tc(theme)
    fig = go.Figure()
    for idx, profile in enumerate(selected_profiles):
        g    = profile["graphs"][graph_name]
        d, i, s, cv = g["D"], g["I"], g["S"], g["C"]
        vals = [(d+i)/2, i, (i+s)/2, s, (s+cv)/2, cv, (cv+d)/2, d]
        color = THEME["radar"][idx % len(THEME["radar"])]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="none",
            name=profile["participant_name"],
            line=dict(color=color, width=2.5),
            opacity=0.9,
            hovertemplate="<b>%{fullData.name}</b><br>%{theta}: %{r:.2f}<extra></extra>",
        ))
    fig.update_layout(**_base_layout(
        f"Radar Comparison — {graph_name.title()}",
        height=520, theme=theme,
        extra=dict(polar=dict(
            bgcolor=c["surface"],
            angularaxis=dict(
                categoryorder="array", categoryarray=categories,
                direction="clockwise", rotation=90,
                gridcolor=c["border"], linecolor=c["border"],
                tickfont=dict(color=c["text"], size=11),
            ),
            radialaxis=dict(
                visible=True, range=[-8, 8],
                tickmode="array", tickvals=[-8, -4, 0, 4, 8],
                ticktext=["-8", "-4", "0", "4", "8"],
                angle=0, tickangle=-90,
                gridcolor=c["border"], linecolor=c["border"],
                tickfont=dict(color=c["muted"], size=9),
            ),
        )),
    ))
    return fig


def build_letter_mean_combo(df: pd.DataFrame, letter: str,
                             anchor_graph: str,
                             theme: str = "dark") -> go.Figure:
    col      = f"{anchor_graph}_{letter}"
    sorted_df = (df[["participant_name", col]]
                 .copy()
                 .sort_values(col, ascending=False))
    names    = sorted_df["participant_name"].tolist()
    scores   = sorted_df[col].tolist()
    mean_val = float(df[col].mean())
    color    = THEME["disc"][letter]
    c        = _tc(theme)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=scores, name=f"{letter} Score",
        marker_color=color, marker_line_width=0, opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=names, y=[mean_val] * len(names),
        mode="lines", name=f"Mean ({mean_val:+.2f})",
        line=dict(color=c["text"], width=1.5, dash="dash"),
        hovertemplate=f"Team Mean: {mean_val:+.2f}<extra></extra>",
    ))
    fig.add_shape(
        type="line", x0=-0.5, x1=len(names) - 0.5, y0=0, y1=0,
        line=dict(color=c["border"], width=1.5),
    )
    fig.update_layout(**_base_layout(
        f"{letter}  —  Scores (Greatest to Least)",
        height=360, theme=theme,
        extra=dict(
            xaxis=_axis(show_grid=False, theme=theme),
            yaxis=_axis("Score", fixed_range=[-8, 8], theme=theme),
        ),
    ))
    return fig


def build_eqi_bar_chart(eqi_scores: dict, theme: str = "dark") -> go.Figure:
    """
    Horizontal bar chart showing all 5 EQ-i composite scores.
    Normative mean (100) shown as a vertical reference line.
    Bars coloured by composite. Scores below 85 shown in red,
    above 115 in green, otherwise in composite colour.
    X-axis range fixed 40–130 to match EQ-i standard scale.
    """
    c = _tc(theme)

    # Prefer actual composite scores from the report; fall back to subscale mean
    composite_scores = {}
    for comp, subscales in EQI_COMPOSITES.items():
        if comp in eqi_scores:                            # actual normed score
            composite_scores[comp] = eqi_scores[comp]
        else:
            available = [eqi_scores[s] for s in subscales if s in eqi_scores]
            if available:
                composite_scores[comp] = round(sum(available) / len(available), 1)

    if not composite_scores:
        return go.Figure()

    labels = list(composite_scores.keys())
    values = [composite_scores[k] for k in labels]

    # Always use the composite's category colour (matches MHS EQ-i report)
    bar_colors = [THEME["eqi"][lbl] for lbl in labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker_color=bar_colors,
        marker_line_width=0,
        opacity=0.88,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        textfont=dict(color=c["text"], size=11),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
        name="EQ-i Composite",
    ))

    # Normative mean reference line at 100
    fig.add_vline(
        x=EQI_NORM,
        line=dict(color=c["muted"], width=1.5, dash="dot"),
        annotation_text="Norm (100)",
        annotation_position="top",
        annotation_font=dict(color=c["muted"], size=10),
    )

    fig.update_layout(**_base_layout(
        "EQ-i 2.0 Composite Scores",
        height=260, theme=theme,
        extra=dict(
            xaxis=_axis("Score", show_grid=True,
                        fixed_range=[40, 140], theme=theme),
            yaxis=dict(
                tickfont=dict(color=c["text"], size=11),
                showgrid=False,
                linecolor=c["border"],
                autorange="reversed",   # top composite at top
            ),
            showlegend=False,
            margin=dict(l=130, r=40, t=48, b=36),
        ),
    ))
    return fig


# ─────────────────────────────────────────
# Component helpers
# ─────────────────────────────────────────

def shift_badge(value: float) -> html.Span:
    """Returns a coloured arrow span reflecting score direction."""
    if abs(value) <= 0.15:
        return html.Span(f"± {abs(value):.2f}",
                         className="shift-neutral",
                         style={"color": THEME["dark"]["muted"],
                                "fontWeight": 600, "fontSize": "11px"})
    if value > 0:
        return html.Span(f"▲ {value:+.2f}",
                         className="shift-positive",
                         style={"color": THEME["dark"]["green"],
                                "fontWeight": 700, "fontSize": "11px"})
    return html.Span(f"▼ {value:+.2f}",
                     className="shift-negative",
                     style={"color": THEME["dark"]["red"],
                            "fontWeight": 700, "fontSize": "11px"})


def _eq_total_bar(eqi_scores: dict) -> Optional[html.Div]:
    """
    Horizontal EQ score bar shown below the participant name when
    a Total EI score is available.  Range 70–130, midline at 100.
    """
    total_ei = eqi_scores.get("total_ei")
    if total_ei is None:
        return None

    pct     = max(0.0, min(100.0, (total_ei - 70) / 60 * 100))
    mid_pct = (100 - 70) / 60 * 100   # 50 %

    bar_color = THEME["dark"]["purple"]

    return html.Div([
        html.Div([
            html.Span("Total EQ", style={
                "color": THEME["dark"]["muted"], "fontSize": "10px",
                "fontWeight": 700, "letterSpacing": "0.06em",
                "textTransform": "uppercase",
            }),
            html.Span(str(total_ei), style={
                "color": bar_color, "fontSize": "14px",
                "fontWeight": 900, "marginLeft": "auto",
            }),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "marginBottom": "4px"}),
        html.Div([
            html.Div(style={
                "width": f"{pct:.1f}%", "height": "100%",
                "backgroundColor": bar_color, "borderRadius": "3px",
                "transition": "width 0.6s ease",
            }),
            html.Div(style={
                "position": "absolute",
                "left": f"{mid_pct:.1f}%",
                "top": "-2px", "bottom": "-2px",
                "width": "2px",
                "backgroundColor": "rgba(255,255,255,0.25)",
                "borderRadius": "1px",
            }),
        ], style={
            "position": "relative", "width": "100%", "height": "8px",
            "backgroundColor": "rgba(255,255,255,0.1)",
            "borderRadius": "4px", "overflow": "visible",
        }),
        html.Div([
            html.Span("70",  style={"color": THEME["dark"]["muted"], "fontSize": "9px"}),
            html.Span("100", style={"color": THEME["dark"]["muted"], "fontSize": "9px",
                                    "marginLeft": "auto"}),
            html.Span("130", style={"color": THEME["dark"]["muted"], "fontSize": "9px"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "marginTop": "3px"}),
    ], style={"marginTop": "12px", "marginBottom": "2px"})


# Explanations shown when hovering over alignment badges in EQI Insights
_BADGE_TOOLTIPS = {
    "Aligned":         "Your DISC score and EQ-i score point in the same direction — the behavioral tendency is supported by the emotional skill.",
    "Gap":             "Your DISC style is strong here, but your EQ-i score reveals a development opportunity. The behavior is present; the emotional skill needs growth.",
    "Inverse Risk":    "A high score on this DISC factor typically suppresses this EQ-i subscale. This is a known blind spot — awareness is the first step.",
    "Counterbalanced": "Despite the inverse pressure from your DISC style, your EQ-i score shows strong development here. This is a meaningful strength.",
    "EQ Strength":     "Your EQ-i score exceeds what your DISC style alone would predict. This subscale is a genuine emotional intelligence asset.",
}


def _eqi_score_color(score: Optional[int]) -> str:
    if score is None:        return THEME["dark"]["muted"]
    if score >= 110:         return THEME["dark"]["green"]
    if score >= 100:         return THEME["dark"]["accent"]
    if score >= 90:          return THEME["dark"]["gold"]
    return THEME["dark"]["red"]


def _eqi_insights_section(profile: dict) -> Optional[html.Details]:
    """
    Collapsible <details> block shown on individual report cards when
    both DISC and EQI data are present.

    Order: Bottom 3 Development Areas (top) → Full subscale correlations
    """
    eqi_scores = profile.get("eqi_scores", {})
    if not eqi_scores:
        return None

    primary = (profile.get("summary", {}).get("top_two") or [None])[0]
    if not primary:
        return None

    # Extract DISC anchor scores so correlations can show alignment
    disc_factor_scores = {}
    for f in ["D", "I", "S", "C"]:
        fp = profile.get("factor_profiles", {}).get(f, {})
        s = fp.get("anchor_score")
        if s is not None:
            disc_factor_scores[f] = s

    ins = generate_insights(primary, eqi_scores, disc_factor_scores)
    if not ins:
        return None

    # ── Style header ────────────────────────────────────────────────────────
    body_children = [
        html.Div([
            html.Span(f"{ins['style_name']} Style", style={
                "fontSize": "12px", "fontWeight": 800,
                "color": THEME["dark"]["text"],
            }),
            html.Span(f"  ·  {ins['style_desc']}", style={
                "fontSize": "11px", "color": THEME["dark"]["muted"],
            }),
        ], style={
            "backgroundColor": THEME["dark"]["bg"],
            "borderLeft": f"3px solid {THEME['dark']['purple']}",
            "borderRadius": "0 6px 6px 0",
            "padding": "10px 14px", "marginBottom": "16px",
        }),
    ]

    # ── BOTTOM 3 DEVELOPMENT AREAS (at top of section) ─────────────────────
    if ins.get("bottom_three"):
        growth_items = []
        for item in ins["bottom_three"]:
            sc = _eqi_score_color(item["score"])
            action_bullets = [
                html.Li(action, style={
                    "fontSize": "11px", "color": THEME["dark"]["muted"],
                    "lineHeight": "1.6", "marginBottom": "4px",
                }) for action in item.get("actions", [])
            ]
            growth_items.append(html.Div([
                html.Div([
                    html.Span(item["label"], style={
                        "fontWeight": 700, "fontSize": "13px",
                        "color": THEME["dark"]["text"],
                    }),
                    html.Span(f"  {item['score']}", style={
                        "fontSize": "12px", "fontWeight": 700,
                        "color": sc, "marginLeft": "8px",
                    }),
                    html.Span(f"  —  {item['level']}", style={
                        "fontSize": "11px", "color": sc,
                    }),
                ], style={"display": "flex", "alignItems": "center",
                          "marginBottom": "8px", "flexWrap": "wrap"}),
                html.Ul(action_bullets, style={"paddingLeft": "16px", "margin": 0}),
            ], style={
                "border": f"1px solid {THEME['dark']['border']}",
                "borderRadius": "8px", "padding": "12px 14px",
                "marginBottom": "10px",
            }))

        body_children += [
            html.Div("BOTTOM 3 DEVELOPMENT AREAS", style={
                "fontSize": "10px", "fontWeight": 700,
                "color": THEME["dark"]["muted"], "letterSpacing": "0.07em",
                "textTransform": "uppercase", "marginBottom": "10px",
            }),
            *growth_items,
            html.Hr(style={"borderColor": THEME["dark"]["border"],
                           "margin": "18px 0"}),
        ]

    # ── FULL SUBSCALE CORRELATIONS — all 15 grouped by composite ───────────
    if ins.get("all_subscale_correlations"):
        comp_blocks = []
        for group in ins["all_subscale_correlations"]:
            comp_name = group["composite"]
            comp_color = THEME["eqi"].get(comp_name, THEME["dark"]["accent"])
            sub_rows = []
            for sub in group["subscales"]:
                eqi_sc   = sub.get("eqi_score")
                disc_ltr = sub.get("disc_letter", "")
                disc_sc  = sub.get("disc_score")
                inverse  = sub.get("inverse", False)
                align_lbl    = sub.get("align_label")
                align_ck     = sub.get("align_color_key")
                align_hex    = THEME["dark"].get(align_ck, "") if align_ck else ""

                eqi_color  = _eqi_score_color(eqi_sc)
                disc_color = (
                    THEME["dark"]["green"] if disc_sc and disc_sc > 0
                    else THEME["dark"]["red"] if disc_sc and disc_sc < 0
                    else THEME["dark"]["muted"]
                )
                disc_color = THEME["disc"].get(disc_ltr, THEME["dark"]["muted"])

                header_items = [
                    html.Span(sub["label"], style={
                        "fontWeight": 700, "fontSize": "12px",
                        "color": THEME["dark"]["text"],
                        "minWidth": "160px",
                    }),
                    # EQI score
                    html.Span(
                        str(eqi_sc) if eqi_sc is not None else "—",
                        style={"color": eqi_color, "fontWeight": 700,
                               "fontSize": "12px", "marginLeft": "10px"},
                    ),
                ]

                # DISC correlation badge
                if disc_ltr:
                    disc_display = (
                        f"{disc_sc:+.2f}" if disc_sc is not None else "—"
                    )
                    header_items += [
                        html.Span(" ↔ ", style={
                            "color": THEME["dark"]["muted"], "fontSize": "11px",
                            "margin": "0 4px",
                        }),
                        html.Span(disc_ltr, style={
                            "color": disc_color, "fontWeight": 900,
                            "fontSize": "12px",
                        }),
                        html.Span(f" {disc_display}", style={
                            "color": disc_color, "fontSize": "11px",
                            "fontWeight": 600,
                        }),
                        html.Span(" (inverse)" if inverse else "", style={
                            "fontSize": "10px", "color": THEME["dark"]["gold"],
                            "marginLeft": "4px",
                        }),
                    ]

                # Alignment badge — with hover tooltip
                if align_lbl and align_hex:
                    tooltip_text = _BADGE_TOOLTIPS.get(align_lbl, "")
                    header_items.append(
                        html.Span(
                            align_lbl,
                            className="badge-tooltip",
                            **{"data-tooltip": tooltip_text},
                            style={
                                "fontSize": "10px", "fontWeight": 700,
                                "color": align_hex,
                                "backgroundColor": f"{align_hex}22",
                                "border": f"1px solid {align_hex}55",
                                "borderRadius": "10px",
                                "padding": "1px 7px",
                                "marginLeft": "8px",
                            },
                        )
                    )

                sub_rows.append(html.Div([
                    html.Div(header_items, style={
                        "display": "flex", "alignItems": "center",
                        "flexWrap": "wrap", "marginBottom": "3px",
                    }),
                    html.P(sub.get("note", ""), style={
                        "fontSize": "11px", "color": THEME["dark"]["muted"],
                        "lineHeight": "1.6", "margin": "0 0 0 4px",
                    }),
                ], style={
                    "borderBottom": f"1px solid {THEME['dark']['border']}",
                    "paddingBottom": "8px", "marginBottom": "8px",
                }))

            comp_blocks.append(html.Div([
                html.Div(comp_name, style={
                    "fontSize": "10px", "fontWeight": 800,
                    "color": comp_color, "letterSpacing": "0.07em",
                    "textTransform": "uppercase",
                    "borderLeft": f"3px solid {comp_color}",
                    "paddingLeft": "8px", "marginBottom": "10px",
                }),
                *sub_rows,
            ], style={"marginBottom": "18px"}))

        body_children += [
            html.Div("DISC — EQ-i 2.0 SUBSCALE CORRELATIONS", style={
                "fontSize": "10px", "fontWeight": 700,
                "color": THEME["dark"]["muted"], "letterSpacing": "0.07em",
                "textTransform": "uppercase", "marginBottom": "14px",
            }),
            html.Div([
                html.Span("Score", style={
                    "fontSize": "10px", "color": THEME["dark"]["muted"],
                    "marginRight": "16px",
                }),
                html.Span("↔  DISC factor + anchor score", style={
                    "fontSize": "10px", "color": THEME["dark"]["muted"],
                }),
            ], style={"marginBottom": "14px"}),
            *comp_blocks,
        ]

    return html.Details([
        html.Summary([
            html.Span("EQI Insights", style={"fontWeight": 700}),
            html.Span(
                f"{ins['style_name']} Style  ×  EQ-i 2.0",
                style={
                    "marginLeft": "auto", "fontSize": "10px",
                    "fontWeight": 500, "color": THEME["dark"]["purple"],
                    "backgroundColor": "rgba(188,140,255,0.1)",
                    "padding": "2px 8px", "borderRadius": "12px",
                    "border": "1px solid rgba(188,140,255,0.3)",
                },
            ),
        ], className="eqi-insights-summary"),
        html.Div(body_children, style={"padding": "4px 20px 20px"}),
    ], className="eqi-insights-details")


def _eqi_comparison_insights(profile: dict) -> Optional[html.Details]:
    """
    Condensed collapsible EQI Insights for comparison cards.
    Shows the bottom 3 underdeveloped subscales with scores only (no action steps).
    """
    eqi_scores = profile.get("eqi_scores", {})
    if not eqi_scores:
        return None

    primary = (profile.get("summary", {}).get("top_two") or [None])[0]
    if not primary:
        return None

    summ = generate_summary_insights(primary, eqi_scores)
    if not summ:
        return None

    body = []
    for item in summ.get("bottom_three", []):
        score = item["score"]
        if score >= 110:
            sc = THEME["dark"]["green"]
        elif score >= 100:
            sc = THEME["dark"]["accent"]
        elif score >= 90:
            sc = THEME["dark"]["gold"]
        else:
            sc = THEME["dark"]["red"]
        body.append(html.Div([
            html.Span(item["label"], style={
                "fontSize": "11px", "fontWeight": 700,
                "color": THEME["dark"]["text"],
            }),
            html.Span(f"  {score}  —  {item['level']}", style={
                "fontSize": "10px", "color": sc,
            }),
        ], style={"marginBottom": "5px"}))

    if not body:
        return None

    return html.Details([
        html.Summary(
            f"EQI Insights  ·  {summ['style_name']} Style",
            className="eqi-insights-summary eqi-insights-summary-sm",
        ),
        html.Div([
            html.Div("BOTTOM 3 DEVELOPMENT AREAS", style={
                "fontSize": "9px", "fontWeight": 700,
                "color": THEME["dark"]["muted"], "letterSpacing": "0.07em",
                "textTransform": "uppercase", "marginBottom": "6px",
            }),
            *body,
        ], style={"padding": "8px 14px 12px"}),
    ], className="eqi-insights-details eqi-insights-details-sm")


def metric_cards(df: pd.DataFrame, anchor_graph: str) -> dbc.Row:
    """
    Four DISC metric tiles — dramatic luxury redesign.
    • 4px coloured top border
    • Faded watermark factor letter in background
    • 56px Playfair Display score number
    • Serif "Team Mean" label
    """
    cols = []
    for f in FACTORS:
        col   = f"{anchor_graph}_{f}"
        mean  = float(df[col].mean())
        color = THEME["disc"][f]
        sign  = "+" if mean >= 0 else ""
        cols.append(dbc.Col(
            html.Div([
                # ── Coloured top border stripe ──────────────────
                html.Div(style={
                    "height": "4px",
                    "backgroundColor": color,
                    "borderRadius": "12px 12px 0 0",
                }),
                # ── Card body (relative so watermark can be absolute) ──
                html.Div([
                    # Watermark factor letter — faded behind content
                    html.Div(f, style={
                        "position":   "absolute",
                        "right":      "10px",
                        "bottom":     "-4px",
                        "fontSize":   "88px",
                        "fontWeight": 900,
                        "fontFamily": "'Playfair Display', Georgia, serif",
                        "color":      color,
                        "opacity":    "0.07",
                        "lineHeight": "1",
                        "pointerEvents": "none",
                        "userSelect": "none",
                        "letterSpacing": "-0.04em",
                    }),
                    # ── Content layer ───────────────────────────
                    html.Div([
                        html.Div(f, style={
                            "color":         color,
                            "fontWeight":    800,
                            "fontSize":      "11px",
                            "letterSpacing": "0.12em",
                            "textTransform": "uppercase",
                            "marginBottom":  "2px",
                        }),
                        html.Div("Team Mean", style={
                            "color":      THEME["dark"]["muted"],
                            "fontSize":   "10px",
                            "fontStyle":  "italic",
                            "fontFamily": "'Playfair Display', Georgia, serif",
                            "marginBottom": "10px",
                        }),
                        html.Div(f"{sign}{mean:.2f}", style={
                            "fontSize":      "56px",
                            "fontWeight":    700,
                            "fontFamily":    "'Playfair Display', Georgia, serif",
                            "color":         THEME["dark"]["text"],
                            "lineHeight":    "1",
                            "letterSpacing": "-0.03em",
                        }),
                    ]),
                ], style={
                    "position": "relative",
                    "padding":  "16px 18px 22px 18px",
                    "overflow": "hidden",
                }),
            ], style={
                "backgroundColor": THEME["dark"]["surface"],
                "border":          f"1px solid {THEME['dark']['border']}",
                "borderRadius":    "12px",
                "boxShadow":       f"0 4px 24px {THEME['dark']['shadow_sm']}",
                "overflow":        "hidden",
            }, className="metric-hover"),
        ))
    return dbc.Row(cols, className="mb-4 g-3")


def ranking_table(df: pd.DataFrame,
                  anchor_graph: str,
                  sort_factor: str) -> html.Div:
    """Collapsible ranking table sorted by chosen DISC factor."""
    col = f"{anchor_graph}_{sort_factor}"
    sorted_df = (
        df[["participant_name", col, f"{sort_factor}_bucket"]]
        .copy()
        .sort_values(col, ascending=False)
        .reset_index(drop=True)
    )
    color = THEME["disc"][sort_factor]
    bg    = THEME["dark"]["bg"]
    surf  = THEME["dark"]["surface"]
    surf2 = THEME["dark"]["surface2"]

    badge_styles = [
        {"backgroundColor": THEME["gold_badge"],   "color": "#000"},
        {"backgroundColor": THEME["silver_badge"], "color": "#000"},
        {"backgroundColor": THEME["bronze_badge"], "color": "#fff"},
    ]
    base_badge = {
        "display": "inline-flex", "alignItems": "center",
        "justifyContent": "center",
        "width": "22px", "height": "22px", "borderRadius": "50%",
        "fontSize": "10px", "fontWeight": 800,
        "backgroundColor": THEME["dark"]["border"],
        "color": THEME["dark"]["text"],
    }

    def _th(label, col_color=None):
        return html.Th(label, style={
            "color": col_color or THEME["dark"]["muted"],
            "fontSize": "10px", "padding": "8px 12px",
            "borderBottom": f"1px solid {THEME['dark']['border']}",
            "backgroundColor": bg,
        })

    header = html.Thead(html.Tr([
        _th("#"), _th("PARTICIPANT"),
        _th(f"{sort_factor} SCORE", color),
        _th("BUCKET"),
    ]))

    tbody_rows = []
    for i, row in sorted_df.iterrows():
        bstyle = {**base_badge, **(badge_styles[i] if i < 3 else {})}
        score  = row[col]
        bg_row = surf if i % 2 == 0 else surf2
        tbody_rows.append(html.Tr([
            html.Td(html.Span(str(i + 1), style=bstyle),
                    style={"padding": "8px 12px", "backgroundColor": bg_row}),
            html.Td(row["participant_name"],
                    style={"padding": "8px 12px", "color": THEME["dark"]["text"],
                           "fontSize": "13px", "backgroundColor": bg_row}),
            html.Td(f"{'+'if score>=0 else ''}{score:.2f}",
                    style={"padding": "8px 12px", "color": color,
                           "fontWeight": 700, "fontSize": "13px",
                           "backgroundColor": bg_row}),
            html.Td(row[f"{sort_factor}_bucket"],
                    style={"padding": "8px 12px",
                           "color": THEME["dark"]["muted"],
                           "fontSize": "11px", "backgroundColor": bg_row}),
        ], className="rank-row"))

    return html.Div([
        html.Table(
            [header, html.Tbody(tbody_rows)],
            style={"width": "100%", "borderCollapse": "collapse"},
        ),
    ], style={**SECTION_STYLE(), "padding": "0", "overflow": "hidden"})


def participant_card(profile: dict) -> html.Div:
    """
    Full operator report card with DISC factor tiles,
    shift indicators, raw scores, and EQ-i composite bar chart.
    """
    top_two    = ", ".join(profile["summary"]["top_two"])
    style_type = profile.get("style_type", "—")
    eqi        = profile.get("eqi_scores", {})

    factor_cols = []
    for idx, f in enumerate(FACTORS):
        fp    = profile["factor_profiles"][f]
        color = THEME["disc"][f]
        traits = [html.Li(t, style={
            "fontSize": "12px", "color": THEME["dark"]["muted"],
            "marginBottom": "3px",
        }) for t in fp["traits"][:4]]

        factor_cols.append(dbc.Col(html.Div([
            html.Div([
                html.Span(f, className="factor-letter",
                          style={"fontSize": "26px", "fontWeight": 900,
                                 "color": color}),
                html.Span(f" {fp['anchor_score']:+.2f}",
                          style={"fontSize": "13px",
                                 "color": THEME["dark"]["muted"],
                                 "marginLeft": "6px"}),
            ], style={"marginBottom": "2px"}),
            html.Div(fp["bucket"].replace("_", " ").title(),
                     className="bucket-label",
                     style={"fontSize": "10px", "color": color,
                            "fontWeight": 700, "letterSpacing": "0.06em",
                            "textTransform": "uppercase",
                            "marginBottom": "10px"}),
            html.Div("Traits", style={
                "color": THEME["dark"]["muted"], "fontSize": "10px",
                "fontWeight": 700, "letterSpacing": "0.06em",
                "textTransform": "uppercase", "marginBottom": "4px",
            }),
            html.Ul(traits, style={"paddingLeft": "14px",
                                    "marginBottom": "10px"}),
            html.Div(style={"height": "1px",
                            "backgroundColor": THEME["dark"]["border"],
                            "marginBottom": "8px"}),
            html.Div([
                html.Span("Public → Stress:  ",
                          style={"color": THEME["dark"]["muted"],
                                 "fontSize": "11px"}),
                shift_badge(fp["delta_public_to_stress"]),
            ], style={"marginBottom": "3px"}),
            html.Div([
                html.Span("Mirror → Stress:  ",
                          style={"color": THEME["dark"]["muted"],
                                 "fontSize": "11px"}),
                shift_badge(fp["delta_mirror_to_stress"]),
            ]),
        ], style={
            "backgroundColor": THEME["dark"]["surface2"],
            "border":          f"1px solid {THEME['dark']['border']}",
            "borderRadius":    "10px",
            "padding":         "14px",
            "height":          "100%",
        }, className=f"factor-tile factor-tile-{idx}")))

    g = profile["graphs"]

    eq_bar = _eq_total_bar(eqi)

    # EQ-i section — composite bar chart, only when scores are present
    eqi_section = []
    if eqi:
        eqi_section = [
            html.Div([
                html.Div("EQ-i 2.0 Composite Scores", style={
                    "color": THEME["dark"]["muted"], "fontSize": "10px",
                    "fontWeight": 700, "letterSpacing": "0.08em",
                    "textTransform": "uppercase", "marginBottom": "8px",
                }),
                dcc.Graph(
                    figure=build_eqi_bar_chart(eqi),
                    config={"displayModeBar": False},
                    style={"height": "260px"},
                ),
            ], style=CARD_STYLE()),
        ]

    # EQI Insights dropdown — only when both DISC and EQI data are present
    insights_section = _eqi_insights_section(profile)

    return html.Div([
        # ── Header card ────────────────────────────────────
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div("SDG DISC Operator Report",
                             className="report-title",
                             style={
                                 "fontSize":      "11px",
                                 "fontWeight":    700,
                                 "fontFamily":    "'Playfair Display', Georgia, serif",
                                 "fontStyle":     "italic",
                                 "color":         THEME["dark"]["muted"],
                                 "letterSpacing": "0.06em",
                                 "textTransform": "uppercase",
                                 "marginBottom":  "6px",
                             }),
                    html.Div(
                        profile.get("participant_name", ""),
                        className="participant-name",
                        style={
                            "fontFamily":    "'Playfair Display', Georgia, serif",
                            "fontSize":      "22px",
                            "fontWeight":    700,
                            "color":         THEME["dark"]["accent"],
                            "lineHeight":    "1.15",
                            "marginBottom":  "4px",
                            "letterSpacing": "0.01em",
                        },
                    ),
                    html.Div(
                        f"Anchor: "
                        f"{profile.get('anchor_graph','stress').title()}"
                        f"  ·  Top Two: {top_two}",
                        style={"color": THEME["dark"]["muted"],
                               "fontSize": "12px"},
                    ),
                    # EQ total bar — shown when EQI report is linked
                    *([eq_bar] if eq_bar else []),
                ]),
                dbc.Col([
                    html.Div("DISC STYLE", style={
                        "color":         THEME["dark"]["muted"],
                        "fontSize":      "9px",
                        "fontWeight":    700,
                        "letterSpacing": "0.12em",
                        "textAlign":     "right",
                        "textTransform": "uppercase",
                        "marginBottom":  "2px",
                    }),
                    html.Div(style_type, style={
                        "fontFamily":    "'Playfair Display', Georgia, serif",
                        "fontSize":      "44px",
                        "fontWeight":    700,
                        "color":         THEME["dark"]["accent"],
                        "textAlign":     "right",
                        "lineHeight":    "1",
                        "letterSpacing": "-0.02em",
                    }),
                ], width="auto"),
            ], align="start"),
        ], style=CARD_STYLE()),

        # ── Factor tiles ────────────────────────────────────
        dbc.Row(factor_cols, className="g-3 mb-3"),

        # ── Raw scores ──────────────────────────────────────
        html.Div([
            html.Div("Raw Graph Scores", style={
                "color": THEME["dark"]["muted"], "fontSize": "10px",
                "fontWeight": 700, "letterSpacing": "0.08em",
                "textTransform": "uppercase", "marginBottom": "12px",
            }),
            dbc.Row([
                dbc.Col([
                    html.Div("PUBLIC", style={
                        "color": THEME["dark"]["muted"], "fontSize": "9px",
                        "fontWeight": 700, "marginBottom": "4px",
                    }),
                    html.Div(
                        f"D {g['public']['D']:+.2f}  I {g['public']['I']:+.2f}  "
                        f"S {g['public']['S']:+.2f}  C {g['public']['C']:+.2f}",
                        style={"color": THEME["dark"]["text"],
                               "fontSize": "12px",
                               "fontFamily": "monospace"},
                    ),
                ]),
                dbc.Col([
                    html.Div("STRESS", style={
                        "color": THEME["dark"]["muted"], "fontSize": "9px",
                        "fontWeight": 700, "marginBottom": "4px",
                    }),
                    html.Div(
                        f"D {g['stress']['D']:+.2f}  I {g['stress']['I']:+.2f}  "
                        f"S {g['stress']['S']:+.2f}  C {g['stress']['C']:+.2f}",
                        style={"color": THEME["dark"]["text"],
                               "fontSize": "12px",
                               "fontFamily": "monospace"},
                    ),
                ]),
                dbc.Col([
                    html.Div("MIRROR", style={
                        "color": THEME["dark"]["muted"], "fontSize": "9px",
                        "fontWeight": 700, "marginBottom": "4px",
                    }),
                    html.Div(
                        f"D {g['mirror']['D']:+.2f}  I {g['mirror']['I']:+.2f}  "
                        f"S {g['mirror']['S']:+.2f}  C {g['mirror']['C']:+.2f}",
                        style={"color": THEME["dark"]["text"],
                               "fontSize": "12px",
                               "fontFamily": "monospace"},
                    ),
                ]),
            ]),
        ], style=CARD_STYLE()),

        # ── EQ-i composite chart (conditional) ───────────────
        *eqi_section,

        # ── EQI Insights dropdown (conditional) ──────────────
        *([insights_section] if insights_section else []),
    ])


def comparison_card(profile: dict) -> html.Div:
    """
    Compact side-by-side card showing DISC mini-tiles,
    shift indicators, raw scores, and EQ-i composite bar chart.
    """
    top_two    = ", ".join(profile["summary"]["top_two"])
    style_type = profile.get("style_type", "—")
    eqi        = profile.get("eqi_scores", {})
    mini_cols  = []

    for f in FACTORS:
        fp    = profile["factor_profiles"][f]
        color = THEME["disc"][f]
        traits = "; ".join(fp["traits"][:2])
        mini_cols.append(dbc.Col(html.Div([
            html.Div([
                html.Span(f, className="factor-letter",
                          style={"fontWeight": 900, "color": color,
                                 "fontSize": "15px"}),
                html.Span(f" {fp['anchor_score']:+.2f}",
                          style={"fontSize": "12px",
                                 "color": THEME["dark"]["muted"]}),
            ], style={"marginBottom": "2px"}),
            html.Div(fp["bucket"].replace("_", " "),
                     className="bucket-label",
                     style={"fontSize": "10px", "color": color,
                            "fontWeight": 600, "marginBottom": "4px"}),
            html.Div(traits, style={
                "fontSize": "10px", "color": THEME["dark"]["muted"],
                "lineHeight": "1.4", "marginBottom": "6px",
            }),
            html.Div([
                html.Span("P→S ", style={
                    "color": THEME["dark"]["muted"], "fontSize": "10px",
                }),
                shift_badge(fp["delta_public_to_stress"]),
                html.Span("  M→S ", style={
                    "color": THEME["dark"]["muted"], "fontSize": "10px",
                }),
                shift_badge(fp["delta_mirror_to_stress"]),
            ]),
        ], style={
            "backgroundColor": THEME["dark"]["bg"],
            "borderRadius": "8px", "padding": "10px",
        }), width=6))

    g = profile["graphs"]

    # EQ-i mini chart — only shown when scores are present
    eqi_row = []
    if eqi:
        eqi_row = [
            html.Div([
                html.Div("EQ-i 2.0", style={
                    "color": THEME["dark"]["muted"], "fontSize": "9px",
                    "fontWeight": 700, "letterSpacing": "0.06em",
                    "textTransform": "uppercase", "marginBottom": "4px",
                }),
                dcc.Graph(
                    figure=build_eqi_bar_chart(eqi),
                    config={"displayModeBar": False},
                    style={"height": "220px"},
                ),
            ], style={"marginTop": "10px"}),
        ]

    cmp_eq_bar = _eq_total_bar(eqi)
    cmp_insights = _eqi_comparison_insights(profile)

    return html.Div([
        # Name and style type header
        dbc.Row([
            dbc.Col([
                html.Div(profile.get("participant_name", ""),
                         className="comp-name",
                         style={"fontSize": "15px", "fontWeight": 800,
                                "color": THEME["dark"]["text"]}),
                html.Div(
                    f"Top Two: {top_two}  ·  "
                    f"Anchor: {profile.get('anchor_graph','stress')}",
                    style={"fontSize": "11px",
                           "color": THEME["dark"]["muted"],
                           "marginTop": "2px"},
                ),
                # EQ total bar — shown when EQI report is linked
                *([cmp_eq_bar] if cmp_eq_bar else []),
            ]),
            dbc.Col(
                html.Div(style_type, style={
                    "fontSize": "22px", "fontWeight": 900,
                    "color": THEME["dark"]["accent"],
                    "textAlign": "right",
                }),
                width="auto",
            ),
        ], align="start", className="mb-3"),

        # DISC mini tiles
        dbc.Row(mini_cols, className="g-2 mb-2"),

        # Raw scores
        html.Div(
            f"Public D{g['public']['D']:+.1f} I{g['public']['I']:+.1f} "
            f"S{g['public']['S']:+.1f} C{g['public']['C']:+.1f}  |  "
            f"Stress D{g['stress']['D']:+.1f} I{g['stress']['I']:+.1f} "
            f"S{g['stress']['S']:+.1f} C{g['stress']['C']:+.1f}  |  "
            f"Mirror D{g['mirror']['D']:+.1f} I{g['mirror']['I']:+.1f} "
            f"S{g['mirror']['S']:+.1f} C{g['mirror']['C']:+.1f}",
            style={"color": THEME["dark"]["muted"], "fontSize": "10px",
                   "fontFamily": "monospace", "marginTop": "4px"},
        ),

        # EQ-i composite chart (conditional)
        *eqi_row,

        # EQI Insights summary (conditional)
        *([cmp_insights] if cmp_insights else []),

    ], style=SECTION_STYLE())


def _graph_card(children, title: str = "", theme: str = "dark") -> html.Div:
    """Wraps a Plotly chart in a themed card container."""
    c = _tc(theme)
    card_style = {
        "backgroundColor": c["surface"],
        "border":          f"1px solid {c['border']}",
        "borderRadius":    "12px",
        "overflow":        "hidden",
        "boxShadow":       f"0 4px 24px {c['shadow_md']}",
        "marginBottom":    "0",
        "transition":      "background-color 0.3s ease, border-color 0.3s ease",
    }
    header = ([html.Div(title, style={
        "padding":       "12px 16px 0 16px",
        "fontSize":      "12px",
        "fontWeight":    700,
        "color":         c["muted"],
        "letterSpacing": "0.04em",
    })] if title else [])
    return html.Div([*header, children], style=card_style)


# ─────────────────────────────────────────
# Layout
# ─────────────────────────────────────────
app.layout = html.Div(
    id="app-root",
    style={"minHeight": "100vh",
           "fontFamily": "Inter, system-ui, sans-serif"},
    children=[

        dcc.Store(id="profiles-store"),
        dcc.Store(id="df-store"),
        dcc.Store(id="theme-store", data="dark"),
        dcc.Download(id="download-csv"),
        dcc.Download(id="download-json"),

        # ── Sticky header ──────────────────────────────────────────
        html.Div([
            dbc.Container([
                dbc.Row([
                    # Logo / title
                    dbc.Col([
                        html.Div([
                            html.Span("SDG", className="sdg-badge", style={
                                "backgroundColor": THEME["dark"]["accent"],
                                "color": THEME["dark"]["bg"],
                                "fontWeight": 900, "fontSize": "11px",
                                "padding": "3px 8px", "borderRadius": "5px",
                                "marginRight": "10px", "letterSpacing": "0.05em",
                            }),
                            html.Span("DISC Dashboard",
                                      className="header-title",
                                      style={"fontWeight": 700,
                                             "fontSize": "15px",
                                             "color": THEME["dark"]["text"]}),
                        ], style={"display": "flex", "alignItems": "center"}),
                    ], width="auto"),

                    # Navigation tabs
                    dbc.Col(
                        dbc.Tabs(
                            id="tabs", active_tab="team",
                            children=[
                                dbc.Tab(label="Team Dashboard",
                                        tab_id="team"),
                                dbc.Tab(label="Individual Results",
                                        tab_id="individual"),
                                dbc.Tab(label="Comparisons",
                                        tab_id="comparisons"),
                            ],
                            style={"borderBottom": "none"},
                        ),
                        style={"display": "flex", "alignItems": "flex-end",
                               "justifyContent": "flex-end"},
                    ),

                    # Light/dark toggle
                    dbc.Col([
                        html.Div([
                            html.Span("🌙", id="theme-icon-moon",
                                      style={"fontSize": "13px",
                                             "color": THEME["dark"]["accent"],
                                             "transition": "color 0.3s"}),
                            html.Div(
                                html.Div(id="theme-knob", style={
                                    "width": "18px", "height": "18px",
                                    "borderRadius": "50%",
                                    "backgroundColor": THEME["dark"]["text"],
                                    "position": "absolute",
                                    "top": "3px", "left": "3px",
                                    "transition":
                                        "transform 0.3s ease, "
                                        "background-color 0.3s",
                                }),
                                id="theme-toggle",
                                n_clicks=0,
                                style={
                                    "width": "44px", "height": "24px",
                                    "backgroundColor": THEME["dark"]["border"],
                                    "borderRadius": "12px",
                                    "position": "relative",
                                    "cursor": "pointer",
                                    "margin": "0 8px",
                                    "transition": "background-color 0.3s",
                                    "flexShrink": "0",
                                },
                            ),
                            html.Span("☀", id="theme-icon-sun",
                                      style={"fontSize": "13px",
                                             "color": THEME["dark"]["muted"],
                                             "transition": "color 0.3s"}),
                        ], style={"display": "flex", "alignItems": "center",
                                  "gap": "4px"}),
                    ], width="auto"),

                ], align="center", justify="between"),
            ], fluid=True),
        ], style={
            "position": "sticky", "top": "0", "zIndex": "1000",
            "backgroundColor": THEME["dark"]["surface"],
            "borderBottom": f"1px solid {THEME['dark']['border']}",
            "padding": "12px 0",
            "boxShadow": f"0 2px 20px {THEME['dark']['shadow_md']}",
        }, className="sticky-header"),

        # ── Landing page (shown before any upload) ─────────────────
        html.Div(
            id="landing-section",
            children=[
                dbc.Container([
                    # SDG logo banner
                    html.Div(
                        html.Img(
                            src="/assets/sdg_logo.svg",
                            style={
                                "maxWidth": "720px", "width": "100%",
                                "borderRadius": "12px",
                                "boxShadow": f"0 8px 40px {THEME['dark']['shadow_lg']}",
                            },
                        ),
                        style={"textAlign": "center", "marginBottom": "40px"},
                    ),

                    # Upload + anchor controls side by side
                    dbc.Row([
                        # Upload area
                        dbc.Col([
                            html.Label(
                                "Upload DISC & EQ-i 2.0 PDFs",
                                style={**LABEL_STYLE, "fontSize": "12px"},
                            ),
                            dcc.Upload(
                                id="upload-pdfs",
                                children=html.Div([
                                    html.Div("↑", style={
                                        "fontSize": "32px",
                                        "color": THEME["dark"]["accent"],
                                        "marginBottom": "8px",
                                    }),
                                    html.Div("Drag & drop or ", style={
                                        "fontSize": "14px",
                                        "color": THEME["dark"]["muted"],
                                        "display": "inline",
                                    }),
                                    html.A("browse", style={
                                        "color": THEME["dark"]["accent"],
                                        "cursor": "pointer",
                                        "fontWeight": 700,
                                        "fontSize": "14px",
                                    }),
                                    html.Div(
                                        "DISC PDFs + EQ-i 2.0 PDFs — upload both in the same batch",
                                        style={
                                            "fontSize": "11px",
                                            "color": THEME["dark"]["muted"],
                                            "marginTop": "6px",
                                        },
                                    ),
                                ]),
                                style={
                                    "width": "100%", "minHeight": "120px",
                                    "lineHeight": "1.4",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "12px",
                                    "textAlign": "center",
                                    "padding": "24px 16px",
                                    "borderColor": THEME["dark"]["border"],
                                    "color": THEME["dark"]["muted"],
                                    "backgroundColor": THEME["dark"]["surface"],
                                    "cursor": "pointer",
                                },
                                multiple=True,
                            ),
                        ], md=7),

                        # Anchor graph dropdown
                        dbc.Col([
                            html.Label(
                                "Anchor Graph",
                                style={**LABEL_STYLE, "fontSize": "12px"},
                            ),
                            dcc.Dropdown(
                                id="anchor-graph",
                                options=[
                                    {"label": "Stress (Adapted)", "value": "stress"},
                                    {"label": "Public (Natural)", "value": "public"},
                                    {"label": "Mirror",           "value": "mirror"},
                                ],
                                value="stress",
                                clearable=False,
                                style=DROPDOWN_STYLE(),
                            ),
                            html.P(
                                "Selects which DISC graph drives scores and "
                                "shift calculations throughout the dashboard.",
                                style={
                                    "fontSize": "11px",
                                    "color": THEME["dark"]["muted"],
                                    "lineHeight": "1.6",
                                    "marginTop": "10px",
                                },
                            ),
                        ], md=4),
                    ], justify="center", className="g-4"),
                ], fluid=True, style={"maxWidth": "860px"}),
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "justifyContent": "center",
                "minHeight": "80vh",
                "padding": "48px 16px",
            },
        ),

        # ── Dashboard body (hidden until upload) ───────────────────
        html.Div(
            id="dashboard-body",
            style={"display": "none"},
            children=[
                dbc.Container([

                    # Compact controls row
                    dbc.Row([
                        dbc.Col([
                            html.Label("Anchor Graph", style=LABEL_STYLE),
                            dcc.Dropdown(
                                id="anchor-graph-dash",
                                options=[
                                    {"label": "Stress (Adapted)", "value": "stress"},
                                    {"label": "Public (Natural)", "value": "public"},
                                    {"label": "Mirror",           "value": "mirror"},
                                ],
                                value="stress", clearable=False,
                                style=DROPDOWN_STYLE(),
                            ),
                        ], width=2),
                        dbc.Col(
                            html.Button(
                                "↺  New Session",
                                id="btn-new-session",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "transparent",
                                    "color": THEME["dark"]["muted"],
                                    "border": f"1px solid {THEME['dark']['border']}",
                                    "borderRadius": "8px",
                                    "padding": "6px 16px",
                                    "fontSize": "12px",
                                    "cursor": "pointer",
                                    "marginTop": "22px",
                                    "fontWeight": 600,
                                },
                            ),
                            width="auto",
                        ),
                    ], className="mb-4 g-3", style={"paddingTop": "24px"}),

                    html.Div(id="upload-errors"),
                    html.Div(id="scan-status"),

                    # Session header banner — shows after upload
                    html.Div(id="session-banner", style={"display": "none"}),

                    html.Div(id="metric-cards"),

                    # Collapsible ranking
                    html.Div([
                        dbc.Row([
                            dbc.Col(
                                dbc.Button(
                                    [html.Span("▶ ", id="rank-chevron",
                                               style={"fontSize": "10px",
                                                      "marginRight": "4px"}),
                                     "Participant Rankings"],
                                    id="rank-toggle", color="link", n_clicks=0,
                                    style={
                                        "color": THEME["dark"]["muted"],
                                        "fontSize": "11px", "fontWeight": 700,
                                        "letterSpacing": "0.08em",
                                        "textTransform": "uppercase",
                                        "textDecoration": "none",
                                        "padding": "8px 0",
                                        "border": "none", "background": "none",
                                    },
                                ), width="auto",
                            ),
                            dbc.Col(
                                dbc.ButtonGroup([
                                    dbc.Button(
                                        f, id=f"rank-btn-{f}", n_clicks=0,
                                        style={
                                            "backgroundColor": THEME["dark"]["surface2"],
                                            "color": THEME["disc"][f],
                                            "border": f"1px solid {THEME['disc'][f]}",
                                            "fontSize": "11px", "fontWeight": 800,
                                            "padding": "4px 12px",
                                            "borderRadius": "6px",
                                        },
                                    ) for f in FACTORS
                                ], style={"gap": "6px"}),
                                width="auto",
                            ),
                        ], align="center", className="mb-1"),
                        dbc.Collapse(
                            html.Div(id="ranking-table"),
                            id="rank-collapse", is_open=False,
                        ),
                    ], className="mb-4"),

                    html.Div(id="tab-content"),

                    html.Hr(style={
                        "borderColor": THEME["dark"]["border"],
                        "marginTop": "32px", "marginBottom": "20px",
                    }),
                    dbc.Row([
                        dbc.Col(html.Button(
                            "↓  Export CSV", id="btn-csv", n_clicks=0,
                            style={
                                "backgroundColor": THEME["dark"]["surface"],
                                "color": THEME["dark"]["text"],
                                "border": f"1px solid {THEME['dark']['border']}",
                                "borderRadius": "8px", "padding": "10px 20px",
                                "fontSize": "12px", "cursor": "pointer",
                                "width": "100%", "fontWeight": 600,
                                "letterSpacing": "0.04em",
                            },
                        ), width=2),
                        dbc.Col(html.Button(
                            "↓  Export JSON", id="btn-json", n_clicks=0,
                            style={
                                "backgroundColor": THEME["dark"]["surface"],
                                "color": THEME["dark"]["text"],
                                "border": f"1px solid {THEME['dark']['border']}",
                                "borderRadius": "8px", "padding": "10px 20px",
                                "fontSize": "12px", "cursor": "pointer",
                                "width": "100%", "fontWeight": 600,
                                "letterSpacing": "0.04em",
                            },
                        ), width=2),
                    ], className="mb-5 g-3"),

                ], fluid=True),
            ],
        ),
    ],
)


# ═══════════════════════════════════════════════════════════════
# CALLBACKS
# Each callback has a one-line comment: Triggered by → Updates
# ═══════════════════════════════════════════════════════════════

# 0a — profiles-store → toggle landing / dashboard visibility
@app.callback(
    Output("landing-section",  "style"),
    Output("dashboard-body",   "style"),
    Input("profiles-store",    "data"),
)
def toggle_layout(profiles_data):
    if profiles_data:
        return {"display": "none"}, {}
    return (
        {
            "display": "flex", "flexDirection": "column",
            "alignItems": "center", "justifyContent": "center",
            "minHeight": "80vh", "padding": "48px 16px",
        },
        {"display": "none"},
    )


# 0b — New Session button → clear stores, return to landing
@app.callback(
    Output("profiles-store",  "data",  allow_duplicate=True),
    Output("df-store",        "data",  allow_duplicate=True),
    Input("btn-new-session",  "n_clicks"),
    prevent_initial_call=True,
)
def new_session(n_clicks):
    return None, None


# 0c — anchor-graph (landing) → sync to anchor-graph-dash (dashboard)
@app.callback(
    Output("anchor-graph-dash", "value"),
    Input("anchor-graph",       "value"),
)
def sync_anchor(val):
    return val or "stress"


# 0d — profiles-store + df-store → session banner
@app.callback(
    Output("session-banner", "children"),
    Output("session-banner", "style"),
    Input("profiles-store",  "data"),
    Input("df-store",        "data"),
)
def update_session_banner(profiles_json, df_json):
    """Build a slim full-width session overview banner after upload."""
    if not profiles_json or not df_json:
        return [], {"display": "none"}

    profiles = json.loads(profiles_json)
    df       = pd.read_json(io.StringIO(df_json))

    # ── Gather stats ────────────────────────────────────────────
    n_participants = len(profiles)
    upload_date    = datetime.now().strftime("%B %d, %Y")

    # Team EQ average — average across all participants that have EQI
    eq_scores = [
        p["eqi_scores"].get("total_ei")
        for p in profiles
        if p.get("eqi_scores") and p["eqi_scores"].get("total_ei") is not None
    ]
    eq_avg = f"{sum(eq_scores)/len(eq_scores):.0f}" if eq_scores else None

    # Dominant DISC style — most common primary style
    primary_styles = [p.get("primary_style") for p in profiles if p.get("primary_style")]
    dominant_style = Counter(primary_styles).most_common(1)[0][0] if primary_styles else "—"
    dominant_color = THEME["disc"].get(dominant_style, THEME["dark"]["accent"])

    # ── Build banner cells ──────────────────────────────────────
    def _cell(label, value, value_color=None, serif=False):
        return html.Div([
            html.Div(label, style={
                "fontSize":      "9px",
                "fontWeight":    700,
                "letterSpacing": "0.12em",
                "textTransform": "uppercase",
                "color":         THEME["dark"]["muted"],
                "marginBottom":  "3px",
            }),
            html.Div(value, style={
                "fontSize":   "18px" if serif else "15px",
                "fontWeight": 700,
                "fontFamily": "'Playfair Display', Georgia, serif" if serif else "inherit",
                "color":      value_color or THEME["dark"]["text"],
                "lineHeight": "1",
            }),
        ], style={"padding": "0 20px", "borderRight": f"1px solid {THEME['dark']['border']}"})

    cohort_input = html.Div([
        html.Div("COHORT NAME", style={
            "fontSize":      "9px",
            "fontWeight":    700,
            "letterSpacing": "0.12em",
            "textTransform": "uppercase",
            "color":         THEME["dark"]["muted"],
            "marginBottom":  "3px",
        }),
        dcc.Input(
            id="cohort-name-input",
            value="SDG Cohort",
            debounce=True,
            style={
                "background":    "transparent",
                "border":        "none",
                "borderBottom":  f"1px solid {THEME['dark']['border']}",
                "color":         THEME["dark"]["accent"],
                "fontSize":      "15px",
                "fontWeight":    700,
                "fontFamily":    "'Playfair Display', Georgia, serif",
                "outline":       "none",
                "padding":       "0",
                "width":         "160px",
                "letterSpacing": "0.01em",
            },
        ),
    ], style={"padding": "0 20px 0 0", "borderRight": f"1px solid {THEME['dark']['border']}"})

    cells = [
        cohort_input,
        _cell("Participants",   str(n_participants), serif=True),
        _cell("Session Date",   upload_date),
        *([_cell("Team EQ Avg", eq_avg, THEME["dark"]["purple"], serif=True)] if eq_avg else []),
        _cell("Dominant Style", dominant_style, dominant_color, serif=True),
    ]

    banner_children = html.Div(
        html.Div(cells, style={
            "display":     "flex",
            "alignItems":  "center",
            "flexWrap":    "wrap",
            "gap":         "0",
        }),
        style={
            "display":         "flex",
            "alignItems":      "center",
            "justifyContent":  "flex-start",
            "padding":         "14px 20px",
        },
    )

    banner_style = {
        "backgroundColor": THEME["dark"]["surface"],
        "border":          f"1px solid {THEME['dark']['border']}",
        "borderTop":       f"3px solid {THEME['dark']['accent']}",
        "borderRadius":    "12px",
        "marginBottom":    "24px",
        "boxShadow":       f"0 2px 16px {THEME['dark']['shadow_sm']}",
        "display":         "block",
    }

    return banner_children, banner_style


# 1 — PDF upload → parse profiles, build dataframe, show errors
@app.callback(
    Output("profiles-store", "data",   allow_duplicate=True),
    Output("df-store",       "data",   allow_duplicate=True),
    Output("upload-errors",  "children"),
    Output("scan-status",    "children"),
    Input("upload-pdfs",     "contents"),
    State("upload-pdfs",     "filename"),
    State("anchor-graph",    "value"),
    prevent_initial_call=True,
)
def process_uploads(contents_list, filenames, anchor_graph):
    if not contents_list:
        return None, None, None, None
    files_data = [decode_upload(c, f)
                  for c, f in zip(contents_list, filenames)]
    profiles, df, errors = process_uploaded_files(files_data, anchor_graph)
    error_banner = None
    if errors:
        error_banner = dbc.Alert(
            [html.B("Parse errors: ")]
            + [html.Div(f"{e['file']}: {e['error']}",
                        style={"fontSize": "11px"}) for e in errors],
            color="warning", dismissable=True,
            style={"fontSize": "12px", "marginBottom": "12px"},
        )
    if df.empty:
        return (None, None,
                dbc.Alert("No valid profiles found.", color="danger"), None)
    # Scan banner cleared (returns None) once processing completes
    return json.dumps(profiles), df.to_json(orient="records"), \
           error_banner, None


# 2 — df-store or anchor-graph-dash change → rebuild four DISC metric tiles
@app.callback(
    Output("metric-cards", "children"),
    Input("df-store", "data"),
    Input("anchor-graph-dash", "value"),
)
def update_metric_cards(df_json, anchor_graph):
    if not df_json:
        return html.P(
            "Upload Maxwell DISC PDFs to begin.",
            style={"color": THEME["dark"]["muted"], "fontSize": "13px",
                   "paddingTop": "20px"},
        )
    df = pd.read_json(io.StringIO(df_json), orient="records")
    return metric_cards(df, anchor_graph)


# 3 — df-store, anchor-graph, or DISC filter button → rebuild ranking table
#     also updates active button highlight style
@app.callback(
    Output("ranking-table",  "children"),
    Output("rank-btn-D",     "style"),
    Output("rank-btn-I",     "style"),
    Output("rank-btn-S",     "style"),
    Output("rank-btn-C",     "style"),
    Input("df-store",           "data"),
    Input("anchor-graph-dash",  "value"),
    Input("rank-btn-D",         "n_clicks"),
    Input("rank-btn-I",         "n_clicks"),
    Input("rank-btn-S",         "n_clicks"),
    Input("rank-btn-C",         "n_clicks"),
)
def update_ranking(df_json, anchor_graph, nd, ni, ns, nc):
    from dash import ctx
    triggered = ctx.triggered_id or "rank-btn-D"
    active = (triggered.replace("rank-btn-", "")
              if triggered in ("rank-btn-D","rank-btn-I",
                               "rank-btn-S","rank-btn-C") else "D")

    def btn_style(f, is_active):
        base = {"fontSize": "11px", "fontWeight": 800,
                "padding": "4px 12px", "borderRadius": "6px"}
        if is_active:
            return {**base,
                    "backgroundColor": THEME["disc"][f],
                    "color": THEME["dark"]["bg"],
                    "border": f"1px solid {THEME['disc'][f]}"}
        return {**base,
                "backgroundColor": THEME["dark"]["surface2"],
                "color": THEME["disc"][f],
                "border": f"1px solid {THEME['disc'][f]}"}

    styles = [btn_style(f, f == active) for f in FACTORS]
    if not df_json:
        return None, *styles
    df = pd.read_json(io.StringIO(df_json), orient="records")
    return ranking_table(df, anchor_graph, active), *styles


# 4 — Ranking toggle button → open/close collapse, update chevron icon
@app.callback(
    Output("rank-collapse", "is_open"),
    Output("rank-chevron",  "children"),
    Input("rank-toggle",    "n_clicks"),
    State("rank-collapse",  "is_open"),
)
def toggle_ranking(n, is_open):
    new_open = not is_open if n else False
    return new_open, ("▼ " if new_open else "▶ ")


# 5 — Tab selection, data stores, anchor-graph-dash, or theme → render tab content
@app.callback(
    Output("tab-content",      "children"),
    Input("tabs",              "active_tab"),
    Input("df-store",          "data"),
    Input("profiles-store",    "data"),
    Input("anchor-graph-dash", "value"),
    Input("theme-store",       "data"),
)
def render_tab(active_tab, df_json, profiles_json, anchor_graph, theme):
    theme = theme or "dark"
    if not df_json:
        return None
    df        = pd.read_json(io.StringIO(df_json), orient="records")
    profiles  = json.loads(profiles_json)
    all_names = df["participant_name"].tolist()

    # ── Team Dashboard ─────────────────────────────────────────
    if active_tab == "team":
        return html.Div([
            dbc.Row([
                dbc.Col(_graph_card(
                    dcc.Graph(
                        figure=build_anchor_comparison_chart(
                            df, anchor_graph, theme),
                        config={"displayModeBar": False},
                    ), theme=theme,
                ), width=8),
                dbc.Col(_graph_card(
                    dcc.Graph(
                        figure=build_disc_type_chart(profiles, theme),
                        config={"displayModeBar": False},
                    ), theme=theme,
                ), width=4),
            ], className="g-3 mb-3"),
            _graph_card(
                dcc.Graph(figure=build_heatmap(df, anchor_graph, theme),
                          config={"displayModeBar": False}),
                theme=theme,
            ),
            html.Div(style={"marginTop": "20px"}),
            html.Div("Per-Factor Mean Charts", style={
                "color": THEME["dark"]["muted"], "fontSize": "11px",
                "fontWeight": 700, "letterSpacing": "0.08em",
                "textTransform": "uppercase", "marginBottom": "10px",
            }),
            dbc.Tabs(
                id="letter-tabs", active_tab="D",
                children=[dbc.Tab(label=f, tab_id=f) for f in FACTORS],
                style={"marginBottom": "12px"},
            ),
            html.Div(id="letter-chart-body"),
        ], className="tab-fade-in")

    # ── Individual Results ─────────────────────────────────────
    if active_tab == "individual":
        return html.Div([
            dbc.Row([dbc.Col([
                html.Label("Select Participant", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="selected-participant",
                    options=[{"label": n, "value": n} for n in all_names],
                    value=all_names[0], clearable=False,
                    style=DROPDOWN_STYLE(),
                ),
            ], width=4)], className="mb-4"),
            html.Div(id="participant-card-body"),
        ], className="tab-fade-in")

    # ── Comparisons ────────────────────────────────────────────
    if active_tab == "comparisons":
        return html.Div([
            html.Div("Radar Profile Comparison",
                     className="section-title",
                     style={"color": THEME["dark"]["text"],
                            "fontWeight": 700, "fontSize": "14px",
                            "marginBottom": "14px"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Graph Source", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="radar-graph-choice",
                        options=[{"label": g.title(), "value": g}
                                 for g in ["public", "stress", "mirror"]],
                        value="stress", clearable=False,
                        style=DROPDOWN_STYLE(),
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Overlay Participants", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="radar-participants",
                        options=[{"label": n, "value": n}
                                 for n in all_names],
                        value=[all_names[0]], multi=True,
                        style=DROPDOWN_STYLE(),
                    ),
                ], width=6),
            ], className="mb-3"),
            _graph_card(
                dcc.Graph(id="radar-chart",
                          config={"displayModeBar": False}),
                theme=theme,
            ),
            html.Hr(style={
                "borderColor": THEME["dark"]["border"],
                "margin": "24px 0",
            }),
            html.Div("Side-by-Side Operator Cards",
                     className="section-title",
                     style={"color": THEME["dark"]["text"],
                            "fontWeight": 700, "fontSize": "14px",
                            "marginBottom": "14px"}),
            html.Label("Select Leaders", style=LABEL_STYLE),
            dcc.Dropdown(
                id="comparison-participants",
                options=[{"label": n, "value": n} for n in all_names],
                value=(all_names[:2] if len(all_names) >= 2
                       else all_names),
                multi=True, style=DROPDOWN_STYLE(),
            ),
            html.Div(id="comparison-cards-body",
                     style={"marginTop": "16px"}),
        ], className="tab-fade-in")

    return None


# 6 — Letter tab selection, df-store, anchor-graph-dash, or theme → per-factor chart
@app.callback(
    Output("letter-chart-body",  "children"),
    Input("letter-tabs",         "active_tab"),
    Input("df-store",            "data"),
    Input("anchor-graph-dash",   "value"),
    Input("theme-store",         "data"),
)
def update_letter_chart(letter, df_json, anchor_graph, theme):
    theme = theme or "dark"
    if not df_json or not letter:
        return None
    df = pd.read_json(io.StringIO(df_json), orient="records")
    return _graph_card(
        dcc.Graph(
            figure=build_letter_mean_combo(df, letter,
                                           anchor_graph, theme),
            config={"displayModeBar": False},
        ),
        theme=theme,
    )


# 7 — Participant dropdown → rebuild full operator report card
@app.callback(
    Output("participant-card-body", "children"),
    Input("selected-participant",   "value"),
    State("profiles-store",         "data"),
)
def update_participant_card(name, profiles_json):
    if not name or not profiles_json:
        return None
    profiles       = json.loads(profiles_json)
    profile_lookup = {p["participant_name"]: p for p in profiles}
    return (participant_card(profile_lookup[name])
            if name in profile_lookup else None)


# 8 — Radar participant/graph selection or theme → rebuild radar chart
@app.callback(
    Output("radar-chart",       "figure"),
    Input("radar-participants", "value"),
    Input("radar-graph-choice", "value"),
    State("profiles-store",     "data"),
    State("theme-store",        "data"),
)
def update_radar(selected_names, graph_choice, profiles_json, theme):
    theme = theme or "dark"
    c = _tc(theme)
    if not selected_names or not profiles_json:
        return go.Figure(layout=dict(
            paper_bgcolor=c["surface"],
            plot_bgcolor=c["surface"],
            font=dict(color=c["text"]),
        ))
    profiles       = json.loads(profiles_json)
    profile_lookup = {p["participant_name"]: p for p in profiles}
    selected       = [profile_lookup[n]
                      for n in selected_names if n in profile_lookup]
    return (build_multi_radar_chart(selected, graph_choice, theme)
            if selected else go.Figure())


# 9 — Comparison participant dropdown → rebuild side-by-side cards
@app.callback(
    Output("comparison-cards-body",   "children"),
    Input("comparison-participants",  "value"),
    State("profiles-store",           "data"),
)
def update_comparison_cards(selected_names, profiles_json):
    if not selected_names or not profiles_json:
        return html.P(
            "Select at least one leader above.",
            style={"color": THEME["dark"]["muted"], "fontSize": "12px"},
        )
    profiles       = json.loads(profiles_json)
    profile_lookup = {p["participant_name"]: p for p in profiles}
    selected       = [profile_lookup[n]
                      for n in selected_names if n in profile_lookup]
    rows = []
    for i in range(0, len(selected), 2):
        pair = selected[i:i + 2]
        cols = [dbc.Col(comparison_card(p), width=6) for p in pair]
        rows.append(dbc.Row(cols, className="g-3 mb-2"))
    return html.Div(rows)


# 10 — CSV export button → trigger file download of team dataframe
@app.callback(
    Output("download-csv", "data"),
    Input("btn-csv",       "n_clicks"),
    State("df-store",      "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, df_json):
    if not df_json:
        return dash.no_update
    df = pd.read_json(io.StringIO(df_json), orient="records")
    return dcc.send_data_frame(df.to_csv,
                               "sdg_disc_team_summary.csv", index=False)


# 11 — JSON export button → trigger file download of full profiles store
@app.callback(
    Output("download-json", "data"),
    Input("btn-json",       "n_clicks"),
    State("profiles-store", "data"),
    prevent_initial_call=True,
)
def download_json(n_clicks, profiles_json):
    if not profiles_json:
        return dash.no_update
    return dict(
        content=json.dumps(json.loads(profiles_json), indent=2),
        filename="sdg_disc_profiles.json",
    )


# ── Clientside: theme toggle button → swap data-theme on body,
#    animate knob/track, update icon colours
app.clientside_callback(
    """
    function(n_clicks, current_theme) {
        if (!n_clicks) return [window.dash_clientside.no_update,
                               window.dash_clientside.no_update,
                               window.dash_clientside.no_update,
                               window.dash_clientside.no_update,
                               window.dash_clientside.no_update];
        var isLight = (current_theme === 'light');
        var newTheme = isLight ? 'dark' : 'light';
        document.body.setAttribute('data-theme', newTheme);
        var knobStyle = {
            width:'18px', height:'18px', borderRadius:'50%',
            position:'absolute', top:'3px', left:'3px',
            transform:       isLight ? 'translateX(0px)'  : 'translateX(20px)',
            backgroundColor: isLight ? '#e6edf3'          : '#facc15',
            transition: 'transform 0.3s ease, background-color 0.3s'
        };
        var trackStyle = {
            width:'44px', height:'24px', borderRadius:'12px',
            position:'relative', cursor:'pointer',
            margin:'0 8px', flexShrink:'0',
            transition: 'background-color 0.3s',
            backgroundColor: isLight ? '#30363d' : '#ca8a04'
        };
        var moonStyle = {fontSize:'13px', transition:'color 0.3s',
                         color: isLight ? '#8b949e' : '#58a6ff'};
        var sunStyle  = {fontSize:'13px', transition:'color 0.3s',
                         color: isLight ? '#8b949e' : '#d29922'};
        return [newTheme, knobStyle, trackStyle, moonStyle, sunStyle];
    }
    """,
    Output("theme-store",    "data"),
    Output("theme-knob",     "style"),
    Output("theme-toggle",   "style"),
    Output("theme-icon-moon","style"),
    Output("theme-icon-sun", "style"),
    Input("theme-toggle",    "n_clicks"),
    State("theme-store",     "data"),
    prevent_initial_call=True,
)

# ── Clientside: file selected in upload → show scan status banner immediately
#    (fires before server round-trip so user sees feedback instantly)
app.clientside_callback(
    """
    function(contents, filenames) {
        if (!contents || contents.length === 0)
            return window.dash_clientside.no_update;
        var count = contents.length;
        var names = filenames ? filenames.join(', ') : '';
        return {
            props: {
                children: [{
                    type:'Div', namespace:'dash_html_components',
                    props: {
                        style: {
                            display:'flex', alignItems:'center', gap:'12px',
                            backgroundColor:'#1c2333',
                            border:'1px solid #30363d',
                            borderLeft:'3px solid #58a6ff',
                            borderRadius:'8px', padding:'12px 16px',
                            marginBottom:'16px', fontSize:'13px',
                            color:'#e6edf3'
                        },
                        children: [
                            {type:'Span', namespace:'dash_html_components',
                             props:{className:'scan-spinner', style:{
                                 display:'inline-block',
                                 width:'16px', height:'16px',
                                 border:'2px solid #30363d',
                                 borderTop:'2px solid #58a6ff',
                                 borderRadius:'50%', flexShrink:0}}},
                            {type:'Span', namespace:'dash_html_components',
                             props:{children:'Scanning '+count+' PDF'
                                    +(count>1?'s':'')+' for DISC & EQ-i data...',
                                    style:{fontWeight:600}}},
                            {type:'Span', namespace:'dash_html_components',
                             props:{children:names, style:{
                                 color:'#8b949e', fontSize:'11px',
                                 overflow:'hidden',
                                 textOverflow:'ellipsis',
                                 whiteSpace:'nowrap',
                                 maxWidth:'400px'}}}
                        ]
                    }
                }]
            },
            type:'Div', namespace:'dash_html_components'
        };
    }
    """,
    Output("scan-status", "children", allow_duplicate=True),
    Input("upload-pdfs",  "contents"),
    State("upload-pdfs",  "filename"),
    prevent_initial_call=True,
)

if __name__ == "__main__":
    app.run(debug=True)
