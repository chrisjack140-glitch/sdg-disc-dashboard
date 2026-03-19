import io
import json
from collections import Counter

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.disc import (
    FACTORS, GRAPHS, FACTOR_COLORS,
    decode_upload, process_uploaded_files,
)

# ─────────────────────────────────────────
# Theme
# ─────────────────────────────────────────
BG       = "#0d1117"
SURFACE  = "#161b22"
SURFACE2 = "#1c2333"
BORDER   = "#30363d"
TEXT     = "#e6edf3"
MUTED    = "#8b949e"
ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
RED_IND  = "#f85149"
PURPLE   = "#bc8cff"
CYAN     = "#39d353"

# ─────────────────────────────────────────
# App init
# ─────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
)
server = app.server

# ─────────────────────────────────────────
# Style dicts
# ─────────────────────────────────────────
CARD_STYLE = {
    "backgroundColor": SURFACE,
    "border": f"1px solid {BORDER}",
    "borderRadius": "12px",
    "padding": "20px",
    "marginBottom": "16px",
    "boxShadow": "0 4px 24px rgba(0,0,0,0.45)",
}
SECTION_STYLE = {
    "backgroundColor": SURFACE2,
    "border": f"1px solid {BORDER}",
    "borderRadius": "10px",
    "padding": "14px",
    "marginBottom": "12px",
}
GRAPH_CARD = {
    "backgroundColor": SURFACE,
    "border": f"1px solid {BORDER}",
    "borderRadius": "12px",
    "overflow": "hidden",
    "boxShadow": "0 4px 24px rgba(0,0,0,0.45)",
    "marginBottom": "0",
}
# Dark dropdown with white text
DROPDOWN_STYLE = {
    "backgroundColor": SURFACE2,
    "color": TEXT,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
}
LABEL_STYLE = {
    "color": MUTED,
    "fontSize": "11px",
    "fontWeight": "600",
    "letterSpacing": "0.06em",
    "textTransform": "uppercase",
    "marginBottom": "6px",
    "display": "block",
}

# ─────────────────────────────────────────
# Chart helpers — theme-aware
# ─────────────────────────────────────────

# Light theme palette
LIGHT = {
    "bg":      "#ffffff",
    "surface": "#f3f4f6",
    "border":  "#d0d7de",
    "text":    "#1f2328",
    "muted":   "#57606a",
}
DARK = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "border":  "#30363d",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
}

def _theme_colors(theme: str) -> dict:
    return LIGHT if theme == "light" else DARK


def _base_layout(title: str, height: int = 380,
                 extra: dict = None, theme: str = "dark") -> dict:
    c = _theme_colors(theme)
    layout = dict(
        paper_bgcolor=c["surface"],
        plot_bgcolor=c["surface"],
        font=dict(color=c["text"], family="Inter, system-ui, sans-serif"),
        title=dict(text=title, font=dict(color=c["muted"], size=12,
                   family="Inter, system-ui"), x=0.01, xanchor="left"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=c["muted"], size=11),
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        height=height,
        margin=dict(l=44, r=16, t=48, b=36),
        transition={"duration": 600, "easing": "cubic-in-out"},
    )
    if extra:
        layout.update(extra)
    return layout


def _axis(title_text: str = "", show_grid: bool = True,
          fixed_range: list = None, theme: str = "dark") -> dict:
    c = _theme_colors(theme)
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


def build_anchor_comparison_chart(df, anchor_graph, theme="dark"):
    c = _theme_colors(theme)
    fig = go.Figure()
    for f in FACTORS:
        fig.add_trace(go.Bar(
            x=df["participant_name"], y=df[f"{anchor_graph}_{f}"],
            name=f, marker_color=FACTOR_COLORS[f], marker_line_width=0, opacity=0.9,
            hovertemplate="<b>%{x}</b><br>" + f + ": %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(**_base_layout(
        f"Anchor Score Comparison — {anchor_graph.title()}", height=340, theme=theme,
        extra=dict(barmode="group", xaxis=_axis(show_grid=False, theme=theme),
                   yaxis=_axis("Score", theme=theme), bargap=0.18, bargroupgap=0.04),
    ))
    return fig


def build_heatmap(df, anchor_graph, theme="dark"):
    c = _theme_colors(theme)
    z = df[[f"{anchor_graph}_{f}" for f in FACTORS]].values
    colorscale = [[0.00,"#1e3a8a"],[0.30,"#3b82f6"],[0.50,"#f9fafb"],[0.70,"#ef4444"],[1.00,"#7f1d1d"]]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=FACTORS, y=df["participant_name"], colorscale=colorscale,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>", xgap=1, ygap=1,
    ))
    fig.update_layout(**_base_layout(
        f"Score Heatmap — {anchor_graph.title()}", height=max(280, 56*len(df)), theme=theme,
        extra=dict(xaxis=_axis(show_grid=False, theme=theme),
                   yaxis=_axis(show_grid=False, theme=theme),
                   plot_bgcolor="#000000", paper_bgcolor=c["surface"]),
    ))
    return fig


def build_disc_type_chart(profiles, theme="dark"):
    c = _theme_colors(theme)
    type_counts = Counter(p.get("style_type", "—") for p in profiles)
    labels = sorted(type_counts.keys(), key=lambda k: -type_counts[k])
    counts = [type_counts[k] for k in labels]
    palette = [FACTOR_COLORS["D"], FACTOR_COLORS["C"], FACTOR_COLORS["S"], FACTOR_COLORS["I"],
               PURPLE, CYAN, "#f97316", "#a78bfa"]
    bar_colors = [palette[i % len(palette)] for i in range(len(labels))]
    fig = go.Figure(go.Bar(
        x=labels, y=counts, marker_color=bar_colors, marker_line_width=0, opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        text=counts, textposition="outside", textfont=dict(color=c["muted"], size=11),
    ))
    fig.update_layout(**_base_layout(
        "DISC Type Distribution (from PDF)", height=340, theme=theme,
        extra=dict(xaxis=_axis("Type", show_grid=False, theme=theme),
                   yaxis=_axis("Count", fixed_range=[0, max(counts)+1.5], theme=theme),
                   showlegend=False),
    ))
    return fig


def build_multi_radar_chart(selected_profiles, graph_name, theme="dark"):
    categories = ["DI","I","IS","S","SC","C","CD","D"]
    palette = [FACTOR_COLORS["D"], FACTOR_COLORS["I"], FACTOR_COLORS["S"], FACTOR_COLORS["C"],
               PURPLE, CYAN, "#fb7185", "#f97316"]
    c = _theme_colors(theme)
    fig = go.Figure()
    for idx, profile in enumerate(selected_profiles):
        g = profile["graphs"][graph_name]
        d, i, s, cv = g["D"], g["I"], g["S"], g["C"]
        vals = [(d+i)/2, i, (i+s)/2, s, (s+cv)/2, cv, (cv+d)/2, d]
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=categories+[categories[0]], fill="none",
            name=profile["participant_name"],
            line=dict(color=palette[idx % len(palette)], width=2), opacity=0.9,
            hovertemplate="<b>%{fullData.name}</b><br>%{theta}: %{r:.2f}<extra></extra>",
        ))
    fig.update_layout(**_base_layout(
        f"Radar Comparison — {graph_name.title()}", height=520, theme=theme,
        extra=dict(polar=dict(
            bgcolor=c["surface"],
            angularaxis=dict(categoryorder="array", categoryarray=categories,
                             direction="clockwise", rotation=90,
                             gridcolor=c["border"], linecolor=c["border"],
                             tickfont=dict(color=c["text"], size=11)),
            radialaxis=dict(visible=True, range=[-8,8], tickmode="array",
                            tickvals=[-8,-4,0,4,8], ticktext=["-8","-4","0","4","8"],
                            angle=0, tickangle=-90,
                            gridcolor=c["border"], linecolor=c["border"],
                            tickfont=dict(color=c["muted"], size=9)),
        )),
    ))
    return fig


def build_letter_mean_combo(df, letter, anchor_graph, theme="dark"):
    col = f"{anchor_graph}_{letter}"
    sorted_df = df[["participant_name", col]].copy().sort_values(col, ascending=False)
    names = sorted_df["participant_name"].tolist()
    scores = sorted_df[col].tolist()
    mean_val = float(df[col].mean())
    color = FACTOR_COLORS[letter]
    c = _theme_colors(theme)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=scores, name=f"{letter} Score",
        marker_color=color, marker_line_width=0, opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=names, y=[mean_val]*len(names), mode="lines",
        name=f"Mean ({mean_val:+.2f})", line=dict(color=c["text"], width=1.5, dash="dash"),
        hovertemplate=f"Team Mean: {mean_val:+.2f}<extra></extra>"))
    fig.add_shape(type="line", x0=-0.5, x1=len(names)-0.5, y0=0, y1=0,
        line=dict(color=c["border"], width=1.5))
    fig.update_layout(**_base_layout(
        f"{letter}  —  Scores (Greatest to Least)", height=360, theme=theme,
        extra=dict(xaxis=_axis(show_grid=False, theme=theme),
                   yaxis=_axis("Score", fixed_range=[-8,8], theme=theme)),
    ))
    return fig


def shift_badge(value: float) -> html.Span:
    if abs(value) <= 0.15:
        return html.Span(f"± {abs(value):.2f}",
                         style={"color": MUTED, "fontWeight": 600, "fontSize": "11px"})
    if value > 0:
        return html.Span(f"▲ {value:+.2f}",
                         style={"color": GREEN, "fontWeight": 700, "fontSize": "11px"})
    return html.Span(f"▼ {value:+.2f}",
                     style={"color": RED_IND, "fontWeight": 700, "fontSize": "11px"})


def metric_cards(df: pd.DataFrame, anchor_graph: str) -> dbc.Row:
    """
    Four aligned metric tiles — coloured accent bar + large score only.
    No sparkline bar chart inside (removed as requested).
    """
    cols = []
    for f in FACTORS:
        col   = f"{anchor_graph}_{f}"
        mean  = float(df[col].mean())
        color = FACTOR_COLORS[f]
        sign  = "+" if mean >= 0 else ""
        cols.append(dbc.Col(
            html.Div([
                # Coloured accent stripe across top
                html.Div(style={
                    "height": "3px",
                    "backgroundColor": color,
                    "borderRadius": "12px 12px 0 0",
                }),
                html.Div([
                    html.Div(f, style={
                        "color": color, "fontWeight": 800, "fontSize": "11px",
                        "letterSpacing": "0.1em", "textTransform": "uppercase",
                        "marginBottom": "4px",
                    }),
                    html.Div("Team Mean", style={
                        "color": MUTED, "fontSize": "10px", "marginBottom": "8px",
                    }),
                    html.Div(f"{sign}{mean:.2f}", style={
                        "fontSize": "34px", "fontWeight": 900, "color": TEXT,
                        "lineHeight": "1", "letterSpacing": "-0.02em",
                    }),
                ], style={"padding": "14px 16px 20px 16px"}),
            ], style={
                "backgroundColor": SURFACE,
                "border": f"1px solid {BORDER}",
                "borderRadius": "12px",
                "boxShadow": "0 4px 20px rgba(0,0,0,0.35)",
                "overflow": "hidden",
                # CSS transition for hover (handled in style.css)
            }, className="metric-hover"),
            # Equal width so all 4 tiles align perfectly
        ))
    return dbc.Row(cols, className="mb-4 g-3")


def ranking_table(df: pd.DataFrame, anchor_graph: str, sort_factor: str) -> html.Div:
    col       = f"{anchor_graph}_{sort_factor}"
    sorted_df = (
        df[["participant_name", col, f"{sort_factor}_bucket"]]
        .copy()
        .sort_values(col, ascending=False)   # true descending (positives first)
        .reset_index(drop=True)
    )
    color = FACTOR_COLORS[sort_factor]

    badge_styles = [
        {"backgroundColor": "#d29922", "color": "#000"},
        {"backgroundColor": "#8b949e", "color": "#000"},
        {"backgroundColor": "#c0640a", "color": "#fff"},
    ]
    base_badge = {
        "display": "inline-flex", "alignItems": "center", "justifyContent": "center",
        "width": "22px", "height": "22px", "borderRadius": "50%",
        "fontSize": "10px", "fontWeight": 800,
        "backgroundColor": BORDER, "color": TEXT,
    }
    header = html.Thead(html.Tr([
        html.Th("#",                    style={"color": MUTED, "fontSize": "10px",
                                               "padding": "8px 12px",
                                               "borderBottom": f"1px solid {BORDER}",
                                               "backgroundColor": BG}),
        html.Th("PARTICIPANT",          style={"color": MUTED, "fontSize": "10px",
                                               "padding": "8px 12px",
                                               "borderBottom": f"1px solid {BORDER}",
                                               "backgroundColor": BG}),
        html.Th(f"{sort_factor} SCORE", style={"color": color, "fontSize": "10px",
                                               "padding": "8px 12px",
                                               "borderBottom": f"1px solid {BORDER}",
                                               "backgroundColor": BG}),
        html.Th("BUCKET",               style={"color": MUTED, "fontSize": "10px",
                                               "padding": "8px 12px",
                                               "borderBottom": f"1px solid {BORDER}",
                                               "backgroundColor": BG}),
    ]))

    tbody_rows = []
    for i, row in sorted_df.iterrows():
        rank   = i + 1
        bstyle = {**base_badge, **(badge_styles[i] if i < 3 else {})}
        score  = row[col]
        sign   = "+" if score >= 0 else ""
        bg     = SURFACE if i % 2 == 0 else SURFACE2
        tbody_rows.append(html.Tr([
            html.Td(html.Span(str(rank), style=bstyle),
                    style={"padding": "8px 12px", "backgroundColor": bg}),
            html.Td(row["participant_name"],
                    style={"padding": "8px 12px", "color": TEXT,
                           "fontSize": "13px", "backgroundColor": bg}),
            html.Td(f"{sign}{score:.2f}",
                    style={"padding": "8px 12px", "color": color,
                           "fontWeight": 700, "fontSize": "13px", "backgroundColor": bg}),
            html.Td(row[f"{sort_factor}_bucket"],
                    style={"padding": "8px 12px", "color": MUTED,
                           "fontSize": "11px", "backgroundColor": bg}),
        ], className="rank-row"))

    return html.Div([
        html.Table(
            [header, html.Tbody(tbody_rows)],
            style={"width": "100%", "borderCollapse": "collapse"},
        ),
    ], style={**SECTION_STYLE, "padding": "0", "overflow": "hidden"})


def participant_card(profile: dict) -> html.Div:
    top_two    = ", ".join(profile["summary"]["top_two"])
    style_type = profile.get("style_type", "—")
    factor_cols = []
    for idx, f in enumerate(FACTORS):
        fp    = profile["factor_profiles"][f]
        color = FACTOR_COLORS[f]
        traits = [html.Li(t, style={"fontSize": "12px", "color": MUTED,
                                     "marginBottom": "3px"})
                  for t in fp["traits"][:4]]
        factor_cols.append(dbc.Col(html.Div([
            html.Div([
                html.Span(f, style={"fontSize": "26px", "fontWeight": 900, "color": color}),
                html.Span(f" {fp['anchor_score']:+.2f}",
                          style={"fontSize": "13px", "color": MUTED, "marginLeft": "6px"}),
            ], style={"marginBottom": "2px"}),
            html.Div(fp["bucket"].replace("_", " ").title(), style={
                "fontSize": "10px", "color": color, "fontWeight": 700,
                "letterSpacing": "0.06em", "textTransform": "uppercase",
                "marginBottom": "10px",
            }),
            html.Div("Traits", style={
                "color": MUTED, "fontSize": "10px", "fontWeight": 700,
                "letterSpacing": "0.06em", "textTransform": "uppercase",
                "marginBottom": "4px",
            }),
            html.Ul(traits, style={"paddingLeft": "14px", "marginBottom": "10px"}),
            html.Div(style={"height": "1px", "backgroundColor": BORDER, "marginBottom": "8px"}),
            html.Div([
                html.Span("Public → Stress:  ", style={"color": MUTED, "fontSize": "11px"}),
                shift_badge(fp["delta_public_to_stress"]),
            ], style={"marginBottom": "3px"}),
            html.Div([
                html.Span("Mirror → Stress:  ", style={"color": MUTED, "fontSize": "11px"}),
                shift_badge(fp["delta_mirror_to_stress"]),
            ]),
        ], style={
            "backgroundColor": SURFACE2,
            "border": f"1px solid {BORDER}",
            "borderRadius": "10px",
            "padding": "14px",
            "height": "100%",
            # Staggered animation via CSS class
        }, className=f"factor-tile factor-tile-{idx}")))

    g = profile["graphs"]
    return html.Div([
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div("SDG DISC Operator Report", style={
                        "fontSize": "17px", "fontWeight": 800, "color": TEXT,
                        "marginBottom": "4px",
                    }),
                    html.Div([
                        html.Span(profile.get("participant_name", ""),
                                  style={"color": ACCENT, "fontWeight": 700, "fontSize": "13px"}),
                        html.Span(
                            f"  ·  Anchor: {profile.get('anchor_graph','stress').title()}"
                            f"  ·  Top Two: {top_two}",
                            style={"color": MUTED, "fontSize": "12px"},
                        ),
                    ]),
                ]),
                dbc.Col([
                    html.Div("DISC STYLE", style={
                        "color": MUTED, "fontSize": "9px", "fontWeight": 700,
                        "letterSpacing": "0.1em", "textAlign": "right",
                    }),
                    html.Div(style_type, style={
                        "fontSize": "26px", "fontWeight": 900,
                        "color": ACCENT, "textAlign": "right",
                    }),
                ], width="auto"),
            ], align="center"),
        ], style=CARD_STYLE),
        dbc.Row(factor_cols, className="g-3 mb-3"),
        html.Div([
            html.Div("Raw Graph Scores", style={
                "color": MUTED, "fontSize": "10px", "fontWeight": 700,
                "letterSpacing": "0.08em", "textTransform": "uppercase",
                "marginBottom": "12px",
            }),
            dbc.Row([
                dbc.Col([
                    html.Div("PUBLIC", style={"color": MUTED, "fontSize": "9px",
                                              "fontWeight": 700, "marginBottom": "4px"}),
                    html.Div(
                        f"D {g['public']['D']:+.2f}  I {g['public']['I']:+.2f}  "
                        f"S {g['public']['S']:+.2f}  C {g['public']['C']:+.2f}",
                        style={"color": TEXT, "fontSize": "12px", "fontFamily": "monospace"},
                    ),
                ]),
                dbc.Col([
                    html.Div("STRESS", style={"color": MUTED, "fontSize": "9px",
                                              "fontWeight": 700, "marginBottom": "4px"}),
                    html.Div(
                        f"D {g['stress']['D']:+.2f}  I {g['stress']['I']:+.2f}  "
                        f"S {g['stress']['S']:+.2f}  C {g['stress']['C']:+.2f}",
                        style={"color": TEXT, "fontSize": "12px", "fontFamily": "monospace"},
                    ),
                ]),
                dbc.Col([
                    html.Div("MIRROR", style={"color": MUTED, "fontSize": "9px",
                                              "fontWeight": 700, "marginBottom": "4px"}),
                    html.Div(
                        f"D {g['mirror']['D']:+.2f}  I {g['mirror']['I']:+.2f}  "
                        f"S {g['mirror']['S']:+.2f}  C {g['mirror']['C']:+.2f}",
                        style={"color": TEXT, "fontSize": "12px", "fontFamily": "monospace"},
                    ),
                ]),
            ]),
        ], style=CARD_STYLE),
    ])


def comparison_card(profile: dict) -> html.Div:
    top_two    = ", ".join(profile["summary"]["top_two"])
    style_type = profile.get("style_type", "—")
    mini_cols  = []
    for f in FACTORS:
        fp    = profile["factor_profiles"][f]
        color = FACTOR_COLORS[f]
        traits = "; ".join(fp["traits"][:2])
        mini_cols.append(dbc.Col(html.Div([
            html.Div([
                html.Span(f, style={"fontWeight": 900, "color": color, "fontSize": "15px"}),
                html.Span(f" {fp['anchor_score']:+.2f}",
                          style={"fontSize": "12px", "color": MUTED}),
            ], style={"marginBottom": "2px"}),
            html.Div(fp["bucket"].replace("_", " "), style={
                "fontSize": "10px", "color": color, "fontWeight": 600, "marginBottom": "4px",
            }),
            html.Div(traits, style={"fontSize": "10px", "color": MUTED,
                                    "lineHeight": "1.4", "marginBottom": "6px"}),
            html.Div([
                html.Span("P→S ", style={"color": MUTED, "fontSize": "10px"}),
                shift_badge(fp["delta_public_to_stress"]),
                html.Span("  M→S ", style={"color": MUTED, "fontSize": "10px"}),
                shift_badge(fp["delta_mirror_to_stress"]),
            ]),
        ], style={"backgroundColor": BG, "borderRadius": "8px", "padding": "10px"}),
        width=6))

    g = profile["graphs"]
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div(profile.get("participant_name", ""), style={
                    "fontSize": "15px", "fontWeight": 800, "color": TEXT,
                }),
                html.Div(
                    f"Top Two: {top_two}  ·  Anchor: {profile.get('anchor_graph','stress')}",
                    style={"fontSize": "11px", "color": MUTED, "marginTop": "2px"},
                ),
            ]),
            dbc.Col(
                html.Div(style_type, style={
                    "fontSize": "22px", "fontWeight": 900,
                    "color": ACCENT, "textAlign": "right",
                }),
                width="auto",
            ),
        ], align="center", className="mb-3"),
        dbc.Row(mini_cols, className="g-2 mb-2"),
        html.Div(
            f"Public D{g['public']['D']:+.1f} I{g['public']['I']:+.1f} "
            f"S{g['public']['S']:+.1f} C{g['public']['C']:+.1f}  |  "
            f"Stress D{g['stress']['D']:+.1f} I{g['stress']['I']:+.1f} "
            f"S{g['stress']['S']:+.1f} C{g['stress']['C']:+.1f}  |  "
            f"Mirror D{g['mirror']['D']:+.1f} I{g['mirror']['I']:+.1f} "
            f"S{g['mirror']['S']:+.1f} C{g['mirror']['C']:+.1f}",
            style={"color": MUTED, "fontSize": "10px",
                   "fontFamily": "monospace", "marginTop": "4px"},
        ),
    ], style=SECTION_STYLE)


def _graph_card(children, title: str = "") -> html.Div:
    header = ([html.Div(title, style={
        "padding": "12px 16px 0 16px",
        "fontSize": "12px", "fontWeight": 700, "color": MUTED,
        "letterSpacing": "0.04em",
    })] if title else [])
    return html.Div([*header, children], style=GRAPH_CARD)

# ─────────────────────────────────────────
# Layout
# ─────────────────────────────────────────
app.layout = html.Div(
    style={"backgroundColor": BG, "minHeight": "100vh",
           "fontFamily": "Inter, system-ui, sans-serif", "color": TEXT},
    children=[

        dcc.Store(id="profiles-store"),
        dcc.Store(id="df-store"),
        dcc.Store(id="theme-store", data="dark"),   # persists current theme
        dcc.Download(id="download-csv"),
        dcc.Download(id="download-json"),

        # ── Sticky header ──────────────────────────────────────────────
        html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("SDG", style={
                                "backgroundColor": ACCENT, "color": BG,
                                "fontWeight": 900, "fontSize": "11px",
                                "padding": "3px 8px", "borderRadius": "5px",
                                "marginRight": "10px", "letterSpacing": "0.05em",
                            }),
                            html.Span("DISC Dashboard", style={
                                "fontWeight": 700, "fontSize": "15px", "color": TEXT,
                            }),
                        ], style={"display": "flex", "alignItems": "center"}),
                    ], width="auto"),
                    dbc.Col(
                        dbc.Tabs(
                            id="tabs", active_tab="team",
                            children=[
                                dbc.Tab(label="Team Dashboard",    tab_id="team"),
                                dbc.Tab(label="Individual Results", tab_id="individual"),
                                dbc.Tab(label="Comparisons",        tab_id="comparisons"),
                            ],
                            style={"borderBottom": "none"},
                        ),
                        style={"display": "flex", "alignItems": "flex-end",
                               "justifyContent": "flex-end"},
                    ),
                    # ── Light switch ──
                    dbc.Col([
                        html.Div([
                            html.Span("🌙", id="theme-icon-moon",
                                      style={"fontSize": "13px", "color": ACCENT,
                                             "transition": "color 0.3s"}),
                            html.Div(
                                html.Div(id="theme-knob", style={
                                    "width": "18px", "height": "18px",
                                    "borderRadius": "50%",
                                    "backgroundColor": "#e6edf3",
                                    "position": "absolute",
                                    "top": "3px", "left": "3px",
                                    "transition": "transform 0.3s ease, background-color 0.3s",
                                }),
                                id="theme-toggle",
                                n_clicks=0,
                                style={
                                    "width": "44px", "height": "24px",
                                    "backgroundColor": "#30363d",
                                    "borderRadius": "12px",
                                    "position": "relative",
                                    "cursor": "pointer",
                                    "margin": "0 8px",
                                    "transition": "background-color 0.3s",
                                    "flexShrink": "0",
                                },
                            ),
                            html.Span("☀", id="theme-icon-sun",
                                      style={"fontSize": "13px", "color": MUTED,
                                             "transition": "color 0.3s"}),
                        ], style={"display": "flex", "alignItems": "center",
                                  "gap": "4px"}),
                    ], width="auto"),
                ], align="center", justify="between"),
            ], fluid=True),
        ], style={
            "position": "sticky", "top": "0", "zIndex": "1000",
            "backgroundColor": SURFACE,
            "borderBottom": f"1px solid {BORDER}",
            "padding": "12px 0",
            "boxShadow": "0 2px 20px rgba(0,0,0,0.5)",
        }, className="sticky-header"),

        # ── Body ────────────────────────────────────────────────────────
        dbc.Container([

            # Controls
            dbc.Row([
                dbc.Col([
                    html.Label("Anchor Graph", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="anchor-graph",
                        options=[{"label": g.title(), "value": g}
                                 for g in ["stress", "mirror", "public"]],
                        value="stress", clearable=False, style=DROPDOWN_STYLE,
                    ),
                ], width=2),
                dbc.Col([
                    html.Label("Upload Maxwell DISC PDFs", style=LABEL_STYLE),
                    dcc.Upload(
                        id="upload-pdfs",
                        children=html.Div([
                            "Drag & drop or ",
                            html.A("browse", style={"color": ACCENT, "cursor": "pointer",
                                                    "fontWeight": 600}),
                        ], style={"fontSize": "12px"}),
                        style={
                            "width": "100%", "height": "38px", "lineHeight": "38px",
                            "borderWidth": "1px", "borderStyle": "dashed",
                            "borderRadius": "8px", "textAlign": "center",
                            "borderColor": BORDER, "color": MUTED,
                            "backgroundColor": SURFACE,
                        },
                        multiple=True,
                    ),
                ], width=4),
            ], className="mb-4 g-3", style={"paddingTop": "24px"}),

            html.Div(id="upload-errors"),

            # PDF scan status banner — visible only while processing
            html.Div(id="scan-status"),

            # Metric cards
            html.Div(id="metric-cards"),

            # Collapsible ranking — DISC filter buttons
            html.Div([
                dbc.Row([
                    dbc.Col(
                        dbc.Button(
                            [html.Span("▶ ", id="rank-chevron",
                                       style={"fontSize": "10px", "marginRight": "4px"}),
                             "Participant Rankings"],
                            id="rank-toggle", color="link", n_clicks=0,
                            style={"color": MUTED, "fontSize": "11px", "fontWeight": 700,
                                   "letterSpacing": "0.08em", "textTransform": "uppercase",
                                   "textDecoration": "none", "padding": "8px 0",
                                   "border": "none", "background": "none"},
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.ButtonGroup([
                            dbc.Button(f, id=f"rank-btn-{f}", n_clicks=0,
                                       style={
                                           "backgroundColor": SURFACE2,
                                           "color": FACTOR_COLORS[f],
                                           "border": f"1px solid {FACTOR_COLORS[f]}",
                                           "fontSize": "11px", "fontWeight": 800,
                                           "padding": "4px 12px", "borderRadius": "6px",
                                       })
                            for f in FACTORS
                        ], style={"gap": "6px"}),
                        width="auto",
                    ),
                ], align="center", className="mb-1"),
                dbc.Collapse(
                    html.Div(id="ranking-table"),
                    id="rank-collapse",
                    is_open=False,
                ),
            ], className="mb-4"),

            html.Div(id="tab-content"),

            # Downloads
            html.Hr(style={"borderColor": BORDER, "marginTop": "32px",
                            "marginBottom": "20px"}),
            dbc.Row([
                dbc.Col(html.Button(
                    "↓  Export CSV", id="btn-csv", n_clicks=0,
                    style={"backgroundColor": SURFACE, "color": TEXT,
                           "border": f"1px solid {BORDER}", "borderRadius": "8px",
                           "padding": "10px 20px", "fontSize": "12px",
                           "cursor": "pointer", "width": "100%",
                           "fontWeight": 600, "letterSpacing": "0.04em"},
                ), width=2),
                dbc.Col(html.Button(
                    "↓  Export JSON", id="btn-json", n_clicks=0,
                    style={"backgroundColor": SURFACE, "color": TEXT,
                           "border": f"1px solid {BORDER}", "borderRadius": "8px",
                           "padding": "10px 20px", "fontSize": "12px",
                           "cursor": "pointer", "width": "100%",
                           "fontWeight": 600, "letterSpacing": "0.04em"},
                ), width=2),
            ], className="mb-5 g-3"),

        ], fluid=True),
    ],
)

# ─────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────

# 1 — Process uploads
@app.callback(
    Output("profiles-store", "data"),
    Output("df-store", "data"),
    Output("upload-errors", "children"),
    Output("scan-status", "children"),
    Input("upload-pdfs", "contents"),
    State("upload-pdfs", "filename"),
    State("anchor-graph", "value"),
    prevent_initial_call=True,
)
def process_uploads(contents_list, filenames, anchor_graph):
    if not contents_list:
        return None, None, None, None
    files_data = [decode_upload(c, f) for c, f in zip(contents_list, filenames)]
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
        return None, None, dbc.Alert("No valid profiles found.", color="danger"), None
    # Processing complete — clear the scan banner
    return json.dumps(profiles), df.to_json(orient="records"), error_banner, None


# 2 — Metric cards
@app.callback(
    Output("metric-cards", "children"),
    Input("df-store", "data"),
    Input("anchor-graph", "value"),
)
def update_metric_cards(df_json, anchor_graph):
    if not df_json:
        return html.P("Upload Maxwell DISC PDFs to begin.",
                      style={"color": MUTED, "fontSize": "13px", "paddingTop": "20px"})
    df = pd.read_json(io.StringIO(df_json), orient="records")
    return metric_cards(df, anchor_graph)


# 3 — Ranking table
@app.callback(
    Output("ranking-table", "children"),
    Output("rank-btn-D", "style"),
    Output("rank-btn-I", "style"),
    Output("rank-btn-S", "style"),
    Output("rank-btn-C", "style"),
    Input("df-store", "data"),
    Input("anchor-graph", "value"),
    Input("rank-btn-D", "n_clicks"),
    Input("rank-btn-I", "n_clicks"),
    Input("rank-btn-S", "n_clicks"),
    Input("rank-btn-C", "n_clicks"),
)
def update_ranking(df_json, anchor_graph, nd, ni, ns, nc):
    from dash import ctx
    triggered = ctx.triggered_id or "rank-btn-D"
    active = triggered.replace("rank-btn-", "") if triggered in (
        "rank-btn-D", "rank-btn-I", "rank-btn-S", "rank-btn-C") else "D"

    def btn_style(f, is_active):
        base = {"fontSize": "11px", "fontWeight": 800,
                "padding": "4px 12px", "borderRadius": "6px"}
        if is_active:
            return {**base, "backgroundColor": FACTOR_COLORS[f],
                    "color": BG, "border": f"1px solid {FACTOR_COLORS[f]}"}
        return {**base, "backgroundColor": SURFACE2,
                "color": FACTOR_COLORS[f], "border": f"1px solid {FACTOR_COLORS[f]}"}

    styles = [btn_style(f, f == active) for f in FACTORS]
    if not df_json:
        return None, *styles

    df = pd.read_json(io.StringIO(df_json), orient="records")
    return ranking_table(df, anchor_graph, active), *styles


# 4 — Toggle ranking collapse
@app.callback(
    Output("rank-collapse", "is_open"),
    Output("rank-chevron", "children"),
    Input("rank-toggle", "n_clicks"),
    State("rank-collapse", "is_open"),
)
def toggle_ranking(n, is_open):
    new_open = not is_open if n else False
    return new_open, ("▼ " if new_open else "▶ ")


# 5 — Tab content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    Input("df-store", "data"),
    Input("profiles-store", "data"),
    Input("anchor-graph", "value"),
    Input("theme-store", "data"),
)
def render_tab(active_tab, df_json, profiles_json, anchor_graph, theme):
    theme = theme or "dark"
    if not df_json:
        return None
    df        = pd.read_json(io.StringIO(df_json), orient="records")
    profiles  = json.loads(profiles_json)
    all_names = df["participant_name"].tolist()

    if active_tab == "team":
        return html.Div([
            dbc.Row([
                dbc.Col(_graph_card(
                    dcc.Graph(figure=build_anchor_comparison_chart(df, anchor_graph, theme),
                              config={"displayModeBar": False}),
                ), width=8),
                dbc.Col(_graph_card(
                    dcc.Graph(figure=build_disc_type_chart(profiles, theme),
                              config={"displayModeBar": False}),
                ), width=4),
            ], className="g-3 mb-3"),
            _graph_card(dcc.Graph(figure=build_heatmap(df, anchor_graph, theme),
                                  config={"displayModeBar": True, "scrollZoom": True})),
            html.Div(style={"marginTop": "20px"}),
            html.Div("Per-Factor Mean Charts", style={
                "color": MUTED, "fontSize": "11px", "fontWeight": 700,
                "letterSpacing": "0.08em", "textTransform": "uppercase",
                "marginBottom": "10px",
            }),
            dbc.Tabs(
                id="letter-tabs", active_tab="D",
                children=[dbc.Tab(label=f, tab_id=f) for f in FACTORS],
                style={"marginBottom": "12px"},
            ),
            html.Div(id="letter-chart-body"),
        ], className="tab-fade-in")

    if active_tab == "individual":
        return html.Div([
            dbc.Row([dbc.Col([
                html.Label("Select Participant", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="selected-participant",
                    options=[{"label": n, "value": n} for n in all_names],
                    value=all_names[0], clearable=False, style=DROPDOWN_STYLE,
                ),
            ], width=4)], className="mb-4"),
            html.Div(id="participant-card-body"),
        ], className="tab-fade-in")

    if active_tab == "comparisons":
        return html.Div([
            html.Div("Radar Profile Comparison", style={
                "color": TEXT, "fontWeight": 700, "fontSize": "14px",
                "marginBottom": "14px",
            }),
            dbc.Row([
                dbc.Col([
                    html.Label("Graph Source", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="radar-graph-choice",
                        options=[{"label": g.title(), "value": g}
                                 for g in ["public", "stress", "mirror"]],
                        value="stress", clearable=False, style=DROPDOWN_STYLE,
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Overlay Participants", style=LABEL_STYLE),
                    dcc.Dropdown(
                        id="radar-participants",
                        options=[{"label": n, "value": n} for n in all_names],
                        value=[all_names[0]], multi=True, style=DROPDOWN_STYLE,
                    ),
                ], width=6),
            ], className="mb-3"),
            _graph_card(dcc.Graph(id="radar-chart", config={"displayModeBar": False})),
            html.Hr(style={"borderColor": BORDER, "margin": "24px 0"}),
            html.Div("Side-by-Side Operator Cards", style={
                "color": TEXT, "fontWeight": 700, "fontSize": "14px",
                "marginBottom": "14px",
            }),
            html.Label("Select Leaders", style=LABEL_STYLE),
            dcc.Dropdown(
                id="comparison-participants",
                options=[{"label": n, "value": n} for n in all_names],
                value=all_names[:2] if len(all_names) >= 2 else all_names,
                multi=True, style=DROPDOWN_STYLE,
            ),
            html.Div(id="comparison-cards-body", style={"marginTop": "16px"}),
        ], className="tab-fade-in")

    return None


# 6 — Letter chart
@app.callback(
    Output("letter-chart-body", "children"),
    Input("letter-tabs", "active_tab"),
    Input("df-store", "data"),
    Input("anchor-graph", "value"),
    Input("theme-store", "data"),
)
def update_letter_chart(letter, df_json, anchor_graph, theme):
    theme = theme or "dark"
    if not df_json or not letter:
        return None
    df = pd.read_json(io.StringIO(df_json), orient="records")
    return _graph_card(
        dcc.Graph(figure=build_letter_mean_combo(df, letter, anchor_graph, theme),
                  config={"displayModeBar": True, "scrollZoom": True}),
    )


# 7 — Participant card
@app.callback(
    Output("participant-card-body", "children"),
    Input("selected-participant", "value"),
    State("profiles-store", "data"),
)
def update_participant_card(name, profiles_json):
    if not name or not profiles_json:
        return None
    profiles       = json.loads(profiles_json)
    profile_lookup = {p["participant_name"]: p for p in profiles}
    return participant_card(profile_lookup[name]) if name in profile_lookup else None


# 8 — Radar chart
@app.callback(
    Output("radar-chart", "figure"),
    Input("radar-participants", "value"),
    Input("radar-graph-choice", "value"),
    State("profiles-store", "data"),
    State("theme-store", "data"),
)
def update_radar(selected_names, graph_choice, profiles_json, theme):
    theme = theme or "dark"
    c = _theme_colors(theme)
    if not selected_names or not profiles_json:
        return go.Figure(layout=dict(paper_bgcolor=c["surface"],
                                     plot_bgcolor=c["surface"],
                                     font=dict(color=c["text"])))
    profiles       = json.loads(profiles_json)
    profile_lookup = {p["participant_name"]: p for p in profiles}
    selected       = [profile_lookup[n] for n in selected_names if n in profile_lookup]
    return build_multi_radar_chart(selected, graph_choice, theme) if selected else go.Figure()


# 9 — Comparison cards
@app.callback(
    Output("comparison-cards-body", "children"),
    Input("comparison-participants", "value"),
    State("profiles-store", "data"),
)
def update_comparison_cards(selected_names, profiles_json):
    if not selected_names or not profiles_json:
        return html.P("Select at least one leader above.",
                      style={"color": MUTED, "fontSize": "12px"})
    profiles       = json.loads(profiles_json)
    profile_lookup = {p["participant_name"]: p for p in profiles}
    selected       = [profile_lookup[n] for n in selected_names if n in profile_lookup]
    rows = []
    for i in range(0, len(selected), 2):
        pair = selected[i:i+2]
        cols = [dbc.Col(comparison_card(p), width=6) for p in pair]
        rows.append(dbc.Row(cols, className="g-3 mb-2"))
    return html.Div(rows)


# 10 — CSV download
@app.callback(
    Output("download-csv", "data"),
    Input("btn-csv", "n_clicks"),
    State("df-store", "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, df_json):
    if not df_json:
        return dash.no_update
    df = pd.read_json(io.StringIO(df_json), orient="records")
    return dcc.send_data_frame(df.to_csv, "sdg_disc_team_summary.csv", index=False)


# 11 — JSON download
@app.callback(
    Output("download-json", "data"),
    Input("btn-json", "n_clicks"),
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



# ── Theme toggle — clientside for instant response ──────────────────
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

        // Swap data-theme on body — triggers all CSS variables instantly
        document.body.setAttribute('data-theme', newTheme);

        // Knob: left = dark mode, right = light mode
        var knobStyle = {
            width: '18px', height: '18px', borderRadius: '50%',
            position: 'absolute', top: '3px', left: '3px',
            transform:       isLight ? 'translateX(0px)'  : 'translateX(20px)',
            backgroundColor: isLight ? '#e6edf3'          : '#facc15',
            transition: 'transform 0.3s ease, background-color 0.3s'
        };

        // Track: dark grey in dark mode, amber/yellow in light mode
        var trackStyle = {
            width: '44px', height: '24px', borderRadius: '12px',
            position: 'relative', cursor: 'pointer',
            margin: '0 8px', flexShrink: '0',
            transition: 'background-color 0.3s',
            backgroundColor: isLight ? '#30363d' : '#ca8a04'
        };

        // Moon icon: bright in dark mode (currently active), dim in light
        var moonStyle = {
            fontSize: '13px', transition: 'color 0.3s',
            color: isLight ? '#8b949e' : '#58a6ff'
        };

        // Sun icon: bright in light mode (currently active), dim in dark
        var sunStyle = {
            fontSize: '13px', transition: 'color 0.3s',
            color: isLight ? '#8b949e' : '#d29922'
        };

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

# Clientside callback — fires the instant files are picked,
# before the server round-trip begins, so the banner appears immediately.
app.clientside_callback(
    """
    function(contents, filenames) {
        if (!contents || contents.length === 0) {
            return window.dash_clientside.no_update;
        }
        var count = contents.length;
        var names = filenames ? filenames.join(', ') : '';
        return {
            props: {
                children: [
                    {
                        type: 'Div',
                        namespace: 'dash_html_components',
                        props: {
                            style: {
                                display: 'flex',
                                alignItems: 'center',
                                gap: '12px',
                                backgroundColor: '#1c2333',
                                border: '1px solid #30363d',
                                borderLeft: '3px solid #58a6ff',
                                borderRadius: '8px',
                                padding: '12px 16px',
                                marginBottom: '16px',
                                fontSize: '13px',
                                color: '#e6edf3'
                            },
                            children: [
                                {
                                    type: 'Span',
                                    namespace: 'dash_html_components',
                                    props: {
                                        className: 'scan-spinner',
                                        style: {
                                            display: 'inline-block',
                                            width: '16px',
                                            height: '16px',
                                            border: '2px solid #30363d',
                                            borderTop: '2px solid #58a6ff',
                                            borderRadius: '50%',
                                            flexShrink: 0
                                        }
                                    }
                                },
                                {
                                    type: 'Span',
                                    namespace: 'dash_html_components',
                                    props: {
                                        children: 'Scanning ' + count + ' PDF' + (count > 1 ? 's' : '') + ' for DISC data...',
                                        style: { fontWeight: 600 }
                                    }
                                },
                                {
                                    type: 'Span',
                                    namespace: 'dash_html_components',
                                    props: {
                                        children: names,
                                        style: {
                                            color: '#8b949e',
                                            fontSize: '11px',
                                            overflow: 'hidden',
                                            textOverflow: 'ellipsis',
                                            whiteSpace: 'nowrap',
                                            maxWidth: '400px'
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            type: 'Div',
            namespace: 'dash_html_components'
        };
    }
    """,
    Output("scan-status", "children", allow_duplicate=True),
    Input("upload-pdfs", "contents"),
    State("upload-pdfs", "filename"),
    prevent_initial_call=True,
)

if __name__ == "__main__":
    app.run(debug=True)
