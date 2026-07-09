# SDG DISC Dashboard — Project README

## Always read BOTH README files before making any changes.

- `SDG_README.md` (this file) — technical spec: file structure, PDF formats, data contracts, EQI pipeline
- `DISC_DASHBOARD_README.md` — full product context: architecture, theme system, planned features, go-to-market, CSS patterns, known issues

---

## Project location

```
C:\Users\Gaming pc\OneDrive\Documents\sdg_disc_dash\sdg_disc_dash\
```

This is the **only** directory to work in. There is an older, unrelated copy at
`SDG_DISC_Dashboard\` — do not touch it.

---

## What this project is

A **Plotly Dash** web dashboard that:
1. Accepts drag-and-drop uploads of Maxwell DISC PDFs and EQ-i 2.0 PDFs
2. Parses both types automatically (classified by page-1 content)
3. Pairs each person's DISC profile with their EQI report by name similarity
4. Displays DISC factor tiles, shift indicators, radar charts, EQ-i composite bar charts, and DISC–EQI correlation insights

---

## File structure

```
sdg_disc_dash/
  app.py               ← Dash app: layout, callbacks, all chart builders
  utils/
    __init__.py
    disc.py            ← Maxwell DISC PDF parser + profile builder + trait library
    eqi_parser.py      ← EQ-i 2.0 PDF parser (Workplace + Leadership Report formats)
    insights.py        ← DISC–EQI correlation engine + action steps
```

---

## PDF formats supported

### Maxwell DISC PDFs
- Score regex: `D = x.xx, I = x.xx, S = x.xx, C = x.xx` (3 rows: public, stress, mirror)
- Name extracted via anchor phrases: `"style:"`, `"maxwell disc personality indicator report"`
- Style type: parsed from `Style: Attainer DCS` → returns `"DCS"`

### EQ-i 2.0 Workplace Report
- Scores on **page 3** (index 2) — all 15 subscales inline on one line each
- Format: `Self-Regard 113`

### EQ-i 2.0 Leadership Report  ← added May 2026
- Scores on **page 6** (index 5) — subscale name and score on *separate* lines
- Format: `Self-Regard\n113\nRespecting oneself...`
- Executive Summary (page 4) has only top-3 / bottom-3 subscales — do not use it
- Detection: `"multi-health systems"` on page 1 triggers EQI classification

The parser scans pages 3–6 and picks whichever yields the most complete subscale set (15/15 wins over 6/6).

---

## PDF data files

All participant PDFs live in:
```
C:\Users\Gaming pc\OneDrive\Documents\DISC EQI Dashboards PDFS\
```

Naming convention:
- `Sonia_De_Escobar.pdf`          ← Maxwell DISC report
- `Sonia_De_Escobar_client.pdf`   ← EQ-i 2.0 Leadership Report

Upload both files for the same person in a single batch for EQI pairing to work.

---

## Key data contracts

### `parse_eqi_bytes()` output
```python
{
  "name":       str,          # "Sonia De Escobar"
  "total_ei":   int,          # 108
  "composites": dict,         # {"Self-Perception": 108, "Self-Expression": 112, ...}
  "subscales":  dict,         # {"self_regard": 113, "optimism": 105, ...}
}
```

### Profile dict (built by `build_profile()` in disc.py)
```python
{
  "participant_name": str,
  "style_type":       str,          # "SC", "DI", etc.
  "anchor_graph":     str,          # "stress" | "mirror" | "public"
  "graphs":           dict,         # {"public": {D,I,S,C}, "stress": {...}, "mirror": {...}}
  "factor_profiles":  dict,         # per-factor: anchor_score, bucket, traits, deltas
  "summary":          dict,         # {"top_two": ["S","C"], "ranked_by_abs": [...]}
  "eqi_scores":       dict,         # flat: snake subscales + display composites + total_ei
}
```

### `eqi_scores` flat dict (stored on profile after pairing)
```python
{
  "self_regard": 113, ...,          # 15 snake_case subscale keys
  "Self-Perception": 108, ...,      # 5 display-name composite keys
  "total_ei": 108,
}
```

---

## EQI composite → subscale mapping (`EQI_COMPOSITES` in app.py)

| Composite | Subscales |
|---|---|
| Self-Perception | self_regard, self_actualization, emotional_self_awareness |
| Self-Expression | emotional_expression, assertiveness, independence |
| Interpersonal | interpersonal_relationships, empathy, social_responsibility |
| Decision Making | problem_solving, reality_testing, impulse_control |
| Stress Management | flexibility, stress_tolerance, optimism |

---

## EQI sections in the UI

These only render when a DISC profile has a non-empty `eqi_scores`:

- **Total EQ bar** (`_eq_total_bar`) — horizontal progress bar, range 70–130, coloured by score
- **EQ-i composite bar chart** (`build_eqi_bar_chart`) — horizontal bars for 5 composites, norm line at 100
- **EQI Insights** (`_eqi_insights_section`) — collapsible; requires DISC primary style to generate DISC↔EQI correlations and bottom-3 development areas with action steps
- **EQI Insights (comparison cards)** (`_eqi_comparison_insights`) — condensed version, bottom-3 scores only

---

## Name matching (DISC ↔ EQI pairing)

`SequenceMatcher` ratio ≥ 0.65 between lowercased names. First match wins; each EQI result is used at most once.

---

## Theme system

All colours live in the `THEME` dict at the top of `app.py`. Never hardcode hex values outside it. Two modes: `"dark"` (default) and `"light"`, toggled by a clientside callback. DISC and EQI palette keys are theme-independent.

---

## Running the app

```bash
cd "C:\Users\Gaming pc\OneDrive\Documents\sdg_disc_dash\sdg_disc_dash"
python app.py
# Opens at http://127.0.0.1:8050
```

---

## Change history (major)

| Date | Change |
|---|---|
| May 2026 | Added EQ-i 2.0 Leadership Report support to `eqi_parser.py` — multi-page scan, two-line subscale format, optional colon in Total EI |
| Jul 2026 | Added Leadership Roadmap tab: generates a personalized 24-page Roadmap & Workshop Guide booklet (PDF + Word in a zip) per selected cohort member, from DISC + EQ-i + Flywheel + Leadership Signature. New modules: `utils/roadmap_content_model.py`, `roadmap_boilerplate.py`, `roadmap_generator.py`, `roadmap_pdf.py`, `roadmap_docx.py`, `flywheel_disc_reference.json` (41-style DISC→Flywheel lookup). New deps: `reportlab`, `python-docx`. People without paired EQ-i data appear disabled in the dropdown; their booklets substitute a "not yet assessed" notice if generated. |
