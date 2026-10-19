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
| Sep 2026 | **Roadmap generation switched to a template.** The booklet is now produced by filling `utils/roadmap_template.docx` — the hand-finished reference (REV12) with one participant's data swapped for `{{TOKENS}}` — so output matches the reference exactly instead of being rebuilt block-by-block. Only 21 of its 712 blocks are person-specific. New: `utils/roadmap_template.py` (runtime filler), `tools/build_roadmap_template.py` (re-authors the template from a revised reference; also applies the reference's own typo fixes). The three variable-row tables (EQ strengths bars, development bars, EQ Dimension) are rebuilt by cloning their prototype row. **Output is DOCX only** — Render has no Word/LibreOffice, so server-side PDF isn't possible; `utils/roadmap_pdf.py` and the block-model renderers are no longer used by the Roadmap tab. |
| Sep 2026 | **Roadmap wording per full DISC style, not per leading letter.** New `utils/roadmap_style_language.json` holds every DISC-driven passage for each of the 41 Maxwell style codes — natural strengths, how staff experience you, blind spot, pressure risk, coaching adjustment, Snapshot risk response and coaching opportunity, the Integration Summary's "this creates", the DISC Connection foundation and development, the 30-60-90 focus titles and DISC steps, Roadmap Focus, the Flywheel pattern and development risk, the success indicator, and the Leadership Signature's qualities, behaviors, trust and momentum. Each passage is different for every code (a C, a CS and a CSD no longer share text); CSD carries the V4 reference wording verbatim, so Courtney still regenerates to V4. `style_language(code)` in `roadmap_generator.py` looks a code up (falling back to its first two letters, then its first letter); `_FACTOR_PLAN` and most of `_FACTOR_LANGUAGE` were removed. The file's `about` note says where each field appears, so SDG can edit a style's wording without touching code. |
| Sep 2026 | **Roadmap template rebuilt from Courtney Stanford's V4 booklet (the approved base).** `tools/build_roadmap_template.py` now only tokenizes — V4 is already the finished layout (table of contents, section divider pages, banner fixes, colour bands, darker Notes gridlines, centred cover), so the REV12 restyling/pagination steps were removed (git history has them). Regenerating Courtney from her own PDFs reproduces V4's text word for word. New per-person content: **RADAR SHIFTS** carries the person's Stress + Mirror radar, drawn server-side by `utils/radar_image.py` with matplotlib (Render has no browser for Plotly image export; new dep `matplotlib`, Inter bundled in `assets/fonts/` under the SIL OFL) from `utils/radar_geometry.py`, whose axis/icon/colour definitions app.py now imports too; **Leadership Derailment Risk** chart fills the four scores, cranberry under 100 and gold at 100+; Leadership Capacities bars and scores at 100+ are green; the TOC's page numbers refresh when Word opens the file (`updateFields`). Generator wording follows V4 with per-style alternates: new `FLYWHEEL_FOR_YOU` paragraph ("grounded in <primary pattern>… increasing <style's activation focus>… your natural <secondary> and <primary>"), `FLY_DEV_RISK` and `ROADMAP_FOCUS` per primary factor (`_FACTOR_PLAN`), the three 30-60-90 EQ cells keyed to the lowest EQ-i capacity (`_EQ_FOCUS` / `_EQ_PRACTICE` / `_EQ_SUSTAIN`; 60-day adds the third-lowest), Flywheel strengths run as one sentence with the shift split "Move from X" / "→ Y" (`FLY_*_FROM`/`_NEXT`, `FLY_*_LABEL`), "…; and …" lists in the Integration Summary, V4's EQ strength/development notes. The secondary-factor add-on sentences (`_SECONDARY_BLEND`) were removed — V4 has none; the secondary still shapes the natural-strengths line, style label and all Flywheel text. `DISC_INTRO`, `EQ_PURPOSE_NOTE`, `FLYWHEEL_INTRO` are gone with the pages V4 dropped. The filler no longer collapses empty paragraphs (they are V4's layout); it removes only body paragraphs a blank token emptied. `render_from_template(..., images=build_template_images(profile))`. |
| Sep 2026 | **Graph Shift Radar no longer lags when graphs are toggled.** Each checklist click used to rebuild the figure on the server (re-sending the whole profiles store) and replay the entrance animation, which tweened the data with `Plotly.restyle` every frame — ~270–320 full chart redraws (grid, icons, labels) and 2.4–4.5 s of frozen page per click, plus a per-frame error once a toggle changed the trace list. Now `build_graph_overlay_radar()` always builds all three traces (`uid` = graph key, `visible` from the checklist); the server callback runs only on participant/anonymize changes, and a clientside callback shows/hides lines in the browser — 1 redraw, no request, no replay. `assets/radar_reveal.js` grows the lines by scaling the polar trace layer about the centre with an SVG transform (identical picture, since the radial axis is linear with its minimum at the centre) — no Plotly redraws while animating; `vector-effect: non-scaling-stroke` keeps line weight constant. The animation replays on scroll-in and for a new participant, not on toggles. |
| Sep 2026 | **Leadership Signature follows the client-approved pattern.** The client preferred the reference booklet's signature ("I lead with precision, confidence, emotional clarity, and adaptive accountability…") over the generated wording, so `build_leadership_signature()` now writes every signature in that pattern: four qualities — the primary DISC factor's natural quality and the stretch it calls for (`_FACTOR_PLAN[f]["signature"]`: C Precision/Confidence, D Decisiveness/Openness, I Energy/Follow-Through, S Steadiness/Candor), the lowest EQ-i development area as a quality (`_EQ_QUALITY_NAME`, e.g. Emotional Expression → Emotional Clarity), and adaptive accountability — then a per-factor trust and momentum clause. The "Make the Signature Observable" table lists the same five elements in order, so statement and table always agree. A C-primary leader whose lowest EQ-i area is Emotional Expression gets the reference wording exactly. The same statement fills both signature boxes (Integration Summary and Leadership Signature Connection). |
| Sep 2026 | **Leadership Signature wording draws on EQ-i growth, per the SDG Leadership Signature methodology** (`Downloads\SDG_Leadership_Signature_Methodology_and_Replication_Guide.docx`). The client-approved sentence pattern is unchanged; what fills it now comes from both reports. DISC gives the style's natural quality, the stretch its overuse calls for, and its trust/momentum words (as before). The EQ-i report gives the growth: the **key growth areas** are the three lowest subscales beneath their leadership bar (the three the 30-60-90 plan works on; someone who clears every bar uses the three closest to theirs). The lowest becomes the third quality; each of the next two reshapes the clause of the EQ-i leadership competency it belongs to on the report's Leadership Potential page — Authenticity → "I build trust through … and ___" (`_TRUST_GROWTH`), Coaching → "develop people through ___" (`_COACHING_GROWTH`), Insight → "pairing … with ___" plus the fifth observable row (`_MOMENTUM_GROWTH`), Innovation → the fourth quality "___ Accountability" (`_ACCOUNTABILITY_GROWTH`). Impulse Control and Stress Tolerance (derailers, in no competency) sit with trust and momentum. A clause no key growth area reaches keeps the reference wording, and a phrase whose lead word repeats one already in the signature is passed over, so Courtney (Emotional Expression, Problem Solving, Flexibility) still gets V4's signature and table word for word. `_signature_parts()` in `roadmap_generator.py`; Flexibility's quality is now "Flexibility" (was "Adaptability", which echoed "adaptive accountability"). Nothing outside the Leadership Signature section changed. |
| Sep 2026 | **Every Roadmap keeps the reference's 42 pages, so the table of contents is always right.** V4 starts several pages by letting a run of empty paragraphs push content down (EQ pages 17 and 18, the Flywheel grid on 25, "Make the Signature Observable" on 32, the 30-60-90 grid on 38) or with a page-break paragraph (19, 26). That only lands right for Courtney's chart lengths and wording: other people's longer charts pushed content onto extra, mostly blank pages (e.g. the EQ Practice box alone on a page), shorter ones pulled the next page's content up leaving gaps, and every later section moved — 9 of 15 real booklets ran to 43 pages, so the stored TOC numbers (3, 8, 15, 23, 30, 36) were wrong unless Word refreshed its fields. `_paginate()` in `utils/roadmap_template.py` now gives each of those pages a real page break at the element that opens it in V4 and drops the spacer run, so each page looks as it does in V4. Variable content is sized to its page: strength bars tighten above 10, capacity bars above 11 (15 when there is no strengths chart, in which case the capacities chart opens page 17 and the score note stays on 16), and the EQ Dimension table's strength rows lose their spare paragraphs, go single-spaced and share the page's remaining height (10pt text if needed; as a last resort the header and Development Areas rows tighten too) when the reference spacing won't fit (`_dimension_layout()`; Word adds the 7pt cell margin on top of a row's height). The reference's page-break paragraphs before pages 19 and 26 became page-break-before too, so a full page 18 or 25 can no longer spill them onto a blank page. Verified in Word across all 15 real people, Courtney, every strengths count 0–15, and all 41 style codes: 42 pages each, with the stored TOC numbers matching where every section actually starts. |
| Sep 2026 | **Full screen for every chart; radar entrance animation.** `assets/fullscreen.js` never added its button: Plotly sets `.js-plotly-plot` after the graph's div is inserted, and the observer only scanned inserted nodes. It now rescans the page (debounced), sizes charts to the screen on entry and restores their original pixel height on exit (fixed-height figures stayed small before), uses SVG icons, and falls back to filling the browser window where the Fullscreen API is refused or missing (iPhone Safari, embedded views) — neutralizing ancestor transforms/entrance animations that would otherwise trap `position: fixed`, holding the card's space and restoring scroll on close. The button is always faintly visible. New `assets/radar_reveal.js`: cards marked `.scroll-reveal` (the Graph Shift Radar) rise in on scroll and the radar lines grow from the centre to their scores, replaying each time the card returns to view; honours reduced-motion. Graph Shift Radar lines are solid, distinguished by colour and marker shape. |
| Sep 2026 | **Roadmap text now varies with every DISC style; EQ-i composites fixed.** An audit generating booklets for all 41 reference style codes found DISC-driven text that never changed: the 30-60-90 focus themes and DISC column, the Snapshot's risk-response and coaching rows, the Integration Summary's "this creates reliability…quality control" and the signature's trust clause — all written for a C/S leader. These now come from `_FACTOR_PLAN` (per primary factor) in `roadmap_generator.py`, and the five DISC profile cells add a secondary-factor clause from `_SECONDARY_BLEND` (16 variants instead of 4). Reference-participant prose that was still fixed in the template is now tokenized: the page-6 Leadership Signature box (another person's signature), Roadmap Focus, the EQ purpose note, the Flywheel development risk, the 90-Day Success Indicator (now second person) and the five-row "Make the Signature Observable" table (`SIG_EL_n` / `SIG_OBS_n`). Also: "a/an" before style codes ("an SC Peacemaker"); `_assert_second_person` only fails on the full name or the first name used as a name, so names that are ordinary words no longer stop generation. `eqi_parser.py` now reads Leadership Report composites whose score sits on the next line — Self-Perception and Interpersonal were being dropped. |
| Sep 2026 | **Style names come from the Maxwell report.** `extract_style_name_from_page1()` in `utils/disc.py` reads the words before the DISC code on the page-1 `Style:` line (e.g. "Style: Contemplator CSD" → "Contemplator", "Style: Logical Thinker C" → "Logical Thinker") and stores it as `profile["style_name"]`. The Roadmap's style label (`_style_label()` in `roadmap_generator.py`) now prints that name — "CSD Contemplator" rather than the Flywheel reference's "CSD Precise Achiever". Profiles without a parsed name (older saved presets) fall back to the reference name. |
| Sep 2026 | **Graph Shift Radar on the Individual Results tab.** Below the report card, one participant's Public, Stress and Mirror graphs are overlaid on the DISC radar (toggle each with the checklist) to show the shifts between them. `build_graph_overlay_radar()` in `app.py`; the icon/axis layout shared with the Comparisons radar was factored into `_radar_figure()` and `_radar_values()`. Line colours avoid the D/I/S/C hues and use distinct dash patterns. |
| Sep 2026 | **Fixed hard-coded "CS" on the DISC Leadership Roadmap Connection page.** The opening paragraph there was never tokenized (the builder's `DISC_INTRO` prefix matched a different paragraph), so every booklet printed the reference participant's "Your CS pattern…" regardless of the person's style. It is now `{{DISC_CONNECTION_INTRO}}`, generated by `_disc_connection_intro()` in `roadmap_generator.py` from the parsed style code plus each top factor's `foundation` words and the primary factor's `development` clause (new keys in `_FACTOR_LANGUAGE`). The builder has the matching prefix; the committed template was patched in place (only that paragraph changed). The PDF parser itself was reading styles correctly (e.g. "Style: Contemplator CSD" → `CSD`). |
| Sep 2026 | Earlier same-month work, now superseded by the template above: **No Executive Summary page** — the Leadership Roadmap Integration Summary carries that role. Banner titles updated (Integration Summary / Snapshot / DISC + Flywheel Leadership Roadmap Connection). Two new content blocks: `ScoreStrip` (DISC D/I/S/C Mirror + Pressure bands) and `BarChart` (EQ-i strengths / development), both rendered by the PDF and DOCX renderers. Cover navy lifted `#101827`→`#1E3A5F` (the old value printed as flat black) and table gridlines darkened `#D8DEE6`→`#94A3B4` via the new `table_grid` palette token. New **Organization** field on the Roadmap tab rewrites the client name throughout the booklet. New module `utils/eqi_benchmarks.py`. |
| Oct 2026 | **Radar Slides for PowerPoint (Comparisons tab).** Pick people in order and a graph (Mirror, Public or Stress) and download a .pptx with one slide per person that builds up: slide 1 is the first person's radar with their name in the key, slide 2 adds the second person, and so on, so the last slide overlays everyone. Reproduces the reference deck SDG presents from (`Downloads\NWF Dec 9th R graphs.pptx`): a 10.67 × 6 in white slide, the eight behaviour symbols at the reference's positions, and the key at the bottom right in its colours (red, green, blue, orange, magenta, then 10 more), marker shapes (circle, diamond, square, ▲, ▼, …) and fonts (Nunito labels, Roboto Condensed key). The radar is drawn as native PowerPoint shapes, grouped as "DISC radar" / "Behaviour symbols" / "Key", so it stays sharp when projected and remains editable; everything sits in the same place on every slide, so clicking through only adds a line and a name. Long names move the key left and then shrink it; groups over seven move it up and tighten its rows (40 people still fit). Speaker notes list the graph and the people on each slide; anonymize mode puts "Participant N" in the key. New `utils/radar_deck.py` (`build_radar_deck()`), new dep `python-pptx`. |
