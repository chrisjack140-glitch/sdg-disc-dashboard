# SDG DISC Personality Assessment Dashboard — README

> This README provides full context for Claude Code sessions working on the SDG DISC Operator Dashboard. Read this before making any changes to the codebase.

---

## Project Overview

The **SDG DISC Operator Dashboard** is a data dashboard built for SDG (Strategic Design Group) facilitators to upload, parse, and analyze Maxwell Method DISC personality assessment PDFs. It enables individual and team-level behavioral analysis, comparison, and reporting.

**Primary users:** SDG facilitators and operators working with DISC assessment data for government and public sector clients.

**Data source:** Maxwell Method DISC PDFs produced by PeopleKeys.

---

## Project Location

```
C:\Users\Gaming pc\OneDrive\Documents\sdg_disc_dash\
```

---

## File Structure

```
sdg_disc_dash/
├── app.py              ← Main Dash application (layout + callbacks)
├── assets/
│   └── style.css       ← All custom CSS, theme system, light/dark mode
└── utils/
    ├── __init__.py
    └── disc.py         ← All parsing logic, DISC profile building, TRAITS library
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Dash by Plotly (Python) |
| UI Components | Dash Bootstrap Components (`dbc`) — CYBORG dark theme base |
| Charts | Plotly (`plotly.graph_objects`) |
| PDF Parsing | pdfplumber |
| Language | Python |
| Editor | VS Code |
| Version Control | Git (commits pushed via VS Code terminal) |

**Install dependencies:**
```bash
pip install dash dash-bootstrap-components pandas pdfplumber plotly
```

**Run the app:**
```bash
python app.py
```
Then open `http://127.0.0.1:8050` in a browser.

---

## Architecture: Key Concepts

### Dash vs Streamlit (this app was migrated FROM Streamlit)
- Dash does NOT rerun the entire script on every interaction
- Only the specific `@app.callback` function triggered by a user action runs
- Layout (what it looks like) is defined once in `app.layout`
- Interactions are handled by `@app.callback` functions with explicit Input/Output declarations
- State is stored in `dcc.Store` components (equivalent of `st.session_state`)

### Streamlit → Dash Equivalents
| Streamlit | Dash |
|---|---|
| `st.selectbox()` | `dcc.Dropdown()` |
| `st.multiselect()` | `dcc.Dropdown(multi=True)` |
| `st.file_uploader()` | `dcc.Upload()` |
| `st.plotly_chart()` | `dcc.Graph(figure=...)` |
| `st.columns()` | `dbc.Row([dbc.Col(...)])` |
| `st.tabs()` | `dbc.Tabs([dbc.Tab(...)])` |
| `st.session_state` | `dcc.Store(id="...")` |

---

## DISC Data Model

### Factors
- **D** — Dominance (color: `#ef4444` red)
- **I** — Influence (color: `#facc15` yellow)
- **S** — Steadiness (color: `#22c55e` green)
- **C** — Conscientiousness (color: `#3b82f6` blue)

### Graphs (score types per factor)
- `public` — Natural/Public behavior graph
- `stress` — Stress/Adapted behavior graph
- `mirror` — Mirror graph

### Key Functions in `utils/disc.py`
- `bucketize(x)` — Converts raw score to bucket label (very_high, high, moderate_high, balanced, moderate_low, low, very_low)
- `shift_label(delta)` — Labels the Public→Stress shift direction
- `extract_text_from_pdf_bytes(file_bytes)` — Parses raw text from PeopleKeys PDF
- `extract_name_from_text(text, fallback)` — Extracts participant name
- `extract_scores(text)` — Extracts D/I/S/C scores for each graph type
- `build_profile(scores, anchor_graph)` — Builds full profile dict for one participant
- `process_uploaded_files(files_data, anchor_graph)` — Processes a batch of uploaded PDFs

### Profile Dict Structure
Each participant profile contains:
- `name` — participant name
- `disc_type` — combined DISC label (e.g. "DC", "SCI")
- `factors` — dict of D/I/S/C with scores and bucket labels for each graph
- `shifts` — Public→Stress delta per factor
- `traits` — behavioral trait descriptions from TRAITS library

---

## App Layout & Tabs

The header is sticky and contains:
- "SDG DISC Dashboard" title (left)
- Tab navigation: **Team Dashboard | Individual Results | Comparison** (right, aligned with title)
- Light/Dark mode toggle switch (far right)

### Tab 1 — Team Dashboard
- Team mean score cards per DISC factor
- DISC type distribution bar chart (tracks composite types: "DC", "SC", "SIC", etc.)
- Radar chart (team composite view)
- Heat map (with black outline on each section)
- Pre-factor Mean charts (static axis: -8 to 8, center line at 0)

### Tab 2 — Individual Results
- Dropdown to select participant
- Individual profile card with DISC factor scores, bucket labels, shift badges, and trait descriptions
- Factor letter spans use CSS classes (`factor-letter`) for color theming

### Tab 3 — Comparison
- Side-by-side operator cards
- Radar comparison chart (moved from Team Dashboard)
- Gap analysis section

### Download Buttons
Located at the **bottom of the page**.

---

## Theme System (Light/Dark Mode)

**How it works:**
- A toggle switch in the header fires a **clientside callback** (pure JS, no server round-trip)
- It sets `data-theme="light"` or `data-theme="dark"` on the `<body>` element
- All colors are defined as CSS custom properties (`var(--bg)`, `var(--surface)`, etc.) in `style.css`
- When `data-theme` changes, all elements update simultaneously with `0.35s ease` transition

**Dark theme:** `#0d1117` background, `#161b22` cards
**Light theme:** `#ffffff` background, `#f3f4f6` cards (soft light grey), `#1f2328` near-black text

**Toggle icons:** 🌙 on the left (dark), ☀ on the right (light). Knob slides accordingly.

**Important CSS pattern:** Never use broad `color: var(--text)` on generic elements — it strips inline DISC colors. Always use specific CSS class + attribute selectors:
```css
body[data-theme="light"] .factor-letter[style*="#f85149"] { color: #cf222e; }
```

**CSS classes used in Python output:**
- `.factor-letter` — D/I/S/C colored letter spans
- `.bucket-label` — factor bucket label text
- `.shift-positive` / `.shift-negative` / `.shift-neutral` — shift direction badges
- `.report-title`, `.comp-name`, `.section-title`, `.header-title` — key title elements

---

## Known Issues & Decisions

- **Animated counters** — originally used inline `<script>` tags per factor (caused race conditions in Streamlit). Resolved in Dash via clientside callbacks.
- **Dropdown backgrounds** — use `rgba(17,24,39,0.85)` (semi-transparent matching dark bg) so dropdowns feel integrated rather than floating.
- **Inline style conflicts** — Python-generated inline `style` dicts can override CSS variables. Solution: always use CSS classes for color-sensitive elements rather than inline styles.
- **Radar chart numbers** — rotated counter-clockwise 90 degrees per design spec.
- **Pre-factor mean chart axes** — static from -8 to 8 with a center line at 0.

---

## Planned Features (Not Yet Built)

Priority order based on product-market fit for government facilitators:

1. **Conflict & compatibility analysis** — FIFA-style drag-and-drop compatibility grid; most-requested insight not available in standard PeopleKeys report
2. **Session persistence** — named sessions (e.g. "City of Austin Leadership Team Q2 2025"); data resets on page refresh currently
3. **PDF report export** — branded PDF summary for clients (the primary deliverable justifying the tool's cost)
4. **Claude API natural language summaries** — per-participant plain-English summary of behavioral tendencies, communication preferences, stress responses, and growth areas
5. **Historical comparison** — track score shifts when a client retakes the assessment (compelling for ongoing coaching)
6. **Role fit scoring** — role template library + succession readiness view
7. **Communication playbook generator** — DISC-based communication guides for leader-subordinate pairs
8. **EQ-i integration** — composite profiles combining DISC behavioral data with EQ-i emotional intelligence scores

---

## Go-To-Market Context

- **Target users:** Maxwell-certified DISC facilitators in government and public sector
- **Pricing model:** Per-session or per-cohort (not per-seat subscription) — maps to how facilitators already bill clients
- **Contract strategy:** Dashboard should appear as a **named line item** within contract price (not absorbed invisibly, not charged separately on top). Example line item: "SDG Leadership Intelligence Platform — 12-Month Access: $28,000–$40,000"
- **White-labeling:** Logo upload + color scheme customization needed for facilitators selling to their own clients
- **Data residency:** Government buyers will ask about data storage — architecture should support self-hosting or FedRAMP-compliant hosting (AWS GovCloud)
- **Distribution channel:** Maxwell Leadership and PeopleKeys certified facilitator networks are the primary go-to-market path

### Proposal Tier Structure (for government clients)
- **Option 1 — DISC only:** $115,000–$125,000 (300 reports, 30 group reports, 12 workshops, platform access)
- **Option 2 — DISC + EQ-i integrated:** $185,000–$225,000 (recommended, dual-assessment with composite profiles)
- **Option 3 — DISC now, EQ-i Phase 2:** $115,000–$125,000 now + $85,000–$110,000 later

### Acquisition Potential
- Early stage (2–3 government engagements, working platform): $150,000–$400,000
- Growth stage ($150K–$300K ARR, 50+ facilitator subscribers): $750,000–$2,000,000
- Mature stage ($500K+ ARR, enterprise clients, EQ-i integrated): $2,500,000–$5,000,000+
- Strategic buyers: Maxwell Leadership, PeopleKeys, MHS (owns EQ-i 2.0)

---

## Git Workflow

Changes are committed and pushed from the VS Code terminal:
```bash
git add .
git commit -m "Description of change"
git push
```

---

*Last updated: May 2026 | SDG Operations*
