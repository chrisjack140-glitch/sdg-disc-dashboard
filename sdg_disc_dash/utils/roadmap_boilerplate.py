"""
Leadership Roadmap — verbatim static content
==============================================
Text transcribed directly from Ariel_Alston_PA_Leadership_Roadmap_Workshop_
Booklet.docx. These pages/sections are identical for every generated
person — only the participant's name is substituted in. Person-specific
analysis (DISC/EQ-i/Flywheel/Leadership Signature interpretation) lives in
utils/roadmap_generator.py, not here.

Each function returns a list of content-model blocks for ONE page (no
trailing PageBreak — the caller in roadmap_generator.py adds that).
"""
from utils.roadmap_content_model import (
    HeaderBand, Paragraph, BulletList, CalloutBox, DataTable, TableRow,
    BlankWorksheetTable, ShadedGroup, Divider, PALETTE, SECTION_TINTS,
    SECTION_TITLES,
)

COHORT_LABEL = "Southern Region PA Leadership Workshop"
BRAND_TAG = "SDG Leadership Framework"


def possessive(name: str) -> str:
    first = name.split()[0] if name else name
    return f"{first}'" if first.endswith("s") else f"{first}'s"


def first_name(name: str) -> str:
    return name.split()[0] if name else name


def _blank_lines_table(n: int = 13) -> BlankWorksheetTable:
    return BlankWorksheetTable(header_row=["Notes"], prompts=[""] * n, n_blank_cols=0)


# ─────────────────────────────────────────
# Cover page
# ─────────────────────────────────────────
def get_cover_boilerplate(person_name: str, cohort_label: str = COHORT_LABEL):
    return [
        ShadedGroup(bg=PALETTE["navy_cover"], lines=[
            Paragraph("THE STRATEGIC DESIGN GROUP", size=15, bold=True,
                      font="Georgia", color=PALETTE["white"]),
            Paragraph("PA LEADERSHIP ROADMAP", size=28, bold=True,
                      font="Georgia", color=PALETTE["white"]),
            Paragraph("& WORKSHOP GUIDE", size=20, bold=True,
                      font="Georgia", color=PALETTE["gold_text"]),
            Paragraph(person_name, size=20, bold=True, color=PALETTE["white"]),
            Paragraph(
                "Using DISC, EQ-i, Flywheel, and Leadership Signature to "
                "coach supervisors and sustain accountability",
                size=13, color=PALETTE["white"],
            ),
            Paragraph(cohort_label, size=13, color=PALETTE["white"]),
            Paragraph("Confidential Leadership Development Use", size=13,
                      color=PALETTE["white"]),
        ]),
        Divider(color=PALETTE["gold_accent"]),
        Paragraph(
            "This booklet is designed for workshop use and post-workshop "
            "coaching implementation.",
            size=13, italic=True, color=PALETTE["body_text"],
        ),
    ]


# ─────────────────────────────────────────
# How To Use This Booklet
# ─────────────────────────────────────────
def get_how_to_use_page(person_name: str):
    poss = possessive(person_name)
    return [
        HeaderBand(title="HOW TO USE THIS BOOKLET",
                   subtitle="Roadmap + workshop application + coaching plan",
                   brand_tag=BRAND_TAG),
        Paragraph(
            "This booklet is designed to be used during the PA Leadership "
            "Workshop and after the session as a practical coaching tool. "
            "During the workshop, each section introduces one component of "
            f"the SDG Leadership Framework, connects it to {poss} individual "
            "Leadership Roadmap, and provides a worksheet for reflection "
            "and application."
        ),
        Paragraph(
            "The worksheets support concept understanding and in-session "
            "practice. The 30-60-90 Coaching Plan at the end is the primary "
            "takeaway and should be used after the workshop to guide "
            "supervisor coaching, accountability conversations, and "
            "leadership follow-through."
        ),
        Paragraph("Booklet Learning Flow", bold=True, size=15),
        DataTable(
            header_row=["Step", "Purpose", f"What {first_name(person_name)} Should Do"],
            rows=[
                TableRow(["1. Understand the Framework",
                          "See how DISC, EQ-i, Flywheel, and Leadership "
                          "Signature work together as one leadership system.",
                          "Read the overview and locate the personal "
                          "leadership themes."]),
                TableRow(["2. Apply the Concepts",
                          f"Use each section to connect the framework to {poss} "
                          "leadership behavior, pressure patterns, and "
                          "coaching role.",
                          "Complete the worksheets during the workshop."]),
                TableRow(["3. Capture Reflection",
                          "Use the notes pages to document insights, "
                          "questions, coaching language, and peer "
                          "accountability commitments.",
                          "Write specific commitments, not general "
                          "intentions."]),
                TableRow(["4. Take Action",
                          "Use the 30-60-90 Coaching Plan as the "
                          "implementation guide after the workshop.",
                          "Coach supervisors, transfer ownership, and "
                          "sustain Flywheel momentum."]),
            ],
        ),
        CalloutBox(
            heading="Confidentiality Note",
            body="This booklet contains individualized leadership "
                 f"development information and should be used only for {poss} "
                 "development, coaching, and workshop participation.",
            tint=PALETTE["tint_gold_1"],
        ),
    ]


# ─────────────────────────────────────────
# SDG Leadership Framework Overview
# ─────────────────────────────────────────
def get_framework_overview_page(person_name: str):
    name = first_name(person_name)
    poss = possessive(person_name)
    return [
        HeaderBand(title="SDG LEADERSHIP FRAMEWORK OVERVIEW",
                   subtitle="One integrated leadership system",
                   brand_tag=BRAND_TAG),
        Paragraph(
            "The SDG Leadership Framework integrates behavioral insight, "
            "emotional intelligence, organizational momentum, and visible "
            "leadership impact. The purpose is not to complete separate "
            "exercises. The purpose is to build a practical leadership "
            "operating system that helps PAs coach supervisors and sustain "
            "accountability."
        ),
        DataTable(
            header_row=["Framework Component", f"What It Helps {name} Understand",
                        "Workshop Application"],
            rows=[
                TableRow(["DISC",
                          f"How {name} naturally leads, communicates, makes "
                          "decisions, and shifts under pressure.",
                          "Recognize behavior patterns and choose "
                          "intentional coaching adjustments."]),
                TableRow(["EQ-i",
                          f"How {name} manages emotions, expresses "
                          "leadership presence, builds trust, and coaches "
                          "through pressure.",
                          "Regulate before responding and use questions "
                          "that develop ownership."]),
                TableRow(["Flywheel",
                          f"How {poss} leadership creates momentum or drag "
                          "across direction, culture, learning, and "
                          "execution.",
                          "Diagnose where supervisor behavior needs "
                          "coaching and where momentum must be "
                          "strengthened."]),
                TableRow(["Leadership Signature",
                          "The consistent leadership experience "
                          f"supervisors and peers should have with {name}.",
                          "Define visible behaviors, commitments, and "
                          "accountability expectations."]),
            ],
        ),
        Paragraph("The Workshop Move", bold=True, size=15),
        DataTable(
            header_row=None,
            rows=[
                TableRow(["Awareness",
                          "What do I need to understand about my "
                          "leadership pattern?"], fill=PALETTE["tint_gold_2"]),
                TableRow(["Application",
                          "How does this show up when I coach "
                          "supervisors?"], fill=PALETTE["tint_teal"]),
                TableRow(["Accountability",
                          "What must become visible, repeatable, and "
                          "sustained?"], fill=PALETTE["tint_mauve"]),
            ],
        ),
        CalloutBox(
            heading="Core Principle",
            body="The framework becomes useful when it changes leadership "
                 "behavior: how expectations are named, how coaching "
                 "questions are asked, how accountability is followed up, "
                 "and how momentum is sustained.",
            tint=PALETTE["tint_slate_2"],
        ),
    ]


# ─────────────────────────────────────────
# Section intro pages (one per section: disc / eqi / flywheel / signature)
# ─────────────────────────────────────────
_SECTION_PURPOSE_STATIC = {
    "disc": "DISC helps PAs understand how natural leadership behavior is "
            "experienced by supervisors and how that behavior may shift "
            "under pressure.",
    "eqi": "EQ is the capacity that helps PAs stay steady, clear, and fair "
           "when coaching supervisors through pressure.",
    "flywheel": "The Flywheel helps PAs diagnose whether leadership "
                "behavior is creating momentum or drag. The four quadrants "
                "provide a shared language for moving supervisor coaching "
                "away from isolated incidents and toward the leadership "
                "patterns that must be strengthened, adapted, or "
                "addressed.",
    "signature": "Leadership Signature translates the Roadmap into a "
                 "visible leadership standard.",
}

_SECTION_HOW_TO_USE = {
    "disc": [
        "Review the Mirror Profile and pressure pattern before completing "
        "the worksheet.",
        "Identify one strength to use more intentionally and one pressure "
        "behavior to manage.",
        "Practice describing a leadership adjustment in observable "
        "language.",
    ],
    "eqi": [
        "Identify the EQ anchor that matters most in the current "
        "leadership context.",
        "Name what needs to be regulated before responding to difficult "
        "supervisor behavior.",
        "Convert reaction into a coaching question that builds ownership.",
    ],
    "flywheel": [
        "Use the four revised quadrants to scan where momentum exists and "
        "where drag is present.",
        "Connect personal leadership behavior to team momentum.",
        "Identify one supervisor behavior and one PA behavior that must "
        "shift in the next 30 days.",
    ],
    "signature": [
        "Review DISC strengths, EQ anchor, and Flywheel priority before "
        "drafting the signature.",
        "Write a signature that describes observable behavior, not just "
        "intention.",
        "Identify how peers can help maintain consistency under pressure.",
    ],
}

_SECTION_BY_END = {
    "disc": [
        "Explain how {poss} style affects supervisor coaching.",
        "Name a pressure shift without defensiveness.",
        "Choose one adjustment that will improve clarity and "
        "accountability.",
    ],
    "eqi": [
        "Use EQ language to prepare for difficult conversations.",
        "Distinguish correction from coaching.",
        "Create one question that helps supervisors own the next step.",
    ],
    "flywheel": [
        "Name which Flywheel quadrant needs attention.",
        "Describe what is creating momentum or drag.",
        "Turn Flywheel insight into a specific coaching conversation.",
    ],
    "signature": [
        "Draft a concise Leadership Signature.",
        "Make the signature observable to supervisors.",
        "Name one peer accountability request.",
    ],
}

_SECTION_SUBTITLES = {
    "disc": "Behavior, communication, and pressure shifts",
    "eqi": "Leadership presence and coaching capacity",
    "flywheel": "Momentum, drag, and supervisor coaching priorities",
    "signature": "The consistent leadership experience I want to create",
}

_SECTION_INTRO_NUMBER = {"disc": 1, "eqi": 2, "flywheel": 3, "signature": 4}


def get_section_intro_page(section_id: str, person_name: str,
                            personalized_sentence: str = ""):
    """`personalized_sentence` is generated by roadmap_generator.py for the
    disc/eqi sections (a sentence tying the section to this person's
    specific traits). Ignored for flywheel (fully generic in the source
    template); for signature it's replaced with a simple name mention."""
    poss = possessive(person_name)
    name = first_name(person_name)
    n = _SECTION_INTRO_NUMBER[section_id]
    purpose = _SECTION_PURPOSE_STATIC[section_id]
    if section_id in ("disc", "eqi") and personalized_sentence:
        purpose = f"{purpose} {personalized_sentence}"
    elif section_id == "signature":
        purpose = (f"{purpose} This section helps {name} define the "
                   "experience supervisors should consistently have when "
                   "receiving direction, coaching, feedback, and "
                   "accountability.")
    by_end = [b.format(poss=poss) for b in _SECTION_BY_END[section_id]]
    return [
        HeaderBand(title=f"SECTION {n}: {SECTION_TITLES[section_id]}",
                   subtitle=_SECTION_SUBTITLES[section_id],
                   brand_tag=BRAND_TAG),
        Paragraph("Purpose of this section", bold=True, size=15),
        Paragraph(purpose),
        Paragraph("How to use this section during the workshop", bold=True,
                  size=15),
        BulletList(_SECTION_HOW_TO_USE[section_id]),
        Paragraph(f"By the end of this section, {name} should be able to:",
                  bold=True, size=15),
        BulletList(by_end),
        CalloutBox(
            heading="Workshop Practice",
            body="Read the information sheet first, review the Roadmap "
                 "Connection page second, complete the worksheet third, "
                 "and use the notes page to capture commitments for the "
                 "30-60-90 Coaching Plan.",
            tint=SECTION_TINTS[section_id],
        ),
    ]


# ─────────────────────────────────────────
# Participant Worksheet pages (blank reflection templates)
# ─────────────────────────────────────────
def get_worksheet_page(section_id: str, person_name: str):
    n = _SECTION_INTRO_NUMBER[section_id]
    title = f"PARTICIPANT WORKSHEET {n}: {SECTION_TITLES[section_id]}"
    header = HeaderBand(title=title, subtitle=_SECTION_SUBTITLES[section_id],
                         brand_tag=BRAND_TAG)

    if section_id == "disc":
        return [
            header,
            CalloutBox(
                body="Use your DISC Mirror Profile to identify how you "
                     "naturally lead and what shifts when pressure rises. "
                     "The goal is to recognize the behavior, name the "
                     "leadership risk, and choose an intentional coaching "
                     "adjustment.",
                tint=PALETTE["tint_slate_1"],
            ),
            Paragraph("A. MY DISC MIRROR PROFILE", bold=True, size=15),
            BlankWorksheetTable(
                header_row=["Prompt", "Reflection / Notes"],
                prompts=["My highest DISC Mirror style(s):",
                         "My natural leadership strengths:",
                         "How supervisors likely experience me when things "
                         "are calm:",
                         "My natural coaching and accountability "
                         "tendency:"],
            ),
            Paragraph("B. MY STRESS / PRESSURE SHIFT", bold=True, size=15),
            BlankWorksheetTable(
                header_row=["Prompt", "Reflection / Notes"],
                prompts=["Under pressure, my profile shifts toward:",
                         "My communication becomes more:",
                         "My decision-making becomes more:",
                         "Supervisors may experience this pressure shift "
                         "as:",
                         "The leadership risk I need to manage:"],
            ),
            Paragraph("C. INTENTIONAL ADJUSTMENT", bold=True, size=15),
            DataTable(
                header_row=["Pressure Pattern", "Leadership Risk",
                            "Adjustment I Will Practice"],
                rows=[
                    TableRow(["Move too fast / become forceful",
                              "Compliance without ownership", ""]),
                    TableRow(["Over-explain / become scattered",
                              "Energy without clarity", ""]),
                    TableRow(["Avoid conflict / become quiet",
                              "Delayed accountability", ""]),
                    TableRow(["Over-focus on details / become rigid",
                              "Fear of mistakes or slow action", ""]),
                ],
            ),
            Paragraph(
                "30-day commitment: When pressure rises, I will pause, "
                "name the pattern, coach the shift, and confirm "
                "accountability.", italic=True,
            ),
        ]

    if section_id == "eqi":
        return [
            header,
            CalloutBox(
                body="Use this worksheet to connect emotional intelligence "
                     "to everyday PA leadership. EQ is the capacity that "
                     "helps you manage yourself, read the moment, build "
                     "trust, and coach supervisors through pressure "
                     "without becoming reactive.",
                tint=PALETTE["tint_slate_1"],
            ),
            Paragraph("A. EQ LEADERSHIP ANCHORS", bold=True, size=15),
            DataTable(
                header_row=["EQ Anchor", "What It Requires of Me",
                            "My Current Reflection"],
                rows=[
                    TableRow(["Authenticity",
                              "Be clear, fair, steady, and consistent.", ""]),
                    TableRow(["Coaching",
                              "Develop the supervisor instead of only "
                              "correcting the issue.", ""]),
                    TableRow(["Insight",
                              "Connect expectations to purpose, impact, "
                              "and direction.", ""]),
                    TableRow(["Innovation",
                              "Create space for ownership, "
                              "problem-solving, and learning.", ""]),
                ],
            ),
            Paragraph("B. MY EQ PRACTICE UNDER PRESSURE", bold=True,
                      size=15),
            BlankWorksheetTable(
                header_row=["Prompt", "Reflection / Notes"],
                prompts=["A situation that tests my emotional "
                         "intelligence:",
                         "My usual reaction under pressure:",
                         "What I need to regulate before responding:",
                         "The EQ anchor I most need to strengthen:",
                         "The leadership response I want supervisors to "
                         "experience:"],
            ),
            Paragraph("C. COACHING QUESTION I WILL USE", bold=True,
                      size=15),
            BlankWorksheetTable(
                header_row=["Prompt", "Reflection / Notes"],
                prompts=["My first coaching question will be:",
                         "The ownership question I will ask:",
                         "The follow-up accountability question I will "
                         "ask:"],
            ),
            Paragraph(
                "EQ commitment: I will manage my response before I manage "
                "the conversation.", italic=True,
            ),
        ]

    if section_id == "flywheel":
        return [
            header,
            CalloutBox(
                body="Use the Flywheel to diagnose whether your leadership "
                     "and your supervisors' leadership are creating "
                     "momentum or drag. The Flywheel turns when direction, "
                     "leadership behavior, learning feedback, adaptation, "
                     "execution, and accountability are aligned.",
                tint=PALETTE["tint_slate_1"],
            ),
            Paragraph("A. MOMENTUM AND DRAG SCAN", bold=True, size=15),
            DataTable(
                header_row=["Flywheel Quadrant", "Where We Have Momentum",
                            "Where We May Have Drag"],
                rows=[
                    TableRow(["Direction & Strategic Intent", "", ""]),
                    TableRow(["Leadership Behavior & Culture", "", ""]),
                    TableRow(["Learning, Feedback & Adaptation", "", ""]),
                    TableRow(["Execution & Accountability", "", ""]),
                ],
            ),
            Paragraph("B. MY PA LEADERSHIP ROLE IN THE FLYWHEEL", bold=True,
                      size=15),
            BlankWorksheetTable(
                header_row=["Prompt", "Reflection / Notes"],
                prompts=["The quadrant most affected by my leadership "
                         "right now:",
                         "The quadrant most affected by my pressure "
                         "behavior:",
                         "One supervisor behavior that is slowing "
                         "momentum:",
                         "One PA behavior that would strengthen "
                         "momentum:",
                         "The accountability conversation I need to "
                         "initiate:"],
            ),
            Paragraph("C. FLYWHEEL COACHING PROMPT", bold=True, size=15),
            BlankWorksheetTable(
                header_row=["Prompt", "Reflection / Notes"],
                prompts=["When coaching supervisors, I will ask:",
                         "The expected shift I want to see in 30 days:"],
            ),
            Paragraph(
                "Flywheel commitment: I will identify what creates "
                "momentum, name what creates drag, and coach the next "
                "shift.", italic=True,
            ),
        ]

    # signature
    return [
        header,
        CalloutBox(
            body="Your Leadership Signature is the observable pattern of "
                 "leadership others should consistently experience from "
                 "you. It should describe how you lead, coach, "
                 "communicate, and hold accountability - especially when "
                 "pressure rises.",
            tint=PALETTE["tint_slate_1"],
        ),
        Paragraph("A. LEADERSHIP SIGNATURE INPUTS", bold=True, size=15),
        BlankWorksheetTable(
            header_row=["Prompt", "Reflection / Notes"],
            prompts=["The leadership experience I want supervisors to "
                     "consistently have:",
                     "The strengths from my DISC Mirror Profile that "
                     "support this:",
                     "The EQ anchor that will make this more credible:",
                     "The Flywheel quadrant my signature must strengthen:",
                     "The pressure behavior that could weaken my "
                     "signature:"],
        ),
        Paragraph("B. DRAFT MY LEADERSHIP SIGNATURE", bold=True, size=15),
        Paragraph(
            "My Leadership Signature is to lead with "
            "____________________, create ____________________, coach "
            "supervisors toward ____________________, and hold myself "
            "and others accountable for ____________________."
        ),
        Paragraph("C. MAKE IT OBSERVABLE", bold=True, size=15),
        DataTable(
            header_row=["Behavior Supervisors Will See",
                        "What I Will Stop / Manage",
                        "How PA Peers Can Hold Me Accountable"],
            rows=[TableRow(["", "", ""]), TableRow(["", "", ""])],
        ),
        Paragraph("D. 30-DAY COMMITMENT", bold=True, size=15),
        BlankWorksheetTable(
            header_row=["Prompt", "Reflection / Notes"],
            prompts=["One leadership behavior I will practice "
                     "consistently:",
                     "One supervisor coaching conversation I will "
                     "initiate:",
                     "One accountability request I am making of my PA "
                     "peers:"],
        ),
        Paragraph(
            "Leadership Signature commitment: My leadership impact must "
            f"be visible, repeatable, and aligned to {COHORT_LABEL.split(' PA')[0]} "
            "momentum.", italic=True,
        ),
    ]


# ─────────────────────────────────────────
# Notes pages (one per section)
# ─────────────────────────────────────────
_SECTION_NOTES_REFLECTION = {
    "disc": ["What did I learn about my natural leadership pattern?",
             "What pressure behavior must I manage?",
             "What coaching adjustment will supervisors notice?"],
    "eqi": ["What situation tests my emotional intelligence?",
            "What do I need to regulate before responding?",
            "What coaching question will I use first?"],
    "flywheel": ["Where do we have momentum?",
                 "Where are we experiencing drag?",
                 "What conversation needs to happen to move the "
                 "Flywheel?"],
    "signature": ["What do I want supervisors to consistently "
                  "experience?",
                  "What will I stop or manage under pressure?",
                  "What peer accountability request will I make?"],
}


def get_notes_page(section_id: str):
    n = _SECTION_INTRO_NUMBER[section_id]
    title = f"{SECTION_TITLES[section_id]} NOTES"
    return [
        HeaderBand(title=title,
                   subtitle="Reflection, insights, and coaching "
                            "application",
                   brand_tag=BRAND_TAG),
        Paragraph("Guided Reflection", bold=True, size=15),
        BulletList(_SECTION_NOTES_REFLECTION[section_id]),
        _blank_lines_table(13),
    ]


# ─────────────────────────────────────────
# Closing commitment page
# ─────────────────────────────────────────
def get_closing_commitment_page(person_name: str):
    return [
        HeaderBand(title="MY LEADERSHIP MOMENTUM COMMITMENT",
                   subtitle="Final reflection and accountability "
                            "statement",
                   brand_tag=BRAND_TAG),
        Paragraph(
            "Over the next 90 days, I commit to using my Leadership "
            "Roadmap to strengthen how I lead, coach supervisors, and "
            "contribute to Southern Region momentum."
        ),
        BlankWorksheetTable(
            header_row=["Commitment Prompt", "My Response"],
            prompts=["One leadership behavior I will practice "
                     "consistently:",
                     "One supervisor coaching conversation I will "
                     "initiate:",
                     "One accountability commitment I am making to my PA "
                     "peers:",
                     "One Flywheel quadrant I will intentionally "
                     "strengthen:",
                     "One way I will make my Leadership Signature more "
                     "visible:"],
        ),
        CalloutBox(
            heading="Closing Commitment",
            body="My leadership impact must be visible, repeatable, and "
                 "aligned to Southern Region momentum.",
            tint=PALETTE["tint_mauve"],
        ),
        Paragraph(
            "Participant Signature: "
            "____________________________________________    "
            "Date: ____________________"
        ),
    ]
