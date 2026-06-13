"""
app.py  -  WC2026 Agentic Match Predictor  (v3)
================================================
Streamlit UI improvements over v2:
  - All emojis removed from auto-generated prediction queries.
  - Host nation boost note corrected: only USA, Canada, Mexico.
  - Model description updated for accuracy.
  - Temperature default lowered to 0.10.
  - "How it works" expander updated to reflect 9-step v4 workflow.
  - Cleaner sidebar labels and help text.
  - Spinner message references the full v5 model pipeline.
  - Error message includes GROQ rate-limit hint.
  - Quick-predict query template passes structured context to agent.
"""

import os
import time
import datetime
from dotenv import load_dotenv
import streamlit as st
from agent import build_agent, run_agent

load_dotenv()

# -- Page config must be the FIRST Streamlit call ----------------------------
st.set_page_config(
    page_title="WC2026 Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- API key guard ------------------------------------------------------------
if not os.getenv("GROQ_API_KEY"):
    st.error(
        "**GROQ_API_KEY not found.**\n\n"
        "Create a `.env` file in the project folder containing:\n"
        "```\nGROQ_API_KEY=your_key_here\n```"
    )
    st.stop()


# -- Session state initialisation --------------------------------------------
def _init_state():
    defaults = {
        "history": [],        # list of (user_msg, bot_msg, timestamp)
        "pending_query": "",  # pre-filled from quick-predict button
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    model = st.selectbox(
        "LLM Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
        ],
        index=0,
        help=(
            "llama-3.1-8b-instant has a much higher free-tier TPM limit "
            "and is recommended to avoid 413 token-limit errors."
        ),
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=0.8,
        value=0.10, step=0.05,
        help=(
            "Lower = more deterministic tool sequencing and reproducible reasoning.\n"
            "Recommended range: 0.05 to 0.20 for match predictions."
        ),
    )

    st.markdown("---")
    st.subheader("Quick Match Predictor")

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.text_input("Team A", value="Argentina")
    with col2:
        team_b = st.text_input("Team B", value="France")

    stage = st.selectbox(
        "Stage",
        ["Group Stage", "Round of 16", "Quarter-Final", "Semi-Final", "Final"],
        help="Knockout stages enable penalty advancement probabilities.",
    )

    host_team = st.selectbox(
        "Host nation (applies home boost)",
        ["None", team_a if team_a else "Team A", team_b if team_b else "Team B"],
        help=(
            "WC2026 host nations (USA, Canada, Mexico) receive a +30 Elo boost "
            "at neutral venues. Only select if one team is a host nation."
        ),
    )

    extra_context = st.text_area(
        "Additional context (optional)",
        placeholder=(
            "e.g. France are missing Mbappe. England played 3 days ago. "
            "Match is in Dallas at altitude 150m."
        ),
        height=90,
        help="Any known injuries, suspensions, rest days, or venue details.",
    )

    predict_clicked = st.button("Predict This Match", use_container_width=True)

    st.markdown("---")
    st.subheader("Example Questions")

    examples = [
        "Predict England vs Germany, knockout round",
        "Who wins Brazil vs France? France is missing Mbappe.",
        "Predict Morocco vs Japan group stage match in Atlanta",
        "Spain vs Netherlands, who has better recent form?",
        "Predict USA vs Mexico, USA are the host nation",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:24]}", use_container_width=True):
            st.session_state.pending_query = ex

    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.pending_query = ""
        st.rerun()

    if st.session_state.history:
        transcript_lines = []
        for u, b, ts in st.session_state.history:
            transcript_lines.append(
                f"[{ts}]\nQuery: {u}\n\nReport:\n{b}\n\n{'=' * 70}"
            )
        transcript = "\n".join(transcript_lines)
        st.download_button(
            "Download Transcript",
            transcript,
            file_name=f"wc2026_predictions_{datetime.date.today()}.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------
st.title("WC2026 Agentic Match Predictor")
st.caption(
    "LangGraph ReAct agent · Groq LLM (llama-3.1-8b-instant) · "
    "DuckDuckGo real-time search · Dixon-Coles Poisson model (v5) · "
    "9-step analytical workflow"
)

with st.expander("How this works", expanded=False):
    st.markdown(
        """
        The agent follows a **9-step analytical workflow** for every prediction:

        1. **Baseline ratings** - Elo, attack, defence, press intensity, set-piece
           threat, and squad depth baseline for each team from a calibrated table.
        2. **Web research** - 4-6 searches: recent form, injuries, H2H history,
           FIFA rankings, and venue/altitude data.
        3. **Form momentum** - Time-decayed scoring of last 5 results with
           opponent-quality and competition-type weighting.
        4. **Elo adjustment** - Combines form momentum, injury penalties, managerial
           changes, and H2H dominance into a final adjusted Elo per team.
        5. **Venue and conditions** - Altitude, humidity, rest days, and travel
           hours retrieved and passed to the model.
        6. **Tactical matchup analysis** - Press differential and style matchups
           converted into a lambda modifier for the scoring model.
        7. **Dixon-Coles Poisson model** - 10-factor expected goals calculation
           with sigmoidal Elo scaling, depth weighting, fatigue, and altitude
           correction.
        8. **Structured report** - Win probabilities with 90% confidence intervals,
           top-8 scorelines, knockout advancement (Monte Carlo, 100k simulations),
           and confidence assessment.
        9. **Parameter audit** - All inputs passed to the model are echoed for
           full reproducibility.

        **Dixon-Coles** is the industry-standard correction for the Poisson model,
        fixing over-prediction of 0-0 and 1-1 draws. Used by Betfair and most
        commercial prediction systems.

        **Host nation note:** Only USA, Canada, and Mexico receive the neutral-venue
        home boost (+30 Elo). No other teams qualify.
        """
    )

# -- Render existing chat history -------------------------------------------
for user_msg, bot_msg, ts in st.session_state.history:
    with st.chat_message("user"):
        st.write(f"*{ts}*\n\n{user_msg}")
    with st.chat_message("assistant"):
        st.code(bot_msg, language=None)


# -- Determine what query to run --------------------------------------------
prefill_query = ""
if predict_clicked and team_a and team_b:
    stage_type = (
        "group stage" if stage == "Group Stage"
        else "knockout stage"
    )
    host_note = ""
    if host_team not in ("None", ""):
        host_note = f" {host_team} is a host nation and should receive the home boost."
    context_note = f" Additional context: {extra_context.strip()}" if extra_context.strip() else ""
    prefill_query = (
        f"Predict the outcome of a FIFA World Cup 2026 {stage_type} match "
        f"between {team_a} and {team_b}.{host_note}{context_note} "
        f"Follow the full 9-step workflow. Provide win/draw/loss probabilities, "
        f"confidence intervals, top-8 scorelines, and all key analytical factors."
    )

# Example-button click
if st.session_state.pending_query:
    prefill_query = st.session_state.pending_query
    st.session_state.pending_query = ""

# Chat input box
typed_input = st.chat_input(
    "e.g. Predict Argentina vs Brazil, group stage, match in Miami"
)

final_input = prefill_query or typed_input


# -- Run agent if there is input -------------------------------------------
if final_input:
    ts_now = datetime.datetime.now().strftime("%H:%M:%S")

    with st.chat_message("user"):
        st.write(f"*{ts_now}*\n\n{final_input}")

    with st.chat_message("assistant"):
        with st.spinner(
            "Running 9-step workflow: ratings -> form research -> "
            "Elo adjustment -> venue -> tactics -> Dixon-Coles model..."
        ):
            try:
                agent = build_agent(temperature=temperature, model=model)
                bot_response = run_agent(
                    agent,
                    [(u, b) for u, b, _ in st.session_state.history[-2:]],
                    final_input,
                )
            except Exception as exc:
                bot_response = (
                    f"An error occurred:\n\n```\n{exc}\n```\n\n"
                    f"Check that your GROQ_API_KEY is valid and that you have "
                    f"internet access for web search. If you see a rate-limit "
                    f"error, wait 30 seconds and try again - the free Groq tier "
                    f"has a per-minute token limit."
                )

        st.code(bot_response, language=None)

    st.session_state.history.append((final_input, bot_response, ts_now))