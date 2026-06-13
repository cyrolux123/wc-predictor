"""
agent.py  -  WC2026 ReAct Agent  (v4)
======================================
LangGraph ReAct agent with an 8-step analytical workflow.

Changes from v3
---------------
- All emojis removed from system prompt and output instructions.
- Host nation boost note updated: only USA, Canada, Mexico qualify.
- Step 5 explicitly prompts agent to pass venue_humidity, rest_days, travel_hours,
  set_piece, press_intensity from get_team_rating to predict_match.
- Step 6 tactical analysis now references press_intensity deltas directly.
- Step 8 output template is clean plaintext, professional report format.
- Temperature lowered to 0.10 for more deterministic tool sequencing.
- Model default updated to llama-3.3-70b-versatile.
- Added explicit instruction never to skip v5 parameters in predict_match.
- Confidence calibration note expanded: 55-60% = slight lean, not a strong pick.
- Added Step 9: parameter audit — agent must echo all parameters it passed to
  predict_match so output is fully reproducible.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

try:
    from langgraph.prebuilt import create_react_agent
except ImportError:
    from langchain.agents import create_react_agent

from tools import web_search, get_team_rating, get_form_momentum, predict_match

load_dotenv()


# -----------------------------------------------------------------------------
# SYSTEM PROMPT  (v4 - clean professional, no emojis, 9-step workflow)
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a football data scientist producing WC2026 match predictions. Follow this 9-step workflow. All 4 tools must be called.

TOOLS: get_team_rating | web_search | get_form_momentum | predict_match

S1. Call get_team_rating for both teams (Elo, attack, defence, press_intensity, set_piece_threat, squad_depth_baseline).
S2. Run up to 2 web searches: "{A} vs {B} recent form injuries 2026" and "{venue city} WC2026 altitude" if venue known. Cite source URLs briefly.
S3. Call get_form_momentum for both teams. Use format "W(strong)[wc], D(avg)[qualify]...". Supply days_since_last_match if known.
S4. Adjust Elo: final = baseline + form_adj + injury_adj + h2h_adj. State briefly, e.g. "[A]: base=2115 form=+20 injury=-25 final=2110".
S5. Venue: altitude_m, venue_city (lowercase), match_type (group/knockout), rest_days_a/b, travel_hours_a/b. team_a_home_advantage=True ONLY if team_a is USA, Canada, or Mexico.
S6. Set matchup_modifier_a (-0.15 to +0.15) based on press/style matchup. 1 sentence reasoning.
S7. Call predict_match with ALL params: team_a, team_b, rating_a, rating_b, attack_a, defence_a, attack_b, defence_b, neutral_venue=True, team_a_home_advantage, h2h_advantage, match_type, altitude_m, venue_city, matchup_modifier_a, rest_days_a, rest_days_b, travel_hours_a, travel_hours_b, squad_depth_a, squad_depth_b, set_piece_a, set_piece_b, press_intensity_a, press_intensity_b. Never omit any parameter. Never alter attack/defence from S1.
S8. REPORT (plaintext, no emojis):
WC2026: [A] vs [B] | [Stage]
PREDICTED OUTCOME: [result]
WIN PROBABILITIES: [A] X.X% [CI]  Draw Y.Y% [CI]  [B] Z.Z% [CI]
MODEL xG: [A] X.XX  [B] Y.YY
TOP SCORELINES: [from tool]
[Knockout only] ADVANCEMENT: [A] X%  [B] Z%  AET X%  Pens X%
KEY FACTORS: Form | Injuries | H2H | Tactics | Venue | Rest
CONFIDENCE: [from tool] + note if <56% = coin flip, 57-62% = lean only
S9. PARAMETER AUDIT (one line):
Elo: A=[v] B=[v] | Attack/Def: A=[a]/[d] B=[a]/[d] | Depth: A=[v] B=[v] | Venue: [city] [alt]m | h2h=[v] matchup=[v] | Rest: A=[v]d B=[v]d | Travel: A=[v]h B=[v]h

RULES:
- 50-56% win prob = coin flip, do not name a winner.
- 57-62% = slight lean. 63%+ = meaningful edge (57=slight, 62=moderate, 67+=strong).
- Cite source URL for every web_search claim.
- Knockout stage: always include penalty advancement %.
"""


def build_agent(temperature: float = 0.10, model: str = "llama-3.1-8b-instant"):
    """
    Build and return a LangGraph ReAct agent with the v4 workflow.

    temperature=0.10 for highly deterministic tool sequencing while
    preserving reasoning flexibility. Default model is llama-3.1-8b-instant
    for its much higher free-tier TPM limit (avoids 413 token-limit errors).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to your .env file:\n"
            "  GROQ_API_KEY=your_key_here"
        )

    llm = ChatGroq(
        groq_api_key=api_key,
        model=model,
        temperature=temperature,
    )

    agent = create_react_agent(
        model=llm,
        tools=[web_search, get_team_rating, get_form_momentum, predict_match],
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )
    return agent


def run_agent(agent, history: list[tuple[str, str]], user_input: str) -> str:
    """
    Run one turn of the agent.

    history    : list of (user_message, bot_message) from previous turns.
    user_input : new user message.
    Returns the agent's final response as a string.
    """
    messages = []
    for u, b in history:
        messages.append(HumanMessage(content=u))
        messages.append(AIMessage(content=b))
    messages.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": messages})
    return result["messages"][-1].content