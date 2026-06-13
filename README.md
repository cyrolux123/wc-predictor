# WC2026 Agentic Match Predictor

A production-grade football match prediction system for FIFA World Cup 2026, built as a LangGraph ReAct agent backed by a Dixon-Coles Poisson model with 10-factor xG calculation.

---

## What it does

You type a natural-language question like:

> "Predict Argentina vs France, knockout round, match in Miami"

The agent autonomously runs a **9-step analytical workflow** — pulling live web data, computing form momentum, adjusting Elo ratings, and running 100,000 Monte Carlo simulations — then returns a structured professional report with win probabilities, confidence intervals, top-8 scorelines, and (for knockout matches) penalty advancement probabilities.

---

## Architecture

```
app.py          Streamlit UI — sidebar controls, chat history, quick predictor
agent.py        LangGraph ReAct agent — 9-step system prompt, LLM wiring
tools.py        Four LangChain tools: web_search, get_team_rating,
                get_form_momentum, predict_match (Dixon-Coles v5 engine)
```

### The four tools

| Tool | Purpose |
|---|---|
| `get_team_rating` | Returns baseline Elo, attack, defence, press intensity, set-piece threat, squad depth for 90+ nations |
| `web_search` | DuckDuckGo real-time search: form, injuries, H2H, rankings, venue data |
| `get_form_momentum` | Time-decayed form scoring (decay=0.80) with opponent-quality and competition-type weighting |
| `predict_match` | Dixon-Coles Poisson model: 10-factor xG, sigmoidal Elo scaling, 100k MC simulations, bootstrapped 90% CI |

---

## 9-Step Prediction Workflow

The agent is hard-prompted to follow this workflow for every single prediction, without exception:

1. **Baseline ratings** — Elo, attack, defence, press intensity, set-piece threat, squad depth for both teams
2. **Web research** — minimum 6 searches: form, injuries, H2H, FIFA rankings, venue altitude, venue humidity
3. **Form momentum** — time-decayed scoring of last 5 results with opponent-quality and competition-type weighting
4. **Elo adjustment** — combines form momentum (+/-55 pts), injury penalties (-10 to -50), managerial changes (-10 to -20), H2H dominance (+8 to +25)
5. **Venue and conditions** — altitude, humidity (auto-resolved from city name), rest days, travel hours
6. **Tactical matchup analysis** — press differential and style matchup converted into a lambda modifier (+/-0.05 to +/-0.15)
7. **Dixon-Coles prediction** — full 10-factor model with sigmoidal Elo scaling
8. **Structured report** — win probabilities with 90% CI, top-8 scorelines, knockout advancement, confidence label
9. **Parameter audit** — all inputs echoed for full reproducibility

---

## Model Details

### Expected Goals (xG) — 10 factors

```
lam = BASE_GOALS(1.08) × attack × (1/defence) × elo_scale
    × tactical_modifier × press_suppression × set_piece_bonus
    × ref_adj × squad_depth × altitude × humidity
```

- **Sigmoidal Elo scaling**: `0.55 + 0.90 × sigmoid(diff/350)` — reduces overreaction to large Elo gaps
- **Press suppression**: up to 12% lambda reduction for high-press vs low-press mismatch
- **Set-piece coefficient**: 0.55 per unit above 0.55 baseline (43% of WC2022 goals from set pieces)
- **Altitude tiers**: sea level (×1.0), 1000m (×0.98), 1500m (×0.96), 2000m (×0.92), 2500m+ (×0.88)
- **Humidity**: coefficient 0.04 (±4% at extremes)
- **Fatigue**: rest <4 days (−20 Elo), travel >12h (−12 Elo)

### Dixon-Coles Correction

The standard Poisson model over-predicts 0-0 and 1-1 draws. Dixon-Coles (1997) applies a tau correction to low-score cells:
- `rho = -0.13` for group stage, `-0.10` for knockout (tighter games)

This correction is used by Betfair and most commercial prediction engines.

### Bootstrapped 90% Confidence Intervals

2,000-sample bootstrap with ±45 Elo Gaussian perturbation. The CI reflects genuine uncertainty in ratings — a team rated 2050 Elo could plausibly be anywhere from ~2005 to ~2095.

### Monte Carlo Knockout Simulation

100,000 simulations of: 90 min → Extra Time (27% of 90-min rate, fatigue-adjusted) → Penalty Shootout (Elo edge regressed 70% toward 50-50, modified by squad depth and momentum).

---

## Confidence Scale

| Win probability margin | Label |
|---|---|
| < 5% | COIN FLIP — do not pick a winner |
| 5–10% | LOW CONFIDENCE — slight lean |
| 10–18% | MODERATE CONFIDENCE — reasonable lean |
| 18–28% | GOOD CONFIDENCE — clear edge |
| > 28% | HIGH CONFIDENCE — dominant favourite |

Win probability 50–56% is explicitly flagged as a coin flip. The model never overstates conviction.

---

## Setup

### Requirements

- Python 3.10+
- A free [Groq API key](https://console.groq.com) (generous free tier, ~6,000 RPM)

### Installation

```bash
# Clone or download the project
git clone <your-repo-url>
cd wc2026-predictor

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY

# Run the app
streamlit run app.py
```

### .env file

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your key at: https://console.groq.com

---

## Usage

### Quick Predictor (sidebar)

1. Enter Team A and Team B in the sidebar
2. Select the match stage (Group Stage / Round of 16 / etc.)
3. If applicable, select the host nation (USA, Canada, or Mexico only)
4. Optionally add context: injuries, rest days, venue city
5. Click **Predict This Match**

### Natural Language Chat

Type any question into the chat box:

```
Predict England vs Germany, knockout round
Who wins Brazil vs France? France is missing Mbappe.
Predict Morocco vs Japan group stage match in Atlanta
Spain vs Netherlands, who has better recent form?
Predict USA vs Mexico, USA are the host nation
```

### Downloading Results

After one or more predictions, a **Download Transcript** button appears in the sidebar. This saves all predictions as a plain-text file.

---

## Team Coverage

The baseline ratings table covers 90+ nations including all likely WC2026 qualifiers:

- **Tier 1**: Argentina, France, Spain, England, Brazil, Portugal, Netherlands
- **Tier 2**: Germany, Belgium, Croatia, Italy, Uruguay, Colombia, Morocco, Japan, Serbia, Austria
- **Tier 3**: USA, Mexico, Canada, Switzerland, Denmark, Senegal, South Korea, Ecuador
- **WC2026 qualifiers**: Qatar, Chile, Peru, China, Jordan, Thailand, Tanzania, Cape Verde, Namibia, Comoros, Cuba, Guatemala, Haiti, Panama, Jamaica, Honduras, Venezuela, Bolivia, and more

For any team not in the table, the model uses calibrated defaults and instructs the agent to search for their FIFA ranking.

---

## WC2026 Venue Data

All 16 host city venues are pre-loaded with altitude and humidity data:

| City | Country | Humidity index |
|---|---|---|
| Miami | USA | 0.85 (very humid) |
| Houston | USA | 0.80 |
| Atlanta | USA | 0.68 |
| Dallas | USA | 0.62 |
| Nashville | USA | 0.65 |
| New York | USA | 0.60 |
| Philadelphia | USA | 0.60 |
| Boston | USA | 0.58 |
| Seattle | USA | 0.55 |
| Kansas City | USA | 0.50 |
| San Francisco | USA | 0.45 |
| Los Angeles | USA | 0.40 |
| Denver | USA | 0.25 (driest) |
| Toronto | Canada | 0.55 |
| Vancouver | Canada | 0.62 |
| Guadalajara | Mexico | 0.52 |
| Monterrey | Mexico | 0.48 |
| Mexico City | Mexico | 0.38 |

Pass `venue_city="miami"` (lowercase) and the model resolves humidity automatically.

---

## Model Limitations

This is a statistical model, not a crystal ball. Football is inherently unpredictable:

- **Upset rate**: even a team with a 75% win probability loses ~25% of the time
- **Injury information**: relies on web search quality; unannounced injuries are invisible
- **Elo ratings**: calibrated to June 2026 but not updated in real-time
- **Tactical novelty**: a surprise formation change mid-tournament is not captured
- **Psychology**: penalty shootout models cannot capture individual player mentality

Always read the confidence label and the 90% CI before drawing conclusions.

---

## File Reference

| File | Version | Description |
|---|---|---|
| `tools.py` | v5.1 | Four LangChain tools, Dixon-Coles engine, 90+ team ratings |
| `agent.py` | v4.1 | LangGraph ReAct agent, 9-step system prompt |
| `app.py` | v3.1 | Streamlit UI, sidebar controls, chat history |
| `requirements.txt` | v3 | Pinned dependencies |
| `.env.example` | — | Environment variable template |

---

## Key Changes in This Version (v5.1 / v4.1 / v3.1)

Improvements over the uploaded originals:

- `predict_match` now accepts `venue_city` and auto-resolves humidity from the built-in table — no more manual lookup required
- LangGraph `create_react_agent` is now imported first (the `langchain.agents` version is deprecated)
- Invalid model `openai/gpt-oss-120b` removed from Streamlit selector; replaced with valid Groq models
- Streamlit `page_icon` fixed from empty string to ⚽
- Bootstrap CI `rho` now correctly reflects `match_type` (group vs knockout)
- 20+ missing WC2026 qualified nations added to baseline ratings table (Qatar, Chile, Peru, China, Jordan, Cuba, Guatemala, Haiti, etc.)
- Nashville added to VENUE_HUMIDITY; `new york new jersey` alias added
- System prompt Step 7 updated to include `venue_city` in parameter list
- Parameter audit block updated to log venue city

---

Predictions are for analytical and educational purposes only.#   w c - p r e d i c t o r  
 