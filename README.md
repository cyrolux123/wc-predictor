# WC2026 Agentic Match Predictor

A production-grade AI football prediction engine for the FIFA World Cup 2026, built with LangGraph, Groq LLMs, and an advanced Dixon-Coles Poisson model featuring a 10-factor expected goals (xG) system.

The system behaves like an autonomous football analyst:

* Researches live information from the web
* Evaluates team strength and momentum
* Adjusts ratings dynamically
* Simulates matches 100,000 times
* Produces professional analytical reports with confidence intervals and knockout probabilities

---

# Features

## Fully Agentic Workflow

The predictor is powered by a LangGraph ReAct Agent that autonomously performs a structured 9-step reasoning pipeline for every match.

Example query:

```text
Predict Argentina vs France, knockout round, match in Miami
```

The agent automatically:

* Collects live contextual data
* Evaluates injuries and form
* Adjusts Elo ratings
* Applies tactical modifiers
* Runs Monte Carlo simulations
* Generates a detailed prediction report

---

# Project Architecture

```text
wc2026-predictor/
│
├── app.py               # Streamlit frontend UI
├── agent.py             # LangGraph ReAct agent
├── tools.py             # Prediction engine + LangChain tools
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
└── README.md
```

---

# Core Components

## `app.py`

Interactive Streamlit interface with:

* Sidebar quick predictor
* Natural-language chat
* Prediction transcript history
* Downloadable reports

---

## `agent.py`

LangGraph ReAct orchestration layer.

Responsible for:

* Tool calling
* Step-by-step reasoning
* Structured prediction workflow
* Prompt enforcement

---

## `tools.py`

Contains all prediction logic and LangChain tools.

### Included Tools

| Tool                | Purpose                                                     |
| ------------------- | ----------------------------------------------------------- |
| `get_team_rating`   | Baseline Elo, attack, defence, pressing, set-piece strength |
| `web_search`        | Live football research via DuckDuckGo                       |
| `get_form_momentum` | Time-decayed form scoring engine                            |
| `predict_match`     | Dixon-Coles Poisson simulator with 100k Monte Carlo runs    |

---

# 9-Step Prediction Pipeline

The agent is hard-constrained to follow this analytical workflow for every prediction.

---

## 1. Baseline Ratings

Retrieves:

* Elo rating
* Attack strength
* Defensive strength
* Press intensity
* Set-piece threat
* Squad depth

---

## 2. Live Web Research

Minimum 6 real-time searches including:

* Recent form
* Injury news
* Head-to-head history
* FIFA rankings
* Venue altitude
* Venue humidity

---

## 3. Form Momentum Analysis

Applies:

* Time-decay weighting (`0.80`)
* Opponent-quality scaling
* Competition weighting

Recent matches matter more than older matches.

---

## 4. Dynamic Elo Adjustment

Adjusts baseline ratings using:

| Factor                 | Adjustment |
| ---------------------- | ---------- |
| Form momentum          | ±55 Elo    |
| Injury penalties       | −10 to −50 |
| Managerial instability | −10 to −20 |
| H2H dominance          | +8 to +25  |

---

## 5. Venue & Environmental Effects

Automatically evaluates:

* Altitude
* Humidity
* Travel fatigue
* Rest days

---

## 6. Tactical Matchup Analysis

Converts tactical interaction into lambda modifiers:

```text
+0.05 to +0.15
```

Includes:

* High press vs low block
* Possession mismatch
* Transition vulnerability

---

## 7. Dixon-Coles Match Simulation

Runs the full statistical engine:

* 10-factor xG model
* Sigmoidal Elo scaling
* Dixon-Coles correction
* 100,000 Monte Carlo simulations

---

## 8. Structured Prediction Report

Outputs:

* Win probabilities
* 90% confidence intervals
* Top 8 scorelines
* Knockout advancement probabilities
* Confidence label

---

## 9. Parameter Audit

Every input parameter is logged for reproducibility.

---

# Expected Goals (xG) Model

The prediction engine calculates expected goals using:

```text
λ = BASE_GOALS × attack × (1/defence) × elo_scale
    × tactical_modifier × press_suppression
    × set_piece_bonus × referee_adjustment
    × squad_depth × altitude × humidity
```

---

## Included Factors

| Factor            | Purpose                     |
| ----------------- | --------------------------- |
| Attack rating     | Offensive quality           |
| Defensive rating  | Defensive resistance        |
| Elo scaling       | Team strength differential  |
| Tactical modifier | Style matchup               |
| Press suppression | Pressing intensity mismatch |
| Set-piece bonus   | Dead-ball efficiency        |
| Squad depth       | Rotation resilience         |
| Altitude          | Environmental fatigue       |
| Humidity          | Energy drain                |
| Ref adjustment    | Match control variation     |

---

# Statistical Enhancements

## Sigmoidal Elo Scaling

```text
0.55 + 0.90 × sigmoid(diff / 350)
```

Prevents unrealistic overreaction to large Elo gaps.

---

## Dixon-Coles Correction

Corrects low-scoring bias in standard Poisson models.

```text
ρ = -0.13   # Group stage
ρ = -0.10   # Knockout stage
```

Widely used in professional betting and analytics systems.

---

## Bootstrapped Confidence Intervals

* 2,000 bootstrap samples
* ±45 Elo Gaussian perturbation
* Produces realistic uncertainty ranges

---

## Knockout Simulation Engine

For knockout matches:

```text
90 mins → Extra Time → Penalties
```

Penalty shootouts include:

* Elo influence
* Squad depth
* Momentum regression
* Fatigue effects

---

# Confidence Labels

| Margin | Label               |
| ------ | ------------------- |
| < 5%   | COIN FLIP           |
| 5–10%  | LOW CONFIDENCE      |
| 10–18% | MODERATE CONFIDENCE |
| 18–28% | GOOD CONFIDENCE     |
| > 28%  | HIGH CONFIDENCE     |

The system intentionally avoids overconfidence.

---

# WC2026 Venue Intelligence

Built-in support for all World Cup 2026 host cities.

Example venue data:

| City        | Humidity |
| ----------- | -------- |
| Miami       | 0.85     |
| Houston     | 0.80     |
| Atlanta     | 0.68     |
| Denver      | 0.25     |
| Mexico City | 0.38     |

Usage:

```python
venue_city="miami"
```

Humidity and environmental modifiers are resolved automatically.

---

# Team Coverage

Supports 90+ national teams including:

## Tier 1

* Argentina
* France
* Spain
* Brazil
* England
* Portugal
* Netherlands

## Tier 2

* Germany
* Italy
* Croatia
* Morocco
* Uruguay
* Colombia
* Japan

## Tier 3

* USA
* Mexico
* Canada
* South Korea
* Denmark
* Senegal
* Switzerland

Also includes many likely WC2026 qualifiers.

---

# Installation

## Requirements

* Python 3.10+
* Groq API key

Get your key here:

https://console.groq.com

---

## Clone Repository

```bash
git clone <your-repo-url>
cd wc2026-predictor
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

## Run the App

```bash
streamlit run app.py
```

---

# Example Queries

```text
Predict Brazil vs France in Miami

England vs Germany knockout round

Who wins Argentina vs Spain?

Predict Morocco vs Japan in Atlanta

USA vs Mexico, USA are the host nation

France vs Portugal, France missing Mbappe
```

---

# Export Predictions

The Streamlit UI includes a Download Transcript button for exporting prediction history as plain text.

---

# Version Information

| File       | Version |
| ---------- | ------- |
| `tools.py` | v5.1    |
| `agent.py` | v4.1    |
| `app.py`   | v3.1    |

---

# Improvements in Current Version

## Added

* Automatic venue humidity resolution
* Additional WC2026 qualified nations
* Nashville venue support
* Better parameter auditing
* Match-type aware Dixon-Coles rho values

## Fixed

* Deprecated LangGraph imports
* Invalid Groq model selection
* Streamlit page icon issue
* Venue alias handling

---

# Limitations

Football remains highly unpredictable.

The model cannot fully account for:

* Sudden tactical surprises
* Locker-room psychology
* Hidden injuries
* Referee volatility
* Penalty shootout mentality

Even a team with a 75% win probability still loses roughly 1 in 4 matches.

Always interpret predictions alongside the confidence interval.

---

# Tech Stack

* LangGraph
* LangChain
* Streamlit
* Groq API
* Dixon-Coles Model
* Monte Carlo Simulation
* DuckDuckGo Search
* Python 3.10+

---

# License

MIT License

---

# Author

Built for advanced football analytics, AI agent experimentation, and World Cup simulation research.
