# v5.1 - Real multi-agent architecture (esg_agents/)

## What was wrong (root causes found by running your v5 code)

| # | Problem | Evidence | Fix |
|---|---------|----------|-----|
| 1 | The "autonomous agents" never ran. `ai_agents/tools.py` imported `BRSRExtractionAgent`, a class that no longer exists (renamed `Agent1BRSRExtractionAgent`). `ai_agents/__init__.py` swallowed the ImportError, so `AGENT_FACTORIES` was `{}`. | `python -c "import ai_agents.tools"` -> ImportError | Layer replaced by `esg_agents/`; old files in `legacy/ai_agents_v4/` |
| 2 | The old agents were one-line factories whose only real tool was "run the whole script". | `ai_agents/agents.py` | Each agent now has its own file, tools, completion criteria, planner, guardrails |
| 3 | XBRL extractor accepted any fact of the right period: period(60)+no-dimension(12)+unit(6-10) >= 75 with **zero** semantic match -> `Total Employees = 1`, `Board Size = 0`, `Energy = 1.5 billion GWh`, all marked VALID. | TCS output | `utils/xbrl_extractor.py`: facts without semantic/label evidence are rejected |
| 4 | Agent 1 CSV only ever contained the last company run; downstream agents then gave every other company default scores (Social = 10, Env = 2). | `processed-data/brsr_extracted_metrics.csv` had 18 rows | Agent 1 now merges all companies, audits and writes one file |
| 5 | Agent 2 found AQI for 3 of 13 cities: historical file has only 5 cities; the real-time file has pollutants but no AQI column; folder name mismatch (`#U2013` vs en-dash) silently returned empty frames. | 49/83 location rows NaN | Agent 2 computes CPCB AQI from pollutants, resolves cities by exact > alias > proxy with confidence |
| 6 | Agent 3 matched company codes as substrings ("SIS" inside "analysis") and credited every "Stocks to watch: A, B, C" round-up to all companies; VADER on full article text saturates at +/-0.99 so everything was "Positive". | 677 "articles", 100% positive | Word-boundary matching, round-up rejection, sentence-level scoring, shrinkage + reliability |
| 7 | Agent 4 **simulated** the ESG-stock correlation (`simulate_esg_correlation` wrote constants such as 0.65). Price data exists for only 7 of 50 companies. | `agents/agent4_stock_correlation.py` | Agent 4 uses real prices only; uncovered companies get no score |
| 8 | Agent 6 defaulted missing scores to 5.0, scored "Social" mainly from news positivity, and still produced master scores for companies with 0 usable metrics. | Quality_Score 0.7 yet Master 6.3 | Agent 6 renormalises over real evidence and **withholds** the score when evidence is insufficient |
| 9 | Housekeeping: truncated `rebuild_xbrl_report.py`, 3 backup copies of the discovery script, `.invalid.pdf` duplicate (15 MB), `__pycache__` in the zip. | | Removed / moved to `legacy/` |

## The agents (one file each)

| File | Agent | What it decides / does |
|------|-------|------------------------|
| `agent1_brsr_extraction.py` | BRSR Extraction & Audit | surveys filings, extracts in an isolated process (timeout + memory cap, XBRL-only retry), audits plausibility (ranges + per-employee intensity), repairs (other source / unit rescale) or quarantines |
| `agent2_environmental.py` | Environmental Context | loads AQI sources, computes CPCB AQI, resolves each city with confidence, scores locations (`ERS = 10(1-AQI/500)`) |
| `agent3_news_sentiment.py` | News Sentiment | inspects corpus, attributes articles, relaxes attribution if coverage is too thin, audits distribution |
| `agent4_financial.py` | Market Performance | maps price coverage, computes FY return/volatility/drawdown from real prices |
| `agent5_greenwashing.py` | Disclosure Consistency | graded, evidence-linked signals; "Insufficient evidence" when data is too thin |
| `agent6_master_scoring.py` | Master Scoring | direction-aware metric scores, pillar formulas from the report, quality gate, withholds weak-evidence scores |
| `agent7_explainability.py` | Explainability | fact sheet + evidence trail per company; optional LLM may only rephrase facts |
| `supervisor.py` | Supervisor | dependency order, prerequisite checks, shared blackboard, run report |
| `framework/` | Runtime | `BaseAgent` loop, `Tool`, `LLMBrain`, `Blackboard` |

### Why these are agents and not scripts
Every agent runs the loop **observe -> decide -> act -> observe -> reflect** until *code-checked* completion criteria hold:
* **Tools**: named, schema-described actions the agent may take.
* **Planner**: the LLM (function calling) when `OPENAI_API_KEY` is set; otherwise the agent's own deterministic `policy()`.
* **Guardrails** (tested in `tests/test_agents.py`): unknown tool / bad arguments / repeated failing call / premature `finish` are rejected and replaced by the policy action.
* **Reflection**: e.g. Agent 3 relaxes its attribution rule when coverage is too thin; Agent 1 re-audits after repairs.
* **Communication**: agents post coverage warnings to `processed-data/agent_blackboard.json`; downstream agents read them.
* **Evidence of autonomy**: `processed-data/agent_traces/<agent>_trace.json` records every decision with its source (`llm` / `policy` / `guardrail`) and observation.

Deterministic Python does all arithmetic (auditable, reproducible); the LLM only chooses tools and, in Agent 7, rephrases facts.

## Run
```
pip install -r requirements.txt
python run_agents.py --offline            # no API key needed
python run_agents.py                      # LLM chooses tools if OPENAI_API_KEY is set in .env
python run_agents.py --companies TCS INFY # subset
python run_agents.py --agents agent6 agent7
python -m unittest discover -s tests -v
python -m uvicorn backend.app:app         # dashboard
```
Agent 1 caches each company in `processed-data/agent1_cache/` (resume after any failure; `--force` to redo).

## Honest limitations
* One FY of BRSR data; no trend claims.
* The bundled stock file covers only NIFTY-50 members (7 of 50 cohort companies); the news corpus is generic market news, so few companies have ESG articles. These companies are scored without those components rather than with invented values.
* Air quality is location context, not proof of company performance.
* PDF extraction still fails on some filings; those metrics are quarantined and flagged, not guessed.
* The OpenAI path could not be executed in the build sandbox (no network/API key); it was tested against a scripted fake client. Run it once with your key and check `agent_traces/*.json`.
