# v5.2 — what was implemented from Sections A, B, C

| Spec | Status | Where |
|---|---|---|
| A1 XBRL primary + PDF evidence/fallback, provenance per metric | done | Agent 1 (`brsr_taxonomy.py`), `Source/Page/Confidence` columns |
| A2 "Analyze Infosys" → find/download filings automatically | done (needs internet; untested here) | `agent0_filing_discovery.py`, reuses `helper-scripts/discover_brsr_xbrl.py` |
| A3/A5 dynamic external data, parallel, dependency-aware | done | `data_providers.py`, `supervisor.py` (Agents 1-4 run in parallel) |
| A4/A9 market data | Yahoo Finance via yfinance (free) + bundled fallback | `data_providers.fetch_market` — NOT an official NSE feed; see note below |
| A7 GDELT news | done (titles only; untested here) | `data_providers.fetch_news`, Agent 3 `--dynamic` |
| A8 CPCB / OpenAQ | bundled CPCB files by default; OpenAQ hook if key set | Agent 2, `fetch_openaq_city` |
| A11 static data kept for tests/benchmarks | done | all agents fall back to bundled data |
| A12 rule-based + autonomous | done | deterministic math + agent planners |
| A14-A16 timing, parallel timing, provenance | done | `telemetry.py` → `run_metrics.json`; dashboard **Performance** |
| A17 data folders | `data/cache`, `data/runs`, `data/benchmark` added | |
| B1-B16 frontend | done: 12 pages | `frontend/`, `backend/app.py` |
| C1 LLM explanation agent, evidence-grounded | done + automatic grounding verification | Agent 7 |
| C2/C3 no ML, no SHAP/LIME | respected | |
| C4 NSE Top 200 validation | comparison code ready; you supply the file | Agent 8 |
| C5 NSE/CFA comparison | comparison code ready; you supply the file | Agent 8 |
| C6 ESG↔Stock (Pearson+Spearman) | done | Agent 8 |
| C7 ESG↔News | done | Agent 8 |
| C8 cross-source greenwashing | done: claim → BRSR trend → news → verdict | Agent 5 |
| C9 ESG↔Air quality (context) | done | Agent 8 |
| C10 Validation & Research page | done | dashboard |

## Decisions I made (change if you disagree)
* **Score scale 0–100** (S&P/LSEG-style, matches your mock-ups). Internal sub-scores stay 0–10; risk: Low ≥ 70, Medium 50–70, High < 50.
* **Free LLM = Ollama (local)** — unlimited, no key, OpenAI-compatible, so you can switch to OpenAI/Groq/Gemini by changing `.env`.
* **Yahoo Finance for prices** because official NSE historical endpoints need session handling/licensing (your A9 note). `MarketProvider` is isolated in `data_providers.py`, so an NSE-bhavcopy provider can replace it.
* **GDELT gives titles, not article text**, so dynamic news sentiment is title-level.

## Not verified in my sandbox (no internet) — please test first
Yahoo Finance fetch, GDELT fetch, NSE XBRL discovery for companies outside your folder, the Ollama tool-calling path, and the browser UI rendering. Unit tests cover the logic with mocks (16 pass), and the full offline pipeline + API job run were executed.

## Known limitations
* Governance pillar is still mostly unscored (independent directors, board meetings, audit committee, training hours are not in the XBRL) — needs a PDF governance-table reader.
* BRSR PDF auto-download for a company outside the cohort only works if its PDF URL appears in the cached NSE listing; otherwise it runs XBRL-only.
* Small local models (3B–7B) may pick odd tools; guardrails substitute the agent's own policy, and `agent_traces` show how often.
