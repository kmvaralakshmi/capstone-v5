# v5.2 — Run guide (Windows, step by step)

## 0. One-time setup
1. Install Python 3.11–3.13 (tick "Add Python to PATH").
2. Open a terminal in this folder and run:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` (keep the defaults for now).
4. Check everything imports and the tests pass:
   ```
   python -m unittest discover -s tests -v
   ```
   Expected: `Ran 16 tests ... OK`.

## 1. Run WITHOUT any LLM (works immediately)
```
python run_agents.py --offline
```
All 50 companies, bundled data. Takes ~1–3 minutes (Agent 1 caches extractions in `processed-data/agent1_cache/`).
Results are in `processed-data/`. Then start the dashboard:
```
python -m uvicorn backend.app:app --port 8000
```
Open http://127.0.0.1:8000 → **Cohort Ranking** → click a company.

## 2. Get a FREE, UNLIMITED LLM (Ollama, runs on your PC)
1. Install Ollama from https://ollama.com and start it.
2. `ollama pull qwen2.5:7b-instruct`   (≈4.7 GB; use `llama3.1:8b` if you prefer; for a weak PC use `qwen2.5:3b-instruct`)
3. In `.env` set `LLM_PROVIDER=ollama` and `LLM_MODEL=qwen2.5:7b-instruct` (or leave `auto`).
4. Run again **without** `--offline`:
   ```
   python run_agents.py --companies TCS INFY WIPRO
   ```
   The first line printed shows `LLM: ollama:qwen2.5:7b-instruct`. Agents then choose tools through the LLM,
   and Agent 7 writes the explanations (each is verified against the evidence; failures fall back automatically).
   Look at `processed-data/agent_traces/*_trace.json`: every decision shows its source `llm` / `policy` / `guardrail`.
   Other free endpoints (Groq, Gemini, LM Studio): `LLM_PROVIDER=compatible` + `LLM_BASE_URL` + `LLM_API_KEY`.
   OpenAI later: `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...`, `LLM_MODEL=<model>`.

## 3. Live data (needs internet)
```
python run_agents.py --dynamic            # all 50: Yahoo Finance prices + GDELT news (GDELT: ~5-10 min, rate-limited)
python run_agents.py --company Infosys --dynamic        # ONE company, own folder data/runs/...
python run_agents.py --company HDFCBANK --hq-city Mumbai --dynamic   # outside the 50-company cohort
```
On-demand runs find/download the BRSR XBRL via `helper-scripts/discover_brsr_xbrl.py`. They can also be started from the
dashboard (**Analyze Company** page) while the backend is running.
After the live batch run, the ESG↔Stock and ESG↔News analyses have far more companies (they say "insufficient sample" otherwise).

## 4. Validation against NSE Top 200 and NSE/CFA (you supply the data)
1. Read `data/benchmark/README.txt`.
2. Copy `nse_top200_esg_template.csv` → `nse_top200_esg.csv` and fill `NSE_ESG_Score` (E/S/G optional).
3. Copy `nse_cfa_reference_template.csv` → `nse_cfa_reference.csv` and fill the reference percentages.
4. Run `python run_agents.py --agents agent8` → `processed-data/validation_*.csv` + `validation_summary.json`
   (also shown on the dashboard page **Validation & Research**).

## 5. Useful commands
| Goal | Command |
|---|---|
| Re-score only | `python run_agents.py --agents agent6 agent7 agent8 --offline` |
| Force re-extraction | `python run_agents.py --force` |
| Subset | `python run_agents.py --companies TCS INFY --offline` |
| Windows double-click | `run_batch.bat`, `run_batch_live.bat`, `run_company.bat`, `start_backend.bat` |

## 6. If something fails — send me
`processed-data/agent_run_report.json`, the relevant `processed-data/agent_traces/<agent>_trace.json`, and the last 30 lines of the terminal.

## 7. (v5.2.2) Before any --dynamic / LLM run: self-check
```
python -m esg_agents.diagnostics
```
It tests Ollama (server running? model installed? answers? can call tools?), Yahoo Finance and GDELT, and prints the exact error and the fix.
`run_agents.py` now also performs an LLM health check at start. If it prints `LLM health check: FAILED`, the run continues WITHOUT the LLM
(it no longer pretends to use it). If the Ollama pull ends with `context deadline exceeded`, run `ollama pull qwen2.5:7b-instruct` again until it prints `success`, then `ollama list`.

## 8. Dashboard with the LLM
```
python -m uvicorn backend.app:app --port 8000      # then open http://127.0.0.1:8000
```
Ollama must be running. The sidebar footer shows `LLM: ollama:<model>`. On **Analyze Company** keep "use LLM" ticked
(the backend starts `run_agents.py --company ... [--dynamic]` without `--offline`). Batch results (Cohort Ranking) come from the last `python run_agents.py ...` batch run.

## 9. (v5.2.3) Using the local LLM sensibly — read this
A local 7B model on a CPU needs seconds to minutes PER decision, and Ollama serves one request at a time, so four agents calling
it in parallel queue up. Giving the LLM every one of ~500 routine steps of a 50-company batch would take hours. v5.2.3 therefore:
* consults the LLM only at planning (first steps), after a failed tool call, at checkpoints (every `LLM_EVERY`=25 steps), and for every step in small runs (<= 3 companies);
* caps each agent at `LLM_MAX_DECISIONS` (10) LLM decisions and `LLM_AGENT_TIME_BUDGET` (300 s); the rest follows the agent's own policy (counted as `policy`, not `guardrail`);
* limits LLM-written explanations to the first `LLM_EXPLAIN_MAX` (5) companies in a batch (all companies in a 1–3 company run);
* rejects LLM choices that are out of scope (a company not in this run), duplicate calls, or in the wrong order (e.g. audit before extraction).

Recommended workflow
```
python run_agents.py --offline                      # full batch, minutes, no LLM  (use this for results)
python run_agents.py --companies TCS INFY WIPRO     # small LLM demo: see llm/policy/guardrail counts, LLM-written explanations
python run_agents.py --company Infosys --dynamic    # one company, live data + LLM
```
If it is still slow: `ollama ps` (is it running on CPU?), use a smaller model (`ollama pull qwen2.5:3b-instruct`, then `LLM_MODEL=qwen2.5:3b-instruct` in .env),
or lower `LLM_MAX_DECISIONS`. Press Ctrl+C to stop a slow run.
