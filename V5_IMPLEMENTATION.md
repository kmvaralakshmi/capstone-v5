# V5 Implementation — Final Improvement Plan

## Implemented

### Phase 1 — Shared foundation
- Canonical `EvidenceRecord` schema in `sources/provenance.py`.
- `run_id`, company traceability, source, URL, retrieval time, financial year, evidence locator, confidence, validation status, fallback and discrepancy fields.
- Stage-level timing/error tracking in `performance/run_tracker.py`.
- Run artifacts saved under `processed-data/run_metrics/`.
- Existing V4 deterministic batch pipeline remains available.

### Phase 2 — BRSR/XBRL source layer
- `sources/filing_discovery.py` resolves the company and checks the existing official-source BRSR registry/local cache first.
- `sources/xbrl_parser.py` parses XML/XBRL ZIP content into provenance-rich facts.
- XBRL is treated as the primary structured source; BRSR remains the evidence/fallback source.
- The source layer is designed for configurable official NSE discovery/download endpoints rather than a brittle undocumented scraper.

### Phase 3 — Dynamic company/filing discovery
- `sources/company_resolver.py` accepts company name, NSE symbol, aliases and fuzzy matching.
- `run_v5_pipeline.py --company <name-or-symbol>` supports single-company analysis.
- Existing 50-company batch configuration is preserved.

### Phase 4 — External-source orchestration
- NSE market data, GDELT news and CPCB/data.gov.in air-quality adapters are provided.
- OpenAQ is a configured fallback and requires `OPENAQ_API_KEY`.
- External calls are dependency-aware and run in parallel when independent.
- JSON caching, timeout handling, status reporting and fallback metadata are included.

### Phase 5 — Analysis and reasoning
- Existing deterministic scoring is preserved for numeric calculations.
- `ai_agents/evidence_explanation.py` provides the Evidence-grounded Explanation Agent using the configurable OpenAI Responses API model.
- Only structured evidence is sent to the explanation agent; it is instructed not to invent facts or recalculate the score.
- `v5_core/greenwashing_agent.py` performs cross-source consistency reasoning and does not make unsupported greenwashing accusations.
- Stock/news/AQI relationships are treated as association/context, not causation.

### Phase 6 — Research and validation
- `v5_core/research_validation.py` computes Pearson/Spearman correlations where sufficient data exist.
- Supports official NSE Top 200 benchmark input and NSE/CFA BRSR disclosure benchmark input through `research-benchmarks/`.
- Reports score differences, mean absolute difference and disclosure-quality comparison when benchmark files are supplied.

### Phase 7 — Frontend
- Company analysis entry point.
- Evidence-grounded explanation display.
- Source/filing status.
- Validation & Research panel.
- Existing company ranking, BRSR download status and legacy outputs remain available.

## Explicitly excluded
- Custom ML training.
- SHAP.
- LIME.

These remain future enhancements only.

## Security
The delivered ZIP intentionally contains a sanitized `.env` with an empty API-key value. The original local key is not included. If the original key was exposed, revoke/rotate it before use.

## OpenAI model configuration
`OPENAI_MODEL` remains configurable. The current example uses `gpt-6-luna`, which is a current API model identifier shown in OpenAI's pricing/model documentation at packaging time. The code does not hard-code the model.

## Commands

### Existing V4 deterministic pipeline
```bash
python demo.py
```

### V5 single-company analysis
```bash
python run_v5_pipeline.py --company Infosys
```

### V5 single-company analysis after refreshing deterministic outputs
```bash
python run_v5_pipeline.py --company INFY --run-deterministic
```

### V5 batch validation
```bash
python run_v5_pipeline.py --batch --run-deterministic
```

### Frontend
```bash
python -m uvicorn backend.app:app --reload
```

## Important source-access note
The source adapters deliberately avoid pretending that NSE/CPCB/OpenAQ are unrestricted anonymous scraping APIs. Configure the exact official/public access mechanism available to the project and respect provider terms, rate limits and authentication requirements.
