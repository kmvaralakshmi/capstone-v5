"""Isolated worker: extract ONE company in its own process.

Why: very large BRSR PDFs (15-20 MB) can exhaust memory or hang. Running the
extraction in a child process lets Agent 1 survive a crash/timeout, retry in
XBRL-only mode, and carry on with the other companies.

    python -m esg_agents.workers.extract_company TCS [--xbrl-only]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("code")
    ap.add_argument("--xbrl-only", action="store_true")
    ap.add_argument("--mem-gb", type=float, default=2.5)
    a = ap.parse_args()
    try:
        import resource
        lim = int(a.mem_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    except Exception:
        pass  # not available on Windows - timeout still protects us
    from agents.agent1_brsr_extractor import Agent1BRSRExtractionAgent
    from utils.config import COMPANIES
    code = a.code.upper()
    eng = Agent1BRSRExtractionAgent()
    if a.xbrl_only:
        eng.run_pdf = lambda *args, **kw: {"status": "SKIPPED", "metrics": {},
                                           "timing_sec": 0.0,
                                           "error": "PDF skipped (xbrl-only fallback)"}
    with contextlib.redirect_stdout(io.StringIO()):
        eng.process_company(code, COMPANIES[code])
    out = ROOT / "processed-data" / "agent1_cache"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{code}.json").write_text(json.dumps([dict(r) for r in eng.rows], default=str),
                                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
