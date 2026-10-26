"""Run the multi-agent ESG pipeline.

BATCH (all 50 companies, bundled data)
    python run_agents.py --offline
BATCH with LIVE data (Yahoo Finance prices + GDELT news)
    python run_agents.py --dynamic
ON-DEMAND (one company: name or NSE symbol; discovers filings, live data, own output folder)
    python run_agents.py --company Infosys --dynamic
    python run_agents.py --company HDFCBANK --hq-city Mumbai --dynamic      # outside the cohort
Subsets / re-runs
    python run_agents.py --agents agent6 agent7 agent8
    python run_agents.py --companies TCS INFY WIPRO
LLM
    --offline          never use an LLM (deterministic policies; still fully working)
    (default)          LLM chosen from .env: Ollama (free, local) / OpenAI / compatible
Other
    --force            ignore the Agent 1 extraction cache
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--company", help="on-demand run for ONE company (name or NSE symbol)")
    ap.add_argument("--hq-city", default="", help="HQ city for companies outside the cohort (air-quality context)")
    ap.add_argument("--companies", nargs="*", help="batch subset, e.g. TCS INFY")
    ap.add_argument("--agents", nargs="*", help="agent0 ... agent8")
    ap.add_argument("--dynamic", action="store_true", help="fetch live market (Yahoo) and news (GDELT) data")
    ap.add_argument("--offline", action="store_true", help="disable the LLM")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", help="output folder (default: processed-data, or data/runs/<symbol>_<time> for --company)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.company and not a.out_dir:
        slug = "".join(ch for ch in a.company if ch.isalnum()).upper()[:20]
        a.out_dir = str(ROOT / "data" / "runs" / f"{slug}_{datetime.now():%Y%m%d_%H%M%S}")
    if a.out_dir:
        os.environ["ESG_OUT_DIR"] = a.out_dir          # must be set BEFORE esg_agents is imported
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        print(f"Output folder: {a.out_dir}")

    from esg_agents.supervisor import Supervisor
    sup = Supervisor(agents=a.agents, companies=[c.upper() for c in a.companies] if a.companies else None,
                     query=a.company, hq_city=a.hq_city, use_llm=not a.offline, force=a.force,
                     dynamic=a.dynamic, verbose=not a.quiet)
    rep = sup.run()
    t = rep["timing"]
    print(f"\n=== RUN SUMMARY ===  LLM: {rep['llm']}  | total {t['total_seconds']}s")
    for r in rep["agents"]:
        print(f"{r['agent_id']:8} {r['status']:10} {r.get('seconds', '-')!s:>8}s  {r['summary']}")
    for name, st in t["stages"].items():
        print(f"  stage {name:12} wall={st['stage_wall_seconds']}s  sum_of_parts={st['sum_seconds']}s")
    if t["bottleneck"]:
        print(f"  bottleneck: {t['bottleneck']['name']} ({t['bottleneck']['seconds']}s)")
    return 0 if all(r["status"] in ("completed", "skipped") for r in rep["agents"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
