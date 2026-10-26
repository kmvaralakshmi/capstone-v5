"""Run all seven ESG agents (esg_agents package) and copy outputs to demo-output.

Uses the LLM to choose tools when OPENAI_API_KEY is set, otherwise each agent
uses its deterministic policy. For options use run_agents.py directly.
"""
from pathlib import Path
from shutil import copy2

from utils.config import PROCESSED_DATA_DIR, OUTPUT_FILES
from esg_agents.supervisor import Supervisor

BASE_DIR = Path(__file__).resolve().parent
DEMO_OUTPUT_DIR = BASE_DIR / "demo-output"
EXTRA = ["agent1_audit_report.csv", "environmental_company_summary.csv",
         "esg_news_company_summary.csv", "greenwashing_signals_detail.csv",
         "esg_metric_scores.csv", "esg_evidence_trails.json",
         "agent_run_report.json", "agent_blackboard.json"]


def collect_outputs() -> None:
    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    names = list(OUTPUT_FILES.values()) + EXTRA
    copied = 0
    for name in names:
        src = Path(PROCESSED_DATA_DIR) / name
        if src.exists():
            copy2(src, DEMO_OUTPUT_DIR / name)
            copied += 1
        else:
            print(f"Missing output: {src}")
    print(f"\nDemo output folder: {DEMO_OUTPUT_DIR}\nFiles copied: {copied}")


def main() -> None:
    report = Supervisor().run()
    failed = [a for a in report["agents"] if a["status"] not in ("completed", "skipped")]
    collect_outputs()
    if failed:
        raise SystemExit(f"Agents failed: {[a['agent_id'] for a in failed]}")


if __name__ == "__main__":
    main()
