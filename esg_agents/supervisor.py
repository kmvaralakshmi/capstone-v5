"""Supervisor: runs the agents as a dependency-aware pipeline with telemetry.

STAGES
    0 discovery        Agent 0          find/download BRSR+XBRL (on-demand runs)
    1 acquisition      Agents 1,2,3,4   run IN PARALLEL (independent data sources)
    2 consistency      Agent 5
    3 scoring          Agent 6
    4 explanation      Agent 7
    5 validation       Agent 8          (batch runs / when requested)

Every agent and every data fetch is timed (RunTracker). The report separates
the stage's wall-clock time from the sum of its parts (parallel gain).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .common import PROCESSED, OUTPUT_FILES, COMPANIES
from .framework import Blackboard, LLMBrain
from .telemetry import RunTracker
from .agent0_filing_discovery import FilingDiscoveryAgent
from .agent1_brsr_extraction import BRSRExtractionAgent
from .agent2_environmental import EnvironmentalAgent
from .agent3_news_sentiment import NewsSentimentAgent
from .agent4_financial import MarketPerformanceAgent
from .agent5_greenwashing import GreenwashingAgent
from .agent6_master_scoring import MasterScoringAgent
from .agent7_explainability import ExplainabilityAgent
from .agent8_validation_research import ValidationResearchAgent

REGISTRY = {"agent0": FilingDiscoveryAgent, "agent1": BRSRExtractionAgent, "agent2": EnvironmentalAgent,
            "agent3": NewsSentimentAgent, "agent4": MarketPerformanceAgent, "agent5": GreenwashingAgent,
            "agent6": MasterScoringAgent, "agent7": ExplainabilityAgent, "agent8": ValidationResearchAgent}
STAGES = [("discovery", ["agent0"]), ("acquisition", ["agent1", "agent2", "agent3", "agent4"]),
          ("consistency", ["agent5"]), ("scoring", ["agent6"]), ("explanation", ["agent7"]),
          ("validation", ["agent8"])]
ORDER = [a for _, group in STAGES for a in group]
DEFAULT_BATCH = [a for a in ORDER if a != "agent0"]          # agent0 is for on-demand runs
DEPENDS = {"agent5": ["agent1", "agent3", "agent4"], "agent6": ["agent1", "agent2", "agent3", "agent4", "agent5"],
           "agent7": ["agent6"], "agent8": ["agent6"]}
NEEDS = {"agent5": [OUTPUT_FILES["brsr_metrics"], "esg_news_company_summary.csv"],
         "agent6": [OUTPUT_FILES["brsr_metrics"]],
         "agent7": [OUTPUT_FILES["master_scores"], "esg_metric_scores.csv"],
         "agent8": [OUTPUT_FILES["master_scores"]]}


class Supervisor:
    def __init__(self, agents: Optional[List[str]] = None, companies: Optional[List[str]] = None,
                 query: Optional[str] = None, hq_city: str = "", use_llm: bool = True, force: bool = False,
                 dynamic: bool = False, verbose: bool = True, run_id: Optional[str] = None):
        self.query, self.hq_city = query, hq_city
        self.selected = agents or ([a for a in ORDER if a != "agent8"] if query else DEFAULT_BATCH)
        bad = [a for a in self.selected if a not in REGISTRY]
        if bad:
            raise ValueError(f"unknown agent(s) {bad}; choose from {ORDER}")
        self.companies, self.force, self.verbose, self.dynamic = companies, force, verbose, dynamic
        self.brain = LLMBrain(enabled=use_llm)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tracker = RunTracker(self.run_id, query or ",".join(companies or ["all"]), PROCESSED)
        self.board = Blackboard(PROCESSED / "agent_blackboard.json")
        if set(self.selected) >= set(DEFAULT_BATCH):
            self.board.clear()
        self.reports: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def _make(self, key: str):
        kw = {"board": self.board, "brain": self.brain, "verbose": self.verbose,
              "tracker": self.tracker, "dynamic": self.dynamic}
        if key == "agent0":
            kw.update(companies=[self.query] if self.query else self.companies, hq_city=self.hq_city)
        elif self.companies:
            kw["companies"] = self.companies
        if key == "agent1":
            kw["force"] = self.force
        return REGISTRY[key](**kw)

    def _blocked(self, key: str) -> Optional[str]:
        bad_up = [r["agent_id"] for r in self.reports
                  if r["agent_id"] in DEPENDS.get(key, []) and r["status"] != "completed"]
        if bad_up:
            return f"upstream agent(s) did not complete: {bad_up}"
        missing = [f for f in NEEDS.get(key, []) if not (PROCESSED / f).exists()]
        return f"missing prerequisite file(s) {missing}" if missing else None

    def _run_one(self, key: str) -> Dict[str, Any]:
        why = self._blocked(key)
        if why:
            msg = f"{key} skipped: {why}"
            self.board.post("supervisor", "skipped", msg, severity="high")
            print(f"[supervisor] {msg}", flush=True)
            return {"agent_id": key, "status": "skipped", "summary": msg}
        agent = self._make(key)
        rep = agent.run().to_dict()
        if key == "agent0" and self.query:
            codes = list(agent.memory["resolved"].values())
            if codes:
                self.companies = codes
            rep["resolved_codes"] = codes
        if rep["status"] != "completed":
            self.board.post("supervisor", "agent_problem",
                            {"agent": key, "status": rep["status"], "summary": rep["summary"]}, severity="high")
        return rep

    def run(self) -> Dict[str, Any]:
        if self.brain.available:
            ok, msg = self.brain.healthcheck()
            print(f"[supervisor] LLM health check: {'OK - ' + msg if ok else 'FAILED - ' + msg}", flush=True)
            if not ok:
                print("[supervisor] continuing with deterministic policies (no LLM).", flush=True)
        print(f"[supervisor] LLM: {self.brain.describe()} | dynamic data: {self.dynamic} | run {self.run_id}", flush=True)
        for stage, group in STAGES:
            todo = [a for a in group if a in self.selected]
            if not todo:
                continue
            with self.tracker.stage(stage):
                if len(todo) > 1:
                    with ThreadPoolExecutor(max_workers=len(todo)) as ex:
                        results = list(ex.map(self._run_one, todo))
                else:
                    results = [self._run_one(todo[0])]
            self.reports.extend(results)
            if stage == "discovery" and any(r["status"] != "completed" for r in results):
                print("[supervisor] discovery did not complete - stopping.", flush=True)
                break
            if stage == "discovery" and self.query and not self.companies:
                break
        summary = self.tracker.summary()
        out = {"run_id": self.run_id, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "llm_mode": self.brain.available, "llm": self.brain.describe(), "llm_problem": self.brain.disabled_reason, "dynamic": self.dynamic,
               "companies": self.companies, "agents": self.reports, "timing": summary}
        (PROCESSED / "agent_run_report.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        self.tracker.flush()
        return out
