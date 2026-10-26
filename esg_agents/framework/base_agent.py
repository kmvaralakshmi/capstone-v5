"""BaseAgent: the perceive -> plan -> act -> observe -> reflect loop.

Every ESG agent subclasses BaseAgent and supplies:

    role / goal / rules   what the agent is and how it must behave
    build_tools()         the tools (actions) it is allowed to use
    observe()             a compact view of the world (state it reasons over)
    is_done()             COMPLETION CRITERIA, evaluated by code, never by the LLM
    policy()              deterministic planner used when no LLM is available,
                          and as the safety net when the LLM misbehaves
    reflect()             optional self-check after each action

The loop enforces guardrails: a step budget, argument validation, protection
against repeated failing calls, and refusal to accept `finish` before the
completion criteria hold. Every decision and observation is written to a trace.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .blackboard import Blackboard
from .llm import Decision, LLMBrain
from .tools import Tool

ROOT = Path(__file__).resolve().parents[2]
import os
TRACE_DIR = (Path(os.getenv("ESG_OUT_DIR")) if os.getenv("ESG_OUT_DIR") else ROOT / "processed-data") / "agent_traces"


@dataclass
class AgentReport:
    agent_id: str
    name: str
    status: str                      # completed | stalled | max_steps | failed
    steps: int
    seconds: float
    decisions_by_source: Dict[str, int]
    summary: str
    findings: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    llm_mode: bool = False
    llm_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseAgent:
    agent_id = "agent"
    name = "Base agent"
    role = ""
    goal = ""
    rules: List[str] = []
    max_steps = 400

    def __init__(self, board: Optional[Blackboard] = None,
                 brain: Optional[LLMBrain] = None, verbose: bool = True,
                 tracker: Any = None, dynamic: bool = False):
        self.tracker = tracker          # RunTracker (timing/provenance) or None
        self.dynamic = dynamic          # fetch live data instead of bundled files
        self.board = board or Blackboard()
        self.brain = brain or LLMBrain(enabled=False)
        self.verbose = verbose
        self.memory: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.findings: List[Dict[str, Any]] = []
        self.artifacts: List[str] = []
        self._fail_counts: Dict[str, int] = {}
        self._blocked: set = set()
        self._ok_calls: set = set()
        self._step, self._last_ok = 0, True
        self._llm_used, self._llm_seconds, self._budget_noted = 0, 0.0, False
        # A local 7B model needs seconds-to-minutes per call, so the LLM is consulted only at
        # planning/failure/checkpoint moments and within a budget; routine loops use the policy.
        self.llm_budget = int(os.getenv("LLM_MAX_DECISIONS", "10"))
        self.llm_every = int(os.getenv("LLM_EVERY", "25"))
        self.llm_time_budget = float(os.getenv("LLM_AGENT_TIME_BUDGET", "300"))
        self.repeatable_tools: set = set()
        self.tools: Dict[str, Tool] = {t.name: t for t in self.build_tools()}

    # ---- to be provided by subclasses ---------------------------------
    def build_tools(self) -> List[Tool]:
        raise NotImplementedError

    def observe(self) -> Dict[str, Any]:
        raise NotImplementedError

    def is_done(self, state: Dict[str, Any]) -> Tuple[bool, str]:
        raise NotImplementedError

    def policy(self, state: Dict[str, Any]) -> Optional[Decision]:
        raise NotImplementedError

    def reflect(self, decision: Decision, result: Dict[str, Any]) -> None:
        """Optional hook: update memory / raise findings after an action."""

    def check_args(self, decision: Decision, state: Dict[str, Any]) -> Optional[str]:
        """Reject LLM choices that are out of scope. Subclasses add ordering rules."""
        code = decision.args.get("company_code")
        allowed = getattr(self, "codes", None) or getattr(self, "target_codes", None)
        if code and allowed and str(code).upper() not in [a.upper() for a in allowed]:
            return f"company '{code}' is not part of this run"
        return None

    def _consult(self) -> bool:
        if not self.brain.available:
            return False
        if self._llm_used >= self.llm_budget or self._llm_seconds >= self.llm_time_budget:
            if not self._budget_noted:
                self._budget_noted = True
                self.add_finding("llm_budget_reached", "low",
                                 f"LLM used {self._llm_used} decisions / {self._llm_seconds:.0f}s; "
                                 "remaining routine steps follow the agent's policy")
            return False
        codes = getattr(self, "codes", None) or getattr(self, "target_codes", None) or []
        small_run = len(codes) <= 3
        return small_run or self._step <= 2 or (not self._last_ok) or self._step % self.llm_every == 0

    def on_start(self) -> None: ...
    def on_finish(self) -> None: ...

    # ---- helpers for subclasses ---------------------------------------
    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.agent_id}] {msg}", flush=True)

    def add_finding(self, kind: str, severity: str, message: str, **data) -> None:
        self.findings.append({"kind": kind, "severity": severity,
                              "message": message, **data})

    def is_blocked(self, tool: str, args: Dict[str, Any]) -> bool:
        return self._key(tool, args) in self._blocked

    @staticmethod
    def _key(tool: str, args: Dict[str, Any]) -> str:
        return tool + json.dumps(args, sort_keys=True, default=str)

    def system_prompt(self) -> str:
        rules = "\n".join(f"- {r}" for r in self.rules)
        return (f"You are {self.name}, an autonomous agent in a multi-agent ESG "
                f"risk analysis system.\nROLE: {self.role}\nGOAL: {self.goal}\n"
                f"RULES:\n{rules}\n- You may only act by calling one of the "
                f"provided tools. Never invent data or numbers.\n- Call `finish` "
                f"only when the completion criteria in the state are all true.")

    # ---- the loop ------------------------------------------------------
    def _choose(self, state: Dict[str, Any]) -> Tuple[Optional[Decision], Optional[str]]:
        decision = None
        err = None
        if self._consult():
            t0 = time.perf_counter()
            decision = self.brain.decide(self.system_prompt(), state,
                                         self.history, list(self.tools.values()))
            self._llm_seconds += time.perf_counter() - t0
            self._llm_used += 1
            if decision is None:
                err = self.brain.last_error
            elif decision.tool == "finish":
                # Guardrail: completion is decided by code, not by the model.
                self.history.append({"note": "guardrail: model tried to finish early; "
                                             "criteria not met"})
                decision, err = None, "premature finish rejected"
            elif decision.tool not in self.tools:
                decision, err = None, f"unknown tool '{decision.tool}'"
            else:
                bad = self.tools[decision.tool].validate(decision.args)
                if bad:
                    decision, err = None, f"bad args for {decision.tool}: {bad}"
                elif self.is_blocked(decision.tool, decision.args):
                    decision, err = None, "model repeated a blocked call"
                elif (self._key(decision.tool, decision.args) in self._ok_calls
                      and decision.tool not in self.repeatable_tools):
                    decision, err = None, f"duplicate call rejected: {decision.tool}"
                else:
                    bad = self.check_args(decision, state)
                    if bad:
                        decision, err = None, bad
        if decision is None:
            decision = self.policy(state)
            if decision is not None and err:
                decision.source = "guardrail"
        return decision, err

    def _execute(self, decision: Decision) -> Dict[str, Any]:
        tool = self.tools[decision.tool]
        started = time.perf_counter()
        try:
            out = tool.fn(**decision.args)
            result = out if isinstance(out, dict) else {"result": out}
            result.setdefault("ok", True)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                      "trace": traceback.format_exc(limit=3)}
        result["_seconds"] = round(time.perf_counter() - started, 2)
        key = self._key(decision.tool, decision.args)
        if result.get("ok") is not False:
            self._ok_calls.add(key)
        if result.get("ok") is False:
            self._fail_counts[key] = self._fail_counts.get(key, 0) + 1
            if self._fail_counts[key] >= 2:
                self._blocked.add(key)
        return result

    @staticmethod
    def _brief(result: Dict[str, Any], limit: int = 600) -> str:
        slim = {k: v for k, v in result.items() if k != "trace"}
        text = json.dumps(slim, default=str)
        return text if len(text) <= limit else text[:limit] + "...[truncated]"

    def run(self, max_steps: Optional[int] = None) -> AgentReport:
        if self.tracker is None:
            return self._run(max_steps)
        with self.tracker.span(self.name, "agents", self.agent_id, kind="agent") as rec:
            report = self._run(max_steps)
            rec["status"] = "success" if report.status == "completed" else "failed"
            rec["notes"] = {"steps": report.steps, "llm": report.llm_mode,
                            "decisions": report.decisions_by_source}
            return report

    def _run(self, max_steps: Optional[int] = None) -> AgentReport:
        budget = max_steps or self.max_steps
        started = time.perf_counter()
        sources = {"llm": 0, "policy": 0, "guardrail": 0}
        llm_errors: List[str] = []
        status, summary, steps = "max_steps", "step budget exhausted", 0
        self.log(f"start | mode={'LLM+policy' if self.brain.available else 'policy'}")
        self.on_start()
        try:
            for steps in range(1, budget + 1):
                self._step = steps
                state = self.observe()
                done, why = self.is_done(state)
                if done:
                    status, summary = "completed", why
                    steps -= 1
                    break
                decision, err = self._choose(state)
                if err:
                    llm_errors.append(err)
                if decision is None:
                    status, summary = "stalled", "no admissible action available"
                    break
                if self.is_blocked(decision.tool, decision.args):
                    status, summary = "stalled", f"only blocked action left: {decision.tool}"
                    break
                sources[decision.source] = sources.get(decision.source, 0) + 1
                self.log(f"step {steps}: {decision.tool} {json.dumps(decision.args, default=str)[:120]}"
                         f" [{decision.source}]")
                result = self._execute(decision)
                self.history.append({
                    "step": steps, "tool": decision.tool, "args": decision.args,
                    "thought": decision.thought, "source": decision.source,
                    "observation": self._brief(result)})
                self._last_ok = result.get("ok") is not False
                if result.get("ok") is False:
                    self.log(f"  ! {result.get('error')}")
                try:
                    self.reflect(decision, result)
                except Exception as exc:  # reflection must never kill the run
                    self.add_finding("reflect_error", "low", str(exc))
        except Exception as exc:
            status, summary = "failed", f"{type(exc).__name__}: {exc}"
            self.log(f"FAILED: {summary}")
        finally:
            try:
                self.on_finish()
            except Exception as exc:
                self.add_finding("on_finish_error", "medium", str(exc))
        report = AgentReport(
            agent_id=self.agent_id, name=self.name, status=status, steps=steps,
            seconds=round(time.perf_counter() - started, 2),
            decisions_by_source=sources, summary=summary, findings=self.findings,
            artifacts=self.artifacts, llm_mode=self.brain.available,
            llm_errors=llm_errors[:10])
        self._write_trace(report)
        self.log(f"{status} in {report.seconds}s ({steps} steps): {summary}")
        return report

    def _write_trace(self, report: AgentReport) -> None:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        path = TRACE_DIR / f"{self.agent_id}_trace.json"
        payload = {"written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "report": report.to_dict(), "history": self.history}
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        self.artifacts.append(str(path.relative_to(ROOT)))
