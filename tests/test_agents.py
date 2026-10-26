"""Tests for the agent framework and key agent logic (no network, no API key).

Run:  python -m unittest discover -s tests -v
"""
import json
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esg_agents.framework import BaseAgent, Blackboard, Decision, LLMBrain, Tool  # noqa: E402
from esg_agents.agent2_environmental import EnvironmentalAgent, cpcb_sub_index  # noqa: E402
from esg_agents.agent1_brsr_extraction import _plausible  # noqa: E402
from esg_agents.agent6_master_scoring import anchor_score  # noqa: E402


class FakeClient:
    """Scripted stand-in for the OpenAI client: yields queued tool calls."""
    def __init__(self, script):
        self.script = list(script)
        self.responses = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        name, args = self.script.pop(0) if self.script else ("finish", {"summary": "x"})
        call = types.SimpleNamespace(type="function_call", name=name, arguments=json.dumps(args))
        return types.SimpleNamespace(output=[call])


class Counter(BaseAgent):
    """Tiny agent: must call `tick` three times, then `finalize`."""
    agent_id, name = "counter", "Counter"
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.memory.update(n=0, fin=False)
        self.repeatable_tools = {"tick"}      # an increment is meant to be called repeatedly
    def build_tools(self):
        return [Tool("tick", "increment", lambda: self.memory.update(n=self.memory["n"] + 1) or {"n": self.memory["n"]}),
                Tool("finalize", "finish", lambda: self.memory.update(fin=True) or {}),
                Tool("boom", "always fails", lambda: 1 / 0)]
    def observe(self): return {"n": self.memory["n"], "fin": self.memory["fin"]}
    def is_done(self, s): return (s["n"] >= 3 and s["fin"], "ok")
    def policy(self, s):
        return Decision("tick") if s["n"] < 3 else Decision("finalize")


class FrameworkTests(unittest.TestCase):
    def test_policy_mode_completes(self):
        r = Counter(verbose=False).run()
        self.assertEqual(r.status, "completed"); self.assertEqual(r.decisions_by_source["policy"], 4)

    def test_llm_mode_uses_model_choices(self):
        brain = LLMBrain(client=FakeClient([("tick", {})] * 3 + [("finalize", {})]))
        r = Counter(brain=brain, verbose=False).run()
        self.assertEqual(r.status, "completed"); self.assertEqual(r.decisions_by_source["llm"], 4)

    def test_guardrails_override_bad_llm_choices(self):
        script = [("finish", {"summary": "done"}),        # premature finish
                  ("nonexistent", {}),                    # unknown tool
                  ("tick", {"bogus": 1}),                 # bad args
                  ("tick", {}), ("tick", {}), ("tick", {}), ("finalize", {})]
        r = Counter(brain=LLMBrain(client=FakeClient(script)), verbose=False).run()
        self.assertEqual(r.status, "completed")
        self.assertGreaterEqual(r.decisions_by_source["guardrail"], 3)

    def test_failing_tool_is_blocked_not_looped(self):
        class Bad(Counter):
            def policy(self, s): return Decision("boom")
        r = Bad(verbose=False).run(max_steps=20)
        self.assertEqual(r.status, "stalled")

    def test_blackboard_roundtrip(self):
        b = Blackboard(); b.post("a", "topic", {"x": 1}, severity="warning")
        self.assertEqual(b.latest("topic"), {"x": 1})


class DomainTests(unittest.TestCase):
    def test_cpcb_pm25(self):
        self.assertAlmostEqual(cpcb_sub_index("PM2.5", 30), 50, places=0)
        self.assertAlmostEqual(cpcb_sub_index("PM2.5", 60), 100, places=0)

    def test_plausibility_catches_known_bad_values(self):
        self.assertFalse(_plausible("Total Employees", 1)[0])
        self.assertFalse(_plausible("Board Size", 0)[0])
        self.assertTrue(_plausible("Total Employees", 607979)[0])
        # 1 GWh for a 600k-person company fails the intensity cross-check
        self.assertFalse(_plausible("Total Energy Consumption", 1.0, 607979)[0])
        self.assertTrue(_plausible("Total Energy Consumption", 427.0, 607979)[0])

    def test_anchor_scoring_directions(self):
        self.assertEqual(anchor_score("Renewable Energy Percentage", 60), 10)
        self.assertEqual(anchor_score("Ethics Policy Violations", 0), 10)
        self.assertEqual(anchor_score("Ethics Policy Violations", 5), 0)

    def test_agent2_never_invents_aqi(self):
        a = EnvironmentalAgent(verbose=False, companies=["TCS"])
        a.t_load()
        self.assertEqual(a.t_resolve("Atlantis")["method"], "unresolved")
        self.assertIsNone(a.t_resolve("Atlantis")["aqi"])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------- v5.2 tests --
class V52Tests(unittest.TestCase):
    def test_grounding_check(self):
        from esg_agents.agent7_explainability import verify_grounding
        ev = {"scores": {"id": "S1", "master": 78.4}, "metrics": [{"id": "M1", "value": 79.2}]}
        self.assertTrue(verify_grounding("Score 78.4 [S1], renewable 79.2% [M1].", ev)["ok"])
        bad = verify_grounding("Score 91.3 [S1] and [M9].", ev)
        self.assertFalse(bad["ok"]); self.assertIn(91.3, bad["unsupported_numbers"]); self.assertIn("M9", bad["unknown_citations"])

    def test_corr_report_requires_sample(self):
        from esg_agents.agent8_validation_research import corr_report
        self.assertIn("insufficient", corr_report([1, 2, 3], [3, 2, 1])["status"])
        r = corr_report(list(range(20)), [x * 2 + (x % 3) for x in range(20)])
        self.assertEqual(r["status"], "ok"); self.assertGreater(r["spearman_rho"], 0.9)

    def test_market_metrics(self):
        import pandas as pd
        from esg_agents.data_providers import market_metrics
        idx = pd.date_range("2022-01-01", periods=900, freq="B")
        m = market_metrics(pd.Series(range(100, 1000), index=idx))
        self.assertGreater(m["Return_1Y_Pct"], 0); self.assertIn("Max_Drawdown_Pct", m)

    def test_chat_style_llm_path(self):
        """Ollama / OpenAI-compatible tool calling via chat.completions."""
        call = types.SimpleNamespace(function=types.SimpleNamespace(name="tick", arguments="{}"))
        msg = types.SimpleNamespace(tool_calls=[call], content="")
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
            create=lambda **kw: types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)]))))
        brain = LLMBrain(client=client); brain.api_style = "chat"
        d = brain.decide("sys", {}, [], [Tool("tick", "x", lambda: 1)])
        self.assertEqual((d.tool, d.source), ("tick", "llm"))

    def test_gdelt_loader_with_mock(self):
        import esg_agents.agent3_news_sentiment as a3
        from esg_agents.data_providers import DataResult
        arts = [{"title": "Infosys wins sustainability award for renewable energy", "url": "http://x", "seendate": "20260901T120000Z", "domain": "x.com"}]
        a3.fetch_news = lambda *a, **k: DataResult("success", "GDELT", arts)
        a3.time.sleep = lambda s: None
        ag = a3.NewsSentimentAgent(verbose=False, companies=["INFY"], dynamic=True)
        out = ag.t_load(); self.assertEqual(out["source"], "GDELT")
        ag.t_attribute(True); sc = ag.t_score()
        self.assertEqual(sc["esg_relevant"], 1)

    def test_telemetry_stage_vs_sum(self):
        from esg_agents.telemetry import RunTracker
        t = RunTracker("t")
        with t.stage("acq"):
            with t.span("a", "agents", kind="agent"): pass
        self.assertIn("acq", t.summary()["stages"])

    def test_agent0_registers_unknown_symbol(self):
        import tempfile, esg_agents.agent0_filing_discovery as a0
        a0.DYN_FILE = Path(tempfile.mkdtemp()) / "dyn.json"
        ag = a0.FilingDiscoveryAgent(verbose=False, companies=["ZZTESTCO"])
        out = ag.t_resolve("ZZTESTCO")
        self.assertTrue(out["ok"]); self.assertEqual(out["company_code"], "ZZTESTCO")
        from utils.config import COMPANIES; COMPANIES.pop("ZZTESTCO", None)


class V522Tests(unittest.TestCase):
    def test_llm_healthcheck_failure_disables_llm(self):
        def boom(**kw): raise RuntimeError("Error code: 404 - model 'x' not found")
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=boom)))
        b = LLMBrain(client=client); b.api_style = "chat"; b.provider = "ollama"; b.model = "x"
        ok, msg = b.healthcheck()
        self.assertFalse(ok); self.assertFalse(b.available); self.assertIn("ollama pull", msg)

    def test_llm_circuit_breaker_after_three_failures(self):
        def boom(**kw): raise RuntimeError("down")
        client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=boom)))
        b = LLMBrain(client=client); b.api_style = "chat"
        for _ in range(3):
            self.assertIsNone(b.decide("s", {}, [], [Tool("t", "d", lambda: 1)]))
        self.assertFalse(b.available); self.assertIn("consecutive", b.disabled_reason)

    def test_gdelt_circuit_opens_and_skips(self):
        import requests, esg_agents.data_providers as dp
        calls = {"n": 0}
        def fake_get(*a, **k):
            calls["n"] += 1; raise requests.exceptions.ConnectTimeout("timeout")
        orig = requests.get; requests.get = fake_get
        try:
            dp.reset_gdelt_circuit()
            r1 = dp.fetch_news("CoA_zz1"); r2 = dp.fetch_news("CoB_zz2"); r3 = dp.fetch_news("CoC_zz3")
            self.assertFalse(r3.ok); self.assertIn("circuit open", r3.error)
            self.assertLessEqual(calls["n"], 6)           # third company made NO network call
        finally:
            requests.get = orig; dp.reset_gdelt_circuit()


class V523Tests(unittest.TestCase):
    def test_llm_budget_then_policy(self):
        class Many(BaseAgent):
            agent_id, name = "many", "Many"
            codes = ["A", "B", "C", "D", "E"]
            def __init__(s, **k):
                super().__init__(**k); s.memory["n"] = 0; s.llm_budget = 2
            def build_tools(s): return [Tool("tick", "t", lambda: s.memory.update(n=s.memory["n"] + 1) or {"n": s.memory["n"]},
                                             {}, []), Tool("finalize", "f", lambda: s.memory.update(f=True) or {})]
            def observe(s): return {"n": s.memory["n"], "f": s.memory.get("f", False)}
            def is_done(s, st): return (st["f"], "ok")
            def policy(s, st): return Decision("tick") if st["n"] < 6 else Decision("finalize")
            repeatable = None
        a = Many(brain=LLMBrain(client=FakeClient([("tick", {})] * 50)), verbose=False)
        a.repeatable_tools = {"tick"}
        r = a.run()
        self.assertEqual(r.status, "completed"); self.assertLessEqual(r.decisions_by_source["llm"], 2)
        self.assertGreater(r.decisions_by_source["policy"], 0)

    def test_out_of_scope_company_rejected(self):
        a = EnvironmentalAgent(verbose=False, companies=["TCS"])
        out = a.t_score("HCLTECH")
        self.assertFalse(out["ok"])
        d = Decision("score_company", {"company_code": "HCLTECH"})
        self.assertIn("not part of this run", a.check_args(d, {}))

    def test_agent1_rejects_audit_before_taxonomy(self):
        from esg_agents.agent1_brsr_extraction import BRSRExtractionAgent
        a = BRSRExtractionAgent(verbose=False, companies=["TCS"])
        st = a.observe()
        self.assertIn("needs extract_company", a.check_args(Decision("audit_company", {"company_code": "TCS"}), st))
