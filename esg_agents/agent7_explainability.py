"""Agent 7 - Explanation Agent (evidence-grounded, LLM-written, self-verifying).

ROLE
    For each company, reason over ALL collected evidence and write a dynamic,
    company-specific explanation of the ESG result (spec C1).

PIPELINE PER COMPANY
    1. build_evidence   assemble an evidence package with citable IDs:
                        scores, BRSR metrics (value/source/page), year-over-year
                        trends, news findings, market performance, air-quality
                        context, consistency signals, limitations
    2. explain          the LLM writes the narrative using ONLY that package and
                        cites evidence IDs like [M3]; without an LLM a
                        deterministic writer produces the text from the same package
    3. verify           every number in the narrative must exist in the evidence
                        (tolerance 1%); every cited ID must exist. On failure the
                        agent asks the LLM to correct it once, then falls back to
                        the deterministic text and records the fallback.

The grounding check is what makes the explanation trustworthy: the model can
phrase and prioritise, but it cannot introduce facts.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .common import (COMPANIES, PROCESSED, ROOT, OUTPUT_FILES, clean, is_num, pd, read_csv_safe)
from .framework import BaseAgent, Decision, Tool

ACTIONS = {
    "Renewable Energy Percentage": "Raise the renewable share of energy via PPAs/on-site solar and disclose the target.",
    "Water Recycled Percentage": "Expand water recycling capacity and report recycled volumes.",
    "Waste Recycled Percentage": "Improve waste segregation and recycler tie-ups; report recovery rate.",
    "Energy Intensity (kWh/employee)": "Cut energy intensity through efficiency retrofits and workplace consolidation.",
    "Water Intensity (KL/employee)": "Reduce water use per employee through fixtures and reuse.",
    "Female Employee Percentage": "Strengthen gender-diversity hiring and retention programmes.",
    "Employee Turnover Rate": "Address attrition drivers: career paths, compensation, engagement.",
    "Training Hours per Employee": "Increase structured training hours and report them per employee.",
    "Safety Incidents per 1000 Employees": "Strengthen safety programmes and incident reporting.",
    "Independent Directors Percentage": "Increase board independence beyond the regulatory minimum.",
    "Female Directors Percentage": "Add women directors to improve board diversity.",
    "Board Meeting Frequency": "Review board meeting cadence for adequate oversight.",
    "Audit Committee Size": "Ensure the audit committee is sized for effective oversight.",
    "Ethics Policy Violations": "Investigate and remediate ethics violations; publish corrective actions.",
}

SYSTEM = (
    "You are an ESG analyst writing for investors and regulators. Write ONLY from the JSON evidence "
    "package. Rules: (1) every number you state must appear in the evidence; (2) cite the evidence IDs "
    "in square brackets, e.g. [M3], [T2], [N1], after each claim; (3) never invent causes, events, "
    "targets or numbers; (4) if evidence for something is missing, say it is not available; "
    "(5) describe market, news and air-quality information as context/association, never as causation; "
    "(6) structure: Overall assessment; Strengths; Areas for improvement; External evidence "
    "(news/market/air quality); Consistency check (greenwashing signals); Limitations. "
    "Keep it under 280 words. Scores are on a 0-100 scale.")

NUM = re.compile(r"(?<![\w\[])-?\d+(?:,\d{3})*(?:\.\d+)?")
CITE = re.compile(r"\[([A-Z]\d+)\]")


def _numbers(text: str) -> List[float]:
    out = []
    for m in NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def verify_grounding(narrative: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Return {ok, unsupported_numbers, unknown_citations}. Pure function (tested)."""
    allowed: Set[float] = set()
    for n in _numbers(json.dumps(evidence, default=str)):
        allowed.add(n)
        allowed.update({round(n), round(n, 1), round(n, 2), abs(n)})
    ids = set(re.findall(r'"id":\s*"([A-Z]\d+)"', json.dumps(evidence)))
    bad_nums = []
    for n in _numbers(CITE.sub("", narrative)):
        if 1900 <= n <= 2100 and float(n).is_integer():
            continue                       # years
        if n in (0, 1, 2, 3, 100):         # structural words ("top 3", "0-100")
            continue
        if not any(abs(n - a) <= max(0.01 * abs(a), 0.051) for a in allowed):
            bad_nums.append(n)
    bad_ids = [c for c in CITE.findall(narrative) if c not in ids]
    return {"ok": not bad_nums and not bad_ids, "unsupported_numbers": bad_nums[:8], "unknown_citations": bad_ids[:8]}


class ExplainabilityAgent(BaseAgent):
    agent_id = "agent7"
    name = "Agent 7 - Explanation Agent"
    role = "Write evidence-grounded, company-specific ESG explanations and verify them."
    goal = "Every company has a verified explanation with an evidence trail; outputs written."
    rules = ["Use only facts from the evidence package.",
             "State what is missing for withheld or partial-evidence companies.",
             "Re-verify after every LLM output; fall back to the deterministic writer if verification fails."]

    def __init__(self, *args, companies: Optional[List[str]] = None, **kw):
        self.codes = [c.upper() for c in companies] if companies else list(COMPANIES)
        super().__init__(*args, **kw)
        self.memory.update({"loaded": False, "evidence": {}, "results": {}, "finalized": False})

    def build_tools(self) -> List[Tool]:
        c = {"company_code": {"type": "string"}}
        return [
            Tool("load_scoring_outputs", "Load every upstream output the explanation may cite.", self.t_load),
            Tool("build_evidence", "Assemble the citable evidence package for ONE company.", self.t_evidence, c, ["company_code"]),
            Tool("explain_company", "Write and verify the explanation for ONE company.", self.t_explain, c, ["company_code"]),
            Tool("finalize", "Write multi_agent_explanations.csv and evidence JSON.", self.t_finalize),
        ]

    # ------------------------------------------------------------------ load
    def t_load(self) -> Dict[str, Any]:
        d = {k: read_csv_safe(PROCESSED / f) for k, f in {
            "scores": OUTPUT_FILES["master_scores"], "metrics": "esg_metric_scores.csv",
            "trends": "brsr_trends.csv", "news": "esg_news_company_summary.csv",
            "articles": OUTPUT_FILES["news_sentiment"], "market": OUTPUT_FILES["stock_correlation"],
            "env": OUTPUT_FILES["environmental_risk"], "gw": OUTPUT_FILES["greenwashing"],
            "claims": "greenwashing_claims.csv", "gwsig": "greenwashing_signals_detail.csv"}.items()}
        self.memory["data"], self.memory["loaded"] = d, True
        ok = not d["scores"].empty
        return {"ok": ok, "companies": len(d["scores"]), **({} if ok else {"error": "Run Agent 6 first"})}

    def _row(self, key: str, code: str):
        df = self.memory["data"][key]
        r = df[df["Company_Code"] == code] if not df.empty and "Company_Code" in df else df.iloc[0:0]
        return r

    # -------------------------------------------------------------- evidence
    def t_evidence(self, company_code: str) -> Dict[str, Any]:
        code = company_code.upper()
        sc = self._row("scores", code)
        if sc.empty:
            return {"ok": False, "error": f"{code} not in scoring output"}
        r = sc.iloc[0]
        ev: Dict[str, Any] = {"company": r["Company_Name"], "code": code, "scale": "0-100"}
        ev["scores"] = {"id": "S1", "master": clean(r["Master_ESG_Score"]), "environmental": clean(r["Environmental_Score"]),
                        "social": clean(r["Social_Score"]), "governance": clean(r["Governance_Score"]),
                        "risk_level": r["Risk_Level"], "evidence_quality": r["Quality_Gate_Status"],
                        "usable_metrics_of_18": int(r["Usable_Metrics"]), "rank": clean(r["Rank"])}
        m = self._row("metrics", code).sort_values("Score_0_10", ascending=False)
        ev["metrics"] = [{"id": f"M{i + 1}", "metric": x["Metric"], "value": round(float(x["Value"]), 3),
                          "score_0_100": clean(x.get("Score_0_100", x["Score_0_10"] * 10)),
                          "source": x["Source"], "page": clean(x["Page"])} for i, (_, x) in enumerate(m.iterrows())]
        t = self._row("trends", code)
        ev["trends"] = [{"id": f"T{i + 1}", "metric": x["Metric"], "change": float(x["Change"]), "unit": x["Change_Unit"],
                         "direction": x["Direction"]} for i, (_, x) in enumerate(t.iterrows())]
        n, a = self._row("news", code), self._row("articles", code)
        ev["news"] = {"id": "N1", "available": False}
        if not n.empty and is_num(n.iloc[0].get("Articles")) and n.iloc[0]["Articles"] > 0:
            nr = n.iloc[0]
            ev["news"] = {"id": "N1", "available": True, "esg_articles": int(nr["Articles"]),
                          "negative_share_pct": round(100 * float(nr.get("Negative_Share") or 0), 1),
                          "environmental_events": int(nr.get("Environmental_Events") or 0),
                          "social_events": int(nr.get("Social_Events") or 0),
                          "governance_events": int(nr.get("Governance_Events") or 0),
                          "reliability": clean(nr.get("Sentiment_Reliability")),
                          "headlines": [str(x)[:110] for x in (a["Article_Title"].head(3).tolist() if not a.empty and "Article_Title" in a else [])]}
        mk = self._row("market", code)
        ev["market"] = {"id": "K1", "available": False}
        if not mk.empty and mk.iloc[0].get("Coverage") == "price_data":
            x = mk.iloc[0]
            ev["market"] = {"id": "K1", "available": True,
                            **{k: clean(x.get(k)) for k in ("FY_Return_Pct", "Return_1Y_Pct", "Volatility_Pct", "Max_Drawdown_Pct") if is_num(x.get(k))},
                            "source": x.get("Data_Source")}
        e = self._row("env", code)
        ev["air_quality"] = {"id": "A1", "available": False}
        if not e.empty and e["Avg_AQI"].notna().any():
            hq = e.dropna(subset=["Avg_AQI"]).iloc[0]
            ev["air_quality"] = {"id": "A1", "available": True, "location": hq["City"], "avg_aqi": round(float(hq["Avg_AQI"]), 0),
                                 "category": hq["Risk_Level"], "match": hq["Match_Method"],
                                 "note": "context only; not proof of company performance"}
        g, cl = self._row("gw", code), self._row("claims", code)
        ev["consistency"] = {"id": "G1", "level": None, "signals": [], "claims": []}
        if not g.empty:
            ev["consistency"].update({"level": g.iloc[0]["Risk_Level"],
                                      "signals": [s for s in str(g.iloc[0].get("Signals") or "").split(" | ") if s]})
        ev["consistency"]["claims"] = [{"claim": x["Claim"], "verdict": x["Verdict"], "evidence": str(x["Evidence"])[:200]}
                                       for _, x in cl.iterrows()][:5]
        lim = []
        if r["Quality_Gate_Status"] != "Pass":
            lim.append(f"only {int(r['Usable_Metrics'])} of 18 BRSR metrics were trusted")
        for p, nm in (("E", "Metrics_E"), ("S", "Metrics_S"), ("G", "Metrics_G")):
            if r[nm] < 2:
                lim.append({"E": "environmental", "S": "social", "G": "governance"}[p] + " pillar has no reliable BRSR evidence")
        if not ev["market"]["available"]:
            lim.append("no market data")
        if not ev["news"]["available"]:
            lim.append("no ESG news found")
        ev["limitations"] = {"id": "L1", "items": lim}
        ev["withheld"] = not is_num(r["Master_ESG_Score"])
        ev["weak_metrics"] = [x["metric"] for x in ev["metrics"][::-1][:3]]
        self.memory["evidence"][code] = ev
        return {"ok": True, "company": code, "metrics": len(ev["metrics"]), "trends": len(ev["trends"]), "withheld": ev["withheld"]}

    # ------------------------------------------------------ deterministic text
    def _template(self, ev: Dict[str, Any]) -> str:
        s = ev["scores"]
        if ev["withheld"]:
            return (f"{ev['company']}: no master ESG score is published [S1]. Only {s['usable_metrics_of_18']} of 18 BRSR "
                    f"metrics could be trusted, which is below the evidence threshold. Limitations: "
                    f"{'; '.join(ev['limitations']['items']) or 'sparse disclosure'} [L1].")
        # rank is blank in single-company runs (a rank needs a cohort of >= 5 scored companies)
        rank_txt = f", rank {int(s['rank'])}" if s.get("rank") is not None else ""
        parts = [f"Overall assessment: {ev['company']} scores {s['master']:.1f}/100 ({s['risk_level']} risk{rank_txt}), "
                 f"with {s['evidence_quality']} evidence quality ({s['usable_metrics_of_18']}/18 metrics) [S1]."]
        top, low = ev["metrics"][:2], ev["metrics"][::-1][:2]
        if top:
            parts.append("Strengths: " + "; ".join(f"{m['metric']} = {m['value']:g} (score {m['score_0_100']:.0f}) [{m['id']}]" for m in top) + ".")
        if low:
            parts.append("Areas for improvement: " + "; ".join(f"{m['metric']} = {m['value']:g} (score {m['score_0_100']:.0f}) [{m['id']}]" for m in low) + ".")
        imp = [t for t in ev["trends"] if t["direction"] in ("improving", "worsening")][:3]
        if imp:
            parts.append("Year-over-year: " + "; ".join(f"{t['metric']} {t['direction']} ({t['change']:+.1f}{t['unit']}) [{t['id']}]" for t in imp) + ".")
        ext = []
        n, k, a = ev["news"], ev["market"], ev["air_quality"]
        if n["available"]:
            ext.append(f"{n['esg_articles']} ESG news articles, {n['negative_share_pct']:.0f}% negative [N1]")
        if k["available"] and "FY_Return_Pct" in k:
            ext.append(f"FY return {k['FY_Return_Pct']:.1f}% and volatility {k.get('Volatility_Pct', 0):.1f}% (association, not causation) [K1]")
        if a["available"]:
            ext.append(f"local air quality context: AQI {a['avg_aqi']:.0f} in {a['location']} [A1]")
        parts.append("External evidence: " + ("; ".join(ext) if ext else "not available") + ".")
        c = ev["consistency"]
        verdicts = "; ".join(f"{x['claim']}: {x['verdict']}" for x in c["claims"][:3])
        parts.append(f"Consistency check: {c['level'] or 'not assessed'}" + (f" ({verdicts})" if verdicts else "") + " [G1].")
        if ev["limitations"]["items"]:
            parts.append("Limitations: " + "; ".join(ev["limitations"]["items"]) + " [L1].")
        return " ".join(parts)

    # ----------------------------------------------------------------- explain
    def t_explain(self, company_code: str) -> Dict[str, Any]:
        code = company_code.upper()
        ev = self.memory["evidence"].get(code)
        if ev is None:
            return {"ok": False, "error": "build_evidence must run first"}
        source, text, check, attempts = "deterministic", self._template(ev), {"ok": True}, 0
        cap = int(os.getenv("LLM_EXPLAIN_MAX", "5" if len(self.codes) > 3 else "999"))
        narrated = sum(str(r["Narrative_Source"]).startswith("llm") or "grounding" in str(r["Narrative_Source"])
                       for r in self.memory["results"].values())
        if self.brain.available and not ev["withheld"] and narrated >= cap:
            source = f"deterministic (LLM narration limited to first {cap} companies; set LLM_EXPLAIN_MAX)"
        if self.brain.available and not ev["withheld"] and narrated < cap:
            prompt = "EVIDENCE PACKAGE:\n" + json.dumps(ev, indent=1, default=str)
            for attempts in (1, 2):
                out = self.brain.write(SYSTEM, prompt)
                if not out:
                    break
                check = verify_grounding(out, ev)
                if check["ok"]:
                    source, text = f"llm({self.brain.describe()})", out
                    break
                prompt += (f"\n\nYour previous answer failed verification: unsupported numbers "
                           f"{check['unsupported_numbers']}, unknown citations {check['unknown_citations']}. "
                           f"Rewrite using only numbers and IDs present in the evidence.")
            else:
                source = "deterministic (LLM failed grounding check twice)"
        recs = [ACTIONS[m] for m in ev["weak_metrics"] if m in ACTIONS][:3]
        self.memory["results"][code] = {
            "Company_Code": code, "Company_Name": ev["company"], "Master_ESG_Score": ev["scores"]["master"],
            "Risk_Level": ev["scores"]["risk_level"], "Quality_Gate_Status": ev["scores"]["evidence_quality"],
            "Score_Explanation": text, "Narrative_Source": source,
            "Grounding_Check": "pass" if check.get("ok") else "fail", "LLM_Attempts": attempts,
            "Strengths": " | ".join(f"{m['metric']}={m['value']:g}" for m in ev["metrics"][:3]),
            "Weaknesses": " | ".join(f"{m['metric']}={m['value']:g}" for m in ev["metrics"][::-1][:3]),
            "Recommendations": " | ".join(recs), "Limitations": " | ".join(ev["limitations"]["items"])}
        return {"ok": True, "company": code, "source": source, "grounding": self.memory["results"][code]["Grounding_Check"]}

    def t_finalize(self) -> Dict[str, Any]:
        rows = list(self.memory["results"].values())
        p = PROCESSED / OUTPUT_FILES["explanations"]
        pd.DataFrame(rows).to_csv(p, index=False)
        (PROCESSED / "esg_evidence_packages.json").write_text(json.dumps(self.memory["evidence"], indent=1, default=str), encoding="utf-8")
        self.memory["finalized"] = True
        self.artifacts.append(str(p.relative_to(p.parents[1])))
        return {"ok": True, "explanations": len(rows),
                "llm_written": sum(str(r["Narrative_Source"]).startswith("llm") for r in rows)}

    # --------------------------------------------------------------- reasoning
    def observe(self) -> Dict[str, Any]:
        ev, res = self.memory["evidence"], self.memory["results"]
        return {"loaded": self.memory["loaded"],
                "need_evidence": [c for c in self.codes if c not in ev][:10],
                "need_explanation": [c for c in self.codes if c in ev and c not in res][:10],
                "remaining": len([c for c in self.codes if c not in res]), "finalized": self.memory["finalized"]}

    def is_done(self, s) -> Tuple[bool, str]:
        if s["finalized"]:
            n = sum(str(r["Narrative_Source"]).startswith("llm") for r in self.memory["results"].values())
            return True, f"{len(self.memory['results'])} explanations ({n} LLM-written, verified)"
        return False, ""

    def policy(self, s) -> Optional[Decision]:
        if not s["loaded"]:
            return Decision("load_scoring_outputs", {}, "load all citable outputs")
        if s["need_explanation"]:
            return Decision("explain_company", {"company_code": s["need_explanation"][0]}, "write + verify")
        if s["need_evidence"]:
            return Decision("build_evidence", {"company_code": s["need_evidence"][0]}, "assemble evidence")
        return Decision("finalize", {}, "all explained")