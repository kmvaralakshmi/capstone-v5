"""Agent 6 - Master ESG Scoring Agent.

ROLE
    Combine trusted evidence into E, S, G pillar scores and a Master ESG score,
    with an explicit evidence-quality gate. Missing evidence is NEVER replaced
    by a neutral default.

SCORING (matches the report, weights renormalised over available components)
    E = 0.5 * BRSR_E + 0.3 * Environmental-context + 0.2 * Trust
    S = 0.6 * BRSR_S + 0.4 * News sentiment
    G = 0.5 * BRSR_G + 0.3 * Market performance + 0.2 * Trust
    Master = 0.33 E + 0.33 S + 0.34 G   (renormalised over available pillars)
    Internals are 0-10; every published pillar/master score is multiplied by 10 -> 0-100.
    Risk level: Low >= 70, Medium 50-70, High < 50.

    BRSR pillar score = mean of direction-aware metric scores (0-10) computed
    from transparent anchors (METRIC_ANCHORS). Only values that are present and
    not flagged for manual review are used.

QUALITY GATE
    Pass          >= 12 usable metrics and >= 3 in every pillar
    Partial       >= 6 usable metrics and >= 2 pillars with BRSR evidence
                  (score published, flagged as partial-evidence)
    Insufficient  otherwise -> NO master score is produced
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .common import (COMPANIES, PROCESSED, ROOT, OUTPUT_FILES, clean, is_num, np, pd, read_csv_safe)
from .framework import BaseAgent, Decision, Tool

# metric: (score-0 anchor, score-10 anchor, pillar)   lower-is-better handled by order
METRIC_ANCHORS: Dict[str, Tuple[float, float, str]] = {
    "Renewable Energy Percentage": (0, 60, "E"),
    "Water Recycled Percentage": (0, 50, "E"),
    "Waste Recycled Percentage": (0, 80, "E"),
    "Energy Intensity (kWh/employee)": (15000, 500, "E"),
    "Water Intensity (KL/employee)": (100, 5, "E"),
    "Female Employee Percentage": (0, 40, "S"),
    "Employee Turnover Rate": (40, 10, "S"),
    "Training Hours per Employee": (0, 40, "S"),
    "Safety Incidents per 1000 Employees": (5, 0, "S"),
    "Independent Directors Percentage": (20, 60, "G"),
    "Female Directors Percentage": (0, 33, "G"),
    "Board Meeting Frequency": (4, 8, "G"),
    "Audit Committee Size": (2, 4, "G"),
    "Ethics Policy Violations": (5, 0, "G"),
}
PILLAR_W = {"E": 0.33, "S": 0.33, "G": 0.34}


def anchor_score(metric: str, value: float) -> float:
    a0, a10, _ = METRIC_ANCHORS[metric]
    return float(np.clip(10 * (value - a0) / (a10 - a0), 0, 10))


def wmean(parts: List[Tuple[Optional[float], float, str]]) -> Tuple[Optional[float], List[str]]:
    avail = [(v, w, n) for v, w, n in parts if v is not None]
    if not avail:
        return None, []
    tw = sum(w for _, w, _ in avail)
    return sum(v * w for v, w, _ in avail) / tw, [n for _, _, n in avail]


class MasterScoringAgent(BaseAgent):
    agent_id = "agent6"
    name = "Agent 6 - Master ESG Scoring Agent"
    role = "Compute pillar and master ESG scores with an evidence-quality gate."
    goal = "Inputs verified, every company scored or explicitly withheld, outputs written."
    rules = ["Never substitute a default for missing evidence.",
             "Withhold the master score when evidence is insufficient.",
             "Use only metric values that are present and not flagged for review.",
             "Check upstream agents' warnings before scoring."]

    def __init__(self, *args, companies: Optional[List[str]] = None, **kw):
        self.codes = [c.upper() for c in companies] if companies else list(COMPANIES)
        super().__init__(*args, **kw)
        self.memory.update({"inputs": None, "results": {}, "metric_rows": [],
                            "finalized": False})

    def build_tools(self) -> List[Tool]:
        return [
            Tool("check_inputs", "Verify upstream files exist and read upstream warnings.", self.t_check),
            Tool("score_company", "Score ONE company (metrics, pillars, gate, contradictions).",
                 self.t_score, {"company_code": {"type": "string"}}, ["company_code"]),
            Tool("rank_and_finalize", "Rank scored companies and write all output files.", self.t_finalize),
        ]

    def t_check(self) -> Dict[str, Any]:
        d = {"brsr": read_csv_safe(PROCESSED / OUTPUT_FILES["brsr_metrics"]),
             "env": read_csv_safe(PROCESSED / "environmental_company_summary.csv"),
             "news": read_csv_safe(PROCESSED / "esg_news_company_summary.csv"),
             "stock": read_csv_safe(PROCESSED / OUTPUT_FILES["stock_correlation"]),
             "gw": read_csv_safe(PROCESSED / OUTPUT_FILES["greenwashing"])}
        missing = [k for k, v in d.items() if v.empty]
        self.memory["inputs"] = d
        warns = [{"from": m["from"], "topic": m["topic"]} for m in self.board.read()
                 if m["severity"] in ("warning", "high")]
        if missing:
            self.add_finding("missing_inputs", "high", f"empty/missing upstream inputs: {missing}")
        return {"ok": not d["brsr"].empty, "missing": missing, "upstream_warnings": warns,
                **({} if not d["brsr"].empty else {"error": "Agent 1 output is required"})}

    def _metric_values(self, code: str) -> Dict[str, Dict[str, Any]]:
        df = self.memory["inputs"]["brsr"]
        sub = df[df["Company_Code"] == code]
        vals = {}
        for _, r in sub.iterrows():
            if is_num(r["Metric_Value"]) and r.get("Needs_Manual_Review") is not True \
                    and str(r.get("Needs_Manual_Review")) != "True":
                vals[r["Metric_Name"]] = {"value": float(r["Metric_Value"]),
                                          "source": r.get("Final_Source"),
                                          "page": r.get("PDF_Page"),
                                          "confidence": r.get("Confidence_Score")}
        return vals

    def t_score(self, company_code: str) -> Dict[str, Any]:
        code = company_code.upper()
        v = self._metric_values(code)
        emp = v.get("Total Employees", {}).get("value")
        derived = {}
        if emp and emp >= 50:
            if "Total Energy Consumption" in v:
                derived["Energy Intensity (kWh/employee)"] = v["Total Energy Consumption"]["value"] * 1e6 / emp
            if "Total Water Consumption" in v:
                derived["Water Intensity (KL/employee)"] = v["Total Water Consumption"]["value"] / emp
            if "Health and Safety Incidents" in v:
                derived["Safety Incidents per 1000 Employees"] = v["Health and Safety Incidents"]["value"] * 1000 / emp
        mscores: Dict[str, Dict[str, Any]] = {}
        for m, (_, _, pillar) in METRIC_ANCHORS.items():
            raw = derived.get(m, v.get(m, {}).get("value"))
            if raw is None:
                continue
            mscores[m] = {"pillar": pillar, "value": raw, "score": anchor_score(m, raw),
                          "source": v.get(m, {}).get("source", "derived"),
                          "page": v.get(m, {}).get("page")}
        for m, s in mscores.items():
            self.memory["metric_rows"].append({"Company_Code": code, "Metric": m, "Pillar": s["pillar"],
                                               "Value": s["value"], "Score_0_10": round(s["score"], 2),
                                               "Score_0_100": round(s["score"] * 10, 1),
                                               "Source": s["source"], "Page": s["page"]})
        pillar_brsr = {p: (float(np.mean([s["score"] for s in mscores.values() if s["pillar"] == p]))
                           if sum(s["pillar"] == p for s in mscores.values()) >= 2 else None)
                       for p in "ESG"}
        counts = {p: sum(s["pillar"] == p for s in mscores.values()) for p in "ESG"}
        inp = self.memory["inputs"]

        def row(df, key="Company_Code"):
            r = df[df[key] == code]
            return r.iloc[0] if not r.empty else None

        env_r, news_r, stock_r, gw_r = row(inp["env"]), row(inp["news"]), row(inp["stock"]), row(inp["gw"])
        env = float(env_r["Environmental_Context_Score"]) if env_r is not None and is_num(env_r["Environmental_Context_Score"]) else None
        news = None
        if news_r is not None:
            if is_num(news_r.get("News_Sentiment_Score_10")) and news_r["Articles"] >= 3:
                news = float(news_r["News_Sentiment_Score_10"])
            elif is_num(news_r.get("General_Sentiment_Shrunk")) and news_r.get("General_Articles", 0) >= 5:
                news = float(5 + 5 * news_r["General_Sentiment_Shrunk"])
        stock = float(stock_r["Stock_Performance_Score"]) if stock_r is not None and is_num(stock_r["Stock_Performance_Score"]) else None
        trust = float(gw_r["Trust_Score"]) if gw_r is not None and is_num(gw_r["Trust_Score"]) else None
        E, ce = wmean([(pillar_brsr["E"], .5, "BRSR"), (env, .3, "AirQualityContext"), (trust, .2, "Trust")])
        S, cs = wmean([(pillar_brsr["S"], .6, "BRSR"), (news, .4, "NewsSentiment")])
        G, cg = wmean([(pillar_brsr["G"], .5, "BRSR"), (stock, .3, "MarketPerformance"), (trust, .2, "Trust")])
        # a pillar that has no BRSR evidence is not scored at all
        for name in "ESG":
            if pillar_brsr[name] is None:
                if name == "E": E = None
                if name == "S": S = None
                if name == "G": G = None
        usable = len(v)
        sub_all = self.memory["inputs"]["brsr"]
        sub_all = sub_all[sub_all["Company_Code"] == code]
        review_flags = int((sub_all["Needs_Manual_Review"].astype(str) == "True").sum()) if not sub_all.empty else 0
        pillars_with = sum(pillar_brsr[p] is not None for p in "ESG")
        if usable >= 12 and all(counts[p] >= 3 for p in "ESG"):
            gate = "Pass"
        elif usable >= 6 and pillars_with >= 2:
            gate = "Partial"
        else:
            gate = "Insufficient"
        master = None
        if gate != "Insufficient":
            parts = [(x, PILLAR_W[p]) for x, p in ((E, "E"), (S, "S"), (G, "G")) if x is not None]
            master = sum(x * w for x, w in parts) / sum(w for _, w in parts)
        contra = []
        if G is not None and news_r is not None and is_num(news_r.get("Shrunk_Sentiment")) \
                and G > 7 and news_r["Shrunk_Sentiment"] < -0.2:
            contra.append("High governance score but clearly negative ESG news")
        if E is not None and env is not None and E > 7.5 and env < 5:
            contra.append("High environmental score but poor local air-quality context (context only)")
        risk = "Insufficient evidence" if master is None else "Low" if master >= 7 else "Medium" if master >= 5 else "High"   # thresholds on the 0-10 internal scale (= 70 / 50 on 0-100)
        res = {"E": E, "S": S, "G": G, "master": master, "gate": gate, "usable": usable,
               "brsr_E": pillar_brsr["E"], "counts": counts, "review_flags": review_flags, "components": {"E": ce, "S": cs, "G": cg},
               "contradictions": contra, "risk": risk, "env": env, "news": news,
               "stock": stock, "trust": trust}
        self.memory["results"][code] = res
        return {"ok": True, "company": code, "gate": gate, "master": clean(master), "usable_metrics": usable}

    def t_finalize(self) -> Dict[str, Any]:
        rows, quality, cross = [], [], []
        x100 = lambda v: None if v is None else round(float(v) * 10, 2)  # noqa: E731
        scored = {c: r["master"] for c, r in self.memory["results"].items() if r["master"] is not None}
        # a rank is meaningless inside a 1-company run; require a cohort of at least 5 scored companies
        ranks = pd.Series(scored).rank(ascending=False, method="min") if len(scored) >= 5 else pd.Series(dtype=float)
        cov = {c: "".join(p for p, v in zip("ESG", (r["E"], r["S"], r["G"])) if v is not None)
               for c, r in self.memory["results"].items()}
        grp = pd.DataFrame({"cov": pd.Series(cov), "m": pd.Series(scored)}).dropna()
        within = grp.groupby("cov")["m"].rank(ascending=False, method="min")
        gsize = grp.groupby("cov")["m"].transform("count")
        within = within.where(gsize >= 3)           # a group of 1-2 companies has no meaningful ranking
        for c, r in self.memory["results"].items():
            rows.append({"Company_Code": c, "Pillars_Scored": cov[c] or None,
                         "Rank_Within_Same_Coverage": clean(within.get(c)), "Company_Name": COMPANIES[c]["full_name"],
                         "Environmental_Score": x100(r["E"]), "Social_Score": x100(r["S"]),
                         "Governance_Score": x100(r["G"]), "Master_ESG_Score": x100(r["master"]),
                         "Score_Scale": "0-100", "BRSR_Environmental_Score": x100(r.get("brsr_E")),
                         "Env_Context_Score_10": clean(r["env"]), "News_Score_10": clean(r["news"]),
                         "Market_Score_10": clean(r["stock"]), "Trust_Score_10": clean(r["trust"]),
                         "Risk_Level": r["risk"], "Quality_Gate_Status": r["gate"],
                         "Usable_Metrics": r["usable"], "Metrics_E": r["counts"]["E"],
                         "Metrics_S": r["counts"]["S"], "Metrics_G": r["counts"]["G"],
                         "BRSR_Completeness_Pct": round(100 * r["usable"] / 18, 1),
                         "Evidence_Confidence": round(r["usable"] / 18, 3),
                         "Components_E": "+".join(r["components"]["E"]), "Components_S": "+".join(r["components"]["S"]),
                         "Components_G": "+".join(r["components"]["G"]),
                         "Rank": clean(ranks.get(c)), "Project_Rank": clean(ranks.get(c)),
                         "Quality_Score": round(10 * r["usable"] / 18, 2),
                         "Review_Flag_Count": r["review_flags"], "Contradiction_Count": len(r["contradictions"]),
                         "Contradiction_Summary": "; ".join(r["contradictions"]) or "none",
                         "Analysis_Date": datetime.now().strftime("%Y-%m-%d")})
            quality.append({"Company_Code": c, "Usable_Metrics": r["usable"], "Of": 18,
                            "Quality_Gate_Status": r["gate"]})
            cross.append({"Company_Code": c, "Rule_ID": "CONTRADICTION" if r["contradictions"] else "NONE",
                          "Description": "; ".join(r["contradictions"]) or "No contradiction detected"})
        df = pd.DataFrame(rows)
        p = PROCESSED / OUTPUT_FILES["master_scores"]
        df.to_csv(p, index=False)
        pd.DataFrame(self.memory["metric_rows"]).to_csv(PROCESSED / "esg_metric_scores.csv", index=False)
        pd.DataFrame(quality).to_csv(PROCESSED / OUTPUT_FILES["data_quality"], index=False)
        pd.DataFrame(cross).to_csv(PROCESSED / OUTPUT_FILES["cross_validation"], index=False)
        meta = {"run_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "inputs": {k: hashlib.sha256((PROCESSED / f).read_bytes()).hexdigest()[:16]
                           for k, f in {"brsr": OUTPUT_FILES["brsr_metrics"],
                                        "stock": OUTPUT_FILES["stock_correlation"]}.items()
                           if (PROCESSED / f).exists()},
                "gate_counts": df["Quality_Gate_Status"].value_counts().to_dict(),
                "method": "esg_agents.agent6_master_scoring v1"}
        (PROCESSED / OUTPUT_FILES["run_metadata"]).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.board.post(self.agent_id, "scoring_summary",
                        {"scored": len(scored), "withheld": len(df) - len(scored),
                         "gate_counts": meta["gate_counts"]})
        self.memory["finalized"] = True
        self.artifacts.append(str(p.relative_to(ROOT)))
        return {"ok": True, "scored": len(scored), "withheld": len(df) - len(scored),
                "gates": meta["gate_counts"]}

    def observe(self) -> Dict[str, Any]:
        todo = [c for c in self.codes if c not in self.memory["results"]]
        return {"inputs_checked": self.memory["inputs"] is not None, "to_score": todo[:10],
                "remaining": len(todo), "finalized": self.memory["finalized"]}

    def is_done(self, state) -> Tuple[bool, str]:
        if state["finalized"]:
            s = sum(r["master"] is not None for r in self.memory["results"].values())
            return True, f"{s}/{len(self.codes)} companies scored; rest withheld for insufficient evidence"
        return False, ""

    def policy(self, state) -> Optional[Decision]:
        if not state["inputs_checked"]:
            return Decision("check_inputs", {}, "verify inputs and upstream warnings")
        if state["to_score"]:
            return Decision("score_company", {"company_code": state["to_score"][0]}, "next company")
        return Decision("rank_and_finalize", {}, "all companies processed")
