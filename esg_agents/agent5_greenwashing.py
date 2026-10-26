"""Agent 5 - Disclosure-Consistency (Greenwashing Signal) Agent.

ROLE
    Cross-check what companies disclose (Agent 1) against independent evidence
    (Agents 2-4) and report *potential inconsistency signals*.

POSITIONING
    The agent does not accuse companies of greenwashing; one year of data
    cannot prove intent. It produces graded, evidence-linked signals and an
    explicit "Insufficient evidence" level when it lacks the data to judge.

SIGNALS (each carries its evidence)
    S1 source_disagreement   XBRL and PDF values for the same metric disagree
    S2 implausible_claims    values quarantined by Agent 1 as implausible
    S3 intensity_outlier     energy/water intensity far from the cohort (|z|>2)
    S4 claim_vs_news         strong governance/environment claims while ESG news
                             is clearly negative (needs >=2 articles)
    S5 disclosure_gap        large share of mandatory metrics undisclosed
                             (transparency concern, weighted low)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .common import (COMPANIES, PROCESSED, BASE_PROCESSED, ROOT, OUTPUT_FILES, clean, is_num, np, pd, read_csv_safe)
from .framework import BaseAgent, Decision, Tool

WEIGHTS = {"source_disagreement": 2.0, "implausible_claims": 1.5, "intensity_outlier": 2.0,
           "claim_vs_news": 3.0, "disclosure_gap": 1.0, "claim_contradicted": 3.0, "claim_offset": 1.0}


class GreenwashingAgent(BaseAgent):
    agent_id = "agent5"
    name = "Agent 5 - Disclosure-Consistency Agent"
    role = "Detect graded, evidence-linked inconsistency signals across sources."
    goal = "Every company has signals evaluated (or an explicit insufficient-evidence verdict) and outputs written."
    rules = ["Signals, not accusations.",
             "Return 'Insufficient evidence' when fewer than 6 metrics are usable.",
             "Each signal must cite its evidence."]

    def __init__(self, *args, companies: Optional[List[str]] = None, **kw):
        self.codes = [c.upper() for c in companies] if companies else list(COMPANIES)
        super().__init__(*args, **kw)
        self.memory.update({"data": None, "cohort": None, "results": {}, "finalized": False})

    def build_tools(self) -> List[Tool]:
        return [
            Tool("load_inputs", "Load Agent 1-4 outputs and read upstream warnings.", self.t_load),
            Tool("build_cohort_baseline", "Compute cohort intensity baselines (mean/std).", self.t_baseline),
            Tool("evaluate_company", "Evaluate all signals for ONE company.", self.t_eval,
                 {"company_code": {"type": "string"}}, ["company_code"]),
            Tool("finalize", "Write greenwashing_detection.csv and signal detail.", self.t_finalize),
        ]

    def t_load(self) -> Dict[str, Any]:
        d = {"brsr": read_csv_safe(PROCESSED / OUTPUT_FILES["brsr_metrics"]),
             "news": read_csv_safe(PROCESSED / "esg_news_company_summary.csv"),
             "trends": read_csv_safe(PROCESSED / "brsr_trends.csv"),
             "env": read_csv_safe(PROCESSED / "environmental_company_summary.csv")}
        self.memory["data"] = d
        warns = [m for m in self.board.read() if m["severity"] in ("warning", "high")]
        return {"ok": True, "brsr_rows": len(d["brsr"]), "news_rows": len(d["news"]),
                "upstream_warnings": len(warns)}

    def _val(self, df, code, metric):
        r = df[(df["Company_Code"] == code) & (df["Metric_Name"] == metric)]
        if r.empty or not is_num(r.iloc[0]["Metric_Value"]):
            return None
        return float(r.iloc[0]["Metric_Value"])

    def _intensities(self, code, df=None):
        df = self.memory["data"]["brsr"] if df is None else df
        emp = self._val(df, code, "Total Employees")
        if not emp or emp < 50:
            return {}
        out = {}
        e = self._val(df, code, "Total Energy Consumption")
        w = self._val(df, code, "Total Water Consumption")
        if e: out["energy_kwh_per_emp"] = e * 1e6 / emp
        if w: out["water_kl_per_emp"] = w / emp
        return out

    def t_baseline(self) -> Dict[str, Any]:
        df = self.memory["data"]["brsr"]
        if len(self.codes) < 8:       # single-company run: use the batch cohort as the baseline
            batch = read_csv_safe(BASE_PROCESSED / OUTPUT_FILES["brsr_metrics"])
            if not batch.empty:
                df = batch
        rows = [self._intensities(c, df) for c in df["Company_Code"].unique()] if not df.empty else []
        base = {}
        for key in ("energy_kwh_per_emp", "water_kl_per_emp"):
            vals = np.array([r[key] for r in rows if key in r], dtype=float)
            if len(vals) >= 5:
                lv = np.log(vals)
                base[key] = (float(lv.mean()), float(lv.std() or 1.0), len(vals))
        self.memory["cohort"] = base
        return {"ok": True, "baselines": {k: {"n": v[2]} for k, v in base.items()}}

    def t_eval(self, company_code: str) -> Dict[str, Any]:
        code = company_code.upper()
        df = self.memory["data"]["brsr"]
        sub = df[df["Company_Code"] == code] if not df.empty else df
        usable = int((sub["Metric_Value"].notna() & (sub["Needs_Manual_Review"] != True)).sum()) if not sub.empty else 0
        signals: List[Dict[str, Any]] = []
        if not sub.empty:
            dis = sub[sub.get("Audit_Status", pd.Series(dtype=str)) == "sources_disagree"] if "Audit_Status" in sub else sub.iloc[0:0]
            if len(dis):
                signals.append({"signal": "source_disagreement", "strength": min(1.0, len(dis) / 4),
                                "evidence": "; ".join(dis["Metric_Name"] + ": " + dis["Audit_Note"].fillna("")).strip()[:400]})
            q = sub[sub.get("Validation_Status", pd.Series(dtype=str)) == "QUARANTINED"] if "Validation_Status" in sub else sub.iloc[0:0]
            if len(q):
                signals.append({"signal": "implausible_claims", "strength": min(1.0, len(q) / 4),
                                "evidence": "quarantined: " + ", ".join(q["Metric_Name"])})
            gap = float(sub["Metric_Value"].isna().mean())
            if gap > 0.5:
                signals.append({"signal": "disclosure_gap", "strength": gap,
                                "evidence": f"{gap:.0%} of the 18 mandatory metrics have no trusted value"})
        for key, (mu, sd, n) in (self.memory["cohort"] or {}).items():
            v = self._intensities(code).get(key)
            if v:
                z = (np.log(v) - mu) / sd
                if abs(z) > 2:
                    signals.append({"signal": "intensity_outlier", "strength": min(1.0, abs(z) / 4),
                                    "evidence": f"{key}={v:.3g}, log-z={z:.2f} vs {n} companies"})
        news = self.memory["data"]["news"]
        nr = news[news["Company_Code"] == code] if not news.empty else news
        if not nr.empty and nr.iloc[0].get("Articles", 0) >= 2 and is_num(nr.iloc[0].get("Negative_Share")) \
                and nr.iloc[0]["Negative_Share"] >= 0.5:
            eth = self._val(df, code, "Ethics Policy Violations") if not df.empty else None
            if eth == 0 or eth is None:
                signals.append({"signal": "claim_vs_news", "strength": float(nr.iloc[0]["Negative_Share"]),
                                "evidence": f"{int(nr.iloc[0]['Articles'])} ESG articles, {nr.iloc[0]['Negative_Share']:.0%} negative, "
                                            f"while disclosed ethics violations = {eth}"})
        claims = self._investigate_claims(code, sub, nr, signals)
        self.memory.setdefault("claims", []).extend(claims)
        if usable < 6:
            level, risk = "Insufficient evidence", None
        else:
            risk = float(min(10.0, sum(WEIGHTS[s["signal"]] * s["strength"] for s in signals)))
            level = "Low" if risk < 2 else "Medium" if risk < 5 else "High"
        self.memory["results"][code] = {"usable": usable, "signals": signals, "risk": risk, "level": level}
        return {"ok": True, "company": code, "level": level, "risk": clean(risk), "signals": len(signals)}

    def _investigate_claims(self, code, sub, nr, signals) -> List[Dict[str, Any]]:
        """Cross-source check of the company's *improvement claims* (spec C8):
        BRSR value + prior-year trend + related-metric consistency + external news."""
        tr = self.memory["data"]["trends"]
        tr = tr[tr["Company_Code"] == code] if not tr.empty else tr
        if tr.empty:
            return []
        trend = {r["Metric"]: r for _, r in tr.iterrows()}
        env_events = neg_share = 0
        if not nr.empty:
            ee, ns = nr.iloc[0].get("Environmental_Events"), nr.iloc[0].get("Negative_Share")
            env_events = int(ee) if is_num(ee) else 0
            neg_share = float(ns) if is_num(ns) else 0.0
        disagree = set()
        if not sub.empty and "Audit_Status" in sub:
            disagree = set(sub[sub["Audit_Status"] == "sources_disagree"]["Metric_Name"])
        claims = []
        for metric, r in trend.items():
            if r["Direction"] != "improving":
                continue                                    # only improvement claims can be 'washed'
            evidence, verdict = [f"BRSR: {metric} {r['Change']:+.1f}{r['Change_Unit']} vs prior year"], "Supported"
            if metric in disagree:
                verdict = "Potential inconsistency"
                evidence.append("XBRL and PDF disagree on this metric")
            if metric == "Renewable Energy Percentage" and "Total Energy Consumption" in trend \
                    and trend["Total Energy Consumption"]["Direction"] == "worsening" \
                    and trend["Total Energy Consumption"]["Change"] > 10:
                verdict = "Partly offset" if verdict == "Supported" else verdict
                evidence.append(f"total energy rose {trend['Total Energy Consumption']['Change']:+.1f}% - "
                                "renewable share improved while absolute use grew")
            if env_events >= 2 and neg_share >= 0.4 and metric in (
                    "Renewable Energy Percentage", "Total Water Consumption", "Total Waste Generated",
                    "Waste Recycled Percentage", "Total Energy Consumption"):
                verdict = "Potential inconsistency"
                evidence.append(f"external news: {env_events} environmental articles, {neg_share:.0%} negative")
            elif env_events == 0:
                evidence.append("external news: no environmental coverage found (no contradiction, no corroboration)")
            claims.append({"Company_Code": code, "Claim": f"Improved {metric}", "Verdict": verdict,
                           "Evidence": " | ".join(evidence), "Metric": metric})
        for c in claims:
            if c["Verdict"] == "Potential inconsistency":
                signals.append({"signal": "claim_contradicted", "strength": 0.5, "evidence": c["Claim"] + ": " + c["Evidence"]})
            elif c["Verdict"] == "Partly offset":
                signals.append({"signal": "claim_offset", "strength": 0.5, "evidence": c["Claim"] + ": " + c["Evidence"]})
        return claims

    def t_finalize(self) -> Dict[str, Any]:
        rows, detail = [], []
        for c, r in self.memory["results"].items():
            rows.append({"Company_Code": c, "Company_Name": COMPANIES[c]["full_name"],
                         "Usable_Metrics": r["usable"], "Signal_Count": len(r["signals"]),
                         "Inconsistency_Risk_Score": r["risk"],
                         "Trust_Score": None if r["risk"] is None else round(10 - r["risk"], 2),
                         "Risk_Level": r["level"],
                         "Signals": " | ".join(s["signal"] for s in r["signals"])})
            for s in r["signals"]:
                detail.append({"Company_Code": c, **s})
        p = PROCESSED / OUTPUT_FILES["greenwashing"]
        pd.DataFrame(rows).to_csv(p, index=False)
        pd.DataFrame(detail).to_csv(PROCESSED / "greenwashing_signals_detail.csv", index=False)
        pd.DataFrame(self.memory.get("claims", [])).to_csv(PROCESSED / "greenwashing_claims.csv", index=False)
        insuff = [r["Company_Code"] for r in rows if r["Risk_Level"] == "Insufficient evidence"]
        self.board.post(self.agent_id, "consistency_summary",
                        {"insufficient_evidence": insuff,
                         "high_risk": [r["Company_Code"] for r in rows if r["Risk_Level"] == "High"]})
        self.memory["finalized"] = True
        self.artifacts.append(str(p.relative_to(ROOT)))
        return {"ok": True, "companies": len(rows), "insufficient_evidence": len(insuff)}

    def observe(self) -> Dict[str, Any]:
        todo = [c for c in self.codes if c not in self.memory["results"]]
        return {"loaded": self.memory["data"] is not None,
                "baseline": self.memory["cohort"] is not None,
                "to_evaluate": todo[:10], "remaining": len(todo),
                "finalized": self.memory["finalized"]}

    def is_done(self, state) -> Tuple[bool, str]:
        return (True, f"{len(self.memory['results'])} companies evaluated") if state["finalized"] else (False, "")

    def policy(self, state) -> Optional[Decision]:
        if not state["loaded"]:
            return Decision("load_inputs", {}, "read upstream outputs")
        if not state["baseline"]:
            return Decision("build_cohort_baseline", {}, "need cohort baselines for outlier tests")
        if state["to_evaluate"]:
            return Decision("evaluate_company", {"company_code": state["to_evaluate"][0]}, "next company")
        return Decision("finalize", {}, "all evaluated")
