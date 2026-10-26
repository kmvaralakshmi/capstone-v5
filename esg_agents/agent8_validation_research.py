"""Agent 8 - Validation & Research Agent (spec C4-C7, C9, C10).

ROLE
    Test whether the system's results hold up against external references and
    explore how ESG performance relates to market, news and air-quality data.

ANALYSES
    nse_top200   our ESG vs the NSE Top-200 ESG benchmark (YOU provide the file)
    nse_cfa      our BRSR disclosure rates vs the NSE-CFA Institute study (YOU provide)
    esg_stock    master score vs FY/1Y return, volatility, drawdown
    esg_news     master score vs news sentiment / negative events
    esg_air      environmental score vs local AQI (context only)

HONESTY RULES
    * Every statistic is reported with n; under 10 paired companies it is marked
      "insufficient sample" instead of presenting a correlation.
    * Results are associations, never causal claims.
    * If a benchmark file is absent the analysis says so and shows how to
      supply it - nothing is fabricated.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .common import COMPANIES, PROCESSED, ROOT, OUTPUT_FILES, clean, is_num, pd, read_csv_safe
from .framework import BaseAgent, Decision, Tool

BENCH = ROOT / "data" / "benchmark"
MIN_N = 10
ANALYSES = ["nse_top200", "nse_cfa", "esg_stock", "esg_news", "esg_air"]


def corr_report(x, y) -> Dict[str, Any]:
    from scipy import stats
    d = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    n = len(d)
    out: Dict[str, Any] = {"n": n}
    if n < MIN_N or d["x"].nunique() < 3 or d["y"].nunique() < 3:
        out["status"] = f"insufficient sample (n={n}, need >= {MIN_N})"
        return out
    pr, pp = stats.pearsonr(d["x"], d["y"])
    sr, sp = stats.spearmanr(d["x"], d["y"])
    out.update({"status": "ok", "pearson_r": round(float(pr), 3), "pearson_p": round(float(pp), 4),
                "spearman_rho": round(float(sr), 3), "spearman_p": round(float(sp), 4),
                "reading": ("weak" if abs(sr) < 0.3 else "moderate" if abs(sr) < 0.6 else "strong")
                           + (" positive" if sr > 0 else " negative") + " association (not causation)"})
    return out


class ValidationResearchAgent(BaseAgent):
    agent_id = "agent8"
    name = "Agent 8 - Validation & Research Agent"
    role = "Validate results against external benchmarks and analyse ESG vs market, news and air quality."
    goal = "All five analyses attempted and written, each stating its sample size and limitations."
    rules = ["Report n for every statistic.", "Never claim causation.",
             "If a benchmark file is missing, say how to supply it instead of inventing numbers."]

    def __init__(self, *args, companies: Optional[List[str]] = None, **kw):
        super().__init__(*args, **kw)           # validation always uses every scored company in the run folder
        self.memory.update({"data": None, "done": {}, "results": {}, "finalized": False})

    def build_tools(self) -> List[Tool]:
        tools = [Tool("load_results", "Load scores, market, news and air-quality outputs.", self.t_load)]
        for name in ANALYSES:
            tools.append(Tool(f"run_{name}", f"Run the {name} analysis.", (lambda n=name: self.t_run(n))))
        tools.append(Tool("finalize", "Write validation_summary.json.", self.t_finalize))
        return tools

    def t_load(self) -> Dict[str, Any]:
        d = {k: read_csv_safe(PROCESSED / f) for k, f in {
            "scores": OUTPUT_FILES["master_scores"], "market": OUTPUT_FILES["stock_correlation"],
            "news": "esg_news_company_summary.csv", "env": OUTPUT_FILES["environmental_risk"],
            "brsr": OUTPUT_FILES["brsr_metrics"]}.items()}
        self.memory["data"] = d
        scored = int(d["scores"]["Master_ESG_Score"].notna().sum()) if not d["scores"].empty else 0
        return {"ok": scored > 0, "scored_companies": scored,
                **({} if scored else {"error": "no scored companies: run the pipeline first"})}

    def _scored(self) -> pd.DataFrame:
        s = self.memory["data"]["scores"]
        return s[s["Master_ESG_Score"].notna()].copy()

    def t_run(self, name: str) -> Dict[str, Any]:
        res = getattr(self, f"_a_{name}")()
        self.memory["results"][name] = res
        self.memory["done"][name] = True
        return {"ok": True, "analysis": name, "status": res.get("status")}

    # ------------------------------------------------------------ analyses
    def _a_nse_top200(self) -> Dict[str, Any]:
        f = BENCH / "nse_top200_esg.csv"
        if not f.exists():
            return {"status": "awaiting benchmark file", "how": "copy data/benchmark/nse_top200_esg_template.csv to "
                    "nse_top200_esg.csv, fill NSE_ESG_Score, and re-run: python run_agents.py --agents agent8"}
        b = pd.read_csv(f)
        ours = self._scored()[["Company_Code", "Company_Name", "Master_ESG_Score", "Environmental_Score",
                               "Social_Score", "Governance_Score"]].copy()
        key = None
        if "Company_Code" in b and b["Company_Code"].notna().any():
            key, b["_k"], ours["_k"] = "code", b["Company_Code"].astype(str).str.upper().str.strip(), ours["Company_Code"]
        elif "Company_Name" in b:
            norm = lambda s: s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)  # noqa: E731
            key, b["_k"], ours["_k"] = "name", norm(b["Company_Name"]), norm(ours["Company_Name"])
        else:
            return {"status": "benchmark file needs Company_Code or Company_Name"}
        if "NSE_ESG_Score" not in b:
            return {"status": "benchmark file needs an NSE_ESG_Score column"}
        for c in ("NSE_ESG_Score", "NSE_E_Score", "NSE_S_Score", "NSE_G_Score"):
            if c in b:
                b[c] = pd.to_numeric(b[c], errors="coerce")
                if b[c].max() <= 10:
                    b[c] = b[c] * 10               # bring to 0-100
        m = ours.merge(b.drop_duplicates("_k"), on="_k", how="inner", suffixes=("", "_nse"))
        m = m.dropna(subset=["NSE_ESG_Score"])
        if m.empty:
            return {"status": "no overlapping companies between our results and the benchmark", "matched_by": key}
        m["Score_Difference"] = m["Master_ESG_Score"] - m["NSE_ESG_Score"]
        m["Our_Rank"] = m["Master_ESG_Score"].rank(ascending=False, method="min")
        m["NSE_Rank"] = m["NSE_ESG_Score"].rank(ascending=False, method="min")
        m["Rank_Difference"] = m["Our_Rank"] - m["NSE_Rank"]
        m.drop(columns=[c for c in ("_k", "Company_Code_nse", "Company_Name_nse") if c in m]).to_csv(
            PROCESSED / "validation_nse_top200.csv", index=False)
        res = {"status": "ok", "matched_by": key, "overlap": int(len(m)),
               "mean_absolute_difference": round(float(m["Score_Difference"].abs().mean()), 2),
               "mean_signed_difference": round(float(m["Score_Difference"].mean()), 2),
               "mean_abs_rank_difference": round(float(m["Rank_Difference"].abs().mean()), 2),
               "overall": corr_report(m["Master_ESG_Score"], m["NSE_ESG_Score"])}
        for ours_c, nse_c, tag in (("Environmental_Score", "NSE_E_Score", "E"), ("Social_Score", "NSE_S_Score", "S"),
                                   ("Governance_Score", "NSE_G_Score", "G")):
            if nse_c in m:
                res[f"pillar_{tag}"] = corr_report(m[ours_c], m[nse_c])
        return res

    def _a_nse_cfa(self) -> Dict[str, Any]:
        brsr = self.memory["data"]["brsr"]
        if brsr.empty:
            return {"status": "no BRSR output"}
        trusted = brsr[brsr["Metric_Value"].notna() & (brsr["Needs_Manual_Review"].astype(str) != "True")]
        n_co = brsr["Company_Code"].nunique()
        ours = (trusted.groupby("Metric_Name")["Company_Code"].nunique() / max(n_co, 1) * 100).round(1)
        table = pd.DataFrame({"Metric": ours.index, "Our_Pct_Companies_With_Trusted_Value": ours.values})
        f = BENCH / "nse_cfa_reference.csv"
        if not f.exists():
            table.to_csv(PROCESSED / "validation_nse_cfa.csv", index=False)
            return {"status": "awaiting reference file", "our_disclosure_rates_written": True, "companies": int(n_co),
                    "how": "copy data/benchmark/nse_cfa_reference_template.csv to nse_cfa_reference.csv and fill "
                           "Reference_Pct_Companies_Reporting"}
        ref = pd.read_csv(f)
        m = table.merge(ref, on="Metric", how="left")
        m["Difference_pp"] = m["Our_Pct_Companies_With_Trusted_Value"] - pd.to_numeric(
            m["Reference_Pct_Companies_Reporting"], errors="coerce")
        m.to_csv(PROCESSED / "validation_nse_cfa.csv", index=False)
        d = m.dropna(subset=["Difference_pp"])
        return {"status": "ok" if len(d) else "reference file has no filled values", "metrics_compared": int(len(d)),
                "mean_abs_difference_pp": None if d.empty else round(float(d["Difference_pp"].abs().mean()), 1),
                "note": "our rate = trusted extracted values, so it is a lower bound on true disclosure"}

    def _pair(self, ours_col: str, other: pd.DataFrame, other_col: str) -> Tuple[pd.DataFrame, str]:
        s = self._scored()
        m = s[["Company_Code", ours_col]].merge(other[["Company_Code", other_col]], on="Company_Code")
        return m, ""

    def _a_esg_stock(self) -> Dict[str, Any]:
        mk = self.memory["data"]["market"]
        if mk.empty:
            return {"status": "no market output"}
        out, rows = {"status": "ok"}, []
        for col in ("FY_Return_Pct", "Return_1Y_Pct", "Volatility_Pct", "Max_Drawdown_Pct"):
            if col in mk:
                m, _ = self._pair("Master_ESG_Score", mk, col)
                r = corr_report(m["Master_ESG_Score"], m[col])
                out[col] = r
                rows.append({"Analysis": f"ESG vs {col}", **r})
        pd.DataFrame(rows).to_csv(PROCESSED / "validation_esg_stock.csv", index=False)
        n = max((v.get("n", 0) for v in out.values() if isinstance(v, dict)), default=0)
        out["status"] = "ok" if n >= MIN_N else f"insufficient sample (n={n}); run in --dynamic mode so more companies have prices"
        return out

    def _a_esg_news(self) -> Dict[str, Any]:
        nw = self.memory["data"]["news"]
        if nw.empty:
            return {"status": "no news output"}
        out, rows = {}, []
        for col in ("Shrunk_Sentiment", "Negative_Share", "Negative_Events"):
            if col in nw:
                m, _ = self._pair("Master_ESG_Score", nw[nw["Articles"] > 0], col)
                r = corr_report(m["Master_ESG_Score"], m[col])
                out[col] = r
                rows.append({"Analysis": f"ESG vs {col}", **r})
        pd.DataFrame(rows).to_csv(PROCESSED / "validation_esg_news.csv", index=False)
        n = max((v.get("n", 0) for v in out.values()), default=0)
        out["status"] = "ok" if n >= MIN_N else f"insufficient sample (n={n}); run in --dynamic mode (GDELT) for more articles"
        return out

    def _a_esg_air(self) -> Dict[str, Any]:
        env = self.memory["data"]["env"]
        if env.empty:
            return {"status": "no air-quality output"}
        e = env.dropna(subset=["Avg_AQI"])
        hq = e[e["Location_Type"] == "Headquarters"].groupby("Company_Code")["Avg_AQI"].mean().reset_index()
        sc = self._scored()
        # use the BRSR-only environmental score: the published E pillar already contains an
        # air-quality component, so correlating it with AQI would be partly circular
        col = "BRSR_Environmental_Score" if "BRSR_Environmental_Score" in sc and sc["BRSR_Environmental_Score"].notna().any() else "Environmental_Score"
        m = sc[["Company_Code", col]].merge(hq, on="Company_Code").rename(columns={col: "Environmental_Score"})
        r = corr_report(m["Environmental_Score"], m["Avg_AQI"])
        pd.DataFrame([{"Analysis": "BRSR-only environmental score vs HQ AQI", **r}]).to_csv(PROCESSED / "validation_esg_air.csv", index=False)
        return {"status": r["status"] if r["status"] != "ok" else "ok", "environmental_vs_hq_aqi": r,
                "note": "air quality is local context, not evidence about the company"}

    def t_finalize(self) -> Dict[str, Any]:
        summary = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "scored_companies": int(len(self._scored())), "analyses": clean(self.memory["results"]),
                   "caveat": "All relationships are statistical associations, not causal claims."}
        (PROCESSED / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        self.memory["finalized"] = True
        return {"ok": True, "analyses": {k: v.get("status") for k, v in self.memory["results"].items()}}

    def observe(self) -> Dict[str, Any]:
        return {"loaded": self.memory["data"] is not None,
                "pending": [a for a in ANALYSES if a not in self.memory["done"]], "finalized": self.memory["finalized"]}

    def is_done(self, s) -> Tuple[bool, str]:
        return (True, "validation_summary.json written") if s["finalized"] else (False, "")

    def policy(self, s) -> Optional[Decision]:
        if not s["loaded"]:
            return Decision("load_results", {}, "load outputs")
        if s["pending"]:
            return Decision(f"run_{s['pending'][0]}", {}, "next analysis")
        return Decision("finalize", {}, "write summary")
