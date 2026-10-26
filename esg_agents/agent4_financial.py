"""Agent 4 - Market Performance Agent.

ROLE
    Compute price-based market performance from REAL price history.

DYNAMIC vs STATIC
    dynamic=True  : Yahoo Finance (via yfinance) for every company, cached, with the
                    bundled NIFTY/SENSEX files as fallback (spec A3/A4/A9).
    dynamic=False : bundled files only (benchmark / regression-test mode).

INTEGRITY
    Nothing is simulated. A company with no price history gets no market score
    (it is not defaulted to a neutral value), and the gap is posted to the
    blackboard so Agent 6 and the report can state it.

METRICS: 1M/3M/6M/1Y/3Y returns, FY2024-25 return, volatility, max drawdown.
SCORE (0-10) = 0.6 * FY-return percentile + 0.4 * (1 - volatility percentile),
percentiles within the companies that have data. With a single company there is
no cohort, so no score is produced (metrics are still reported).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from .common import (COMPANIES, PROCESSED, BASE_PROCESSED, read_csv_safe, ROOT, OUTPUT_FILES, clean, np, pd)
from .data_providers import fetch_market
from .framework import BaseAgent, Decision, Tool


class MarketPerformanceAgent(BaseAgent):
    agent_id = "agent4"
    name = "Agent 4 - Market Performance Agent"
    role = "Compute price-based market performance for each company from real data."
    goal = "Prices fetched (or fallback used) for every company, metrics computed, scores written."
    rules = ["Never simulate or default a value.",
             "Record which source (live / cache / static fallback) each company's data came from.",
             "Report companies without price history."]

    def __init__(self, *args, companies: Optional[List[str]] = None, **kw):
        self.codes = [c.upper() for c in companies] if companies else list(COMPANIES)
        super().__init__(*args, **kw)
        self.memory.update({"fetched": False, "metrics": {}, "sources": {}, "errors": {},
                            "finalized": False})

    def build_tools(self) -> List[Tool]:
        return [
            Tool("fetch_prices", "Fetch price history for ALL target companies in parallel "
                 "(live Yahoo Finance when dynamic, with static fallback).", self.t_fetch),
            Tool("retry_failed", "Retry companies whose price fetch failed, using the static fallback.", self.t_retry),
            Tool("finalize", "Score by percentile and write the market CSV.", self.t_finalize),
        ]

    def _alternates(self, code: str) -> List[str]:
        """data/ticker_overrides.json lists alternative Yahoo symbols (renamed/relisted companies)."""
        try:
            import json
            f = ROOT / "data" / "ticker_overrides.json"
            return json.loads(f.read_text(encoding="utf-8")).get(code, []) if f.exists() else []
        except Exception:
            return []

    def _one(self, code: str, dynamic: bool):
        return code, fetch_market(COMPANIES[code]["ticker"], dynamic=dynamic, tracker=self.tracker,
                                  alternates=self._alternates(code))

    def _apply(self, results):
        for code, res in results:
            if res.ok and res.payload:
                self.memory["metrics"][code] = res.payload
                self.memory["sources"][code] = res.source
                self.memory["errors"].pop(code, None)
            else:
                self.memory["errors"][code] = res.error or "no data"

    def t_fetch(self) -> Dict[str, Any]:
        with ThreadPoolExecutor(max_workers=8 if self.dynamic else 1) as ex:
            results = list(ex.map(lambda c: self._one(c, self.dynamic), self.codes))
        self._apply(results)
        self.memory["fetched"] = True
        by_src: Dict[str, int] = {}
        for src in self.memory["sources"].values():
            by_src[src] = by_src.get(src, 0) + 1
        return {"ok": True, "with_data": len(self.memory["metrics"]),
                "without_data": len(self.memory["errors"]), "by_source": by_src}

    def t_retry(self) -> Dict[str, Any]:
        todo = list(self.memory["errors"])
        self._apply([self._one(c, False) for c in todo])
        self.memory["retried"] = True
        return {"ok": True, "retried": len(todo), "still_missing": len(self.memory["errors"])}

    def t_finalize(self) -> Dict[str, Any]:
        m = pd.DataFrame.from_dict(self.memory["metrics"], orient="index")
        pool = m
        if len(m) < 8 and PROCESSED != BASE_PROCESSED:        # single-company run: rank against the batch cohort
            batch = read_csv_safe(BASE_PROCESSED / OUTPUT_FILES["stock_correlation"])
            if not batch.empty and "FY_Return_Pct" in batch:
                b = batch.dropna(subset=["FY_Return_Pct", "Volatility_Pct"]).set_index("Company_Code")[["FY_Return_Pct", "Volatility_Pct"]]
                pool = pd.concat([m[["FY_Return_Pct", "Volatility_Pct"]], b[~b.index.isin(m.index)]])
        have_score = len(pool) >= 3 and "FY_Return_Pct" in pool
        if have_score:
            ok = pool.dropna(subset=["FY_Return_Pct", "Volatility_Pct"])
            ret_pct, vol_pct = ok["FY_Return_Pct"].rank(pct=True), ok["Volatility_Pct"].rank(pct=True)
            sc = (10 * (0.6 * ret_pct + 0.4 * (1 - vol_pct))).round(2)
            m["Stock_Performance_Score"] = sc.reindex(m.index)
        rows = []
        for c in self.codes:
            base = {"Company_Code": c, "Company_Name": COMPANIES[c]["full_name"],
                    "Stock_Ticker": COMPANIES[c]["ticker"]}
            if c in m.index:
                r = m.loc[c]
                rows.append({**base,
                             **{k: (None if pd.isna(r.get(k)) else round(float(r[k]), 2))
                                for k in ("Return_1M_Pct", "Return_3M_Pct", "Return_6M_Pct", "Return_1Y_Pct",
                                          "Return_3Y_Pct", "FY_Return_Pct", "Volatility_Pct", "Max_Drawdown_Pct")
                                if k in m.columns},
                             "Stock_Performance_Score": (None if not have_score or pd.isna(r.get("Stock_Performance_Score"))
                                                         else r["Stock_Performance_Score"]),
                             "Data_Points": int(r["Data_Points"]), "Window_End": r.get("Window_End"),
                             "Data_Source": self.memory["sources"].get(c), "Coverage": "price_data"})
            else:
                rows.append({**base, "Stock_Performance_Score": None, "Coverage": "no_price_history",
                             "Error": self.memory["errors"].get(c)})
        path = PROCESSED / OUTPUT_FILES["stock_correlation"]
        pd.DataFrame(rows).to_csv(path, index=False)
        missing = [r["Company_Code"] for r in rows if r["Coverage"] != "price_data"]
        self.board.post(self.agent_id, "market_coverage",
                        {"companies_without_prices": missing, "covered": len(rows) - len(missing),
                         "dynamic": self.dynamic},
                        severity="warning" if missing else "info")
        if missing:
            self.add_finding("market_coverage_gap", "high" if len(missing) > len(rows) / 2 else "low",
                             f"{len(missing)}/{len(rows)} companies have no price history")
        if self.tracker:
            for c, src in self.memory["sources"].items():
                self.tracker.record_provenance(f"{c}:market", "price history", src, fallback=("static" in src))
        self.memory["finalized"] = True
        self.artifacts.append(str(path.relative_to(path.parents[1])))
        return {"ok": True, "covered": len(rows) - len(missing), "without_prices": len(missing)}

    def observe(self) -> Dict[str, Any]:
        return {"fetched": self.memory["fetched"], "failed": list(self.memory["errors"])[:10],
                "failed_count": len(self.memory["errors"]), "retried": self.memory.get("retried", False),
                "finalized": self.memory["finalized"], "dynamic": self.dynamic}

    def is_done(self, state) -> Tuple[bool, str]:
        return (True, f"{len(self.memory['metrics'])}/{len(self.codes)} companies have market data") \
            if state["finalized"] else (False, "")

    def policy(self, state) -> Optional[Decision]:
        if not state["fetched"]:
            return Decision("fetch_prices", {}, "fetch all price histories in parallel")
        if state["failed_count"] and state["dynamic"] and not state["retried"]:
            return Decision("retry_failed", {}, "fall back to bundled data for failed fetches")
        return Decision("finalize", {}, "score and write")
