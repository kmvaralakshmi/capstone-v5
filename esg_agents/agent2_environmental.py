"""Agent 2 - Environmental Context Agent (AQI evidence).

ROLE
    Provide an independent, location-based environmental signal for each
    company from CPCB air-quality data, so BRSR environmental claims can be
    compared with what the surrounding air actually looks like.

WHAT MAKES IT AN AGENT
    * It inspects the AQI sources first and discovers what is really there
      (the historical file covers only 5 cities; the real-time file has
      pollutant readings but NO AQI column).
    * It computes the missing AQI itself from raw pollutants with the CPCB
      sub-index method, instead of silently returning "no data".
    * For every company location it resolves the city through a ladder -
      exact name -> spelling alias -> metro-region proxy -> unresolved - and
      records the method and a confidence, so a proxy is never presented as
      a direct measurement. Unresolved locations are NOT given invented scores.
    * It reports coverage problems to the blackboard for Agents 5-7.

FORMULA (matches report section 4.2)
    ERS = 10 * (1 - AQI / 500), clipped to [0, 10]; higher = better air.
    Company score = weighted mean over locations (HQ weight 2, offices 1).
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from .common import (COMPANIES, PROCESSED, ROOT, OUTPUT_FILES, clean, is_num, np, pd)
from .framework import BaseAgent, Decision, Tool

# CPCB breakpoints: (conc_lo, conc_hi, aqi_lo, aqi_hi)
BREAKPOINTS: Dict[str, List[Tuple[float, float, int, int]]] = {
    "PM2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
              (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    "PM10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
             (251, 350, 201, 300), (351, 430, 301, 400), (431, 800, 401, 500)],
    "NO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
            (181, 280, 201, 300), (281, 400, 301, 400), (401, 800, 401, 500)],
    "SO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
            (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 3200, 401, 500)],
    "CO": [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200),
           (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 68, 401, 500)],
    "OZONE": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
              (169, 208, 201, 300), (209, 748, 301, 400), (749, 1500, 401, 500)],
    "NH3": [(0, 200, 0, 50), (201, 400, 51, 100), (401, 800, 101, 200),
            (801, 1200, 201, 300), (1201, 1800, 301, 400), (1801, 3600, 401, 500)],
}

# spelling variants and metro-region proxies. A proxy is a *different* city
# whose air shed the location shares; it always carries lower confidence.
ALIASES = {"bengaluru": "bangalore", "bangalore": "bangalore", "mysore": "mysuru",
           "new delhi": "delhi", "bombay": "mumbai", "madras": "chennai",
           "gurgaon": "gurugram", "calcutta": "kolkata"}
PROXIES = {"navi mumbai": "mumbai", "noida": "delhi", "gurugram": "delhi",
           "kochi": "eloor", "thane": "mumbai"}


def cpcb_sub_index(pollutant: str, conc: float) -> Optional[float]:
    for lo, hi, ilo, ihi in BREAKPOINTS.get(pollutant, []):
        if lo <= conc <= hi:
            return ilo + (conc - lo) * (ihi - ilo) / (hi - lo) if hi > lo else float(ilo)
    return None


def risk_level(aqi: float) -> str:
    for cap, name in [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
                      (300, "Poor"), (400, "Very Poor")]:
        if aqi <= cap:
            return name
    return "Severe"


class EnvironmentalAgent(BaseAgent):
    agent_id = "agent2"
    name = "Agent 2 - Environmental Context Agent"
    role = "Derive location-based air-quality risk for every company location."
    goal = ("Every location of every company is resolved to AQI evidence "
            "(direct, alias or proxy) or explicitly flagged unresolved, scored, "
            "and written out.")
    rules = [
        "Load and inspect the AQI sources before scoring anything.",
        "Never invent an AQI. If a city cannot be resolved, leave it unscored.",
        "Prefer direct city data, then aliases, then proxies; record the method.",
        "Score all companies, then finalize.",
    ]

    def __init__(self, *args, companies: Optional[List[str]] = None, **kw):
        self.codes = [c.upper() for c in companies] if companies else list(COMPANIES)
        super().__init__(*args, **kw)
        self.memory.update({"loaded": False, "hist": None, "rt": None,
                            "scored": {}, "rows": [], "finalized": False,
                            "city_cache": {}})

    # ------------------------------------------------------------------ tools
    def build_tools(self) -> List[Tool]:
        return [
            Tool("load_aqi_sources",
                 "Load historical city-day AQI (3-year window) and the real-time "
                 "pollutant file, computing CPCB AQI from raw pollutants.",
                 self.t_load),
            Tool("inspect_coverage",
                 "Report which configured office cities are resolvable and by "
                 "what method.", self.t_inspect),
            Tool("resolve_city", "Resolve ONE city to AQI evidence.",
                 self.t_resolve, {"city": {"type": "string"}}, ["city"]),
            Tool("score_company",
                 "Compute per-location ERS and the company-level environmental "
                 "score for ONE company.", self.t_score,
                 {"company_code": {"type": "string"}}, ["company_code"]),
            Tool("finalize", "Write the environmental risk CSV and publish "
                 "coverage to the blackboard.", self.t_finalize),
        ]

    # ----------------------------------------------------------------- loading
    def _realtime_path(self):
        base = ROOT / "raw-datasets"
        for d in sorted(base.glob("Real-Time Air Quality*")):
            f = d / "AQI.csv"
            if f.exists():
                return f
        return None

    def _hist_path(self):
        base = ROOT / "raw-datasets"
        for d in sorted(base.glob("Air Quality Data in India*")):
            f = d / "city_day.csv"
            if f.exists():
                return f
        return None

    def t_load(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        hp = self._hist_path()
        if hp is None:
            info["historical"] = "MISSING"
            hist = pd.DataFrame(columns=["city", "aqi", "n", "max", "min"])
        else:
            df = pd.read_csv(hp)
            dcol = "Datetime" if "Datetime" in df.columns else "Date"
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
            df = df.dropna(subset=["AQI"])
            window = df[df[dcol] >= df[dcol].max() - pd.Timedelta(days=3 * 365)]
            g = window.groupby("City")["AQI"]
            hist = pd.DataFrame({"aqi": g.mean(), "n": g.size(),
                                 "max": g.max(), "min": g.min()}).reset_index()
            hist["city"] = hist["City"].str.lower()
            info["historical"] = {"cities": sorted(hist["City"]),
                                  "window_end": str(window[dcol].max().date())}
        rp = self._realtime_path()
        if rp is None:
            info["realtime"] = "MISSING"
            rt = pd.DataFrame(columns=["city", "aqi", "n", "max", "min"])
        else:
            raw = pd.read_csv(rp)
            raw["pollutant_avg"] = pd.to_numeric(raw["pollutant_avg"], errors="coerce")
            raw = raw.dropna(subset=["pollutant_avg"])
            raw["sub"] = [cpcb_sub_index(p, v) for p, v in
                          zip(raw["pollutant_id"], raw["pollutant_avg"])]
            raw = raw.dropna(subset=["sub"])
            rows = []
            for (city, station), grp in raw.groupby(["city", "station"]):
                pols = set(grp["pollutant_id"])
                # CPCB rule: >=3 pollutants and at least one particulate
                if len(pols) >= 3 and ({"PM2.5", "PM10"} & pols):
                    rows.append({"city": city.lower(), "station": station,
                                 "aqi": grp["sub"].max()})
            st = pd.DataFrame(rows)
            if st.empty:
                rt = pd.DataFrame(columns=["city", "aqi", "n", "max", "min"])
            else:
                g = st.groupby("city")["aqi"]
                rt = pd.DataFrame({"aqi": g.mean(), "n": g.size(), "max": g.max(),
                                   "min": g.min()}).reset_index()
            info["realtime"] = {"cities_with_computed_aqi": int(len(rt)),
                                "note": "single-hour snapshot; lower confidence"}
        self.memory.update({"hist": hist, "rt": rt, "loaded": True})
        return {"ok": True, **info}

    # --------------------------------------------------------------- resolving
    def t_resolve(self, city: str) -> Dict[str, Any]:
        key = city.strip().lower()
        if key in self.memory["city_cache"]:
            return self.memory["city_cache"][key]
        hist, rt = self.memory["hist"], self.memory["rt"]

        def lookup(name: str):
            h = hist[hist["city"] == name]
            if not h.empty:
                r = h.iloc[0]
                return {"source": "CPCB city_day 3-yr", "aqi": float(r["aqi"]),
                        "max": float(r["max"]), "min": float(r["min"]),
                        "records": int(r["n"]), "base_conf": 0.95}
            r_ = rt[rt["city"] == name]
            if not r_.empty:
                r = r_.iloc[0]
                return {"source": "CPCB real-time snapshot (computed AQI)",
                        "aqi": float(r["aqi"]), "max": float(r["max"]),
                        "min": float(r["min"]), "records": int(r["n"]),
                        "base_conf": 0.6}
            return None

        steps = [("exact", key, 1.0), ("alias", ALIASES.get(key), 0.95),
                 ("proxy", PROXIES.get(key), 0.55)]
        result: Dict[str, Any] = {"ok": True, "city": city, "method": "unresolved",
                                  "matched_city": None, "aqi": None, "confidence": 0.0}
        best = None
        for method, name, factor in steps:
            if not name:
                continue
            found = lookup(name)
            if found:
                conf = round(found["base_conf"] * factor, 2)
                # keep the most trustworthy evidence, not merely the first hit
                # (a 3-year city_day series via alias beats a 1-hour snapshot)
                if best is None or conf > best["confidence"]:
                    best = {**found, "method": method,
                            "matched_city": name.title(), "confidence": conf}
        if best:
            result.update(best)
        self.memory["city_cache"][key] = result
        return result

    def _locations(self, code: str) -> List[Dict[str, str]]:
        info = COMPANIES[code]
        locs = [{"type": "Headquarters", **info["headquarters"]}]
        seen = {(info["headquarters"]["city"], info["headquarters"].get("area"))}
        for o in info.get("major_offices", []):
            k = (o["city"], o.get("area"))
            if k not in seen:
                seen.add(k)
                locs.append({"type": "Major Office", **o})
        return locs

    def t_inspect(self) -> Dict[str, Any]:
        cities = sorted({l["city"] for c in self.codes for l in self._locations(c)})
        table = {c: {k: v for k, v in self.t_resolve(c).items()
                     if k in ("method", "matched_city", "confidence")} for c in cities}
        unresolved = [c for c, v in table.items() if v["method"] == "unresolved"]
        return {"ok": True, "cities": table, "unresolved": unresolved}

    # ----------------------------------------------------------------- scoring
    def t_score(self, company_code: str) -> Dict[str, Any]:
        code = company_code.upper()
        if code not in self.codes:
            return {"ok": False, "error": f"{code} is not part of this run ({self.codes[:5]}...)"}
        today = date.today().isoformat()
        rows, wsum, wtot = [], 0.0, 0.0
        for loc in self._locations(code):
            ev = self.t_resolve(loc["city"])
            aqi = ev.get("aqi")
            ers = None if aqi is None else float(np.clip(10 * (1 - aqi / 500.0), 0, 10))
            rows.append({
                "Company_Code": code, "Company_Name": COMPANIES[code]["full_name"],
                "Location_Type": loc["type"], "City": loc["city"], "Area": loc.get("area"),
                "Avg_AQI": aqi, "Max_AQI": ev.get("max"), "Min_AQI": ev.get("min"),
                "Environmental_Risk_Score": ers,
                "Risk_Level": risk_level(aqi) if aqi is not None else None,
                "Data_Records": ev.get("records", 0), "Analysis_Date": today,
                "AQI_Source": ev.get("source"), "Match_Method": ev["method"],
                "Matched_City": ev.get("matched_city"),
                "Evidence_Confidence": ev.get("confidence", 0.0)})
            if ers is not None:
                w = 2.0 if loc["type"] == "Headquarters" else 1.0
                wsum += w * ers
                wtot += w
        self.memory["rows"].extend(rows)
        score = (wsum / wtot) if wtot else None
        self.memory["scored"][code] = {"score": score, "locations": len(rows),
                                       "resolved": sum(r["Environmental_Risk_Score"] is not None for r in rows)}
        return {"ok": True, "company": code, "score": clean(score),
                **self.memory["scored"][code]}

    def t_finalize(self) -> Dict[str, Any]:
        df = pd.DataFrame(self.memory["rows"])
        path = PROCESSED / OUTPUT_FILES["environmental_risk"]
        df.to_csv(path, index=False)
        summary = pd.DataFrame([{"Company_Code": c, "Environmental_Context_Score": v["score"],
                                 "Locations": v["locations"], "Resolved_Locations": v["resolved"]}
                                for c, v in self.memory["scored"].items()])
        summary.to_csv(PROCESSED / "environmental_company_summary.csv", index=False)
        no_evidence = [c for c, v in self.memory["scored"].items() if v["score"] is None]
        proxy_only = df[df["Match_Method"] == "proxy"]["Company_Code"].unique().tolist() if not df.empty else []
        self.board.post(self.agent_id, "environmental_coverage",
                        {"companies_without_aqi": no_evidence,
                         "companies_using_proxy_cities": proxy_only,
                         "unresolved_cities": sorted(df[df["Match_Method"] == "unresolved"]["City"].unique())})
        self.memory["finalized"] = True
        self.artifacts.append(str(path.relative_to(ROOT)))
        return {"ok": True, "rows": len(df), "companies_without_aqi": no_evidence}

    # --------------------------------------------------------------- reasoning
    def observe(self) -> Dict[str, Any]:
        pending = [c for c in self.codes if c not in self.memory["scored"]]
        return {"sources_loaded": self.memory["loaded"],
                "coverage_inspected": self.memory.get("inspected", False),
                "companies_pending": pending[:15], "pending_count": len(pending),
                "finalized": self.memory["finalized"]}

    def is_done(self, state) -> Tuple[bool, str]:
        if state["sources_loaded"] and state["pending_count"] == 0 and state["finalized"]:
            n = sum(v["score"] is not None for v in self.memory["scored"].values())
            return True, f"{n}/{len(self.codes)} companies have an environmental score"
        return False, ""

    def policy(self, state) -> Optional[Decision]:
        if not state["sources_loaded"]:
            return Decision("load_aqi_sources", {}, "sources must be inspected first")
        if not state["coverage_inspected"]:
            return Decision("inspect_coverage", {}, "check which cities resolve")
        if state["pending"] if "pending" in state else state["companies_pending"]:
            return Decision("score_company", {"company_code": state["companies_pending"][0]},
                            "score next company")
        if not state["finalized"]:
            return Decision("finalize", {}, "all companies scored")
        return None

    def reflect(self, decision: Decision, result: Dict[str, Any]) -> None:
        if decision.tool == "inspect_coverage" and result.get("ok"):
            self.memory["inspected"] = True
            if result["unresolved"]:
                self.add_finding("unresolved_cities", "medium",
                                 f"No AQI evidence for: {result['unresolved']}",
                                 cities=result["unresolved"])
        if decision.tool == "score_company" and result.get("score") is None:
            self.add_finding("no_environmental_evidence", "high",
                             f"{result.get('company')}: no location could be resolved",
                             company=result.get("company"))
