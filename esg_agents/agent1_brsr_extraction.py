"""Agent 1 - BRSR Extraction & Audit Agent.

ROLE
    Turn each company's BRSR filing (XBRL first, PDF fallback) into 18 trusted
    ESG metrics, and be the first line of defence against bad data.

WHAT MAKES IT AN AGENT (not a script)
    * It surveys the world first (which companies have an XBRL file / a PDF,
      which are already cached) and decides what to work on.
    * After extracting a company it AUDITS the numbers against domain
      plausibility rules (an employee count of 1, a board of size 0, energy of
      1.5 billion GWh are all caught here).
    * For every suspicious value it chooses a repair strategy -
      prefer the other source, rescale units, or quarantine - verifies the
      repaired value and records exactly what it did and why.
    * Values it cannot trust are *quarantined* (value removed, flagged for
      manual review) rather than silently passed to the scorer.
    * It tells downstream agents what it found via the blackboard.

DETERMINISTIC TOOLS UNDERNEATH
    utils.xbrl_extractor / utils.brsr_extractor (via Agent1BRSRExtractionAgent)
    do the low-level XBRL parsing and PDF text/table mining.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .common import (ROOT, COMPANIES, ESG_METRICS, PROCESSED, BASE_PROCESSED, BRSR_PDF_DIR, XBRL_RAW_DIR,
                     OUTPUT_FILES, clean, is_num, pd)
from .framework import BaseAgent, Decision, Tool

CACHE_DIR = BASE_PROCESSED / "agent1_cache"

# (low, high, hard_zero_invalid) - plausible envelopes for listed Indian
# technology / telecom companies. Used for audit, never for scoring.
PLAUSIBLE: Dict[str, Tuple[float, float, bool]] = {
    "Total Energy Consumption": (0.05, 60000.0, True),        # GWh
    "Renewable Energy Percentage": (0.0, 100.0, False),
    "Total Water Consumption": (100.0, 5e9, True),            # KL
    "Water Recycled Percentage": (0.0, 100.0, False),
    "Total Waste Generated": (0.1, 1e6, True),                # MT
    "Waste Recycled Percentage": (0.0, 100.0, False),
    "Total Employees": (50.0, 1.5e6, True),
    "Female Employee Percentage": (2.0, 80.0, True),
    "Employee Turnover Rate": (0.5, 100.0, True),
    "Training Hours per Employee": (3.0, 400.0, True),
    "Health and Safety Incidents": (0.0, 5000.0, False),
    "CSR Expenditure": (0.01, 5000.0, True),                  # crore INR
    "Board Size": (3.0, 25.0, True),
    "Independent Directors Percentage": (20.0, 100.0, True),
    "Female Directors Percentage": (1.0, 70.0, True),         # SEBI: >=1 woman
    "Board Meeting Frequency": (2.0, 40.0, True),
    "Audit Committee Size": (2.0, 12.0, True),
    "Ethics Policy Violations": (0.0, 2000.0, False),
}
PERCENT_METRICS = {m for m, (lo, hi, _) in PLAUSIBLE.items()
                   if "Percentage" in m or m == "Employee Turnover Rate"}

# unit rescaling candidates tried by the 'rescale' strategy, in order.
RESCALE: Dict[str, List[Tuple[float, str]]] = {
    "Total Energy Consumption": [(1 / 3.6e6, "MJ->GWh"), (1 / 3600.0, "GJ->GWh"),
                                 (1 / 1e6, "kWh->GWh"), (1 / 1e3, "MWh->GWh")],
    "Total Water Consumption": [(1 / 1e3, "litres->KL"), (1e3, "ML->KL"),
                                (1 / 1e6, "mL->KL")],
    "Total Waste Generated": [(1 / 1e3, "kg->MT"), (1e3, "kt->MT")],
    "CSR Expenditure": [(1 / 1e7, "INR->crore"), (1 / 100.0, "lakh->crore"),
                        (1 / 1e5, "INR(lakh scale)->crore")],
}


# per-employee intensity envelopes (cross-metric consistency)
INTENSITY: Dict[str, Tuple[float, float, str, float]] = {
    # metric: (low, high, label, multiplier applied to value/employees)
    "Total Energy Consumption": (100.0, 40000.0, "kWh/employee", 1e6),
    "Total Water Consumption": (0.5, 3000.0, "KL/employee", 1.0),
    "Total Waste Generated": (0.005, 5000.0, "kg/employee", 1e3),
}


def _plausible(metric: str, value: Optional[float],
               employees: Optional[float] = None) -> Tuple[bool, str]:
    if not is_num(value):
        return False, "missing"
    lo, hi, zero_bad = PLAUSIBLE[metric]
    v = float(value)
    if v == 0 and zero_bad:
        return False, "zero_not_plausible"
    if v < lo or v > hi:
        return False, f"outside_range[{lo:g},{hi:g}]"
    if metric in INTENSITY and is_num(employees) and float(employees) >= 50:
        ilo, ihi, label, mult = INTENSITY[metric]
        intensity = v * mult / float(employees)
        if intensity < ilo or intensity > ihi:
            return False, f"intensity_implausible({intensity:.3g} {label})"
    return True, "ok"


class BRSRExtractionAgent(BaseAgent):
    agent_id = "agent1"
    name = "Agent 1 - BRSR Extraction & Audit Agent"
    role = ("Extract the 18 BRSR ESG metrics per company from XBRL/PDF filings "
            "and audit them for plausibility before anyone else uses them.")
    goal = ("Every company that has a filing is extracted, audited, repaired "
            "or quarantined, and the final metrics file is written.")
    rules = [
        "Extract a company before auditing it; audit before repairing.",
        "Repair a value only with a strategy that produces a plausible number "
        "from evidence already extracted (other source or a unit rescale).",
        "If no strategy yields a plausible value, quarantine the metric - "
        "never keep a number you cannot defend.",
        "Finalize only after all companies with filings are processed.",
    ]

    def __init__(self, *args, companies: Optional[List[str]] = None,
                 force: bool = False, timeout_s: int = 420, **kw):
        self.target_codes = [c.upper() for c in companies] if companies else list(COMPANIES)
        unknown = [c for c in self.target_codes if c not in COMPANIES]
        if unknown:
            raise ValueError(f"Unknown company code(s): {unknown}")
        self.force = force
        # ~15 actions per company worst case (extract, audit x2, repair/quarantine x6)
        self.max_steps = 60 + 18 * len(self.target_codes)
        self.timeout_s = timeout_s
        self._engine = None
        super().__init__(*args, **kw)
        self.memory.update({"rows": {}, "status": {}, "issues": {}, "repairs": [],
                            "finalized": False, "timings": {}})
        self.repeatable_tools = {"audit_company"}      # re-audit after repairs is legitimate

    # ------------------------------------------------------------------ tools
    def build_tools(self) -> List[Tool]:
        code = {"company_code": {"type": "string",
                                 "description": "Company code, e.g. INFY"}}
        return [
            Tool("survey_sources",
                 "List each company with whether its BRSR PDF / XBRL file exist "
                 "and its current processing status.", self.t_survey),
            Tool("extract_company",
                 "Run the XBRL-first, PDF-fallback extraction engine for ONE "
                 "company and store its 18 metric rows (uses cache when valid).",
                 self.t_extract, code, ["company_code"]),
            Tool("taxonomy_extract",
                 "Read exact BRSR XBRL taxonomy concepts (employees, water, waste, "
                 "board, turnover, ...) and override fuzzy-matched values with them.",
                 self.t_taxonomy, code, ["company_code"]),
            Tool("audit_company",
                 "Check a company's extracted metrics against plausibility "
                 "rules and list every suspicious metric with the reason.",
                 self.t_audit, code, ["company_code"]),
            Tool("repair_metric",
                 "Try to repair ONE suspicious metric. strategy: auto (try all "
                 "in order), prefer_pdf, prefer_xbrl, or rescale (unit fix).",
                 self.t_repair,
                 {**code, "metric_name": {"type": "string"},
                  "strategy": {"type": "string",
                               "enum": ["auto", "prefer_pdf", "prefer_xbrl", "rescale"]}},
                 ["company_code", "metric_name", "strategy"]),
            Tool("quarantine_metric",
                 "Remove an untrustworthy value and flag it for manual review.",
                 self.t_quarantine,
                 {**code, "metric_name": {"type": "string"},
                  "reason": {"type": "string"}},
                 ["company_code", "metric_name", "reason"]),
            Tool("finalize",
                 "Write brsr_extracted_metrics.csv + audit report and publish "
                 "coverage to the blackboard. Only after all companies are done.",
                 self.t_finalize),
        ]

    def check_args(self, decision, state):
        bad = super().check_args(decision, state)
        if bad:
            return bad
        code = str(decision.args.get("company_code", "")).upper()
        t = decision.tool
        status = self.memory["status"].get(code, "pending")
        if t == "taxonomy_extract" and status == "pending":
            return f"{code} must be extracted before taxonomy_extract"
        if t == "audit_company" and (status == "pending" or code not in self.memory.get("taxonomized", set())):
            return f"{code} needs extract_company and taxonomy_extract before audit"
        if t in ("repair_metric", "quarantine_metric") and decision.args.get("metric_name") not in \
                [i["metric"] for i in self.memory["issues"].get(code, [])]:
            return f"{code}/{decision.args.get('metric_name')} is not an open audit issue"
        if t == "finalize" and (state["pending_count"] or state["extracted_not_audited"] or state["open_issues"]):
            return "finalize before all companies are processed"
        return None

    # ----------------------------------------------------------------- engine
    def _get_engine(self):
        if self._engine is None:
            from agents.agent1_brsr_extractor import Agent1BRSRExtractionAgent
            self._engine = Agent1BRSRExtractionAgent()
        return self._engine

    def _has_pdf(self, code: str) -> bool:
        return (BRSR_PDF_DIR / COMPANIES[code]["brsr_file"]).exists()

    def _has_xbrl(self, code: str) -> bool:
        return (XBRL_RAW_DIR / f"{code}_BRSR_2024_25.xml").exists()

    # ------------------------------------------------------------ tool bodies
    def t_survey(self) -> Dict[str, Any]:
        rows = []
        for code in self.target_codes:
            rows.append({"company": code, "pdf": self._has_pdf(code),
                         "xbrl": self._has_xbrl(code),
                         "cached": (CACHE_DIR / f"{code}.json").exists(),
                         "status": self.memory["status"].get(code, "pending")})
        no_source = [r["company"] for r in rows if not (r["pdf"] or r["xbrl"])]
        return {"companies": rows, "without_any_source": no_source}

    def t_extract(self, company_code: str) -> Dict[str, Any]:
        code = company_code.upper()
        if code not in COMPANIES:
            return {"ok": False, "error": f"unknown company {code}"}
        if not (self._has_pdf(code) or self._has_xbrl(code)):
            self.memory["status"][code] = "no_source"
            self.memory["rows"][code] = []
            self.add_finding("no_source", "high",
                             f"{code}: neither BRSR PDF nor XBRL file found", company=code)
            return {"ok": True, "company": code, "status": "no_source", "metrics": 0}
        cache = CACHE_DIR / f"{code}.json"
        started = time.perf_counter()
        if cache.exists() and not self.force:
            rows = json.loads(cache.read_text(encoding="utf-8"))
            origin = "cache"
        else:
            rows, origin = self._extract_isolated(code, cache)
            if rows is None:
                self.memory["status"][code] = "failed"
                self.memory["rows"][code] = []
                self.add_finding("extraction_failed", "high",
                                 f"{code}: extraction crashed or timed out in all modes", company=code)
                return {"ok": False, "error": f"{code}: extraction failed in all modes"}
        for r in rows:
            r.setdefault("Audit_Status", "not_audited")
            r.setdefault("Audit_Note", "")
        self.memory["rows"][code] = rows
        self.memory["status"][code] = "extracted"
        self.memory["timings"][code] = round(time.perf_counter() - started, 2)
        got = sum(1 for r in rows if is_num(r.get("Metric_Value")))
        return {"ok": True, "company": code, "origin": origin,
                "metrics_with_value": got, "metrics_total": len(rows),
                "seconds": self.memory["timings"][code]}

    def _extract_isolated(self, code: str, cache: Path):
        """Run the engine in a child process; retry XBRL-only on crash/timeout."""
        import subprocess
        import sys
        for mode, extra in (("full", []), ("xbrl_only_fallback", ["--xbrl-only"])):
            if mode == "xbrl_only_fallback" and not self._has_xbrl(code):
                break
            try:
                cache.unlink(missing_ok=True)
                proc = subprocess.run(
                    [sys.executable, "-m", "esg_agents.workers.extract_company", code, *extra],
                    cwd=str(ROOT), capture_output=True, text=True, timeout=self.timeout_s)
                if proc.returncode == 0 and cache.exists():
                    if mode != "full":
                        self.add_finding("pdf_skipped", "medium",
                                         f"{code}: PDF extraction failed/timed out; used XBRL only",
                                         company=code)
                    return json.loads(cache.read_text(encoding="utf-8")), mode
                self.log(f"  {code} [{mode}] failed rc={proc.returncode}: {proc.stderr[-160:]!r}")
            except subprocess.TimeoutExpired:
                self.log(f"  {code} [{mode}] timed out after {self.timeout_s}s")
        return None, "failed"

    def t_taxonomy(self, company_code: str) -> Dict[str, Any]:
        from .brsr_taxonomy import extract_exact, convert_energy
        code = company_code.upper()
        self.memory.setdefault("taxonomized", set()).add(code)
        xml = XBRL_RAW_DIR / f"{code}_BRSR_2024_25.xml"
        if not xml.exists():
            return {"ok": True, "company": code, "applied": 0, "note": "no XBRL file"}
        exact = extract_exact(xml)
        emp = exact.get("Total Employees", {}).get("value")
        raw = exact.pop("_energy_raw", None)
        if raw:
            gwh, note = convert_energy(raw["value"], emp)
            if gwh is not None:
                exact["Total Energy Consumption"] = {"value": gwh, "concept": raw["concept"], "note": note}
            else:
                self.add_finding("energy_unit_unresolved", "low", f"{code}: {note}", company=code)
        self._compute_trends(code, xml, exact, raw)
        applied = []
        for metric, info in exact.items():
            row = self._row(code, metric)
            if row is None or metric not in PLAUSIBLE:
                continue
            old = row.get("Metric_Value")
            row.update({"Metric_Value": info["value"], "Final_Source": "XBRL_TAXONOMY_EXACT",
                        "Validation_Status": "VALID", "Confidence_Score": 0.95,
                        "Needs_Manual_Review": False, "Audit_Status": "taxonomy_exact",
                        "Audit_Note": f"concept={info['concept']}; {info['note']}; previous={old}"})
            applied.append(metric)
        return {"ok": True, "company": code, "applied": len(applied), "metrics": applied}

    def _compute_trends(self, code: str, xml: Path, exact: Dict[str, Any], raw_energy) -> None:
        """Year-over-year change from the prior-year facts in the same XBRL file."""
        from .brsr_taxonomy import extract_prior, POLARITY
        prior = extract_prior(xml)
        rows = []
        for metric, pol in POLARITY.items():
            cur = exact.get(metric, {}).get("value")
            if metric == "Total Energy Consumption" and raw_energy:
                cur = raw_energy["value"]                   # raw unit vs raw unit
            prv = prior.get(metric)
            if cur is None or prv in (None, 0):
                continue
            change = (cur - prv) / abs(prv) * 100
            if metric.endswith("Percentage") or metric == "Employee Turnover Rate":
                change = cur - prv                         # percentage points
            better = (change * pol) > 0
            rows.append({"Company_Code": code, "Metric": metric, "Current": cur, "Prior": prv,
                         "Change": round(change, 2),
                         "Change_Unit": "pp" if (metric.endswith("Percentage") or metric == "Employee Turnover Rate") else "%",
                         "Direction": "improving" if better and abs(change) > 0.5 else
                                      "worsening" if (not better) and abs(change) > 0.5 else "stable"})
        self.memory.setdefault("trends", {})[code] = rows

    def _rows(self, code: str) -> List[Dict[str, Any]]:
        return self.memory["rows"].get(code.upper(), [])

    def _employees(self, code: str) -> Optional[float]:
        r = self._row(code, "Total Employees")
        if r and r.get("Audit_Status") != "quarantined" \
                and _plausible("Total Employees", r.get("Metric_Value"))[0]:
            return float(r["Metric_Value"])
        return None

    def _row(self, code: str, metric: str) -> Optional[Dict[str, Any]]:
        for r in self._rows(code):
            if r.get("Metric_Name") == metric:
                return r
        return None

    def t_audit(self, company_code: str) -> Dict[str, Any]:
        code = company_code.upper()
        issues = []
        emp = self._employees(code)
        self.memory.setdefault("audit_passes", {})
        self.memory["audit_passes"][code] = self.memory["audit_passes"].get(code, 0) + 1
        for r in self._rows(code):
            m = r["Metric_Name"]
            if m not in PLAUSIBLE or r.get("Audit_Status") in ("quarantined", "taxonomy_exact"):
                continue
            val = r.get("Metric_Value")
            ok, reason = _plausible(m, val, emp)
            if reason == "missing":
                continue
            if not ok:
                issues.append({"metric": m, "value": clean(val), "reason": reason,
                               "source": r.get("Final_Source"),
                               "xbrl_value": clean(r.get("XBRL_Value")),
                               "pdf_value": clean(r.get("PDF_Value"))})
                r["Audit_Status"] = "suspicious"
            elif r.get("Audit_Status") == "repaired":
                continue
            else:
                # plausible but the two sources disagree strongly
                xv, pv = r.get("XBRL_Value"), r.get("PDF_Value")
                if is_num(xv) and is_num(pv) and max(abs(xv), abs(pv)) > 0:
                    rel = abs(xv - pv) / max(abs(xv), abs(pv))
                    if rel > 0.35:
                        r["Audit_Status"] = "sources_disagree"
                        r["Audit_Note"] = f"XBRL={xv:g} vs PDF={pv:g} (rel diff {rel:.0%})"
                        r["Needs_Manual_Review"] = True
                        continue
                r["Audit_Status"] = "passed"
        order = {m: i for i, m in enumerate(PLAUSIBLE)}
        issues.sort(key=lambda i: (i["metric"] != "Total Employees", order[i["metric"]]))
        self.memory["issues"][code] = issues
        self.memory["status"][code] = "audited"
        return {"ok": True, "company": code, "issue_count": len(issues),
                "issues": issues}

    def t_repair(self, company_code: str, metric_name: str,
                 strategy: str = "auto") -> Dict[str, Any]:
        code = company_code.upper()
        row = self._row(code, metric_name)
        if row is None:
            return {"ok": False, "error": f"no such metric row {code}/{metric_name}"}
        original = row.get("Metric_Value")
        order = ["prefer_pdf", "prefer_xbrl", "rescale"] if strategy == "auto" else [strategy]
        for strat in order:
            cand, note = self._candidate(row, metric_name, strat, code)
            if cand is None:
                continue
            ok, _ = _plausible(metric_name, cand, self._employees(code))
            if not ok:
                continue
            row["Metric_Value"] = cand
            row["Final_Source"] = f"{row.get('Final_Source')}->repaired:{strat}"
            row["Validation_Status"] = "REPAIRED"
            row["Confidence_Score"] = min(float(row.get("Confidence_Score") or 0.6), 0.7)
            row["Audit_Status"] = "repaired"
            row["Audit_Note"] = f"{strat}: {note}; original={original}"
            # Trust a repair only if independent evidence supports it: the
            # per-employee intensity check passed against a trusted headcount,
            # or the adopted source itself reported high confidence.
            emp = self._employees(code)
            corroborated = (metric_name in INTENSITY and is_num(emp)) or \
                float(row.get("PDF_Confidence") or 0) >= 0.75 and strat == "prefer_pdf" \
                or float(row.get("XBRL_Confidence") or 0) >= 0.9 and strat == "prefer_xbrl"
            row["Needs_Manual_Review"] = not corroborated
            self.memory["repairs"].append({"company": code, "metric": metric_name,
                                           "strategy": strat, "from": clean(original),
                                           "to": clean(cand), "note": note})
            return {"ok": True, "repaired": True, "strategy": strat,
                    "old": clean(original), "new": clean(cand), "note": note}
        return {"ok": True, "repaired": False, "tried": order,
                "message": "no strategy produced a plausible value"}

    def _candidate(self, row, metric, strat, code="") -> Tuple[Optional[float], str]:
        xv, pv = row.get("XBRL_Value"), row.get("PDF_Value")
        cur = row.get("Metric_Value")
        if strat == "prefer_pdf":
            if is_num(pv) and (not is_num(cur) or float(pv) != float(cur)):
                return float(pv), "adopted PDF-extracted value"
        elif strat == "prefer_xbrl":
            if is_num(xv) and (not is_num(cur) or float(xv) != float(cur)):
                return float(xv), "adopted XBRL-extracted value"
        elif strat == "rescale":
            emp = self._employees(code) if code else None
            base = [float(v) for v in (cur, xv, pv) if is_num(v) and float(v) != 0]
            for b in base:
                if metric in RESCALE:
                    for factor, label in RESCALE[metric]:
                        if _plausible(metric, b * factor, emp)[0]:
                            return b * factor, f"rescaled {label} (x{factor:.3g})"
            # XBRL stores percentages as fractions; a PDF value is never rescaled.
            if metric in PERCENT_METRICS and is_num(xv) and 0 < float(xv) <= 1 \
                    and _plausible(metric, float(xv) * 100)[0]:
                return float(xv) * 100.0, "XBRL ratio->percent (x100)"
        return None, ""

    def t_quarantine(self, company_code: str, metric_name: str,
                     reason: str) -> Dict[str, Any]:
        code = company_code.upper()
        row = self._row(code, metric_name)
        if row is None:
            return {"ok": False, "error": f"no such metric row {code}/{metric_name}"}
        row["Audit_Note"] = f"quarantined: {reason}; removed value={row.get('Metric_Value')}"
        row["Metric_Value"] = None
        row["Validation_Status"] = "QUARANTINED"
        row["Audit_Status"] = "quarantined"
        row["Needs_Manual_Review"] = True
        row["Confidence_Score"] = 0.0
        return {"ok": True, "quarantined": True, "metric": metric_name}

    def t_finalize(self) -> Dict[str, Any]:
        all_rows, audit_rows = [], []
        for code in self.target_codes:
            rows = self._rows(code)
            if not rows and self.memory["status"].get(code) == "no_source":
                for group in ESG_METRICS.values():
                    for m in group:
                        all_rows.append({"Company_Code": code,
                                         "Company_Name": COMPANIES[code]["full_name"],
                                         "Metric_Category": next(k for k, v in ESG_METRICS.items() if m in v),
                                         "Metric_Name": m["name"], "Unit": m["unit"],
                                         "Metric_Value": None, "Validation_Status": "NOT_FOUND",
                                         "Needs_Manual_Review": True,
                                         "Audit_Status": "no_source",
                                         "Audit_Note": "no BRSR filing available"})
            all_rows.extend(rows)
            for r in rows:
                if r.get("Audit_Status") not in (None, "passed", "not_audited"):
                    audit_rows.append({k: r.get(k) for k in
                        ("Company_Code", "Metric_Name", "Metric_Value", "Final_Source",
                         "Validation_Status", "Audit_Status", "Audit_Note")})
        df = pd.DataFrame(all_rows)
        path = PROCESSED / OUTPUT_FILES["brsr_metrics"]
        df.to_csv(path, index=False)
        audit_path = PROCESSED / "agent1_audit_report.csv"
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
        trends = [r for rows_ in self.memory.get("trends", {}).values() for r in rows_]
        pd.DataFrame(trends).to_csv(PROCESSED / "brsr_trends.csv", index=False)
        timing = PROCESSED / "agent1_extraction_timing.csv"
        pd.DataFrame([{"Company_Code": c, "Seconds": s}
                      for c, s in self.memory["timings"].items()]).to_csv(timing, index=False)
        cov = self._coverage(df)
        self.board.post(self.agent_id, "brsr_coverage", cov)
        self.memory["finalized"] = True
        self.artifacts += [str(path.relative_to(path.parents[1])),
                           str(audit_path.relative_to(audit_path.parents[1]))]
        return {"ok": True, "rows": len(df), "coverage": cov,
                "files": [str(path), str(audit_path)]}

    def _coverage(self, df: "pd.DataFrame") -> Dict[str, Any]:
        if df.empty:
            return {"companies": 0}
        per = df.groupby("Company_Code")["Metric_Value"].apply(lambda s: int(s.notna().sum()))
        return {"companies": int(per.shape[0]),
                "mean_metrics_per_company": round(float(per.mean()), 2),
                "companies_below_9_metrics": sorted(per[per < 9].index.tolist()),
                "repairs": len(self.memory["repairs"])}

    # ------------------------------------------------------------- reasoning
    def observe(self) -> Dict[str, Any]:
        st = self.memory["status"]
        buckets: Dict[str, List[str]] = {"pending": [], "extracted": [], "audited": [],
                                         "no_source": []}
        for c in self.target_codes:
            buckets.setdefault(st.get(c, "pending"), []).append(c)
        open_issues = {c: [i["metric"] for i in v]
                       for c, v in self.memory["issues"].items() if v}
        return {"companies_total": len(self.target_codes),
                "pending": buckets["pending"][:12], "pending_count": len(buckets["pending"]),
                "extracted_not_audited": buckets["extracted"],
                "audited": len(buckets["audited"]), "no_source": buckets["no_source"],
                "open_issues": open_issues, "repairs_done": len(self.memory["repairs"]),
                "finalized": self.memory["finalized"]}

    def is_done(self, state) -> Tuple[bool, str]:
        if state["pending_count"] == 0 and not state["extracted_not_audited"] \
                and not state["open_issues"] and state["finalized"]:
            return True, (f"{state['audited']} companies audited, "
                          f"{len(self.memory['repairs'])} metrics repaired, output written")
        return False, ""

    def policy(self, state) -> Optional[Decision]:
        if "surveyed" not in self.memory:
            self.memory["surveyed"] = True
            return Decision("survey_sources", {}, "Survey which filings exist first")
        for c in state["extracted_not_audited"]:
            if c not in self.memory.get("taxonomized", set()):
                return Decision("taxonomy_extract", {"company_code": c},
                                f"{c}: prefer exact taxonomy concepts over fuzzy matches")
            return Decision("audit_company", {"company_code": c},
                            f"{c} extracted; audit before trusting it")
        for c, metrics in list(state["open_issues"].items()):
            m = metrics[0]
            key = ("tried", c, m)
            if key not in self.memory:
                self.memory[key] = 1
                return Decision("repair_metric",
                                {"company_code": c, "metric_name": m, "strategy": "auto"},
                                f"try to repair {c}/{m}")
            return Decision("quarantine_metric",
                            {"company_code": c, "metric_name": m,
                             "reason": "no repair strategy produced a plausible value"},
                            f"{c}/{m} cannot be defended")
        if state["pending"]:
            c = state["pending"][0]
            return Decision("extract_company", {"company_code": c}, f"next pending company {c}")
        if not state["finalized"]:
            return Decision("finalize", {}, "all companies processed")
        return None

    def reflect(self, decision: Decision, result: Dict[str, Any]) -> None:
        if decision.tool == "repair_metric" and result.get("repaired") is False:
            # keep the issue open so the policy quarantines it next
            return
        if decision.tool in ("repair_metric", "quarantine_metric") and result.get("ok"):
            code = decision.args["company_code"].upper()
            m = decision.args["metric_name"]
            self.memory["issues"][code] = [i for i in self.memory["issues"].get(code, [])
                                           if i["metric"] != m]
            if not self.memory["issues"][code] \
                    and self.memory.get("audit_passes", {}).get(code, 0) < 2 \
                    and any(r["company"] == code for r in self.memory["repairs"]):
                self.memory["status"][code] = "extracted"   # verify the repairs
        if decision.tool == "audit_company" and result.get("issue_count"):
            self.add_finding("audit_issues", "medium",
                             f"{result['company']}: {result['issue_count']} implausible metrics",
                             company=result["company"])
