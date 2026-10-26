"""Agent 0 - Filing Discovery Agent.

ROLE
    Turn "Analyze Infosys" (a name or NSE symbol) into verified BRSR PDF + XBRL
    files on disk, without the user hunting for URLs (spec A2).

HOW IT WORKS
    resolve_company   -> cohort lookup (exact/alias/fuzzy); an unknown NSE symbol
                         is registered on demand in data/dynamic_companies.json
    check_local       -> which filings already exist (nothing is re-downloaded)
    discover_xbrl     -> runs helper-scripts/discover_brsr_xbrl.py for the symbol
                         (NSE BRSR listing -> matching XBRL -> validated download)
    fetch_pdf         -> BRSR PDF from the NSE listing's ATTACHMENT url when the
                         company has no PDF yet
    validate_filings  -> XBRL parses and has the target-year context; PDF has
                         the %PDF header and a plausible size
    finalize          -> writes filing_discovery.json and posts the result

Failures degrade gracefully: if only XBRL exists the pipeline runs in
XBRL-only mode (Agent 1 already supports it); if nothing exists the run stops
with a clear message instead of producing empty scores.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .common import BRSR_PDF_DIR, COMPANIES, PROCESSED, ROOT, XBRL_RAW_DIR
from .framework import BaseAgent, Decision, Tool

DYN_FILE = ROOT / "data" / "dynamic_companies.json"
LISTING_DEBUG = ROOT / "processed-data" / "nse_brsr_api_debug.txt"


def _norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", x.lower())


class FilingDiscoveryAgent(BaseAgent):
    agent_id = "agent0"
    name = "Agent 0 - Filing Discovery Agent"
    role = "Find, download and validate the BRSR PDF and XBRL for a company."
    goal = "Company resolved and every available filing validated, or a clear 'not found' verdict."
    rules = ["Never re-download a filing that already validates.",
             "Prefer XBRL (structured); the PDF is evidence and fallback.",
             "If a source fails, record it and continue with what exists."]

    def __init__(self, *args, companies: Optional[List[str]] = None, query: Optional[str] = None,
                 hq_city: str = "", **kw):
        self.queries = [query] if query else list(companies or [])
        self.hq_city = hq_city
        super().__init__(*args, **kw)
        self.memory.update({"resolved": {}, "local": {}, "valid": {}, "attempts": {}, "finalized": False})

    def build_tools(self) -> List[Tool]:
        q = {"query": {"type": "string", "description": "company name or NSE symbol"}}
        return [
            Tool("resolve_company", "Resolve a name/symbol to a company code (registers unknown symbols).",
                 self.t_resolve, q, ["query"]),
            Tool("check_local", "Check which filings already exist for a company code.",
                 self.t_local, {"company_code": {"type": "string"}}, ["company_code"]),
            Tool("discover_xbrl", "Discover and download the target-year XBRL from NSE.",
                 self.t_xbrl, {"company_code": {"type": "string"}}, ["company_code"]),
            Tool("fetch_pdf", "Download the BRSR PDF from the NSE listing attachment.",
                 self.t_pdf, {"company_code": {"type": "string"}}, ["company_code"]),
            Tool("validate_filings", "Validate downloaded XBRL/PDF files.",
                 self.t_validate, {"company_code": {"type": "string"}}, ["company_code"]),
            Tool("finalize", "Write filing_discovery.json.", self.t_finalize),
        ]

    # ---------------------------------------------------------------- tools
    def t_resolve(self, query: str) -> Dict[str, Any]:
        from sources.company_resolver import resolve_company
        try:
            r = resolve_company(query)
            code = r["company_code"]
            self.memory["resolved"][query] = code
            return {"ok": True, "company_code": code, "name": r["company_name"], "match": r["match"]}
        except ValueError:
            pass
        sym = re.sub(r"[^A-Za-z0-9&-]", "", query).upper()
        if not (2 <= len(sym) <= 20):
            return {"ok": False, "error": f"cannot interpret '{query}' as an NSE symbol"}
        info = {"full_name": query.strip().title() if " " in query else sym, "ticker": f"{sym}.NS",
                "sector": "Unknown", "brsr_file": f"{sym}_BR_24-25.pdf",
                "headquarters": {"city": self.hq_city or "Unknown", "area": ""}, "major_offices": [],
                "registered_dynamically": True}
        DYN_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(DYN_FILE.read_text(encoding="utf-8")) if DYN_FILE.exists() else {}
        data[sym] = info
        DYN_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        COMPANIES.setdefault(sym, info)
        self.memory["resolved"][query] = sym
        self.add_finding("dynamic_company", "low",
                         f"{sym} is outside the 50-company cohort: registered on demand "
                         f"(HQ city '{info['headquarters']['city']}' -> air-quality context may be unavailable)")
        return {"ok": True, "company_code": sym, "name": info["full_name"], "match": "registered_new"}

    def _paths(self, code: str):
        brsr_file = COMPANIES[code].get("brsr_file", f"{code}_BR_24-25.pdf")
        return BRSR_PDF_DIR / brsr_file, XBRL_RAW_DIR / f"{code}_BRSR_2024_25.xml"

    def t_local(self, company_code: str) -> Dict[str, Any]:
        code = company_code.upper()
        pdf, xml = self._paths(code)
        self.memory["local"][code] = {"pdf": pdf.exists() and pdf.stat().st_size > 50_000,
                                      "xbrl": xml.exists() and xml.stat().st_size > 5_000}
        return {"ok": True, "company": code, **self.memory["local"][code]}

    def _run_script(self, args: List[str], timeout: int = 240) -> Tuple[int, str]:
        try:
            p = subprocess.run([sys.executable, *args], cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
            return p.returncode, (p.stdout + p.stderr)[-400:]
        except subprocess.TimeoutExpired:
            return 124, "timed out"

    def t_xbrl(self, company_code: str) -> Dict[str, Any]:
        code = company_code.upper()
        self.memory["attempts"][("xbrl", code)] = True
        rc, tail = self._run_script(["helper-scripts/discover_brsr_xbrl.py", "--company", code])
        _, xml = self._paths(code)
        return {"ok": rc == 0 and xml.exists(), "returncode": rc, "found": xml.exists(), "detail": tail}

    def _listing_url(self, code: str) -> Optional[str]:
        if not LISTING_DEBUG.exists():
            return None
        want = _norm(COMPANIES[code]["full_name"])
        text = LISTING_DEBUG.read_text(encoding="utf-8-sig", errors="ignore")
        for row in csv.reader(text.splitlines()[1:]):
            if len(row) > 3 and (want in _norm(row[0]) or _norm(row[0]) in want) and row[3].startswith("http"):
                return row[3]
        return None

    def t_pdf(self, company_code: str) -> Dict[str, Any]:
        import requests
        code = company_code.upper()
        self.memory["attempts"][("pdf", code)] = True
        url = self._listing_url(code)
        if not url:
            return {"ok": False, "error": "no BRSR PDF url found in the NSE listing cache; add the PDF manually "
                                          f"as brsr-pdfs/{self._paths(code)[0].name}"}
        pdf, _ = self._paths(code)
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or not r.content.startswith(b"%PDF"):
            return {"ok": False, "error": f"download failed (HTTP {r.status_code}, not a PDF)"}
        pdf.write_bytes(r.content)
        return {"ok": True, "saved": str(pdf.relative_to(ROOT)), "bytes": len(r.content)}

    def t_validate(self, company_code: str) -> Dict[str, Any]:
        code = company_code.upper()
        pdf, xml = self._paths(code)
        res = {"pdf": False, "xbrl": False}
        if pdf.exists():
            res["pdf"] = pdf.read_bytes()[:5] == b"%PDF-" and pdf.stat().st_size > 50_000
        if xml.exists():
            head = xml.read_text(encoding="utf-8", errors="ignore")
            res["xbrl"] = "<xbrli:xbrl" in head[:5000].lower() or "<xbrl" in head[:5000].lower()
            res["xbrl"] = res["xbrl"] and "2024-04-01" in head and "2025-03-31" in head
        self.memory["valid"][code] = res
        return {"ok": True, "company": code, **res}

    def t_finalize(self) -> Dict[str, Any]:
        out = {"companies": {c: {"valid": self.memory["valid"].get(c, {}), "local": self.memory["local"].get(c, {})}
                             for c in self.memory["valid"]},
               "queries": self.memory["resolved"]}
        (PROCESSED / "filing_discovery.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        none = [c for c, v in self.memory["valid"].items() if not (v["pdf"] or v["xbrl"])]
        self.board.post(self.agent_id, "filings", out,
                        severity="high" if none else "info")
        for c in none:
            self.add_finding("no_filings", "high", f"{c}: no valid BRSR PDF or XBRL could be found")
        self.memory["finalized"] = True
        return {"ok": True, "no_filings": none}

    # ------------------------------------------------------------ reasoning
    def observe(self) -> Dict[str, Any]:
        unresolved = [q for q in self.queries if q not in self.memory["resolved"]]
        codes = list(self.memory["resolved"].values())
        need_check = [c for c in codes if c not in self.memory["local"]]
        need_x = [c for c in codes if c in self.memory["local"] and not self.memory["local"][c]["xbrl"]
                  and ("xbrl", c) not in self.memory["attempts"]]
        need_p = [c for c in codes if c in self.memory["local"] and not self.memory["local"][c]["pdf"]
                  and ("pdf", c) not in self.memory["attempts"] and self._listing_url(c)]
        need_val = [c for c in codes if c in self.memory["local"] and not need_x and not need_p
                    and c not in self.memory["valid"]]
        return {"unresolved": unresolved, "need_check": need_check, "need_xbrl": need_x,
                "need_pdf": need_p, "need_validation": need_val, "finalized": self.memory["finalized"]}

    def is_done(self, s) -> Tuple[bool, str]:
        if s["finalized"]:
            return True, f"{len(self.memory['valid'])} company/companies have validated filing status"
        return False, ""

    def policy(self, s) -> Optional[Decision]:
        if s["unresolved"]:
            return Decision("resolve_company", {"query": s["unresolved"][0]}, "resolve the name/symbol")
        if s["need_check"]:
            return Decision("check_local", {"company_code": s["need_check"][0]}, "see what exists locally")
        if s["need_xbrl"]:
            return Decision("discover_xbrl", {"company_code": s["need_xbrl"][0]}, "XBRL is the primary source")
        if s["need_pdf"]:
            return Decision("fetch_pdf", {"company_code": s["need_pdf"][0]}, "PDF is the evidence layer")
        if s["need_validation"]:
            # re-check local state first so validation sees freshly downloaded files
            return Decision("validate_filings", {"company_code": s["need_validation"][0]}, "verify files")
        return Decision("finalize", {}, "record the discovery result")

    def reflect(self, decision: Decision, result: Dict[str, Any]) -> None:
        if decision.tool in ("discover_xbrl", "fetch_pdf"):
            self.t_local(decision.args["company_code"])
        if decision.tool in ("discover_xbrl", "fetch_pdf") and not result.get("ok"):
            self.add_finding("download_failed", "medium",
                             f"{decision.args['company_code']}: {decision.tool} failed: "
                             f"{result.get('error') or result.get('detail', '')[:160]}")
