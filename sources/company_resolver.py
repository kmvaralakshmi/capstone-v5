"""Resolve company names/symbols to the configured 50-company cohort."""
from __future__ import annotations
import re
from difflib import get_close_matches
from utils.config import COMPANIES

ALIASES = {
    "infosys limited": "INFY", "infosys": "INFY", "tata consultancy services": "TCS",
    "tcs": "TCS", "wipro limited": "WIPRO", "wipro": "WIPRO", "hcl technologies": "HCLTECH",
    "hcltech": "HCLTECH", "tech mahindra": "TECHM", "ltimindtree": "LTIM", "ltim": "LTIM",
    "mphasis limited": "MPHASIS", "persistent systems": "PERSISTENT", "coforge limited": "COFORGE",
    "l&t technology services": "LTTS", "tata elxsi": "TATAELXSI", "oracle financial services software": "OFSS",
    "kpit technologies": "KPITTECH", "cyient": "CYIENT", "birlasoft": "BSOFT",
    "zensar technologies": "ZENSARTECH", "tata technologies": "TATATECH", "happiest minds": "HAPPSTMNDS",
    "sonata software": "SONATSOFTW", "hexaware technologies": "HEXAWARE",
}

def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower().replace(".ns", ""))

def resolve_company(value: str) -> dict:
    raw = value.strip()
    if not raw:
        raise ValueError("Company name or NSE symbol is required")
    key = normalize(raw)
    for code, info in COMPANIES.items():
        candidates = [code, info["full_name"], info.get("ticker", "")]
        if any(normalize(c) == key for c in candidates):
            return {"company_code": code, "company_name": info["full_name"], **info, "match": "exact"}
    alias = ALIASES.get(raw.lower())
    if alias and alias in COMPANIES:
        return {"company_code": alias, "company_name": COMPANIES[alias]["full_name"], **COMPANIES[alias], "match": "alias"}
    names = {normalize(c): code for code, info in COMPANIES.items() for c in [code, info["full_name"]]}
    close = get_close_matches(key, list(names), n=1, cutoff=0.82)
    if close:
        code = names[close[0]]
        return {"company_code": code, "company_name": COMPANIES[code]["full_name"], **COMPANIES[code], "match": "fuzzy"}
    raise ValueError(f"Company '{value}' was not found in the configured 50-company cohort")
