"""Compare structured XBRL facts with BRSR-extracted metric records."""
from __future__ import annotations
import re
from typing import Any

def _num(v):
    try:
        s=str(v).replace(",","").replace("%","").strip()
        return float(s)
    except Exception: return None

def match_xbrl_to_brsr(xbrl_records:list[dict[str,Any]], brsr_records:list[dict[str,Any]], tolerance:float=0.05)->list[dict[str,Any]]:
    out=[]
    for x in xbrl_records:
        xn=re.sub(r"[^a-z0-9]","",str(x.get("metric_name","")).lower())
        candidates=[]
        for b in brsr_records:
            bn=re.sub(r"[^a-z0-9]","",str(b.get("Metric_Name",b.get("metric_name",""))).lower())
            if xn and (xn in bn or bn in xn): candidates.append(b)
        best=candidates[0] if candidates else None
        xv=_num(x.get("value")); bv=_num(best.get("Metric_Value")) if best else None
        discrepancy=False
        if xv is not None and bv is not None:
            scale=max(abs(xv),abs(bv),1.0); discrepancy=abs(xv-bv)/scale>tolerance
        out.append({"metric_name":x.get("metric_name"),"xbrl_value":x.get("value"),"brsr_value":best.get("Metric_Value") if best else None,"match_status":"matched" if best else "no_brsr_match","discrepancy_flag":discrepancy,"tolerance":tolerance,"xbrl_evidence":x.get("evidence_locator"),"brsr_page":best.get("Page_Number") if best else None})
    return out
