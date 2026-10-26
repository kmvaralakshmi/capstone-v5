"""Exact-concept reader for NSE BRSR XBRL files.

The generic fuzzy matcher in utils/xbrl_extractor.py guesses by keyword and
mistakes related facts for each other (e.g. it reported *renewable* energy as
*total* energy). BRSR XBRL uses a fixed taxonomy, so for the metrics that have
a clean concept we read that concept directly. Anything not in this map is left
to the fuzzy extractor / PDF fallback.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

FACT = re.compile(r'<[A-Za-z0-9\-]+:([A-Za-z0-9_]+)\s[^>]*?contextRef="([^"]+)"[^>]*>([^<]*)<')

# energy unit candidates to convert the filing's unit to GWh
ENERGY_UNITS = [("MJ", 1 / 3.6e6), ("GJ", 1 / 3600.0), ("kWh", 1 / 1e6), ("MWh", 1 / 1e3),
                ("J", 1 / 3.6e12)]


def _num(v: str) -> Optional[float]:
    try:
        return float(v.strip())
    except Exception:
        return None


_FACT_CACHE: Dict[Tuple[str, float], Dict] = {}


def read_facts(xml_path: Path) -> Dict[Tuple[str, str], float]:
    """(concept, contextRef) -> first numeric value (first = current year). Cached per file version."""
    key = (str(xml_path), Path(xml_path).stat().st_mtime)
    if key in _FACT_CACHE:
        return _FACT_CACHE[key]
    _FACT_CACHE[key] = _read_facts(xml_path)
    return _FACT_CACHE[key]


def _read_facts(xml_path: Path) -> Dict[Tuple[str, str], float]:
    text = Path(xml_path).read_text(encoding="utf-8", errors="ignore")
    out: Dict[Tuple[str, str], float] = {}
    for concept, ctx, val in FACT.findall(text):
        n = _num(val)
        if n is not None and (concept, ctx) not in out:
            out[(concept, ctx)] = n
    return out


def first_ctx(facts, concept: str, ctx_prefix: str) -> Optional[float]:
    for (c, ctx), v in facts.items():
        if c == concept and ctx.startswith(ctx_prefix):
            return v
    return None


def extract_exact(xml_path: Path) -> Dict[str, Dict[str, Any]]:
    """Return {metric: {value, concept, note}} for metrics with exact concepts.
    Energy is returned in the filing's own unit under key '_energy_raw'."""
    f = read_facts(xml_path)
    g = lambda concept, ctx="DCYMain": f.get((concept, ctx))  # noqa: E731
    out: Dict[str, Dict[str, Any]] = {}

    def put(metric, value, concept, note=""):
        if value is not None:
            out[metric] = {"value": float(value), "concept": concept, "note": note}

    emp_c = "NumberOfEmployeesOrWorkersIncludingDifferentlyAbled"
    pct_c = "PercentageOfEmployeesOrWorkersIncludingDifferentlyAbled"
    put("Total Employees", first_ctx(f, emp_c, "D_Gender_PermanentEmployees"), emp_c, "permanent employees (all genders)")
    fem = first_ctx(f, pct_c, "D_Female_PermanentEmployees")
    put("Female Employee Percentage", None if fem is None else fem * 100, pct_c, "female share of permanent employees")
    tr = first_ctx(f, "TurnoverRate", "D_Gender_PermanentEmployees")
    put("Employee Turnover Rate", None if tr is None else tr * 100, "TurnoverRate", "permanent employees, current year")
    inj = first_ctx(f, "TotalRecordableWorkRelatedInjuries", "D_Employees")
    put("Health and Safety Incidents", inj, "TotalRecordableWorkRelatedInjuries", "employees, current year")
    put("Total Water Consumption", g("TotalVolumeOfWaterConsumption"), "TotalVolumeOfWaterConsumption", "KL")
    waste, rec = g("TotalWasteGenerated"), g("TotalWasteRecovered")
    put("Total Waste Generated", waste, "TotalWasteGenerated", "MT")
    if waste and rec is not None and waste > 0:
        put("Waste Recycled Percentage", 100.0 * rec / waste, "TotalWasteRecovered/TotalWasteGenerated", "recovered / generated")
    tot, ren = g("TotalEnergyConsumedFromRenewableAndNonRenewableSources"), g("TotalEnergyConsumedFromRenewableSources")
    if tot:
        out["_energy_raw"] = {"value": tot, "concept": "TotalEnergyConsumedFromRenewableAndNonRenewableSources", "note": ""}
        if ren is not None:
            put("Renewable Energy Percentage", 100.0 * ren / tot, "TotalEnergyConsumedFromRenewableSources/Total", "same unit cancels")
    put("Board Size", g("TotalNumberOfBoardOfDirectors"), "TotalNumberOfBoardOfDirectors")
    fd = g("PercentageOfFemaleBoardOfDirectors")
    put("Female Directors Percentage", None if fd is None else fd * 100, "PercentageOfFemaleBoardOfDirectors")
    return out


def convert_energy(raw: float, employees: Optional[float]) -> Tuple[Optional[float], str]:
    """Pick the unit whose result gives a plausible kWh/employee (100..40000)."""
    if not employees or employees < 50:
        return None, "employees unknown: cannot infer energy unit"
    for label, factor in ENERGY_UNITS:
        gwh = raw * factor
        intensity = gwh * 1e6 / employees
        if 100 <= intensity <= 40000:
            return gwh, f"unit inferred as {label} (intensity {intensity:.0f} kWh/employee)"
    return None, "no energy unit gives a plausible per-employee intensity"


def nth_ctx(facts, concept: str, ctx_prefix: str, n: int) -> Optional[float]:
    hits = [v for (c, ctx), v in facts.items() if c == concept and ctx.startswith(ctx_prefix)]
    return hits[n] if len(hits) > n else None


# metric -> (polarity)  +1 = higher is better, -1 = lower is better
POLARITY = {"Total Energy Consumption": -1, "Renewable Energy Percentage": +1,
            "Total Water Consumption": -1, "Total Waste Generated": -1,
            "Waste Recycled Percentage": +1, "Employee Turnover Rate": -1}


def extract_prior(xml_path: Path) -> Dict[str, float]:
    """Prior-year values (same unit as current) for trend analysis."""
    f = read_facts(xml_path)
    g = lambda concept, ctx="DPYMain": f.get((concept, ctx))  # noqa: E731
    out: Dict[str, float] = {}
    for metric, concept in (("Total Water Consumption", "TotalVolumeOfWaterConsumption"),
                            ("Total Waste Generated", "TotalWasteGenerated")):
        v = g(concept)
        if v is not None:
            out[metric] = v
    tot = g("TotalEnergyConsumedFromRenewableAndNonRenewableSources")
    ren = g("TotalEnergyConsumedFromRenewableSources")
    if tot:
        out["Total Energy Consumption"] = tot            # raw unit, ratio used for trend
        if ren is not None:
            out["Renewable Energy Percentage"] = 100.0 * ren / tot
    w, rec = g("TotalWasteGenerated"), g("TotalWasteRecovered")
    if w and rec is not None and w > 0:
        out["Waste Recycled Percentage"] = 100.0 * rec / w
    tr = nth_ctx(f, "TurnoverRate", "D_Gender_PermanentEmployees_TableB", 1)
    if tr is not None:
        out["Employee Turnover Rate"] = tr * 100
    return out


def energy_raw_current(xml_path: Path) -> Optional[float]:
    return read_facts(xml_path).get(("TotalEnergyConsumedFromRenewableAndNonRenewableSources", "DCYMain"))
