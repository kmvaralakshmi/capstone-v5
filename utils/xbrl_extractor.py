"""
Taxonomy-aware BRSR XBRL extractor.

Design:
- Parses the company-specific NSE BRSR XBRL instance.
- Preserves XBRL contexts, dimensions, units and source provenance.
- Uses taxonomy labels when a local BRSR taxonomy directory/XSD is supplied.
- Uses BRSR-specific semantic aliases as a fallback when labels are unavailable.
- Selects current-FY facts using period, duration/instant, dimensional and unit signals.
- Performs safe unit normalization (GJ->GWh, INR->crore, percentages).
- Supports evidence-backed derived metrics when a direct fact is absent.
- Produces a complete fact inventory for audit/debugging.

The canonical 18-metric interface is intentionally compatible with Agent 1.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TARGET_FY_START = "2024-04-01"
TARGET_FY_END = "2025-03-31"


def lname(tag: str) -> str:
    return (tag or "").rsplit("}", 1)[-1]


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def number(value: Any) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if not s:
        return None
    try:
        x = float(s)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


@dataclass
class XBRLContext:
    context_id: str
    start_date: Optional[str]
    end_date: Optional[str]
    instant: Optional[str]
    dimensions: List[str]
    dimension_count: int

    @property
    def is_target_end(self) -> bool:
        return self.end_date == TARGET_FY_END or self.instant == TARGET_FY_END

    @property
    def is_target_duration(self) -> bool:
        return self.start_date == TARGET_FY_START and self.end_date == TARGET_FY_END


@dataclass
class XBRLFact:
    concept: str
    qualified_concept: str
    value: Optional[float]
    raw_value: str
    unit: str
    unit_ref: str
    context_ref: str
    start_date: Optional[str]
    end_date: Optional[str]
    instant: Optional[str]
    dimensions: str
    dimension_count: int
    decimals: Optional[str]
    source_file: str
    label: str = ""
    taxonomy_role: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class XBRLMetricResult:
    metric_name: str
    unit: str
    value: Optional[float]
    source: str
    concept: Optional[str]
    context_ref: Optional[str]
    period_end: Optional[str]
    confidence: float
    status: str
    match_method: str
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class XBRLExtractor:
    """
    BRSR-aware extractor.

    The important improvement over a plain keyword matcher is that a metric
    can be obtained from:
      1. exact/near-exact taxonomy concept/label,
      2. BRSR semantic concept aliases,
      3. a defensible formula using other extracted facts.

    Direct facts always outrank derived facts.
    """

    # These are intentionally broader than the original 18 aliases.
    # They reflect BRSR terminology rather than a single company's wording.
    RULES: Dict[str, Dict[str, Sequence[str]]] = {
        "Total Energy Consumption": {
            "include": (
                "totalenergyconsumedingigajoules",
                "totalenergyconsumption",
                "totalenergyconsumed",
                "energyconsumption",
                "energyconsumed",
                "totalenergy",
            ),
            "exclude": (
                "intensity", "renewable", "nonrenewable",
                "percentage", "percent", "percrore", "turnover",
            ),
        },
        "Renewable Energy Percentage": {
            "include": (
                "renewableenergyconsumed",
                "renewableenergyconsumption",
                "renewableenergy",
                "energyfromrenewables",
                "percentageofenergyfromrenewables",
                "renewableenergypercentage",
                "renewableenergyratio",
            ),
            "exclude": (
                "totalenergy", "nonrenewable", "intensity",
            ),
        },
        "Total Water Consumption": {
            "include": (
                "totalwaterconsumption",
                "totalwaterwithdrawal",
                "totalwaterwithdrawnin",
                "waterconsumption",
                "waterwithdrawal",
                "waterwithdrawn",
                "totalwater",
            ),
            "exclude": (
                "intensity", "recycled", "reused",
                "percentage", "percent", "discharge",
                "effluent", "percrore",
            ),
        },
        "Water Recycled Percentage": {
            "include": (
                "waterrecycled",
                "waterreused",
                "recycledwater",
                "reusedwater",
                "percentageofwaterrecycled",
                "waterrecycling",
                "waterreuseratio",
            ),
            "exclude": (
                "totalwater", "withdrawal", "intensity",
            ),
        },
        "Total Waste Generated": {
            "include": (
                "totalwastegenerated",
                "wastegenerated",
                "totalwaste",
                "wastegeneration",
                "wastegeneratedin",
            ),
            "exclude": (
                "recycled", "recovered", "reused",
                "percentage", "percent", "intensity",
            ),
        },
        "Waste Recycled Percentage": {
            "include": (
                "wasterecycled",
                "wasterecovered",
                "wastereused",
                "recycledwaste",
                "percentageofwasterecycled",
                "wasterecycling",
                "wasterecovery",
            ),
            "exclude": (
                "totalwaste", "generated", "intensity",
            ),
        },
        "Total Employees": {
            "include": (
                "totalemployees",
                "numberofemployees",
                "totalpermanentemployees",
                "permanentemployees",
                "totalworkforce",
                "totalemployee",
            ),
            "exclude": (
                "female", "male", "women", "turnover",
                "attrition", "training", "hours",
                "percentage", "percent", "workers",
            ),
        },
        "Female Employee Percentage": {
            "include": (
                "femaleemployees",
                "womenemployees",
                "femaleemployee",
                "womenemployee",
                "percentageoffemaleemployees",
                "femaleworkforce",
                "womenworkforce",
                "femaleemployeespercentage",
            ),
            "exclude": (
                "director", "board", "worker",
                "number", "count", "wage",
            ),
        },
        "Employee Turnover Rate": {
            "include": (
                "employeeturnoverrate",
                "employeeturnover",
                "turnoverrate",
                "attritionrate",
                "employeeattrition",
                "turnoverpercentage",
            ),
            "exclude": (
                "director", "board", "female", "male",
            ),
        },
        "Training Hours per Employee": {
            "include": (
                "traininghoursperemployee",
                "averagetraininghoursperemployee",
                "traininghours",
                "traininganddevelopmenthours",
                "learninganddevelopmenthours",
                "averagehoursperemployee",
            ),
            "exclude": (
                "percentage", "percent", "coverage",
            ),
        },
        "Health and Safety Incidents": {
            "include": (
                "healthandsafetyincidents",
                "workrelatedinjuries",
                "workrelatedinjury",
                "occupationalinjuries",
                "losttimeinjuries",
                "losttimeinjury",
                "fatalities",
                "workrelatedfatalities",
                "safetyincidents",
                "injuries",
            ),
            "exclude": (
                "rate", "frequency", "severity",
                "percentage", "percent",
            ),
        },
        "CSR Expenditure": {
            "include": (
                "csrexpenditure",
                "csrspent",
                "csrspending",
                "corporatesocialresponsibilityexpenditure",
                "amountspentoncsr",
                "amountspentoncsrc",
                "totalcsr",
            ),
            "exclude": (
                "percentage", "percent", "policy",
                "committee", "obligation",
            ),
        },
        "Board Size": {
            "include": (
                "boardsize",
                "numberofdirectors",
                "numberofboardmembers",
                "boardmembers",
                "totaldirectors",
                "totalboardmembers",
            ),
            "exclude": (
                "independent", "female", "women",
                "percentage", "percent", "meeting",
                "committee",
            ),
        },
        "Independent Directors Percentage": {
            "include": (
                "independentdirectorspercentage",
                "percentageofindependentdirectors",
                "independentdirectorpercentage",
                "independentdirectors",
                "independentdirectorsratio",
            ),
            "exclude": (
                "boardmeeting", "committee",
            ),
        },
        "Female Directors Percentage": {
            "include": (
                "femaledirectorspercentage",
                "percentageoffemaledirectors",
                "womendirectorspercentage",
                "womendirectors",
                "femaleboardmembers",
                "femaleonboard",
            ),
            "exclude": (
                "employees", "employee", "workers",
                "number", "count",
            ),
        },
        "Board Meeting Frequency": {
            "include": (
                "numberofboardmeetings",
                "boardmeetingsheld",
                "boardmeetingsconducted",
                "boardmeetingfrequency",
                "meetingsheldbyboard",
                "boardmeetings",
            ),
            "exclude": (
                "attendance", "percentage", "percent",
            ),
        },
        "Audit Committee Size": {
            "include": (
                "auditcommitteesize",
                "numberofauditcommitteemembers",
                "auditcommitteemembers",
                "numberofmembersofauditcommittee",
            ),
            "exclude": (
                "meeting", "attendance", "percentage",
            ),
        },
        "Ethics Policy Violations": {
            "include": (
                "ethicsviolations",
                "ethicspolicyviolations",
                "codeofconductviolations",
                "violationsrelatedtoethics",
                "ethicscomplaints",
                "whistleblowercomplaints",
                "corruptionincidents",
                "briberyincidents",
            ),
            "exclude": (
                "policyexists", "policy",
                "resolvedpercentage", "percentage",
            ),
        },
    }

    LABELS: Dict[str, Sequence[str]] = {
        "Total Energy Consumption": (
            "total energy consumed", "total energy consumption",
            "energy consumed", "energy consumption",
            "total energy consumed in gigajoules",
        ),
        "Renewable Energy Percentage": (
            "renewable energy", "renewable energy consumed",
            "energy from renewable sources", "renewable energy percentage",
        ),
        "Total Water Consumption": (
            "total water consumption", "total water withdrawal",
            "water withdrawn", "water withdrawal", "water consumption",
        ),
        "Water Recycled Percentage": (
            "water recycled", "water reused",
            "percentage of water recycled", "recycled water",
        ),
        "Total Waste Generated": (
            "total waste generated", "waste generated",
            "total waste", "waste generation",
        ),
        "Waste Recycled Percentage": (
            "waste recycled", "waste recovered",
            "waste reused", "percentage of waste recycled",
        ),
        "Total Employees": (
            "total employees", "total permanent employees",
            "number of employees", "total workforce",
        ),
        "Female Employee Percentage": (
            "female employees", "women employees",
            "female workforce", "percentage of female employees",
        ),
        "Employee Turnover Rate": (
            "employee turnover rate", "turnover rate",
            "employee turnover", "attrition rate",
        ),
        "Training Hours per Employee": (
            "training hours per employee", "average training hours",
            "training and development hours",
        ),
        "Health and Safety Incidents": (
            "health and safety incidents", "work related injuries",
            "lost time injury", "occupational injuries", "fatalities",
        ),
        "CSR Expenditure": (
            "csr expenditure", "csr spent", "csr spending",
            "corporate social responsibility expenditure",
        ),
        "Board Size": (
            "board size", "number of directors",
            "number of board members", "board members",
        ),
        "Independent Directors Percentage": (
            "independent directors", "percentage of independent directors",
        ),
        "Female Directors Percentage": (
            "female directors", "women directors",
            "percentage of female directors",
        ),
        "Board Meeting Frequency": (
            "number of board meetings", "board meetings held",
            "board meetings conducted",
        ),
        "Audit Committee Size": (
            "audit committee members", "audit committee size",
        ),
        "Ethics Policy Violations": (
            "ethics violations", "code of conduct violations",
            "ethics complaints", "whistleblower complaints",
        ),
    }

    TARGET_UNITS = {
        "Total Energy Consumption": "GWh",
        "Renewable Energy Percentage": "%",
        "Total Water Consumption": "KL",
        "Water Recycled Percentage": "%",
        "Total Waste Generated": "MT",
        "Waste Recycled Percentage": "%",
        "Total Employees": "Count",
        "Female Employee Percentage": "%",
        "Employee Turnover Rate": "%",
        "Training Hours per Employee": "Hours",
        "Health and Safety Incidents": "Count",
        "CSR Expenditure": "Crores INR",
        "Board Size": "Count",
        "Independent Directors Percentage": "%",
        "Female Directors Percentage": "%",
        "Board Meeting Frequency": "Count",
        "Audit Committee Size": "Count",
        "Ethics Policy Violations": "Count",
    }

    def __init__(
        self,
        target_year: str = "2024-25",
        target_end_date: str = TARGET_FY_END,
        target_start_date: str = TARGET_FY_START,
        taxonomy_dir: Optional[Path] = None,
    ) -> None:
        self.target_year = target_year
        self.target_end_date = target_end_date
        self.target_start_date = target_start_date
        self.taxonomy_dir = Path(taxonomy_dir) if taxonomy_dir else None
        self.taxonomy_labels: Dict[str, str] = {}
        self.taxonomy_roles: Dict[str, str] = {}
        self.schema_refs: List[str] = []

        if self.taxonomy_dir and self.taxonomy_dir.exists():
            self._load_taxonomy(self.taxonomy_dir)

    # -------------------- taxonomy --------------------

    def _load_taxonomy(self, directory: Path) -> None:
        """
        Load labels from local BRSR XSD/linkbase files.

        This is deliberately permissive: taxonomy packages differ in how
        schema/linkbase files are laid out. We collect every label resource
        that can be found and never fail the instance parser because a label
        file is missing.
        """
        try:
            files = list(directory.rglob("*.xsd")) + list(directory.rglob("*.xml"))
        except Exception:
            return

        label_by_href: Dict[str, str] = {}

        for path in files:
            try:
                root = ET.parse(path).getroot()
            except Exception:
                continue

            for loc in root.iter():
                if lname(loc.tag) == "loc":
                    href = loc.attrib.get("{http://www.w3.org/1999/xlink}href", "")
                    label = loc.attrib.get("{http://www.w3.org/1999/xlink}label", "")
                    if href and label:
                        label_by_href[label] = href

            for label_el in root.iter():
                if lname(label_el.tag) != "label":
                    continue
                text = " ".join("".join(label_el.itertext()).split())
                if not text:
                    continue
                label_ref = label_el.attrib.get("{http://www.w3.org/1999/xlink}label", "")
                if label_ref:
                    href = label_by_href.get(label_ref, "")
                    if href:
                        concept = href.split("#")[-1].split(":")[-1]
                        self.taxonomy_labels[concept] = text

        # Also capture schema element annotations where available.
        for path in list(directory.rglob("*.xsd")):
            try:
                root = ET.parse(path).getroot()
            except Exception:
                continue
            for element in root.iter():
                if lname(element.tag) != "element":
                    continue
                name = element.attrib.get("name", "")
                if not name or name in self.taxonomy_labels:
                    continue
                documentation = []
                for d in element.iter():
                    if lname(d.tag) in {"documentation", "definition"} and d.text:
                        documentation.append(" ".join(d.text.split()))
                if documentation:
                    self.taxonomy_labels[name] = " ".join(documentation)

    def inspect_schema_refs(self, xml_path: Path) -> List[str]:
        refs: List[str] = []
        try:
            root = ET.parse(xml_path).getroot()
        except Exception:
            return refs
        for el in root.iter():
            if lname(el.tag) == "schemaref":
                href = el.attrib.get(
                    "{http://www.w3.org/1999/xlink}href",
                    "",
                )
                if href:
                    refs.append(href)
        self.schema_refs = refs
        return refs

    # -------------------- instance parsing --------------------

    def _parse_contexts(self, root: ET.Element) -> Dict[str, XBRLContext]:
        contexts: Dict[str, XBRLContext] = {}

        for el in root.iter():
            if lname(el.tag) != "context":
                continue

            cid = el.attrib.get("id", "")
            period = next(
                (
                    child for child in list(el)
                    if lname(child.tag) == "period"
                ),
                None,
            )

            start = end = instant = None
            if period is not None:
                for child in list(period):
                    n = lname(child.tag)
                    val = (child.text or "").strip() or None
                    if n == "startdate":
                        start = val
                    elif n == "enddate":
                        end = val
                    elif n == "instant":
                        instant = val
                if instant:
                    end = instant

            dimensions: List[str] = []
            for d in el.iter():
                n = lname(d.tag)
                if n in {"explicitmember", "typedmember"}:
                    text = " ".join((d.text or "").split())
                    if text:
                        dimensions.append(text)

            contexts[cid] = XBRLContext(
                context_id=cid,
                start_date=start,
                end_date=end,
                instant=instant,
                dimensions=dimensions,
                dimension_count=len(dimensions),
            )

        return contexts

    def _parse_units(self, root: ET.Element) -> Dict[str, str]:
        units: Dict[str, str] = {}
        for el in root.iter():
            if lname(el.tag) != "unit":
                continue
            uid = el.attrib.get("id", "")
            pieces = []
            for child in el.iter():
                if child is el:
                    continue
                if child.text and child.text.strip():
                    pieces.append(" ".join(child.text.split()))
            units[uid] = " ".join(pieces)
        return units

    def parse_facts(self, xml_path: Path) -> List[XBRLFact]:
        self.inspect_schema_refs(xml_path)
        root = ET.parse(xml_path).getroot()

        contexts = self._parse_contexts(root)
        units = self._parse_units(root)
        facts: List[XBRLFact] = []

        for el in root.iter():
            if "contextRef" not in el.attrib:
                continue

            raw = (el.text or "").strip()
            value = number(raw)
            if value is None:
                continue

            context_ref = el.attrib.get("contextRef", "")
            ctx = contexts.get(context_ref)
            unit_ref = el.attrib.get("unitRef", "")
            unit = units.get(unit_ref, unit_ref)

            concept = lname(el.tag)
            label = self.taxonomy_labels.get(concept, "")
            dimensions = "|".join(ctx.dimensions) if ctx else ""

            facts.append(
                XBRLFact(
                    concept=concept,
                    qualified_concept=el.tag,
                    value=value,
                    raw_value=raw,
                    unit=unit,
                    unit_ref=unit_ref,
                    context_ref=context_ref,
                    start_date=ctx.start_date if ctx else None,
                    end_date=ctx.end_date if ctx else None,
                    instant=ctx.instant if ctx else None,
                    dimensions=dimensions,
                    dimension_count=ctx.dimension_count if ctx else 0,
                    decimals=el.attrib.get("decimals"),
                    source_file=str(xml_path),
                    label=label,
                    taxonomy_role=self.taxonomy_roles.get(concept, ""),
                )
            )

        return facts

    # -------------------- scoring / units --------------------

    def _period_score(self, fact: XBRLFact, flow: bool = False) -> float:
        score = 0.0

        if fact.end_date == self.target_end_date or fact.instant == self.target_end_date:
            score += 60
        elif fact.end_date and fact.end_date.startswith("2025"):
            score += 20

        if flow and fact.start_date == self.target_start_date and fact.end_date == self.target_end_date:
            score += 35

        if fact.dimension_count == 0:
            score += 12
        else:
            # A dimensional fact can be correct, but consolidated facts are
            # preferred when the target metric is a total.
            score -= min(15, fact.dimension_count * 3)

        return score

    def _unit_norm(self, unit: str) -> str:
        return norm(unit)

    def _convert(self, metric: str, value: float, unit: str) -> Tuple[Optional[float], str]:
        u = self._unit_norm(unit)

        if metric == "Total Energy Consumption":
            if "gigajoule" in u or u in {"gj", "gigajoules"}:
                return value / 3600.0, "GJ->GWh"
            if "megajoule" in u or u == "mj":
                return value / 3_600_000.0, "MJ->GWh"
            if "kwh" in u:
                return value / 1_000_000.0, "kWh->GWh"
            if "gwh" in u:
                return value, "native-GWh"
            return value, "unit-unresolved"

        if metric == "CSR Expenditure":
            if "inr" in u or "indianrupee" in u or u == "rs":
                return value / 10_000_000.0, "INR->crore"
            if "crore" in u:
                return value, "native-crore"
            if "lakh" in u:
                return value / 100.0, "lakh->crore"
            return value, "unit-unresolved"

        if metric.endswith("Percentage") or metric in {
            "Employee Turnover Rate",
            "Female Employee Percentage",
            "Independent Directors Percentage",
            "Female Directors Percentage",
        }:
            if 0 <= value <= 1:
                return value * 100.0, "ratio->percent"
            return value, "native-percent-or-unknown"

        return value, "native"

    def _score_fact(self, fact: XBRLFact, metric: str) -> Tuple[float, str, Optional[float], str]:
        concept = norm(fact.concept)
        label = norm(fact.label)
        combined = f"{concept} {label}"

        rules = self.RULES.get(metric, {})
        score = self._period_score(
            fact,
            flow=metric in {
                "Total Energy Consumption", "Total Water Consumption",
                "Total Waste Generated", "CSR Expenditure",
                "Training Hours per Employee",
                "Employee Turnover Rate", "Board Meeting Frequency",
            },
        )
        methods: List[str] = []

        for token in rules.get("include", ()):
            if norm(token) and norm(token) in combined:
                score += 48
                methods.append(f"semantic:{token}")

        for alias in self.LABELS.get(metric, ()):
            a = norm(alias)
            if a and a in combined:
                score += 32
                methods.append(f"label:{alias}")

        for token in rules.get("exclude", ()):
            t = norm(token)
            if t and t in combined:
                score -= 42

        converted, conversion_method = self._convert(metric, fact.value or 0.0, fact.unit)

        # Unit compatibility.
        target = self.TARGET_UNITS[metric]
        un = self._unit_norm(fact.unit)

        if target == "%":
            if 0 <= (fact.value or 0) <= 1 or "percent" in un or "pure" in un:
                score += 10
        elif target == "Count":
            if "pure" in un or "count" in un or not un:
                score += 6
        elif target == "Hours" and ("hour" in un or "hr" in un):
            score += 10
        elif target == "KL" and ("kilolit" in un or un == "kl"):
            score += 10
        elif target == "MT" and ("metricton" in un or un in {"mt", "ton"}):
            score += 10

        method = ";".join(methods[:3]) if methods else "period/unit-only"
        return score, method, converted, conversion_method

    def _direct_candidates(
        self,
        facts: List[XBRLFact],
        metric: str,
    ) -> List[Tuple[float, XBRLFact, str, Optional[float], str]]:
        out = []
        for fact in facts:
            score, method, converted, conversion = self._score_fact(fact, metric)
            # FIX (v5.1): require real semantic/label evidence. Period +
            # dimension + unit bonuses alone reach ~80 and used to admit ANY
            # fact of the right period (e.g. Board Size = 0, Employees = 1).
            if method == "period/unit-only":
                continue
            if score >= 75:
                out.append((score, fact, method, converted, conversion))
        out.sort(key=lambda x: x[0], reverse=True)
        return out

    # -------------------- derivations --------------------

    def _best_value(
        self,
        facts: List[XBRLFact],
        metric: str,
    ) -> Optional[float]:
        c = self._direct_candidates(facts, metric)
        return c[0][3] if c else None

    def _derive(
        self,
        facts: List[XBRLFact],
        metric: str,
        cache: Dict[str, XBRLMetricResult],
    ) -> Optional[XBRLMetricResult]:

        def direct(name: str) -> Optional[float]:
            if name in cache and cache[name].value is not None:
                return cache[name].value
            return self._best_value(facts, name)

        if metric == "Renewable Energy Percentage":
            renewable = self._best_value(facts, "Renewable Energy Percentage")
            total = direct("Total Energy Consumption")
            # If a direct percentage exists it should already have won.
            if renewable is not None and 0 <= renewable <= 100:
                return None
            if total:
                # Find a renewable quantity-like fact without percentage/intensity.
                candidates = []
                for f in facts:
                    c = norm(f.concept + " " + f.label)
                    if ("renewable" in c and "energy" in c
                            and "percent" not in c and "percentage" not in c
                            and "intensity" not in c):
                        v, _ = self._convert("Total Energy Consumption", f.value or 0, f.unit)
                        if v is not None and 0 <= v <= total * 2:
                            candidates.append((self._period_score(f), v, f))
                if candidates:
                    _, rv, rf = sorted(candidates, reverse=True, key=lambda x: x[0])[0]
                    return XBRLMetricResult(
                        metric, "%", round((rv / total) * 100, 4),
                        "xbrl_derived", rf.concept, rf.context_ref,
                        rf.end_date or rf.instant, 0.78,
                        "Derived", "renewable_energy/total_energy",
                        f"Derived from XBRL facts: {rf.concept} / Total Energy Consumption; "
                        f"unit conversion={self._convert('Total Energy Consumption', rf.value or 0, rf.unit)[1]}."
                    )

        if metric == "Female Employee Percentage":
            total = direct("Total Employees")
            if total and total > 0:
                candidates = []
                for f in facts:
                    c = norm(f.concept + " " + f.label)
                    if ("female" in c or "women" in c) and "employee" in c:
                        if not any(x in c for x in ("percentage", "percent", "wage", "salary")):
                            candidates.append((self._period_score(f), f))
                if candidates:
                    _, f = sorted(candidates, reverse=True, key=lambda x: x[0])[0]
                    if f.value is not None and 0 <= f.value <= total * 1.5:
                        return XBRLMetricResult(
                            metric, "%", round(f.value / total * 100, 4),
                            "xbrl_derived", f.concept, f.context_ref,
                            f.end_date or f.instant, 0.78,
                            "Derived", "female_employees/total_employees",
                            f"Derived from {f.concept} / Total Employees."
                        )

        if metric == "Independent Directors Percentage":
            board = direct("Board Size")
            if board and board > 0:
                candidates = []
                for f in facts:
                    c = norm(f.concept + " " + f.label)
                    if "independent" in c and "director" in c and not any(
                        x in c for x in ("percentage", "percent", "meeting")
                    ):
                        candidates.append((self._period_score(f), f))
                if candidates:
                    _, f = sorted(candidates, reverse=True, key=lambda x: x[0])[0]
                    if f.value is not None and 0 <= f.value <= board * 1.5:
                        return XBRLMetricResult(
                            metric, "%", round(f.value / board * 100, 4),
                            "xbrl_derived", f.concept, f.context_ref,
                            f.end_date or f.instant, 0.78,
                            "Derived", "independent_directors/board_size",
                            f"Derived from {f.concept} / Board Size."
                        )

        if metric == "Female Directors Percentage":
            board = direct("Board Size")
            if board and board > 0:
                candidates = []
                for f in facts:
                    c = norm(f.concept + " " + f.label)
                    if ("female" in c or "women" in c) and "director" in c:
                        if not any(x in c for x in ("percentage", "percent")):
                            candidates.append((self._period_score(f), f))
                if candidates:
                    _, f = sorted(candidates, reverse=True, key=lambda x: x[0])[0]
                    if f.value is not None and 0 <= f.value <= board * 1.5:
                        return XBRLMetricResult(
                            metric, "%", round(f.value / board * 100, 4),
                            "xbrl_derived", f.concept, f.context_ref,
                            f.end_date or f.instant, 0.78,
                            "Derived", "female_directors/board_size",
                            f"Derived from {f.concept} / Board Size."
                        )

        if metric == "Training Hours per Employee":
            employees = direct("Total Employees")
            if employees and employees > 0:
                candidates = []
                for f in facts:
                    c = norm(f.concept + " " + f.label)
                    if "training" in c and "hour" in c and "employee" not in c:
                        if f.value is not None and f.value >= 0:
                            candidates.append((self._period_score(f), f))
                if candidates:
                    _, f = sorted(candidates, reverse=True, key=lambda x: x[0])[0]
                    return XBRLMetricResult(
                        metric, "Hours", round(f.value / employees, 6),
                        "xbrl_derived", f.concept, f.context_ref,
                        f.end_date or f.instant, 0.72,
                        "Derived", "total_training_hours/total_employees",
                        f"Derived from {f.concept} / Total Employees."
                    )

        if metric == "Water Recycled Percentage":
            total = direct("Total Water Consumption")
            if total and total > 0:
                candidates = []
                for f in facts:
                    c = norm(f.concept + " " + f.label)
                    if "water" in c and ("recycled" in c or "reused" in c):
                        if not any(x in c for x in ("percentage", "percent", "intensity")):
                            v, _ = self._convert("Total Water Consumption", f.value or 0, f.unit)
                            if v is not None:
                                candidates.append((self._period_score(f), v, f))
                if candidates:
                    _, rv, f = sorted(candidates, reverse=True, key=lambda x: x[0])[0]
                    return XBRLMetricResult(
                        metric, "%", round(rv / total * 100, 4),
                        "xbrl_derived", f.concept, f.context_ref,
                        f.end_date or f.instant, 0.72,
                        "Derived", "recycled_water/total_water",
                        f"Derived from {f.concept} / Total Water Consumption."
                    )

        if metric == "Waste Recycled Percentage":
            total = direct("Total Waste Generated")
            if total and total > 0:
                candidates = []
                for f in facts:
                    c = norm(f.concept + " " + f.label)
                    if "waste" in c and any(x in c for x in ("recycled", "recovered", "reused")):
                        if not any(x in c for x in ("percentage", "percent", "intensity")):
                            v, _ = self._convert("Total Waste Generated", f.value or 0, f.unit)
                            if v is not None:
                                candidates.append((self._period_score(f), v, f))
                if candidates:
                    _, rv, f = sorted(candidates, reverse=True, key=lambda x: x[0])[0]
                    return XBRLMetricResult(
                        metric, "%", round(rv / total * 100, 4),
                        "xbrl_derived", f.concept, f.context_ref,
                        f.end_date or f.instant, 0.72,
                        "Derived", "recycled_waste/total_waste",
                        f"Derived from {f.concept} / Total Waste Generated."
                    )

        return None

    # -------------------- public matching API --------------------

    def match_metrics(
        self,
        facts: List[XBRLFact],
        metrics: Dict[str, List[Dict[str, Any]]],
    ) -> List[XBRLMetricResult]:

        metric_list = [
            m for group in metrics.values() for m in group
        ]

        results: List[XBRLMetricResult] = []
        cache: Dict[str, XBRLMetricResult] = {}

        # Direct extraction first. This prevents a derived value from
        # overriding an explicit reported KPI.
        for info in metric_list:
            name = info["name"]
            target_unit = info["unit"]
            candidates = self._direct_candidates(facts, name)

            if candidates:
                score, fact, method, converted, conversion = candidates[0]
                confidence = min(0.99, 0.62 + max(0.0, score - 75) / 180)

                result = XBRLMetricResult(
                    metric_name=name,
                    unit=target_unit,
                    value=converted,
                    source="xbrl",
                    concept=fact.concept,
                    context_ref=fact.context_ref,
                    period_end=fact.end_date or fact.instant,
                    confidence=round(confidence, 2),
                    status="Extracted",
                    match_method=method,
                    notes=(
                        f"Direct XBRL fact; label={fact.label or 'unavailable'}; "
                        f"source_unit={fact.unit or 'unspecified'}; "
                        f"normalization={conversion}; "
                        f"dimensions={fact.dimension_count}."
                    ),
                )
            else:
                result = XBRLMetricResult(
                    metric_name=name,
                    unit=target_unit,
                    value=None,
                    source="xbrl",
                    concept=None,
                    context_ref=None,
                    period_end=None,
                    confidence=0.0,
                    status="Not Found",
                    match_method="no_direct_candidate",
                    notes="No sufficiently strong direct XBRL fact matched this metric.",
                )

            results.append(result)
            cache[name] = result

        # Derived pass.
        for i, result in enumerate(results):
            if result.value is not None:
                continue
            derived = self._derive(facts, result.metric_name, cache)
            if derived:
                results[i] = derived
                cache[result.metric_name] = derived

        return results

    def extract_metrics(
        self,
        xml_path: Path,
        metrics: Dict[str, List[Dict[str, Any]]],
        company_code: str = "",
        company_name: str = "",
    ) -> Tuple[List[XBRLMetricResult], List[XBRLFact]]:
        facts = self.parse_facts(xml_path)
        return self.match_metrics(facts, metrics), facts

    # -------------------- inventories / diagnostics --------------------

    @staticmethod
    def save_fact_inventory(
        facts: Iterable[XBRLFact],
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [f.to_dict() for f in facts]
        fields = [
            "concept", "qualified_concept", "value", "raw_value",
            "unit", "unit_ref", "context_ref", "start_date", "end_date",
            "instant", "dimensions", "dimension_count", "decimals",
            "source_file", "label", "taxonomy_role",
        ]
        with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return output_path

    @staticmethod
    def save_metric_results(
        results: Iterable[XBRLMetricResult],
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_dict() for r in results]
        fields = [
            "metric_name", "unit", "value", "source", "concept",
            "context_ref", "period_end", "confidence", "status",
            "match_method", "notes",
        ]
        with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Taxonomy-aware SEBI/NSE BRSR XBRL extractor"
    )
    parser.add_argument("xml", type=Path)
    parser.add_argument("--facts-output", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument(
        "--taxonomy-dir",
        type=Path,
        default=None,
        help="Optional extracted SEBI/NSE BRSR taxonomy directory.",
    )
    args = parser.parse_args()

    # Import only for standalone diagnostics; Agent 1 passes its own config.
    try:
        from utils.config import ESG_METRICS
    except Exception:
        ESG_METRICS = {}

    ex = XBRLExtractor(taxonomy_dir=args.taxonomy_dir)
    facts = ex.parse_facts(args.xml)
    results = ex.match_metrics(facts, ESG_METRICS) if ESG_METRICS else []

    print("=" * 78)
    print("TAXONOMY-AWARE BRSR XBRL EXTRACTOR")
    print("=" * 78)
    print(f"File              : {args.xml}")
    print(f"Facts             : {len(facts)}")
    print(f"Schema references : {len(ex.schema_refs)}")
    for ref in ex.schema_refs:
        print(f"  - {ref}")

    if results:
        found = sum(1 for r in results if r.value is not None)
        derived = sum(1 for r in results if r.value is not None and r.source == "xbrl_derived")
        print(f"Direct/derived    : {found}/18")
        print(f"Derived metrics   : {derived}")
        for r in results:
            print(
                f"{r.metric_name:38s} | "
                f"{str(r.value):>12s} | "
                f"{r.source:12s} | "
                f"{r.concept or '-'}"
            )

    if args.facts_output:
        ex.save_fact_inventory(facts, args.facts_output)
        print(f"\nFact inventory    : {args.facts_output}")

    if args.metrics_output:
        ex.save_metric_results(results, args.metrics_output)
        print(f"Metric inventory   : {args.metrics_output}")


if __name__ == "__main__":
    main()
