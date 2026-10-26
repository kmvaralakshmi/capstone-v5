"""
BRSR PDF Extractor
------------------
Specialized PDF extraction engine for Agent 1.

Responsibilities:
- Extract the configured BRSR ESG metrics from PDF reports.
- Search the complete PDF, not just the first few pages.
- Locate the BRSR section automatically.
- Prefer table-row evidence over loose text matches.
- Select values associated with the target financial year.
- Record page, evidence, extraction method and confidence.
- Handle large/problematic PDFs safely.
- Keep PDF extraction completely separate from Agent 1 orchestration.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.config import ESG_METRICS, TARGET_FINANCIAL_YEAR

try:
    import pdfplumber
except Exception:
    pdfplumber = None


class BRSRPDFExtractor:
    """
    Dedicated BRSR PDF extraction engine.

    Agent 1 should use this class instead of containing PDF parsing logic
    itself.
    """

    # ------------------------------------------------------------------
    # BRSR document markers
    # ------------------------------------------------------------------

    BRSR_MARKERS = (
        "business responsibility and sustainability report",
        "business responsibility & sustainability report",
        "business responsibility sustainability report",
        "brsr",
        "section a: general disclosures",
        "section b: management and process disclosures",
        "section c: principle",
        "principle 1",
        "principle 2",
        "essential indicators",
        "leadership indicators",
    )

    # ------------------------------------------------------------------
    # Metric aliases
    # ------------------------------------------------------------------

    ALIASES = {
        "Total Energy Consumption": [
            "total energy consumption",
            "total energy consumed",
            "energy consumption",
            "total energy",
        ],

        "Renewable Energy Percentage": [
            "renewable energy",
            "renewable sources",
            "percentage of energy consumed from renewable sources",
            "energy consumed from renewable sources",
        ],

        "Total Water Consumption": [
            "total water consumption",
            "water consumption",
            "total water consumed",
        ],

        "Water Recycled Percentage": [
            "water recycled",
            "recycled water",
            "water recycling",
            "percentage of water recycled",
            "total water recycled",
        ],

        "Total Waste Generated": [
            "total waste generated",
            "waste generated",
            "total waste",
        ],

        "Waste Recycled Percentage": [
            "waste recycled",
            "recycled waste",
            "waste recycling",
            "percentage of waste recycled",
        ],

        "GHG Emissions Scope 1": [
            "scope 1 emissions",
            "gross scope 1 emissions",
            "scope 1 ghg emissions",
            "scope 1",
        ],

        "GHG Emissions Scope 2": [
            "scope 2 emissions",
            "gross scope 2 emissions",
            "scope 2 ghg emissions",
            "scope 2",
        ],

        "Total GHG Emissions": [
            "total ghg emissions",
            "total greenhouse gas emissions",
            "greenhouse gas emissions",
            "scope 1 and scope 2",
        ],

        "Female Employee Percentage": [
            "female employees",
            "women employees",
            "female employee",
            "women in workforce",
            "female workforce",
        ],

        "Training Hours per Employee": [
            "training hours per employee",
            "average training hours",
            "training hours",
            "hours of training",
        ],

        "CSR Spending": [
            "csr spending",
            "csr expenditure",
            "amount spent on csr",
            "corporate social responsibility expenditure",
        ],

        "Independent Directors Percentage": [
            "independent directors",
            "independent director",
            "percentage of independent directors",
        ],

        "Female Directors Percentage": [
            "female directors",
            "women directors",
            "women on the board",
            "female members on the board",
        ],

        "Board Meeting Attendance": [
            "board meeting attendance",
            "attendance of directors",
            "board meetings attended",
        ],

        "Ethics/Compliance Training": [
            "ethics training",
            "compliance training",
            "business ethics training",
            "anti-corruption training",
        ],

        "Data Privacy Incidents": [
            "data privacy incidents",
            "data privacy",
            "privacy breaches",
            "data breaches",
        ],

        "Customer Complaints": [
            "customer complaints",
            "consumer complaints",
            "complaints received from customers",
        ],
    }

    NUMBER_RE = re.compile(
        r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?%?"
    )

    def __init__(
        self,
        pdf_path: Path,
        metrics: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        financial_year: str = TARGET_FINANCIAL_YEAR,
    ):
        self.pdf_path = Path(pdf_path)
        self.metrics = metrics or ESG_METRICS
        self.financial_year = financial_year

        self.pages: Dict[int, str] = {}
        self.tables: Dict[int, List] = {}
        self.brsr_pages: List[int] = []

        self._text_loaded = False
        self._tables_loaded = False

    # ==================================================================
    # NORMALIZATION
    # ==================================================================

    @staticmethod
    def _norm(value: Any) -> str:
        if value is None:
            return ""

        text = str(value)
        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text)

        return text.strip().lower()

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        if value is None:
            return None

        text = str(value).strip()

        if not text:
            return None

        if text.lower() in {
            "na",
            "n/a",
            "-",
            "--",
            "nil",
            "none",
            "not applicable",
        }:
            return None

        negative = text.startswith("(") and text.endswith(")")

        text = text.replace(",", "")

        match = re.search(
            r"-?\d+(?:\.\d+)?",
            text
        )

        if not match:
            return None

        try:
            value = float(match.group(0))

            if negative:
                value = -value

            return value

        except ValueError:
            return None

    @staticmethod
    def _plausible(
        value: float,
        unit: str,
    ) -> bool:

        unit = str(unit).strip().lower()

        if unit == "%":
            return 0 <= value <= 100

        if value < 0:
            return False

        # Avoid accidentally selecting years/page numbers.
        if value in {
            2022,
            2023,
            2024,
            2025,
            2026,
            2027,
        }:
            return False

        return True

    # ==================================================================
    # PDF TEXT
    # ==================================================================

    def load_text(self) -> Dict[int, str]:
        """
        Extract text from the complete PDF once.

        This is important because many annual reports contain the BRSR
        hundreds of pages after the beginning of the document.
        """

        if self._text_loaded:
            return self.pages

        if not self.pdf_path.exists():
            return {}

        if pdfplumber is None:
            raise RuntimeError(
                "pdfplumber is required for BRSR PDF extraction."
            )

        pages = {}

        with pdfplumber.open(self.pdf_path) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1,
            ):

                try:
                    text = page.extract_text() or ""

                except Exception:
                    text = ""

                pages[page_number] = text

        self.pages = pages
        self._text_loaded = True

        return pages

    # ==================================================================
    # BRSR SECTION DISCOVERY
    # ==================================================================

    @classmethod
    def _is_brsr_page(
        cls,
        text: str,
    ) -> bool:

        normalized = cls._norm(text)

        if not normalized:
            return False

        if (
            "section a: general disclosures"
            in normalized
        ):
            return True

        if (
            "section b: management and process disclosures"
            in normalized
        ):
            return True

        score = 0

        for marker in cls.BRSR_MARKERS:
            if marker in normalized:
                score += 1

        return score >= 2

    def locate_brsr_pages(self) -> List[int]:
        """
        Find BRSR pages throughout the complete document.

        Does not assume BRSR starts on page 1.
        """

        self.load_text()

        if not self.pages:
            return []

        hits = []

        for page_number, text in self.pages.items():

            if self._is_brsr_page(text):
                hits.append(page_number)

        # Secondary search if primary markers were not sufficient.
        if not hits:

            secondary = []

            secondary_markers = [
                "general disclosures",
                "management and process disclosures",
                "principle 1",
                "principle 2",
                "principle 3",
                "principle 4",
                "principle 5",
                "principle 6",
                "principle 7",
                "principle 8",
                "principle 9",
                "essential indicators",
                "leadership indicators",
            ]

            for page_number, text in self.pages.items():

                normalized = self._norm(text)

                score = sum(
                    marker in normalized
                    for marker in secondary_markers
                )

                if score >= 2:
                    secondary.append(page_number)

            hits = secondary

        # If no BRSR section can be detected, safely fall back to
        # complete-document searching.
        if not hits:
            self.brsr_pages = list(self.pages.keys())
            return self.brsr_pages

        # BRSR pages are normally contiguous. Add a small neighborhood
        # because some pages may have weak text extraction.
        expanded = set(hits)

        for page_number in hits:

            for offset in range(1, 4):

                before = page_number - offset
                after = page_number + offset

                if before >= 1:
                    expanded.add(before)

                if after <= len(self.pages):
                    expanded.add(after)

        self.brsr_pages = sorted(expanded)

        return self.brsr_pages

    # ==================================================================
    # TABLE EXTRACTION
    # ==================================================================

    def load_tables(
        self,
        page_numbers: Sequence[int],
    ) -> Dict[int, List]:

        if pdfplumber is None:
            return {}

        if not self.pages:
            self.load_text()

        if not page_numbers:
            return {}

        # Reuse tables already extracted for this document.
        missing_pages = [
            page_number
            for page_number in page_numbers
            if page_number not in self.tables
        ]

        if not missing_pages:
            return {
                page_number: self.tables.get(page_number, [])
                for page_number in page_numbers
            }

        result = {}

        try:

            with pdfplumber.open(self.pdf_path) as pdf:

                for page_number in missing_pages:

                    if (
                        page_number < 1
                        or page_number > len(pdf.pages)
                    ):
                        continue

                    try:

                        page = pdf.pages[
                            page_number - 1
                        ]

                        tables = (
                            page.extract_tables()
                            or []
                        )

                        result[page_number] = tables

                    except Exception:

                        result[page_number] = []

        except Exception:

            return {}

        self.tables.update(result)

        return result

    # ==================================================================
    # METRIC VOCABULARY
    # ==================================================================

    def aliases_for(
        self,
        metric_name: str,
        metric_info: Dict[str, Any],
    ) -> List[str]:

        aliases = list(
            self.ALIASES.get(
                metric_name,
                [],
            )
        )

        aliases.append(metric_name)

        description = metric_info.get(
            "description",
            "",
        )

        if description:

            pieces = re.split(
                r"[.;,]",
                str(description),
            )

            for piece in pieces:

                piece = piece.strip()

                if len(piece) >= 8:
                    aliases.append(piece)

        normalized = {
            self._norm(alias)
            for alias in aliases
            if alias
        }

        return sorted(
            normalized,
            key=len,
            reverse=True,
        )

    # ==================================================================
    # NUMBER SEARCH
    # ==================================================================

    def numbers_near(
        self,
        text: str,
        keyword: str,
        window: int = 450,
    ) -> List[Tuple[float, str]]:

        normalized = self._norm(text)

        position = normalized.find(
            self._norm(keyword)
        )

        if position < 0:
            return []

        snippet = normalized[
            max(0, position - 100):
            min(
                len(normalized),
                position + window,
            )
        ]

        candidates = []

        for match in self.NUMBER_RE.finditer(
            snippet
        ):

            token = match.group(0)

            value = self._number(token)

            if value is not None:
                candidates.append(
                    (
                        value,
                        token,
                    )
                )

        return candidates

    # ==================================================================
    # TEXT EXTRACTION
    # ==================================================================

    def extract_from_text(
        self,
        metric_name: str,
        metric_info: Dict[str, Any],
        page_numbers: Sequence[int],
    ) -> Optional[Dict[str, Any]]:

        aliases = self.aliases_for(
            metric_name,
            metric_info,
        )

        candidates = []

        target_unit = str(
            metric_info.get(
                "unit",
                "",
            )
        ).strip()

        for page_number in page_numbers:

            text = self.pages.get(
                page_number,
                "",
            )

            if not text:
                continue

            normalized = self._norm(text)

            for alias in aliases:

                if alias not in normalized:
                    continue

                numbers = self.numbers_near(
                    text,
                    alias,
                )

                for value, token in numbers:

                    if not self._plausible(
                        value,
                        target_unit,
                    ):
                        continue

                    score = 40.0

                    # Longer/more specific label.
                    score += min(
                        20.0,
                        len(alias) / 5.0,
                    )

                    # Percentage metric.
                    if target_unit == "%":

                        if "%" in token:
                            score += 30

                        elif 0 <= value <= 100:
                            score += 8

                    else:

                        if "%" not in token:
                            score += 10

                    # Target financial year.
                    year_numbers = re.findall(
                        r"20\d\d",
                        str(
                            self.financial_year
                        ),
                    )

                    if any(
                        year in normalized
                        for year in year_numbers
                    ):
                        score += 12

                    candidates.append(
                        {
                            "value": value,
                            "page": page_number,
                            "keyword": alias,
                            "token": token,
                            "score": score,
                            "evidence": text[:1500],
                            "method": "pdf_text_context",
                        }
                    )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item["score"],
                -item["page"],
            ),
            reverse=True,
        )

        return candidates[0]

    # ==================================================================
    # TABLE EXTRACTION
    # ==================================================================

    def extract_from_tables(
        self,
        metric_name: str,
        metric_info: Dict[str, Any],
        page_numbers: Sequence[int],
    ) -> Optional[Dict[str, Any]]:

        aliases = self.aliases_for(
            metric_name,
            metric_info,
        )

        tables = self.load_tables(
            page_numbers
        )

        candidates = []

        target_unit = str(
            metric_info.get(
                "unit",
                "",
            )
        ).strip()

        for page_number in page_numbers:

            page_tables = tables.get(
                page_number,
                [],
            )

            for table in page_tables:

                if not table:
                    continue

                rows = []

                for row in table:

                    if not row:
                        continue

                    cells = [
                        ""
                        if cell is None
                        else str(cell).strip()
                        for cell in row
                    ]

                    rows.append(cells)

                for row_index, row in enumerate(
                    rows
                ):

                    row_text = " | ".join(
                        cell
                        for cell in row
                        if cell
                    )

                    normalized_row = self._norm(
                        row_text
                    )

                    matched_alias = None

                    for alias in aliases:

                        if alias in normalized_row:
                            matched_alias = alias
                            break

                    if not matched_alias:
                        continue

                    for cell_index, cell in enumerate(
                        row
                    ):

                        for match in self.NUMBER_RE.finditer(
                            cell
                        ):

                            token = match.group(0)

                            value = self._number(
                                token
                            )

                            if value is None:
                                continue

                            if not self._plausible(
                                value,
                                target_unit,
                            ):
                                continue

                            score = 60.0

                            # Exact table-row label.
                            score += min(
                                15.0,
                                len(matched_alias) / 5.0,
                            )

                            # Percentage.
                            if target_unit == "%":

                                if "%" in token:
                                    score += 25

                                elif 0 <= value <= 100:
                                    score += 5

                            else:

                                if "%" not in token:
                                    score += 10

                            # Try to identify the year from header rows.
                            header_parts = []

                            for header_index in range(
                                min(3, len(rows))
                            ):

                                if (
                                    cell_index
                                    < len(
                                        rows[
                                            header_index
                                        ]
                                    )
                                ):

                                    header_parts.append(
                                        rows[
                                            header_index
                                        ][
                                            cell_index
                                        ]
                                    )

                            header_text = self._norm(
                                " ".join(
                                    header_parts
                                )
                            )

                            if (
                                "2024-25"
                                in header_text
                                or "2025"
                                in header_text
                            ):
                                score += 25

                            elif (
                                "2023-24"
                                in header_text
                                or "2024"
                                in header_text
                            ):
                                score -= 5

                            candidates.append(
                                {
                                    "value": value,
                                    "page": page_number,
                                    "keyword": matched_alias,
                                    "token": token,
                                    "score": score,
                                    "evidence": row_text[:2000],
                                    "method": "pdf_table_row",
                                    "row_index": row_index,
                                }
                            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item["score"],
                -item["page"],
            ),
            reverse=True,
        )

        return candidates[0]

    # ==================================================================
    # SINGLE METRIC
    # ==================================================================

    def extract_metric(
        self,
        metric_name: str,
        metric_info: Dict[str, Any],
        page_numbers: Optional[Sequence[int]] = None,
        use_tables: bool = True,
    ) -> Dict[str, Any]:

        started = time.perf_counter()

        try:
            if not self.pages:
                self.load_text()

            pages = (
                list(page_numbers)
                if page_numbers is not None
                else self.locate_brsr_pages()
            )

            if not pages:
                pages = list(self.pages.keys())

            table_result = None

            if use_tables:
                table_result = self.extract_from_tables(
                    metric_name,
                    metric_info,
                    pages,
                )

            text_result = self.extract_from_text(
                metric_name,
                metric_info,
                pages,
            )

            candidates = [
                result
                for result in (table_result, text_result)
                if result is not None
            ]

            if not candidates:
                return {
                    "value": None,
                    "unit": metric_info.get("unit"),
                    "source": "BRSR_PDF",
                    "method": "not_found",
                    "confidence": 0.0,
                    "page": None,
                    "keyword": None,
                    "evidence": None,
                    "timing_sec": time.perf_counter() - started,
                }

            candidates.sort(
                key=lambda item: (
                    item.get("method") == "pdf_table_row",
                    item.get("score", 0),
                ),
                reverse=True,
            )

            best = candidates[0]
            score = float(best.get("score", 0))

            return {
                "value": best["value"],
                "unit": metric_info.get("unit"),
                "source": "BRSR_PDF",
                "method": best.get("method"),
                "confidence": min(0.99, max(0.35, score / 100.0)),
                "page": best.get("page"),
                "keyword": best.get("keyword"),
                "evidence": best.get("evidence"),
                "timing_sec": time.perf_counter() - started,
            }

        except Exception as exc:
            return {
                "value": None,
                "unit": metric_info.get("unit"),
                "source": "BRSR_PDF",
                "method": "error",
                "confidence": 0.0,
                "page": None,
                "keyword": None,
                "evidence": None,
                "error": str(exc),
                "timing_sec": time.perf_counter() - started,
            }


    # ALL METRICS
    # ==================================================================

    def extract_all(
        self,
    ) -> Dict[str, Dict[str, Any]]:

        started = time.perf_counter()

        self.load_text()
        pages = self.locate_brsr_pages()

        if not pages:
            pages = list(self.pages.keys())

        print(
            f"[PDF] Pages scanned      : "
            f"{len(self.pages)}"
        )

        if self.brsr_pages:
            print(
                f"[PDF] BRSR page range   : "
                f"{self.brsr_pages[0]}-"
                f"{self.brsr_pages[-1]}"
            )

        results: Dict[str, Dict[str, Any]] = {}

        # PASS 1: fast text extraction only.
        text_start = time.perf_counter()

        for category, metric_list in self.metrics.items():
            for metric_info in metric_list:
                metric_name = metric_info["name"]

                results[metric_name] = self.extract_metric(
                    metric_name,
                    metric_info,
                    page_numbers=pages,
                    use_tables=False,
                )

        text_time = time.perf_counter() - text_start

        text_found = sum(
            result.get("value") is not None
            for result in results.values()
        )

        print(
            f"[PDF] Text pass        : "
            f"{text_time:.2f} sec "
            f"({text_found}/{len(results)} metrics)"
        )

        # PASS 2: targeted table extraction only for missing metrics.
        missing = [
            metric_name
            for metric_name, result in results.items()
            if result.get("value") is None
        ]

        table_start = time.perf_counter()

        if missing:
            print(
                f"[PDF] Table fallback  : "
                f"{len(missing)} metrics"
            )

            for category, metric_list in self.metrics.items():
                for metric_info in metric_list:
                    metric_name = metric_info["name"]

                    if metric_name not in missing:
                        continue

                    aliases = self.aliases_for(
                        metric_name,
                        metric_info,
                    )

                    candidate_pages = []

                    for page_number in pages:
                        normalized = self._norm(
                            self.pages.get(page_number, "")
                        )

                        if any(
                            alias in normalized
                            for alias in aliases
                            if alias
                        ):
                            candidate_pages.append(page_number)

                    # Include a small neighborhood around matched pages.
                    expanded = set(candidate_pages)

                    for page_number in candidate_pages:
                        for offset in (1, 2):
                            if page_number - offset >= 1:
                                expanded.add(page_number - offset)
                            if page_number + offset <= len(self.pages):
                                expanded.add(page_number + offset)

                    candidate_pages = sorted(expanded)

                    # Do not fall back to all 88 BRSR pages when no
                    # relevant text page was found.
                    if not candidate_pages:
                        continue

                    table_result = self.extract_from_tables(
                        metric_name,
                        metric_info,
                        candidate_pages,
                    )

                    if table_result is not None:
                        score = float(
                            table_result.get("score", 0)
                        )

                        results[metric_name] = {
                            "value": table_result["value"],
                            "unit": metric_info.get("unit"),
                            "source": "BRSR_PDF",
                            "method": table_result.get(
                                "method",
                                "pdf_table_row",
                            ),
                            "confidence": min(
                                0.99,
                                max(0.35, score / 100.0),
                            ),
                            "page": table_result.get("page"),
                            "keyword": table_result.get("keyword"),
                            "evidence": table_result.get("evidence"),
                            "timing_sec": (
                                time.perf_counter()
                                - table_start
                            ),
                        }

        table_time = time.perf_counter() - table_start

        final_found = sum(
            result.get("value") is not None
            for result in results.values()
        )

        print(
            f"[PDF] Table pass       : "
            f"{table_time:.2f} sec"
        )
        print(
            f"[PDF] Final metrics    : "
            f"{final_found}/{len(results)}"
        )

        total_time = time.perf_counter() - started

        for result in results.values():
            result["document_brsr_pages"] = (
                f"{self.brsr_pages[0]}-"
                f"{self.brsr_pages[-1]}"
                if self.brsr_pages
                else None
            )
            result["document_extraction_time_sec"] = total_time

        return results


def extract_brsr_metrics(
    pdf_path: Path,
    metrics: Optional[
        Dict[str, List[Dict[str, Any]]]
    ] = None,
    financial_year: str =
        TARGET_FINANCIAL_YEAR,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function for Agent 1.
    """

    extractor = BRSRPDFExtractor(
        pdf_path=pdf_path,
        metrics=metrics,
        financial_year=financial_year,
    )

    return extractor.extract_all()