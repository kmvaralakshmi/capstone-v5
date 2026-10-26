"""
Agent 1: XBRL-first BRSR Extraction Agent
------------------------------------------

Agent 1 orchestrates two specialized source extractors:

1. XBRL
   utils/xbrl_extractor.py

2. BRSR PDF
   utils/brsr_extractor.py

XBRL is the primary structured source.

BRSR PDF is used as:
- evidence
- fallback
- independent validation source
- discrepancy detection source

Agent 1 itself is responsible for:
- orchestration
- source comparison
- fallback decision
- validation
- provenance
- timing
- output generation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ----------------------------------------------------------------------
# PROJECT ROOT
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


# ----------------------------------------------------------------------
# PROJECT IMPORTS
# ----------------------------------------------------------------------

from utils.config import (
    COMPANIES,
    ESG_METRICS,
    BRSR_PDF_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_FILES,
    TARGET_FINANCIAL_YEAR,
)

XBRL_DIR = ROOT / "raw-datasets" / "xbrl"
from utils.xbrl_extractor import XBRLExtractor

from utils.brsr_extractor import (
    BRSRPDFExtractor,
)


class Agent1BRSRExtractionAgent:
    """
    Agent 1 orchestration layer.

    The actual XBRL and PDF extraction algorithms live in separate
    specialized modules.
    """

    def __init__(self):

        self.companies = COMPANIES

        self.metrics = ESG_METRICS

        self.brsr_dir = BRSR_PDF_DIR

        self.output_dir = PROCESSED_DATA_DIR

        self.xbrl_dir = XBRL_DIR

        self.run_id = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + "_"
            + uuid.uuid4().hex[:6]
        )

        self.rows: List[
            Dict[str, Any]
        ] = []

        self.timing_rows: List[
            Dict[str, Any]
        ] = []

        self.run_records: List[
            Dict[str, Any]
        ] = []

    # ==================================================================
    # METRIC DEFINITIONS
    # ==================================================================

    def metric_definitions(
        self,
    ) -> List[Dict[str, Any]]:

        definitions = []

        for category, metrics in (
            self.metrics.items()
        ):

            for metric in metrics:

                item = dict(metric)

                item["category"] = (
                    category
                )

                definitions.append(
                    item
                )

        return definitions

    # ==================================================================
    # VALUE NORMALIZATION
    # ==================================================================

    @staticmethod
    def get_value(
        result: Any,
    ) -> Optional[float]:

        if result is None:
            return None

        if isinstance(
            result,
            dict,
        ):

            value = result.get(
                "value"
            )

            if value is None:
                value = result.get(
                    "Metric_Value"
                )

        else:

            value = result

        try:

            if value is None:
                return None

            if pd.isna(value):
                return None

            return float(value)

        except (
            TypeError,
            ValueError,
        ):

            return None

    @staticmethod
    def relative_difference(
        first: float,
        second: float,
    ) -> float:

        denominator = max(
            abs(first),
            abs(second),
            1e-9,
        )

        return abs(
            first - second
        ) / denominator

    # ==================================================================
    # XBRL RESULT NORMALIZATION
    # ==================================================================

    def normalize_xbrl_results(
        self,
        results: Any,
    ) -> Dict[str, Dict[str, Any]]:

        normalized = {}

        if not isinstance(
            results,
            dict,
        ):
            return normalized

        for metric_name, result in (
            results.items()
        ):

            if isinstance(
                result,
                dict,
            ):

                normalized[
                    metric_name
                ] = dict(result)

            else:

                normalized[
                    metric_name
                ] = {
                    "value": result
                }

        return normalized

    # ==================================================================
    # XBRL SOURCE
    # ==================================================================

    def run_xbrl(
        self,
        company_code: str,
        company_info: Dict[str, Any],
    ) -> Dict[str, Any]:

        started = time.perf_counter()

        try:
            # The V3 XBRLExtractor accepts taxonomy_dir as its constructor
            # argument. Company-specific XML is supplied to extract_metrics().
            taxonomy_dir = self.xbrl_dir / "taxonomy"

            xml_candidates = sorted(
                self.xbrl_dir.glob(f"{company_code}_*.xml"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            # Also support filenames that contain the company code but do not
            # start with it.
            if not xml_candidates:
                xml_candidates = sorted(
                    [
                        p for p in self.xbrl_dir.glob("*.xml")
                        if company_code.lower() in p.name.lower()
                    ],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )

            if not xml_candidates:
                return {
                    "status": "NOT_FOUND",
                    "metrics": {},
                    "timing_sec": time.perf_counter() - started,
                    "error": f"No XBRL instance found for {company_code} in {self.xbrl_dir}",
                }

            xml_path = xml_candidates[0]

            extractor = XBRLExtractor(
                target_year=TARGET_FINANCIAL_YEAR,
                taxonomy_dir=(
                    taxonomy_dir
                    if taxonomy_dir.exists()
                    else None
                ),
            )

            results, facts = extractor.extract_metrics(
                xml_path=xml_path,
                metrics=self.metrics,
                company_code=company_code,
                company_name=company_info.get(
                    "full_name",
                    company_code,
                ),
            )

            normalized = {}

            for result in results:
                item = result.to_dict()

                # Keep Agent 1's canonical interface while preserving the
                # richer XBRL extractor fields.
                item["value"] = item.get("value")
                item["source"] = item.get("source", "xbrl")
                item["confidence"] = item.get("confidence", 0.0)
                item["method"] = item.get(
                    "match_method",
                    "xbrl_direct",
                )
                item["evidence"] = item.get("notes")
                item["unit"] = item.get("unit")
                item["concept"] = item.get("concept")
                item["context_ref"] = item.get("context_ref")
                item["period_end"] = item.get("period_end")

                normalized[item["metric_name"]] = item

            # Save a fact inventory for audit/debugging.
            facts_output = (
                self.xbrl_dir.parent.parent
                / "processed-data"
                / "xbrl"
                / f"{company_code}_xbrl_facts.csv"
            )
            extractor.save_fact_inventory(
                facts,
                facts_output,
            )

            return {
                "status": "SUCCESS",
                "metrics": normalized,
                "timing_sec": time.perf_counter() - started,
                "error": None,
                "xml_path": str(xml_path),
                "fact_count": len(facts),
                "facts_output": str(facts_output),
            }

        except Exception as exc:
            return {
                "status": "ERROR",
                "metrics": {},
                "timing_sec": time.perf_counter() - started,
                "error": str(exc),
            }

    # ==================================================================
    # BRSR PDF SOURCE
    # ==================================================================

    def run_pdf(
        self,
        company_code: str,
        company_info: Dict[str, Any],
    ) -> Dict[str, Any]:

        started = time.perf_counter()

        pdf_name = company_info.get(
            "brsr_file"
        )

        if not pdf_name:

            return {
                "status": "NOT_FOUND",
                "metrics": {},
                "timing_sec":
                    time.perf_counter()
                    - started,
                "error":
                    "No BRSR PDF configured.",
            }

        pdf_path = (
            self.brsr_dir
            / pdf_name
        )

        if not pdf_path.exists():

            return {
                "status": "NOT_FOUND",
                "metrics": {},
                "timing_sec":
                    time.perf_counter()
                    - started,
                "error":
                    f"PDF not found: "
                    f"{pdf_path}",
            }

        try:

            extractor = BRSRPDFExtractor(
                pdf_path=pdf_path,
                metrics=self.metrics,
                financial_year=
                    TARGET_FINANCIAL_YEAR,
            )

            results = (
                extractor.extract_all()
            )

            return {
                "status": "SUCCESS",
                "metrics": results,
                "timing_sec":
                    time.perf_counter()
                    - started,
                "error": None,
            }

        except Exception as exc:

            return {
                "status": "ERROR",
                "metrics": {},
                "timing_sec":
                    time.perf_counter()
                    - started,
                "error": str(exc),
            }

    # ==================================================================
    # FINAL METRIC DECISION
    # ==================================================================

    def build_metric_row(
        self,
        company_code: str,
        company_info: Dict[str, Any],
        metric: Dict[str, Any],
        xbrl: Dict[str, Any],
        pdf: Dict[str, Any],
    ) -> Dict[str, Any]:

        metric_name = metric[
            "name"
        ]

        xbrl_result = xbrl.get(
            metric_name,
            {},
        )

        pdf_result = pdf.get(
            metric_name,
            {},
        )

        xbrl_value = (
            self.get_value(
                xbrl_result
            )
        )

        pdf_value = (
            self.get_value(
                pdf_result
            )
        )

        # --------------------------------------------------------------
        # DISCREPANCY DETECTION
        # --------------------------------------------------------------

        discrepancy = False

        discrepancy_reason = None

        if (
            xbrl_value is not None
            and pdf_value is not None
        ):

            unit = str(
                metric.get(
                    "unit",
                    "",
                )
            ).strip()

            if unit == "%":

                # Percentage differences are measured in percentage
                # points.
                discrepancy = (
                    abs(
                        xbrl_value
                        - pdf_value
                    )
                    > 2.0
                )

            else:

                # For absolute quantities allow 5% difference.
                discrepancy = (
                    self.relative_difference(
                        xbrl_value,
                        pdf_value,
                    )
                    > 0.05
                )

            if discrepancy:

                discrepancy_reason = (
                    "XBRL_PDF_VALUE_MISMATCH"
                )

        # --------------------------------------------------------------
        # SOURCE PRIORITY
        # --------------------------------------------------------------

        if xbrl_value is not None:

            final_value = (
                xbrl_value
            )

            final_source = "XBRL"

            final_method = (
                xbrl_result.get(
                    "method",
                    "xbrl_direct",
                )
            )

            confidence = float(
                xbrl_result.get(
                    "confidence",
                    xbrl_result.get(
                        "Confidence_Score",
                        0.90,
                    ),
                )
                or 0.90
            )

            fallback_used = False

        elif pdf_value is not None:

            final_value = (
                pdf_value
            )

            final_source = (
                "BRSR_PDF"
            )

            final_method = (
                pdf_result.get(
                    "method",
                    "pdf_fallback",
                )
            )

            confidence = float(
                pdf_result.get(
                    "confidence",
                    0.70,
                )
                or 0.70
            )

            fallback_used = True

        else:

            final_value = None

            final_source = (
                "NOT_FOUND"
            )

            final_method = (
                "not_found"
            )

            confidence = 0.0

            fallback_used = False

        # --------------------------------------------------------------
        # VALIDATION STATUS
        # --------------------------------------------------------------

        if final_value is None:

            validation_status = (
                "NOT_FOUND"
            )

        elif discrepancy:

            validation_status = (
                "DISCREPANCY_REVIEW"
            )

        else:

            validation_status = (
                "VALID"
            )

        return {
            "Run_ID":
                self.run_id,

            "Company_Code":
                company_code,

            "Company_Name":
                company_info.get(
                    "full_name"
                ),

            "Financial_Year":
                TARGET_FINANCIAL_YEAR,

            "Metric_Category":
                metric.get(
                    "category"
                ),

            "Metric_Name":
                metric_name,

            "Unit":
                metric.get(
                    "unit"
                ),

            "Metric_Value":
                final_value,

            "Final_Source":
                final_source,

            "Extraction_Method":
                final_method,

            "Confidence_Score":
                round(
                    confidence,
                    4,
                ),

            "Validation_Status":
                validation_status,

            "Fallback_Used":
                fallback_used,

            # ----------------------------------------------------------
            # XBRL PROVENANCE
            # ----------------------------------------------------------

            "XBRL_Value":
                xbrl_value,

            "XBRL_Source":
                (
                    xbrl_result.get(
                        "source",
                        "NSE_XBRL",
                    )
                    if xbrl_value
                    is not None
                    else None
                ),

            "XBRL_Confidence":
                (
                    xbrl_result.get(
                        "confidence"
                    )
                    if xbrl_value
                    is not None
                    else None
                ),

            "XBRL_Evidence":
                (
                    xbrl_result.get(
                        "evidence"
                    )
                    if xbrl_value
                    is not None
                    else None
                ),

            # ----------------------------------------------------------
            # PDF PROVENANCE
            # ----------------------------------------------------------

            "PDF_Value":
                pdf_value,

            "PDF_Page":
                pdf_result.get(
                    "page"
                ),

            "PDF_Keyword":
                pdf_result.get(
                    "keyword"
                ),

            "PDF_Method":
                pdf_result.get(
                    "method"
                ),

            "PDF_Confidence":
                pdf_result.get(
                    "confidence"
                ),

            "PDF_Evidence":
                pdf_result.get(
                    "evidence"
                ),

            # ----------------------------------------------------------
            # VALIDATION
            # ----------------------------------------------------------

            "Discrepancy":
                discrepancy,

            "Discrepancy_Reason":
                discrepancy_reason,

            "Needs_Manual_Review":
                (
                    discrepancy
                    or final_value
                    is None
                ),

            "Source_Priority":
                "XBRL > BRSR_PDF",
        }

    # ==================================================================
    # PROCESS COMPANY
    # ==================================================================

    def process_company(
        self,
        company_code: str,
        company_info: Dict[str, Any],
    ) -> None:

        company_started = (
            time.perf_counter()
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"PROCESSING: "
            f"{company_code} - "
            f"{company_info.get('full_name', company_code)}"
        )

        print(
            "=" * 70
        )

        # --------------------------------------------------------------
        # XBRL
        # --------------------------------------------------------------

        print(
            "\n[XBRL] "
            "Primary structured source..."
        )

        xbrl_result = (
            self.run_xbrl(
                company_code,
                company_info,
            )
        )

        xbrl_count = sum(
            self.get_value(
                xbrl_result[
                    "metrics"
                ].get(
                    metric["name"]
                )
            )
            is not None
            for metric in
            self.metric_definitions()
        )

        print(
            f"  XBRL status    : "
            f"{xbrl_result['status']}"
        )

        print(
            f"  XBRL metrics   : "
            f"{xbrl_count}/18"
        )

        print(
            f"  XBRL time      : "
            f"{xbrl_result['timing_sec']:.2f} sec"
        )

        if xbrl_result["error"]:
            print(
                f"  XBRL error     : "
                f"{xbrl_result['error']}"
            )

        # --------------------------------------------------------------
        # PDF
        # --------------------------------------------------------------

        print(
            "\n[PDF] "
            "BRSR evidence/fallback..."
        )

        pdf_result = (
            self.run_pdf(
                company_code,
                company_info,
            )
        )

        pdf_count = sum(
            self.get_value(
                pdf_result[
                    "metrics"
                ].get(
                    metric["name"]
                )
            )
            is not None
            for metric in
            self.metric_definitions()
        )

        print(
            f"  PDF status     : "
            f"{pdf_result['status']}"
        )

        print(
            f"  PDF metrics    : "
            f"{pdf_count}/18"
        )

        print(
            f"  PDF time       : "
            f"{pdf_result['timing_sec']:.2f} sec"
        )

        if pdf_result["error"]:
            print(
                f"  PDF error      : "
                f"{pdf_result['error']}"
            )

        # --------------------------------------------------------------
        # FINAL METRIC DECISIONS
        # --------------------------------------------------------------

        company_rows = []

        for metric in (
            self.metric_definitions()
        ):

            row = (
                self.build_metric_row(
                    company_code,
                    company_info,
                    metric,
                    xbrl_result[
                        "metrics"
                    ],
                    pdf_result[
                        "metrics"
                    ],
                )
            )

            company_rows.append(
                row
            )

            self.rows.append(
                row
            )

        final_count = sum(
            row["Metric_Value"]
            is not None
            for row in company_rows
        )

        fallback_count = sum(
            bool(
                row["Fallback_Used"]
            )
            for row in company_rows
        )

        discrepancy_count = sum(
            bool(
                row["Discrepancy"]
            )
            for row in company_rows
        )

        total_time = (
            time.perf_counter()
            - company_started
        )

        # --------------------------------------------------------------
        # TIMING RECORD
        # --------------------------------------------------------------

        self.timing_rows.append(
            {
                "Run_ID":
                    self.run_id,

                "Company_Code":
                    company_code,

                "Company_Name":
                    company_info.get(
                        "full_name"
                    ),

                "Financial_Year":
                    TARGET_FINANCIAL_YEAR,

                "XBRL_Time_Sec":
                    round(
                        xbrl_result[
                            "timing_sec"
                        ],
                        4,
                    ),

                "PDF_Time_Sec":
                    round(
                        pdf_result[
                            "timing_sec"
                        ],
                        4,
                    ),

                "Company_Total_Time_Sec":
                    round(
                        total_time,
                        4,
                    ),

                "XBRL_Metrics":
                    xbrl_count,

                "PDF_Metrics":
                    pdf_count,

                "Final_Metrics":
                    final_count,

                "PDF_Fallback_Metrics":
                    fallback_count,

                "Discrepancies":
                    discrepancy_count,

                "Status":
                    (
                        "SUCCESS"
                        if final_count
                        else
                        "NO_METRICS"
                    ),
            }
        )

        # --------------------------------------------------------------
        # RUN RECORD
        # --------------------------------------------------------------

        self.run_records.append(
            {
                "run_id":
                    self.run_id,

                "company_code":
                    company_code,

                "company_name":
                    company_info.get(
                        "full_name"
                    ),

                "financial_year":
                    TARGET_FINANCIAL_YEAR,

                "xbrl": {
                    "status":
                        xbrl_result[
                            "status"
                        ],

                    "time_sec":
                        xbrl_result[
                            "timing_sec"
                        ],

                    "metrics":
                        xbrl_count,

                    "error":
                        xbrl_result[
                            "error"
                        ],
                },

                "pdf": {
                    "status":
                        pdf_result[
                            "status"
                        ],

                    "time_sec":
                        pdf_result[
                            "timing_sec"
                        ],

                    "metrics":
                        pdf_count,

                    "error":
                        pdf_result[
                            "error"
                        ],
                },

                "final_metrics":
                    final_count,

                "pdf_fallback_metrics":
                    fallback_count,

                "discrepancies":
                    discrepancy_count,

                "company_total_time_sec":
                    total_time,
            }
        )

        # --------------------------------------------------------------
        # CONSOLE SUMMARY
        # --------------------------------------------------------------

        print(
            "\nFINAL SOURCE DECISION"
        )

        print(
            f"  XBRL primary metrics : "
            f"{xbrl_count}/18"
        )

        print(
            f"  PDF evidence metrics : "
            f"{pdf_count}/18"
        )

        print(
            f"  PDF fallback metrics : "
            f"{fallback_count}/18"
        )

        print(
            f"  Final metrics        : "
            f"{final_count}/18"
        )

        print(
            f"  Discrepancies        : "
            f"{discrepancy_count}"
        )

        print(
            f"  Company total        : "
            f"{total_time:.2f} sec"
        )

    # ==================================================================
    # OUTPUTS
    # ==================================================================

    def save_outputs(
        self,
    ) -> None:

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metrics_filename = (
            OUTPUT_FILES.get(
                "brsr_metrics",
                "brsr_extracted_metrics.csv",
            )
        )

        metrics_path = (
            self.output_dir
            / metrics_filename
        )

        timing_path = (
            self.output_dir
            / "agent1_extraction_timing.csv"
        )

        pd.DataFrame(
            self.rows
        ).to_csv(
            metrics_path,
            index=False,
        )

        pd.DataFrame(
            self.timing_rows
        ).to_csv(
            timing_path,
            index=False,
        )

        runs_dir = (
            self.output_dir
            / "agent1_runs"
        )

        runs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        run_path = (
            runs_dir
            / f"agent1_{self.run_id}.json"
        )

        run_path.write_text(
            json.dumps(
                self.run_records,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        print(
            "\nOUTPUTS:"
        )

        print(
            f"  {metrics_path}"
        )

        print(
            f"  {timing_path}"
        )

        print(
            f"  {run_path}"
        )

    # ==================================================================
    # RUN
    # ==================================================================

    def run(
        self,
        company_code: Optional[str] = None,
    ) -> None:

        started = (
            time.perf_counter()
        )

        if company_code:

            code = (
                company_code
                .upper()
                .strip()
            )

            if code not in self.companies:

                raise ValueError(
                    f"Unknown company "
                    f"'{company_code}'."
                )

            companies = {
                code:
                    self.companies[
                        code
                    ]
            }

            mode = (
                f"SINGLE COMPANY "
                f"({code})"
            )

        else:

            companies = (
                self.companies
            )

            mode = (
                f"BATCH "
                f"({len(companies)} companies)"
            )

        print(
            "=" * 70
        )

        print(
            "AGENT 1: "
            "XBRL-FIRST BRSR "
            "EXTRACTION AGENT"
        )

        print(
            "=" * 70
        )

        print(
            "Primary source : NSE XBRL"
        )

        print(
            "Fallback       : BRSR PDF"
        )

        print(
            f"Financial year : "
            f"{TARGET_FINANCIAL_YEAR}"
        )

        print(
            "Metrics        : 18"
        )

        print(
            f"Mode           : {mode}"
        )

        print(
            f"Run ID         : "
            f"{self.run_id}"
        )

        print(
            "=" * 70
        )

        for company_code, company_info in (
            companies.items()
        ):

            self.process_company(
                company_code,
                company_info,
            )

        self.save_outputs()

        total_time = (
            time.perf_counter()
            - started
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "AGENT 1 OVERALL PERFORMANCE"
        )

        print(
            "=" * 70
        )

        print(
            f"Companies processed : "
            f"{len(companies)}"
        )

        print(
            f"Total time          : "
            f"{total_time:.2f} sec"
        )

        if self.rows:

            valid = sum(
                row["Metric_Value"]
                is not None
                for row in self.rows
            )

            discrepancies = sum(
                bool(
                    row["Discrepancy"]
                )
                for row in self.rows
            )

            print(
                f"Metric values       : "
                f"{valid}/{len(self.rows)}"
            )

            print(
                f"Discrepancies       : "
                f"{discrepancies}"
            )


# ======================================================================
# CLI
# ======================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Agent 1: XBRL-first "
            "BRSR extraction"
        )
    )

    parser.add_argument(
        "--company",
        help=(
            "Run a single company, "
            "for example INFY"
        ),
        default=None,
    )

    args = parser.parse_args()

    agent = (
        Agent1BRSRExtractionAgent()
    )

    agent.run(
        company_code=args.company
    )


if __name__ == "__main__":

    main()