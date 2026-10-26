"""
Discover and download company-specific NSE BRSR XBRL instances.

Run from the capstone-v5 project root:
    python helper-scripts/discover_brsr_xbrl.py --company BSOFT
    python helper-scripts/discover_brsr_xbrl.py --all

This script queries the NSE BRSR CSV endpoint using filing-date windows, parses
its multiline CSV headers safely, filters company rows by the requested FY, then
downloads and validates the XML against the exact XBRL duration context.

It never treats the generic NSE RSS feed as an XBRL instance and never
overwrites an existing file.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from utils.config import COMPANIES, TARGET_FINANCIAL_YEAR
except Exception as exc:
    raise SystemExit(
        f"Run this script inside capstone-v5; cannot import utils.config: {exc}"
    ) from exc

PAGE = (
    "https://www.nseindia.com/companies-listing/"
    "corporate-filings-bussiness-sustainabilitiy-reports"
)
API_URL = "https://www.nseindia.com/api/corporate-bussiness-sustainabilitiy"
XDIR = ROOT / "raw-datasets" / "xbrl"
OUT = ROOT / "processed-data"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/html,application/json,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": PAGE,
}

BAD_URL_PARTS = ("brsr.xml", "/content/rss/", "rss.xml", "feed.xml")
GOOD_URL_HINTS = ("/corporate/xbrl/", ".xml", ".xbrl", "xbrl")


def norm(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def norm_legal_name(value: Any) -> str:
    value_norm = norm(value)
    # Strip common legal suffixes only at the end; this is not fuzzy matching.
    suffixes = ("LIMITED", "LTD", "PRIVATE", "PVT", "PLC", "INCORPORATED", "INC")
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if value_norm.endswith(suffix) and len(value_norm) > len(suffix) + 2:
                value_norm = value_norm[:-len(suffix)]
                changed = True
    return value_norm


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def get_companies() -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    if isinstance(COMPANIES, dict):
        for key, value in COMPANIES.items():
            if isinstance(value, dict):
                result.append((str(key), value))
    elif isinstance(COMPANIES, list):
        for value in COMPANIES:
            if isinstance(value, dict):
                key = value.get("symbol") or value.get("code") or value.get("nse_symbol")
                if key:
                    result.append((str(key), value))
    return result


def target_year_pair() -> tuple[int, int] | None:
    match = re.search(r"(20\d{2})\s*[-/–]\s*(\d{2}|20\d{2})", str(TARGET_FINANCIAL_YEAR))
    if not match:
        return None
    start = int(match.group(1))
    end_token = match.group(2)
    end = int(end_token) if len(end_token) == 4 else (start // 100) * 100 + int(end_token)
    if end < start:
        end += 100
    return start, end


def target_period() -> tuple[str, str] | None:
    pair = target_year_pair()
    if not pair:
        return None
    return f"{pair[0]}-04-01", f"{pair[1]}-03-31"


def target_submission_window() -> tuple[str, str] | None:
    """BRSR for FY Y-(Y+1) is generally filed in the following financial year."""
    pair = target_year_pair()
    if not pair:
        return None
    filing_start = date(pair[1], 4, 1)
    filing_end = date(pair[1] + 1, 3, 31)
    return filing_start.strftime("%d-%m-%Y"), filing_end.strftime("%d-%m-%Y")


def get_context_periods(path: Path) -> list[tuple[str, str]]:
    periods: list[tuple[str, str]] = []
    root = ET.parse(path).getroot()
    for element in root.iter():
        if not isinstance(element.tag, str) or element.tag.rsplit("}", 1)[-1].lower() != "context":
            continue
        start_date = end_date = None
        for child in element.iter():
            if not isinstance(child.tag, str):
                continue
            local = child.tag.rsplit("}", 1)[-1].lower()
            if local == "startdate" and child.text:
                start_date = child.text.strip()
            elif local == "enddate" and child.text:
                end_date = child.text.strip()
        if start_date and end_date:
            periods.append((start_date, end_date))
    return sorted(set(periods))


def inspect_xbrl(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid_structure": False,
        "target_year_match": False,
        "periods": [],
        "context_count": 0,
        "fact_count": 0,
        "entity_identifiers": [],
        "detail": "",
    }
    try:
        if not path.is_file() or path.stat().st_size < 1000:
            result["detail"] = "File missing or too small to be an XBRL instance."
            return result

        root = ET.parse(path).getroot()
        names = {
            el.tag.rsplit("}", 1)[-1].lower()
            for el in root.iter()
            if isinstance(el.tag, str)
        }
        context_elements = [
            el for el in root.iter()
            if isinstance(el.tag, str) and el.tag.rsplit("}", 1)[-1].lower() == "context"
        ]
        fact_elements = [
            el for el in root.iter()
            if isinstance(el.tag, str)
            and any(key.rsplit("}", 1)[-1].lower() == "contextref" for key in el.attrib)
        ]
        has_context = bool(context_elements)
        has_fact = bool(fact_elements)
        entity_identifiers = sorted({
            (child.text or "").strip()
            for context in context_elements
            for child in context.iter()
            if isinstance(child.tag, str)
            and child.tag.rsplit("}", 1)[-1].lower() == "identifier"
            and (child.text or "").strip()
        })
        result["context_count"] = len(context_elements)
        result["fact_count"] = len(fact_elements)
        result["entity_identifiers"] = entity_identifiers
        is_feed = (
            root.tag.rsplit("}", 1)[-1].lower() in {"rss", "feed", "html"}
            or "channel" in names
        )
        if not (has_context and has_fact and not is_feed):
            result["detail"] = "Not a valid XBRL instance (context/fact/feed check failed)."
            return result

        periods = get_context_periods(path)
        expected = target_period()
        result["valid_structure"] = True
        result["periods"] = periods
        result["target_year_match"] = bool(expected and expected in periods)
        if result["target_year_match"]:
            result["detail"] = f"Valid XBRL; exact target duration context found: {expected[0]} to {expected[1]}."
        else:
            sample = ", ".join(f"{a} to {b}" for a, b in periods[:6]) or "no duration contexts"
            result["detail"] = (
                f"Valid XBRL structure, but target FY {TARGET_FINANCIAL_YEAR} "
                f"context was not found. Context sample: {sample}"
            )
        return result
    except (ET.ParseError, OSError, ValueError) as exc:
        result["detail"] = f"XML parse/validation error: {type(exc).__name__}: {exc}"
        return result


def valid_xbrl(path: Path) -> bool:
    check = inspect_xbrl(path)
    return bool(check["valid_structure"] and check["target_year_match"])


def is_candidate_url(url: str) -> bool:
    if not url:
        return False
    decoded = unquote(url).lower()
    host = urlparse(url).netloc.lower()
    if any(part in decoded for part in BAD_URL_PARTS):
        return False
    if not any(hint in decoded for hint in GOOD_URL_HINTS):
        return False
    return not host or host.endswith("nseindia.com") or host.endswith("nseindia.in")


def parse_nse_csv(text: str) -> list[list[str]]:
    """
    NSE CSV has header labels containing embedded newline characters.
    Parse the complete CSV stream and skip its short one-cell header preamble;
    do not slice physical lines before passing them to csv.reader.
    """
    rows: list[list[str]] = []
    reader = csv.reader(io.StringIO(text.lstrip("\ufeff"), newline=""))
    for raw in reader:
        row = [str(value).strip().replace("\r", " ").replace("\n", " ") for value in raw]
        # The seven multiline header cells arrive as seven one-cell rows.
        if len(row) >= 7:
            rows.append(row)
    return rows


def company_name(info: dict[str, Any], code: str) -> str:
    return str(info.get("full_name") or info.get("name") or info.get("company_name") or code)



def company_matches(
    code: str,
    name: str,
    listing_name: str,
    info: dict[str, Any] | None = None,
) -> bool:
    """Match NSE listing names using configured and explicit legal-name aliases.

    Legacy Mindtree is intentionally not mapped to LTIMindtree here:
    both codes exist in the target CSV, and matching them together could
    produce duplicate downloads for the same current company.
    """
    listing = norm_legal_name(listing_name)
    if not listing:
        return False

    allowed_values: set[str] = {code, name}

    # Include aliases configured in the company record.
    for key in ("aliases", "nse_aliases", "former_names", "search_alias"):
        values = (info or {}).get(key, [])
        if isinstance(values, str):
            values = [values]
        if isinstance(values, (list, tuple, set)):
            allowed_values.update(
                str(value).strip()
                for value in values
                if value and str(value).strip()
            )

    # Explicit legal/previous-name variations seen in NSE filing rows.
    explicit_aliases = {
        "AFFLE": [
            "Affle India",
            "Affle (India) Limited",
            "Affle 3i Limited",
            "Affle 3i Limited (Formerly known as Affle (India Limited)",
        ],
        "ZOMATO": [
            "Zomato",
            "Zomato Limited",
            "Eternal",
            "Eternal Limited",
            "Eternal (formerly Zomato)",
        ],
        "HEXAWARE": [
            "Hexaware",
            "Hexaware Technologies Limited",
        ],
        "HCL-INFO": [
            "HCL Infosystems",
            "HCL Infosystems Limited",
            "HCL-INSYS",
        ],
    }

    allowed_values.update(explicit_aliases.get(code.upper(), []))

    normalized_allowed = {
        normalized
        for value in allowed_values
        if (normalized := norm_legal_name(value))
    }

    return listing in normalized_allowed



def fetch_csv_windows(session: requests.Session) -> tuple[list[dict[str, str]], list[str]]:
    """Fetch candidate rows from the target FY filing window, then fallback windows."""
    pair = target_year_pair()
    if not pair:
        return [], ["Cannot parse TARGET_FINANCIAL_YEAR."]
    target_start, target_end = pair
    submission_window = target_submission_window()

    windows: list[tuple[str, str, str]] = []
    if submission_window:
        windows.append(("target_filing_window", submission_window[0], submission_window[1]))
    # Some NSE deployments interpret the date filter as reporting-period years.
    windows.append((
        "target_period_window",
        f"01-04-{target_start}",
        f"31-03-{target_end}",
    ))
    # Final fallback: inspect the complete current CSV response.
    windows.append(("unfiltered", "", ""))

    all_rows: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    diagnostics: list[str] = []
    for label, from_date, to_date in windows:
        params: dict[str, str] = {"index": "equities", "csv": "true"}
        if from_date and to_date:
            params["from_date"] = from_date
            params["to_date"] = to_date
        try:
            response = session.get(API_URL, params=params, timeout=(8, 30))
            response.raise_for_status()
            rows = parse_nse_csv(response.text)
            diagnostics.append(f"{label}: HTTP {response.status_code}, parsed_rows={len(rows)}")
            for values in rows:
                if len(values) < 7:
                    continue
                listing, from_year, to_year, attachment, xbrl, original_date, revision_date = values[:7]
                key = (listing, from_year, to_year, attachment, xbrl)
                all_rows[key] = {
                    "company_name": listing,
                    "from_year": from_year,
                    "to_year": to_year,
                    "attachment": attachment,
                    "xbrl": xbrl,
                    "original_submission_date": original_date,
                    "latest_revision_date": revision_date,
                    "query_window": label,
                }
        except requests.RequestException as exc:
            diagnostics.append(f"{label}: request_failed={type(exc).__name__}: {exc}")
    return list(all_rows.values()), diagnostics


def row_matches_target_year(row: dict[str, str]) -> bool:
    pair = target_year_pair()
    if not pair:
        return False
    try:
        return int(row.get("from_year", "")) == pair[0] and int(row.get("to_year", "")) == pair[1]
    except (TypeError, ValueError):
        return False


def parse_nse_date(value: str) -> date:
    from datetime import datetime
    value = str(value or "").strip()
    if not value or value in {"-", "—", "NA", "N/A"}:
        return date.min
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return date.min


def find_api_candidates(
    code: str,
    info: dict[str, Any],
    session: requests.Session,
) -> tuple[list[dict[str, str]], str]:
    name = company_name(info, code)
    rows, diagnostics = fetch_csv_windows(session)
    company_rows = [
        row for row in rows
        if company_matches(code, name, row.get("company_name", ""), info)
    ]
    target_rows = [row for row in company_rows if row_matches_target_year(row)]
    filing_window = target_submission_window()
    dated_rows: list[dict[str, str]] = []
    rejected_date_rows = 0
    if filing_window:
        start = date.fromisoformat(f"{filing_window[0][6:10]}-{filing_window[0][3:5]}-{filing_window[0][0:2]}")
        end = date.fromisoformat(f"{filing_window[1][6:10]}-{filing_window[1][3:5]}-{filing_window[1][0:2]}")
        for row in target_rows:
            submitted = parse_nse_date(row.get("original_submission_date", ""))
            revised = parse_nse_date(row.get("latest_revision_date", ""))
            if any(d != date.min and start <= d <= end for d in (submitted, revised)):
                dated_rows.append(row)
            else:
                rejected_date_rows += 1
    else:
        dated_rows = target_rows
    target_rows = dated_rows

    candidates: list[dict[str, str]] = []
    for row in target_rows:
        for field, label in (("xbrl", "XBRL"), ("attachment", "Attachment")):
            url = row.get(field, "")
            if is_candidate_url(url):
                candidates.append({
                    "url": url,
                    "company_name": row["company_name"],
                    "from_year": row["from_year"],
                    "to_year": row["to_year"],
                    "original_submission_date": row["original_submission_date"],
                    "latest_revision_date": row["latest_revision_date"],
                    "query_window": row["query_window"],
                    "row_text": (
                        f"{row['company_name']} {row['from_year']} {row['to_year']} "
                        f"submitted={row['original_submission_date']} "
                        f"revised={row['latest_revision_date']} query={row['query_window']}"
                    ),
                    "link_text": label,
                })

    unique: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        unique.setdefault(candidate["url"], candidate)
    candidates = list(unique.values())
    # Prefer XBRL over PDF, then latest revision and original submission dates.
    candidates.sort(key=lambda item: (
        0 if "/corporate/xbrl/" in item["url"].lower() else 1,
        -parse_nse_date(item.get("latest_revision_date", "")).toordinal(),
        -parse_nse_date(item.get("original_submission_date", "")).toordinal(),
        item["url"],
    ))

    company_years = sorted({
        f"{row.get('from_year')}-{str(row.get('to_year', ''))[-2:]}"
        for row in company_rows
    })
    if candidates:
        status = f"api_loaded; company_years={','.join(company_years)}; target_rows={len(target_rows)}"
    elif company_rows:
        status = (
            f"api_loaded; company_years={','.join(company_years)}; "
            f"no_exact_target_year_row={TARGET_FINANCIAL_YEAR}"
        )
    else:
        status = "api_loaded; company_not_in_returned_rows"
    if rejected_date_rows:
        status += f"; rejected_outside_or_unknown_submission_window={rejected_date_rows}"
    return candidates, status + "; " + " | ".join(diagnostics)


def local_valid(code: str, info: dict[str, Any]) -> Path | None:
    if not XDIR.exists():
        return None
    # Respect filename boundaries: SIS must never match MPHASIS.
    code_pattern = re.compile(rf"^{re.escape(str(code).strip())}(?:[_ .-]|$)", re.IGNORECASE)
    paths = [
        path for path in XDIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".xml", ".xbrl"}
    ]
    candidates = sorted(
        (path for path in paths if code_pattern.match(path.stem)),
        key=lambda path: path.name.lower(),
    )
    valid = [path for path in candidates if valid_xbrl(path)]
    # Multiple valid local files are ambiguous; force source discovery/review.
    return valid[0] if len(valid) == 1 else None


def download_valid_xbrl(
    session: requests.Session,
    candidates: list[dict[str, str]],
    dest: Path,
) -> tuple[str, str, Path | None]:
    failures: list[str] = []
    for candidate in candidates:
        url = candidate["url"]
        tmp = dest.with_name(dest.name + ".part")
        try:
            with session.get(url, timeout=(8, 40), stream=True) as response:
                response.raise_for_status()
                size = 0
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(128 * 1024):
                        if chunk:
                            handle.write(chunk)
                            size += len(chunk)

            check = inspect_xbrl(tmp)
            if size < 1000 or not check["valid_structure"] or not check["target_year_match"]:
                tmp.unlink(missing_ok=True)
                failures.append(f"{url}: rejected ({size} bytes); {check['detail']}")
                continue

            final_dest = dest
            if final_dest.exists():
                existing = inspect_xbrl(final_dest)
                if existing["valid_structure"] and existing["target_year_match"]:
                    tmp.unlink(missing_ok=True)
                    return url, f"Existing valid target-year file retained: {final_dest.name}", final_dest
                stem, suffix, counter = dest.stem, dest.suffix, 1
                while final_dest.exists():
                    final_dest = dest.with_name(f"{stem}_{counter}{suffix}")
                    counter += 1

            tmp.replace(final_dest)
            return url, (
                f"Downloaded and validated {size} bytes; {check['detail']}; "
                f"source_row={candidate.get('row_text', '')[:220]}"
            ), final_dest
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
    return "", " | ".join(failures[:4]) or "No candidate XBRL URLs found for the target FY.", None


def discover(
    code: str,
    info: dict[str, Any],
    session: requests.Session,
) -> dict[str, Any]:
    started = time.perf_counter()
    name = company_name(info, code)
    local = local_valid(code, info)
    if local:
        check = inspect_xbrl(local)
        return {
            "company_code": code,
            "company_name": name,
            "status": "LOCAL_VALID",
            "path": str(local.relative_to(ROOT)),
            "url": "",
            "reporting_periods": json.dumps(check["periods"]),
            "detail": check["detail"],
            "elapsed_sec": round(time.perf_counter() - started, 2),
        }

    XDIR.mkdir(parents=True, exist_ok=True)
    dest = XDIR / f"{safe(code)}_BRSR_{str(TARGET_FINANCIAL_YEAR).replace('-', '_')}.xml"
    try:
        candidates, source_status = find_api_candidates(code, info, session)
        url, detail, saved_path = download_valid_xbrl(session, candidates, dest)
        if saved_path:
            check = inspect_xbrl(saved_path)
            return {
                "company_code": code,
                "company_name": name,
                "status": "DOWNLOADED_VALID" if url else "LOCAL_VALID",
                "path": str(saved_path.relative_to(ROOT)),
                "url": url,
                "reporting_periods": json.dumps(check["periods"]),
                "detail": f"{source_status}; {detail}",
                "elapsed_sec": round(time.perf_counter() - started, 2),
            }

        status = "NOT_FOUND" if not candidates else "INVALID_CANDIDATES"
        return {
            "company_code": code,
            "company_name": name,
            "status": status,
            "path": "",
            "url": "",
            "reporting_periods": "",
            "detail": f"{source_status}; {detail}",
            "elapsed_sec": round(time.perf_counter() - started, 2),
        }
    except Exception as exc:
        return {
            "company_code": code,
            "company_name": name,
            "status": "DISCOVERY_ERROR",
            "path": "",
            "url": "",
            "reporting_periods": "",
            "detail": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": round(time.perf_counter() - started, 2),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover valid company-specific NSE BRSR XBRL instances"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--company", help="One configured NSE symbol, e.g. BSOFT")
    group.add_argument("--all", action="store_true", help="Run all configured companies")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between companies")
    args = parser.parse_args()

    items = get_companies()
    if not items:
        raise SystemExit("COMPANIES is not a dict/list of company records in utils.config.")
    if args.company:
        items = [(code, info) for code, info in items if norm(code) == norm(args.company)]
        if not items:
            raise SystemExit(f"{args.company} is not a key/symbol in utils.config.COMPANIES.")

    XDIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"NSE BRSR XBRL discovery | FY={TARGET_FINANCIAL_YEAR} | companies={len(items)}")
    print(f"Required XBRL duration context: {target_period()}")
    filing_window = target_submission_window()
    print(f"Expected submission-date window: {filing_window}")
    rows: list[dict[str, Any]] = []
    path_owner: dict[str, str] = {}

    for index, (code, info) in enumerate(items, 1):
        print(f"\n[{index}/{len(items)}] {code} - {company_name(info, code)}")
        row = discover(code, info, session)
        # Cross-company duplicate-path guard. One physical XML cannot silently
        # count as valid for two different configured companies.
        if row.get("path"):
            path_key = str(Path(row["path"]).resolve()).casefold()
            if path_key in path_owner and path_owner[path_key] != code:
                row["status"] = "DUPLICATE_MAPPING"
                row["detail"] = (
                    f"Validation failed: same XBRL file already assigned to "
                    f"{path_owner[path_key]}; rejected reuse for {code}. " + str(row.get("detail", ""))
                )
            else:
                path_owner[path_key] = code
        row.setdefault("xbrl_structure_valid", False)
        row.setdefault("target_period_valid", False)
        row.setdefault("context_count", 0)
        row.setdefault("fact_count", 0)
        row.setdefault("entity_identifiers", "[]")
        row.setdefault("unique_file_mapping", row.get("status") != "DUPLICATE_MAPPING")
        if row.get("path") and row.get("status") != "DUPLICATE_MAPPING":
            check = inspect_xbrl(ROOT / row["path"])
            row["xbrl_structure_valid"] = bool(check["valid_structure"])
            row["target_period_valid"] = bool(check["target_year_match"])
            row["context_count"] = check.get("context_count", 0)
            row["fact_count"] = check.get("fact_count", 0)
            row["entity_identifiers"] = json.dumps(check.get("entity_identifiers", []))
            row["unique_file_mapping"] = True
        else:
            row["unique_file_mapping"] = False
        rows.append(row)
        print(f"  {row['status']} | {row['elapsed_sec']:.2f}s | {row['path'] or row['detail'][:350]}")
        if index < len(items):
            time.sleep(max(0.0, args.delay))

    fields = [
        "company_code", "company_name", "status", "path", "url",
        "reporting_periods", "xbrl_structure_valid", "target_period_valid",
        "context_count", "fact_count", "entity_identifiers", "unique_file_mapping",
        "detail", "elapsed_sec",
    ]
    csv_path = OUT / "xbrl_discovery_coverage.csv"
    json_path = OUT / "xbrl_discovery_coverage.json"
    validation_path = OUT / "xbrl_validation_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    validation_fields = [
        "company_code", "company_name", "status", "path", "xbrl_structure_valid",
        "target_period_valid", "reporting_periods", "context_count", "fact_count",
        "entity_identifiers", "unique_file_mapping", "url", "detail",
    ]
    with validation_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=validation_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    good = sum(
        row["status"] in {"LOCAL_VALID", "DOWNLOADED_VALID"}
        and row.get("xbrl_structure_valid") is True
        and row.get("target_period_valid") is True
        and row.get("unique_file_mapping") is True
        for row in rows
    )
    downloaded = sum(row["status"] == "DOWNLOADED_VALID" for row in rows)
    not_found = sum(row["status"] == "NOT_FOUND" for row in rows)
    errors = sum(row["status"] in {"DISCOVERY_ERROR", "INVALID_CANDIDATES", "DUPLICATE_MAPPING"} for row in rows)
    print("\nSUMMARY")
    print(
        f"Valid local/downloaded: {good}/{len(rows)} | Newly downloaded: {downloaded} | "
        f"Not found: {not_found} | Errors/invalid candidates: {errors}"
    )
    print(f"Reports: {csv_path} ; {json_path} ; {validation_path}")


if __name__ == "__main__":
    main()
