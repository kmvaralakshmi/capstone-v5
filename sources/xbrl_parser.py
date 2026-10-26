"""Generic XBRL/XBRL-zip parser with provenance-preserving output."""
from __future__ import annotations
import re, zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import Any


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def parse_xbrl(path: str | Path, company_code: str, company_name: str, run_id: str, financial_year: str = "2024-25") -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    xml_files = []
    if p.suffix.lower() == '.zip':
        with zipfile.ZipFile(p) as z:
            xml_files = [(n, z.read(n)) for n in z.namelist() if n.lower().endswith(('.xml', '.xbrl'))]
    else:
        xml_files = [(p.name, p.read_bytes())]
    records = []
    for filename, raw in xml_files:
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            continue
        facts = []
        for el in root.iter():
            if list(el):
                continue
            name = _local(el.tag)
            text = (el.text or '').strip()
            if not text or name in {'schemaRef', 'linkbaseRef', 'footnoteLink'}:
                continue
            if re.search(r'(emission|energy|water|waste|renew|employee|female|board|director|grievance|training|scope)', name, re.I):
                facts.append({
                    'run_id': run_id, 'company_code': company_code, 'company_name': company_name,
                    'metric_name': name, 'value': text, 'unit': el.attrib.get('unitRef'),
                    'financial_year': financial_year, 'source': 'NSE_XBRL',
                    'source_url': None, 'evidence_source': 'XBRL', 'evidence_locator': filename,
                    'confidence': 'high', 'validation_status': 'parsed', 'fallback_used': False,
                    'discrepancy_flag': False,
                })
        records.extend(facts)
    return records
