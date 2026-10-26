"""BRSR/XBRL filing discovery.

Uses the local official-source registry when a filing is already available, and
supports configurable NSE discovery/download endpoints without hard-coding a
fragile private scraping API. The production endpoint can be supplied through
NSE_BRSR_DISCOVERY_URL_TEMPLATE / NSE_XBRL_DISCOVERY_URL_TEMPLATE.
"""
from __future__ import annotations
import csv, os, re, requests
from pathlib import Path
from typing import Any
from sources.company_resolver import resolve_company
from utils.config import BASE_DIR, BRSR_PDF_DIR, TARGET_FINANCIAL_YEAR

TIMEOUT = int(os.getenv('HTTP_TIMEOUT_SECONDS', '30'))
REGISTRY = BRSR_PDF_DIR / 'brsr_source_registry.csv'

class FilingDiscovery:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'ESG-Multi-Agent-Research/5.0'})

    def _local_registry(self, code: str) -> dict[str, Any] | None:
        if not REGISTRY.exists(): return None
        with REGISTRY.open('r', encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        matches = [r for r in rows if str(r.get('company_code','')).upper() == code]
        if not matches: return None
        return matches[-1]

    def discover(self, company_input: str, financial_year: str = TARGET_FINANCIAL_YEAR) -> dict[str, Any]:
        company = resolve_company(company_input)
        code = company['company_code']
        local = self._local_registry(code) or {}
        brsr_path = BRSR_PDF_DIR / company.get('brsr_file', f'{code}_BR_24-25.pdf')
        result = {
            'run_id': self.run_id, 'company_code': code, 'company_name': company['company_name'],
            'financial_year': financial_year, 'brsr': None, 'xbrl': None, 'source_priority': ['NSE_XBRL','NSE_BRSR','BSE','COMPANY_OFFICIAL'],
            'status': 'discovered_local_or_registry', 'fallback_used': False,
        }
        if brsr_path.exists() and brsr_path.stat().st_size > 50_000:
            result['brsr'] = {'path': str(brsr_path), 'source': 'NSE_BRSR', 'source_url': local.get('source_url') or local.get('url')}
        # XBRL may be pre-cached locally under data/downloads/xbrl.
        xdir = BASE_DIR / 'data' / 'downloads' / 'xbrl'
        xdir.mkdir(parents=True, exist_ok=True)
        candidates = sorted(xdir.glob(f'{code}*'))
        if candidates:
            result['xbrl'] = {'path': str(candidates[-1]), 'source': 'NSE_XBRL'}
        result['xbrl_discovery_url_template'] = os.getenv('NSE_XBRL_DISCOVERY_URL_TEMPLATE')
        result['brsr_discovery_url_template'] = os.getenv('NSE_BRSR_DISCOVERY_URL_TEMPLATE')
        if not result['brsr'] and not result['xbrl']:
            result['status'] = 'not_found'
        return result
