"""Build compact, provenance-rich evidence packages for the explanation agent."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from utils.config import PROCESSED_DATA_DIR, OUTPUT_FILES


def _read(name):
    p=PROCESSED_DATA_DIR/name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def _rows(df, code):
    if df.empty: return []
    col='Company_Code' if 'Company_Code' in df.columns else None
    return df[df[col].astype(str).str.upper()==code.upper()].head(40).to_dict('records') if col else []

def build_evidence(company_code: str, run_id: str, external: dict | None = None) -> dict:
    scores=_read(OUTPUT_FILES['master_scores']); metrics=_read(OUTPUT_FILES['brsr_metrics'])
    env=_read(OUTPUT_FILES['environmental_risk']); news=_read(OUTPUT_FILES['news_sentiment'])
    stock=_read(OUTPUT_FILES['stock_correlation']); green=_read(OUTPUT_FILES['greenwashing'])
    row=scores[scores.get('Company_Code',pd.Series(dtype=str)).astype(str).str.upper()==company_code.upper()]
    summary=row.iloc[0].to_dict() if not row.empty else {}
    return {
        'run_id':run_id,'company_code':company_code,'company_name':summary.get('Company_Name'),
        'score_summary':summary,
        'brsr_xbrl_evidence':_rows(metrics,company_code),
        'environment_context':_rows(env,company_code),
        'news_analysis':_rows(news,company_code),
        'stock_analysis':_rows(stock,company_code),
        'greenwashing_analysis':_rows(green,company_code),
        'external_sources':external or {},
    }
