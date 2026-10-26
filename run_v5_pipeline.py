"""V5 ESG intelligence pipeline.

Modes:
  python run_v5_pipeline.py --company Infosys
  python run_v5_pipeline.py --company INFY --skip-deterministic
  python run_v5_pipeline.py --batch --run-deterministic

The existing deterministic V4 pipeline remains available. V5 adds provenance,
dynamic source orchestration, evidence-grounded explanations, cross-source
reasoning, research validation, and frontend-ready run metadata.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from performance.run_tracker import RunTracker
from sources.company_resolver import resolve_company
from sources.filing_discovery import FilingDiscovery
from sources.external_orchestrator import fetch_external_parallel
from sources.xbrl_parser import parse_xbrl
from sources.xbrl_matcher import match_xbrl_to_brsr
from v5_core.evidence_builder import build_evidence
from v5_core.research_validation import run_validation
from v5_core.greenwashing_agent import GreenwashingAgent
from ai_agents.evidence_explanation import EvidenceExplanationAgent

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'processed-data'; OUT.mkdir(exist_ok=True)


def run_deterministic(tracker: RunTracker):
    with tracker.stage('deterministic_v4_agents'):
        result=subprocess.run([sys.executable,str(ROOT/'demo.py')],cwd=ROOT,text=True,capture_output=True)
        (ROOT/'processed-data'/'v5_deterministic_pipeline.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode!=0: raise RuntimeError(f'Existing deterministic pipeline failed with exit code {result.returncode}')


def save_json(name,payload):
    (OUT/name).write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str),encoding='utf-8')


def analyze_company(company_input: str, run_deterministic_first: bool=False):
    tracker=RunTracker(company=company_input,mode='single')
    try:
        with tracker.stage('company_resolution'):
            company=resolve_company(company_input)
        with tracker.stage('filing_discovery',source='NSE'):
            filings=FilingDiscovery(tracker.run_id).discover(company['company_code'])
        if run_deterministic_first:
            run_deterministic(tracker)
        with tracker.stage('external_fetch_parallel'):
            external=fetch_external_parallel(company)
        with tracker.stage('evidence_package'):
            evidence=build_evidence(company['company_code'],tracker.run_id,external)
            evidence['filing_discovery']=filings
            if filings.get('xbrl'):
                xfacts=parse_xbrl(filings['xbrl']['path'], company['company_code'], company['company_name'], tracker.run_id)
                evidence['xbrl_facts']=xfacts
                evidence['xbrl_brsr_match']=match_xbrl_to_brsr(xfacts, evidence.get('brsr_xbrl_evidence', []))
            else:
                evidence['xbrl_facts']=[]
                evidence['xbrl_brsr_match']=[]
        with tracker.stage('greenwashing_agent'):
            green=GreenwashingAgent().analyze(evidence)
            evidence['greenwashing_llm']=green
        with tracker.stage('explanation_agent',source='OpenAI'):
            explanation=EvidenceExplanationAgent().explain(evidence)
        save_json(f'evidence_{company["company_code"]}.json',evidence)
        save_json(f'explanation_{company["company_code"]}.json',explanation)
        tracker.finish('completed',filing_discovery=filings,explanation_status=explanation.get('status'),greenwashing_status=green.get('status'))
        return {'company':company,'filings':filings,'external':external,'explanation':explanation,'greenwashing':green}
    except Exception as exc:
        tracker.add_error('pipeline',exc); tracker.finish('failed'); raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--company'); p.add_argument('--batch',action='store_true')
    p.add_argument('--run-deterministic',action='store_true',help='Run existing Agents 1-7 before V5 analysis')
    p.add_argument('--skip-deterministic',action='store_true')
    args=p.parse_args()
    if not args.company and not args.batch: p.error('Provide --company NAME/SYMBOL or --batch')
    if args.batch:
        tracker=RunTracker(mode='batch')
        try:
            if not args.skip_deterministic: run_deterministic(tracker)
            with tracker.stage('research_validation'):
                report=run_validation()
            save_json('v5_batch_run_summary.json',{'run_id':tracker.run_id,'validation':report})
            tracker.finish('completed',validation=report)
            print(json.dumps(report,indent=2,default=str))
        except Exception as exc:
            tracker.add_error('batch_pipeline',exc); tracker.finish('failed'); raise
    else:
        result=analyze_company(args.company,run_deterministic_first=args.run_deterministic)
        print(json.dumps(result,indent=2,ensure_ascii=False,default=str))

if __name__=='__main__': main()
