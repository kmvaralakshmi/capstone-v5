"""Evidence-grounded Explanation Agent.

The LLM interprets structured evidence; it does not calculate the ESG score.
Only evidence supplied in the payload may be used in the explanation.
"""
from __future__ import annotations
import json, os, re
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

ROOT=Path(__file__).resolve().parents[1]
load_dotenv(ROOT/'.env')

SYSTEM_PROMPT="""You are the Evidence-grounded Explanation Agent in an ESG research system.
Your task is to explain an already-computed ESG analysis using ONLY the supplied evidence.
Do not invent values, dates, sources, events, causes, or recommendations that are not supported by the evidence.
Do not recalculate the master ESG score. Clearly distinguish reported facts from interpretation.
If evidence is missing, say that it is unavailable. Mention source/provenance when supplied.
Return concise professional prose suitable for an ESG research dashboard with these headings:
Overall assessment; Key strengths; Key weaknesses; Cross-source signals; Greenwashing/consistency; Recommended focus; Evidence confidence.
Avoid causal claims from correlations. Air quality is contextual and must never be described as proof of company ESG performance.
"""

class EvidenceExplanationAgent:
    def __init__(self, model: str | None = None):
        self.model=model or os.getenv('OPENAI_MODEL')
        self.api_key=os.getenv('OPENAI_API_KEY')
        self.client=OpenAI(api_key=self.api_key) if (self.api_key and OpenAI is not None) else None

    def explain(self, evidence: dict) -> dict:
        if not self.client:
            return {'status':'skipped','reason':'OPENAI_API_KEY is not configured','model':self.model,'explanation':None}
        payload=json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
        response=self.client.responses.create(model=self.model, instructions=SYSTEM_PROMPT, input=payload)
        return {
            'status':'success','model':self.model,'response_id':getattr(response,'id',None),
            'explanation':getattr(response,'output_text','').strip(),
            'generated_at':datetime.now(timezone.utc).isoformat()
        }
