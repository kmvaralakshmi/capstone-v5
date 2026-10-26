"""Cross-source Greenwashing Agent using evidence-grounded LLM reasoning."""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from dotenv import load_dotenv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; load_dotenv(ROOT/'.env')
PROMPT="""You are a cross-source ESG consistency investigator. Compare only the supplied company claims, BRSR/XBRL evidence, historical metrics, news, market signals and environmental context. Do not accuse a company of greenwashing as a fact. Classify the evidence as CONSISTENT, POTENTIAL_INCONSISTENCY, or INSUFFICIENT_EVIDENCE. Explain which supplied facts support the classification and list missing evidence. Never infer causation from stock/news/AQI correlation."""
class GreenwashingAgent:
    def __init__(self):
        self.key=os.getenv('OPENAI_API_KEY'); self.model=os.getenv('OPENAI_MODEL'); self.client=OpenAI(api_key=self.key) if self.key else None
    def analyze(self,evidence):
        if not self.client: return {'status':'skipped','classification':'INSUFFICIENT_EVIDENCE','reason':'OPENAI_API_KEY is not configured'}
        r=self.client.responses.create(model=self.model,instructions=PROMPT,input=json.dumps(evidence,ensure_ascii=False,default=str))
        return {'status':'success','model':self.model,'classification':'LLM_ASSESSMENT','analysis':r.output_text.strip(),'generated_at':datetime.now(timezone.utc).isoformat()}
