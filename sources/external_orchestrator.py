"""Dependency-aware external data orchestration with caching and fallbacks."""
from __future__ import annotations
import json, os, re, time, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import requests
from utils.config import BASE_DIR

CACHE = BASE_DIR / 'data' / 'cache'; CACHE.mkdir(parents=True, exist_ok=True)
TIMEOUT = int(os.getenv('HTTP_TIMEOUT_SECONDS', '20'))

class Cache:
    def __init__(self, name: str): self.dir = CACHE / name; self.dir.mkdir(parents=True, exist_ok=True)
    def path(self, key: str) -> Path: return self.dir / (hashlib.sha256(key.encode()).hexdigest() + '.json')
    def get(self, key: str, max_age: int = 3600):
        p=self.path(key)
        if not p.exists() or time.time()-p.stat().st_mtime > max_age: return None
        try: return json.loads(p.read_text(encoding='utf-8'))
        except Exception: return None
    def put(self, key: str, value):
        p=self.path(key); p.write_text(json.dumps(value, ensure_ascii=False, default=str), encoding='utf-8'); return p

def _request(url, params=None, headers=None):
    r=requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
    r.raise_for_status(); return r

def fetch_gdelt(company_name: str, lookback_hours: int = 168) -> dict:
    cache=Cache('news'); key=f'gdelt:{company_name}:{lookback_hours}'
    cached=cache.get(key, 900)
    if cached: cached['cached']=True; return cached
    query=f'"{company_name}" (ESG OR sustainability OR emissions OR governance OR environment OR social)'
    url='https://api.gdeltproject.org/api/v2/doc/doc'
    params={'query':query,'mode':'ArtList','format':'json','maxrecords':50,'sort':'HybridRel','timespan':f'{lookback_hours}h'}
    try:
        data=_request(url,params=params).json(); out={'source':'GDELT','status':'success','retrieved_at':datetime.now(timezone.utc).isoformat(),'articles':data.get('articles',[]),'query':query,'cached':False}
    except Exception as exc: out={'source':'GDELT','status':'error','error':str(exc),'articles':[],'cached':False}
    cache.put(key,out); return out

def fetch_cpcb(city: str) -> dict:
    cache=Cache('air_quality'); key=f'cpcb:{city}'
    cached=cache.get(key, 1800)
    if cached: cached['cached']=True; return cached
    # data.gov.in resources vary; keep resource configurable instead of hard-coding a brittle endpoint.
    url=os.getenv('DATA_GOV_AQI_URL')
    if not url: return {'source':'CPCB/data.gov.in','status':'not_configured','city':city,'records':[],'cached':False}
    try:
        r=_request(url,params={'city':city}); data=r.json(); out={'source':'CPCB/data.gov.in','status':'success','city':city,'records':data.get('records',data if isinstance(data,list) else []),'cached':False}
    except Exception as exc: out={'source':'CPCB/data.gov.in','status':'error','city':city,'error':str(exc),'records':[],'cached':False}
    cache.put(key,out); return out

def fetch_openaq(city: str) -> dict:
    cache=Cache('air_quality'); key=f'openaq:{city}'
    cached=cache.get(key,1800)
    if cached: cached['cached']=True; return cached
    key_api=os.getenv('OPENAQ_API_KEY')
    if not key_api: return {'source':'OpenAQ','status':'not_configured','city':city,'records':[],'cached':False}
    try:
        headers={'X-API-Key':key_api}; r=_request('https://api.openaq.org/v3/locations',params={'city':city,'limit':10},headers=headers)
        out={'source':'OpenAQ','status':'success','city':city,'records':r.json().get('results',[]),'cached':False}
    except Exception as exc: out={'source':'OpenAQ','status':'error','city':city,'error':str(exc),'records':[],'cached':False}
    cache.put(key,out); return out

def fetch_nse_market(ticker: str) -> dict:
    cache=Cache('market'); key=f'nse:{ticker}'
    cached=cache.get(key,1800)
    if cached: cached['cached']=True; return cached
    url=os.getenv('NSE_MARKET_DATA_URL')
    if not url: return {'source':'NSE','status':'not_configured','ticker':ticker,'records':[],'cached':False}
    try:
        r=_request(url,params={'symbol':ticker.replace('.NS','')}); out={'source':'NSE','status':'success','ticker':ticker,'records':r.json().get('records',r.json() if isinstance(r.json(),list) else []),'cached':False}
    except Exception as exc: out={'source':'NSE','status':'error','ticker':ticker,'error':str(exc),'records':[],'cached':False}
    cache.put(key,out); return out

def fetch_external_parallel(company: dict, include_air: bool = True) -> dict:
    tasks={'market': lambda: fetch_nse_market(company.get('ticker','')),'news':lambda:fetch_gdelt(company['full_name'])}
    city=(company.get('headquarters') or {}).get('city')
    if include_air and city:
        tasks['air_cpcb']=lambda:fetch_cpcb(city)
        tasks['air_openaq']=lambda:fetch_openaq(city)
    out={}
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futures={ex.submit(fn):name for name,fn in tasks.items()}
        for fut in as_completed(futures):
            name=futures[fut]
            try: out[name]=fut.result()
            except Exception as exc: out[name]={'status':'error','error':str(exc)}
    return out
