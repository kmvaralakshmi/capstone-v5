"""Research/validation calculations for Section C."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from utils.config import PROCESSED_DATA_DIR, OUTPUT_FILES
ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/'research-benchmarks'

def _read(name):
    p=PROCESSED_DATA_DIR/name; return pd.read_csv(p) if p.exists() else pd.DataFrame()

def _corr(a,b):
    x=pd.to_numeric(a,errors='coerce'); y=pd.to_numeric(b,errors='coerce'); m=x.notna()&y.notna()
    if m.sum()<3: return {'n':int(m.sum()),'pearson':None,'spearman':None}
    return {'n':int(m.sum()),'pearson':round(float(pearsonr(x[m],y[m]).statistic),6),'spearman':round(float(spearmanr(x[m],y[m]).statistic),6)}

def _benchmark(scores):
    p=BENCH/'nse_top200.csv'
    if not p.exists() or scores.empty: return {'status':'not_configured','message':'Add research-benchmarks/nse_top200.csv from an official NSE source.'}
    b=pd.read_csv(p); merged=scores.merge(b,on='Company_Code',suffixes=('_ours','_nse'))
    if merged.empty: return {'status':'no_overlap','n':0}
    result={'status':'success','n':len(merged)}
    if 'NSE_ESG_Score' in merged: result['score_correlation']=_corr(merged['Master_ESG_Score'],merged['NSE_ESG_Score'])
    if 'NSE_ESG_Score' in merged:
        merged['absolute_score_difference']=(pd.to_numeric(merged['Master_ESG_Score'],errors='coerce')-pd.to_numeric(merged['NSE_ESG_Score'],errors='coerce')).abs()
        result['mean_absolute_difference']=float(merged['absolute_score_difference'].mean())
    return result

def _cfa_brsr():
    p=BENCH/'nse_cfa_brsr.csv'
    if not p.exists(): return {'status':'not_configured','message':'Add research-benchmarks/nse_cfa_brsr.csv from the official NSE/CFA research.'}
    df=pd.read_csv(p); return {'status':'success','rows':len(df),'disclosure_rate_by_metric':df.groupby('Metric_Name')['Disclosure_Status'].apply(lambda s:(s.astype(str).str.lower().isin(['yes','reported','disclosed','true','1'])).mean()).round(4).to_dict() if {'Metric_Name','Disclosure_Status'}<=set(df.columns) else {}}

def run_validation(output_dir: Path | None=None):
    out=output_dir or PROCESSED_DATA_DIR; out.mkdir(parents=True,exist_ok=True)
    scores=_read(OUTPUT_FILES['master_scores']); result={'generated_at':pd.Timestamp.utcnow().isoformat(),'n_companies':int(len(scores))}
    result['nse_top200']=_benchmark(scores); result['nse_cfa_brsr']=_cfa_brsr()
    stock=_read(OUTPUT_FILES['stock_correlation'])
    if not scores.empty and not stock.empty and 'Company_Code' in stock.columns:
        merged=scores.merge(stock,on='Company_Code',suffixes=('_esg','_stock')); metrics={}
        for c in ['One_Year_Return','Annual_Return','Volatility','Max_Drawdown','Stock_Return']:
            if c in merged: metrics[c]=_corr(merged['Master_ESG_Score'],merged[c])
        result['esg_vs_stock']=metrics
    else: result['esg_vs_stock']={}
    news=_read(OUTPUT_FILES['news_sentiment'])
    if not scores.empty and not news.empty and 'Company_Code' in news.columns:
        nums=[c for c in news.columns if c.lower() in {'sentiment_score','news_sentiment_score','esg_relevance_score','article_count','negative_count'}]
        agg=news.groupby('Company_Code')[nums].mean(numeric_only=True).reset_index() if nums else pd.DataFrame(); merged=scores.merge(agg,on='Company_Code') if not agg.empty else pd.DataFrame()
        result['esg_vs_news']={c:_corr(merged['Master_ESG_Score'],merged[c]) for c in nums if c in merged}
    else: result['esg_vs_news']={}
    env=_read(OUTPUT_FILES['environmental_risk'])
    if not scores.empty and not env.empty and 'Company_Code' in env.columns:
        nums=[c for c in env.columns if any(k in c.lower() for k in ['aqi','pm25','pm10','environmental_score'])]
        agg=env.groupby('Company_Code')[nums].mean(numeric_only=True).reset_index() if nums else pd.DataFrame(); merged=scores.merge(agg,on='Company_Code') if not agg.empty else pd.DataFrame()
        result['esg_vs_air_quality_context']={c:_corr(merged['Environmental_Score'],merged[c]) for c in nums if c in merged}
        result['air_quality_note']='Contextual association only; air quality does not directly change ESG score.'
    else: result['esg_vs_air_quality_context']={}
    (out/'validation_research_report.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8'); return result
