"""ESG Intelligence backend (v5.2).

    python -m uvicorn backend.app:app --reload        then open http://127.0.0.1:8000

On-demand analysis spawns `run_agents.py --company ...` in its own process and
output folder (data/runs/<symbol>_<time>), so a single-company run never
overwrites the batch (50-company) results in processed-data/.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BATCH = ROOT / "processed-data"
RUNS = ROOT / "data" / "runs"
FRONTEND = ROOT / "frontend"

app = FastAPI(title="ESG Intelligence System", version="5.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
JOBS: Dict[str, Dict[str, Any]] = {}


# ----------------------------------------------------------------- helpers --
def csv_rows(folder: Path, name: str, code: Optional[str] = None) -> list:
    p = folder / name
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p)
    except Exception:
        return []
    if code and "Company_Code" in df:
        df = df[df["Company_Code"].astype(str).str.upper() == code.upper()]
    return json.loads(df.to_json(orient="records"))      # NaN -> null


def json_file(folder: Path, name: str) -> Any:
    p = folder / name
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None


def resolve_folder(key: str) -> Path:
    if key in ("batch", "", None):
        return BATCH
    p = RUNS / key
    if not p.is_dir():
        raise HTTPException(404, f"run '{key}' not found")
    return p


def latest_run_for(code: str) -> Optional[Path]:
    if not RUNS.exists():
        return None
    runs = sorted([d for d in RUNS.iterdir() if d.is_dir() and d.name.upper().startswith(code.upper())])
    return runs[-1] if runs else None


# ---------------------------------------------------------------- endpoints --
@app.get("/api/health")
def health():
    from esg_agents.framework import LLMBrain
    b = LLMBrain(enabled=True)
    return {"status": "ok", "llm": b.describe(), "batch_ready": (BATCH / "esg_master_scores.csv").exists(),
            "runs": len(list(RUNS.iterdir())) if RUNS.exists() else 0}


@app.get("/api/cohort")
def cohort():
    rows = csv_rows(BATCH, "esg_master_scores.csv")
    return {"companies": sorted(rows, key=lambda r: (r.get("Rank") is None, r.get("Rank") or 1e9))}


@app.get("/api/runs")
def runs():
    out = []
    if RUNS.exists():
        for d in sorted(RUNS.iterdir(), reverse=True):
            if d.is_dir():
                rep = json_file(d, "agent_run_report.json") or {}
                out.append({"run": d.name, "companies": rep.get("companies"), "finished": rep.get("finished"),
                            "total_seconds": (rep.get("timing") or {}).get("total_seconds")})
    return {"runs": out}


class AnalyzeRequest(BaseModel):
    company: str
    hq_city: str = ""
    dynamic: bool = True
    use_llm: bool = True


def _run_job(job_id: str, cmd: list):
    job = JOBS[job_id]
    try:
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:                                   # type: ignore[union-attr]
            job["log"].append(line.rstrip())
            job["log"] = job["log"][-400:]
            m = re.match(r"Output folder:\s*(.+)", line)
            if m:
                job["run"] = Path(m.group(1).strip()).name
        p.wait()
        job["status"] = "completed" if p.returncode == 0 else "failed"
    except Exception as exc:
        job["status"], job["error"] = "failed", str(exc)


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if not req.company.strip():
        raise HTTPException(400, "company is required")
    job_id = uuid.uuid4().hex[:10]
    cmd = [sys.executable, "run_agents.py", "--company", req.company.strip()]
    if req.dynamic:
        cmd.append("--dynamic")
    if not req.use_llm:
        cmd.append("--offline")
    if req.hq_city:
        cmd += ["--hq-city", req.hq_city]
    JOBS[job_id] = {"id": job_id, "company": req.company, "status": "running", "log": [], "run": None}
    threading.Thread(target=_run_job, args=(job_id, cmd), daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/batch/run")
def batch_run(dynamic: bool = False, use_llm: bool = False):
    job_id = uuid.uuid4().hex[:10]
    cmd = [sys.executable, "run_agents.py"] + (["--dynamic"] if dynamic else []) + ([] if use_llm else ["--offline"])
    JOBS[job_id] = {"id": job_id, "company": "ALL (batch)", "status": "running", "log": [], "run": "batch"}
    threading.Thread(target=_run_job, args=(job_id, cmd), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    folder = resolve_folder(j["run"]) if j.get("run") else None
    metrics = json_file(folder, "run_metrics.json") if folder else None
    agents_done = [json.loads(re.search(r"\{.*\}", l).group(0)) if False else l for l in j["log"] if re.search(r"\[agent\d\] (completed|stalled|failed)", l)]
    return {**{k: j[k] for k in ("id", "company", "status", "run")}, "log_tail": j["log"][-25:],
            "agents_finished": agents_done, "metrics": metrics, "error": j.get("error")}


@app.get("/api/report/{key}")
def report(key: str, company: Optional[str] = None):
    """key = run folder name | 'batch'. For batch pass ?company=CODE."""
    folder = resolve_folder(key)
    code = company
    if not code:
        rows = csv_rows(folder, "esg_master_scores.csv")
        if len(rows) != 1:
            raise HTTPException(400, "pass ?company=CODE for multi-company folders")
        code = rows[0]["Company_Code"]
    code = code.upper()
    scores = csv_rows(folder, "esg_master_scores.csv", code)
    if not scores:
        raise HTTPException(404, f"{code} not found in {key}")
    traces = {}
    tdir = folder / "agent_traces"
    if tdir.exists():
        for f in sorted(tdir.glob("*_trace.json")):
            t = json.loads(f.read_text(encoding="utf-8"))
            r = t["report"]
            traces[r["agent_id"]] = {"name": r["name"], "status": r["status"], "seconds": r["seconds"], "steps": r["steps"],
                                     "decisions": r["decisions_by_source"], "summary": r["summary"], "llm": r["llm_mode"],
                                     "findings": r["findings"], "history": t["history"][-60:]}
    packages = json_file(folder, "esg_evidence_packages.json") or {}
    return {"run": key, "company": code, "scores": scores[0],
            "metrics": csv_rows(folder, "esg_metric_scores.csv", code),
            "brsr": csv_rows(folder, "brsr_extracted_metrics.csv", code),
            "trends": csv_rows(folder, "brsr_trends.csv", code),
            "news_summary": csv_rows(folder, "esg_news_company_summary.csv", code),
            "news_articles": csv_rows(folder, "esg_news_sentiment.csv", code)[:25],
            "market": csv_rows(folder, "stock_esg_correlation.csv", code),
            "air": csv_rows(folder, "company_location_environmental_risk.csv", code),
            "consistency": csv_rows(folder, "greenwashing_detection.csv", code),
            "claims": csv_rows(folder, "greenwashing_claims.csv", code),
            "signals": csv_rows(folder, "greenwashing_signals_detail.csv", code),
            "explanation": csv_rows(folder, "multi_agent_explanations.csv", code),
            "evidence_package": packages.get(code), "filings": json_file(folder, "filing_discovery.json"),
            "agents": traces, "run_metrics": json_file(folder, "run_metrics.json"),
            "blackboard": json_file(folder, "agent_blackboard.json")}


@app.get("/api/company/{query}")
def company_lookup(query: str):
    """Latest on-demand run for a company code, else the batch result."""
    run = latest_run_for(query)
    if run:
        return report(run.name, company=query)
    return report("batch", company=query)


@app.get("/api/validation")
def validation():
    summary = json_file(BATCH, "validation_summary.json")
    files = {n: csv_rows(BATCH, n) for n in ("validation_nse_top200.csv", "validation_nse_cfa.csv", "validation_esg_stock.csv",
                                             "validation_esg_news.csv", "validation_esg_air.csv")}
    return {"summary": summary, "tables": files,
            "benchmark_files_present": {"nse_top200_esg.csv": (ROOT / "data/benchmark/nse_top200_esg.csv").exists(),
                                        "nse_cfa_reference.csv": (ROOT / "data/benchmark/nse_cfa_reference.csv").exists()}}


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
