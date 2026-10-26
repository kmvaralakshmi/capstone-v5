"""Dynamic data providers with caching, fallback and telemetry.

    market  : Yahoo Finance via `yfinance` (free, no key, covers every NSE ticker
              as SYMBOL.NS)  ->  fallback: bundled NIFTY/SENSEX CSVs
    news    : GDELT DOC 2.0 API (free, no key)  ->  fallback: bundled news CSV
    air     : bundled CPCB CSVs (default); OpenAQ v3 if OPENAQ_API_KEY is set

NOTE: Yahoo is not an official NSE feed. It is used because the official NSE
historical endpoints need session handling/licensing; MarketProvider is a small
interface, so an NSE-bhavcopy provider can be added without touching agents.

Each fetch returns a DataResult and (when a RunTracker is given) records a span
with source, duration, status and whether a fallback was used.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .common import ROOT

CACHE = ROOT / "data" / "cache"
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT_SECONDS", "25"))


@dataclass
class DataResult:
    status: str                       # success | cached | fallback | failed
    source: str
    payload: Any = None
    seconds: float = 0.0
    fallback_used: bool = False
    error: Optional[str] = None
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def ok(self) -> bool:
        return self.status in ("success", "cached", "fallback") and self.payload is not None


def _cache_path(kind: str, key: str) -> Path:
    d = CACHE / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")


def _cache_get(kind: str, key: str, max_age_s: int):
    p = _cache_path(kind, key)
    if p.exists() and time.time() - p.stat().st_mtime < max_age_s:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _cache_put(kind: str, key: str, value) -> None:
    _cache_path(kind, key).write_text(json.dumps(value, default=str), encoding="utf-8")


# ----------------------------------------------------------------- market ---
def market_metrics(close: pd.Series, fy_start="2024-04-01", fy_end="2025-03-31") -> Dict[str, Any]:
    """Return/volatility/drawdown figures from a date-indexed close series."""
    s = close.dropna().sort_index()
    if len(s) < 60:
        return {}
    end = s.index.max()

    def ret(days):
        w = s[s.index >= end - pd.Timedelta(days=days)]
        return None if len(w) < 5 else float(w.iloc[-1] / w.iloc[0] - 1) * 100

    fy = s[(s.index >= fy_start) & (s.index <= fy_end)]
    fy_ret = None if len(fy) < 100 else float(fy.iloc[-1] / fy.iloc[0] - 1) * 100
    base = fy if len(fy) >= 100 else s[s.index >= end - pd.Timedelta(days=365)]
    daily = base.pct_change().dropna()
    return {"Return_1M_Pct": ret(30), "Return_3M_Pct": ret(91), "Return_6M_Pct": ret(182),
            "Return_1Y_Pct": ret(365), "Return_3Y_Pct": ret(365 * 3),
            "FY_Return_Pct": fy_ret,
            "Volatility_Pct": float(daily.std() * np.sqrt(252) * 100) if len(daily) > 20 else None,
            "Max_Drawdown_Pct": float((base / base.cummax() - 1).min() * 100) if len(base) > 20 else None,
            "Data_Points": int(len(s)), "Window_End": str(end.date())}


_STATIC = {"df": None}


def _static_frame() -> Optional[pd.DataFrame]:
    if _STATIC["df"] is None:
        base = next(iter((ROOT / "raw-datasets").glob("Stock Market*")), None)
        frames = []
        for f in ("NIFTY_50_COMPANIES.csv", "SENSEX_COMPANIES.csv"):
            if base is not None and (base / f).exists():
                d = pd.read_csv(base / f, usecols=["Date", "Close", "Ticker"])
                d["Ticker"] = d["Ticker"].str.replace(".BO", ".NS", regex=False)
                frames.append(d)
        _STATIC["df"] = pd.concat(frames).drop_duplicates(["Ticker", "Date"]) if frames else pd.DataFrame()
    return _STATIC["df"]


def _static_close(ticker: str) -> Optional[pd.Series]:
    all_ = _static_frame()
    if all_ is None or all_.empty:
        return None
    d = all_[all_["Ticker"] == ticker]
    if d.empty:
        return None
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).set_index("Date")["Close"]
    return d if len(d) else None


def fetch_market(ticker: str, dynamic: bool = True, tracker=None, stage: str = "data_acquisition",
                 alternates: Optional[List[str]] = None) -> DataResult:
    t0 = time.perf_counter()
    span = tracker.span(f"market:{ticker}", stage, "yfinance" if dynamic else "static") if tracker else None
    rec = span.__enter__() if span else {}
    try:
        res = _fetch_market_inner(ticker, dynamic, alternates or [])
        res.seconds = round(time.perf_counter() - t0, 3)
        rec.update({"status": "success" if res.ok else "failed", "fallback_used": res.fallback_used,
                    "source": res.source, "error": res.error})
        if span:
            span.__exit__(None, None, None)
        return res
    except Exception as exc:  # pragma: no cover
        if span:
            span.__exit__(type(exc), exc, None)
        return DataResult("failed", "market", error=str(exc))


def _fetch_market_inner(ticker: str, dynamic: bool, alternates: List[str]) -> DataResult:
    err = None
    if dynamic:
        for tk in [ticker] + list(alternates):
            cached = _cache_get("market", tk, 6 * 3600)
            if cached:
                s = pd.Series(cached["close"], index=pd.to_datetime(cached["dates"]))
                return DataResult("cached", "yfinance(cache)", market_metrics(s) | {"Source": f"yfinance(cache) {tk}"})
            try:
                import yfinance as yf
                h = yf.Ticker(tk).history(period="4y", auto_adjust=True)
                if h is not None and len(h) > 60:
                    s = h["Close"].copy()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    _cache_put("market", tk, {"dates": [str(i.date()) for i in s.index], "close": [float(x) for x in s.values]})
                    return DataResult("success", "yfinance", market_metrics(s) | {"Source": f"yfinance {tk}"})
                err = f"yfinance: no data for {tk}"
            except Exception as exc:
                err = f"yfinance {tk}: {type(exc).__name__}: {exc}"
    s = _static_close(ticker)
    if s is not None:
        m = market_metrics(s)
        if m:
            return DataResult("fallback", "static NIFTY/SENSEX csv", m | {"Source": "static"}, fallback_used=True, error=err)
    return DataResult("failed", "market", None, error=err or "no price data available")


# ------------------------------------------------------------------- news ---
GDELT_TIMEOUT = int(os.getenv("GDELT_TIMEOUT_SECONDS", "12"))
GDELT_TIMESPANS = [os.getenv("GDELT_TIMESPAN", "3m"), "12w", "90d"]
_GDELT = {"fails": 0, "ok": 0, "last_error": None}


def reset_gdelt_circuit() -> None:
    _GDELT.update({"fails": 0, "ok": 0, "last_error": None})


def gdelt_last_error() -> Optional[str]:
    return _GDELT["last_error"]


def fetch_news(query_name: str, tracker=None, stage: str = "data_acquisition",
               timespan: str = "", maxrecords: int = 120) -> DataResult:
    """GDELT DOC 2.0 article list (titles/urls/dates only).

    Fails fast: short timeout, and after 2 failures with no success the circuit opens so a
    50-company batch does not spend 25+ minutes waiting on an unreachable service."""
    t0 = time.perf_counter()
    span = tracker.span(f"news:{query_name}", stage, "GDELT") if tracker else None
    rec = span.__enter__() if span else {}
    key = f"{query_name}|v2"
    cached = _cache_get("news", key, 3 * 3600)
    if cached is not None:
        res = DataResult("cached", "GDELT(cache)", cached)
    elif _GDELT["fails"] >= 2 and _GDELT["ok"] == 0:
        res = DataResult("failed", "GDELT", None, error=f"circuit open (GDELT unreachable): {_GDELT['last_error']}")
    else:
        try:
            import requests
            q = (f'"{query_name}" (ESG OR sustainability OR emissions OR pollution OR governance '
                 f'OR fraud OR penalty OR lawsuit OR diversity OR layoffs OR climate)')
            arts, err = None, None
            for ts in ([timespan] if timespan else GDELT_TIMESPANS):
                params = {"query": q, "mode": "ArtList", "format": "json", "maxrecords": maxrecords,
                          "timespan": ts, "sort": "DateDesc"}
                r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=GDELT_TIMEOUT)
                if r.status_code == 429:
                    time.sleep(6)
                    r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=GDELT_TIMEOUT)
                if r.status_code != 200:
                    err = f"HTTP {r.status_code}: {r.text[:120]}"
                    continue
                try:
                    arts = (r.json() or {}).get("articles", [])
                    break
                except ValueError:          # GDELT answers plain text for invalid parameters
                    err = f"non-JSON reply for timespan={ts}: {r.text[:120]!r}"
            if arts is None:
                raise RuntimeError(err or "no usable reply")
            _cache_put("news", key, arts)
            _GDELT["ok"] += 1
            res = DataResult("success", "GDELT", arts)
        except Exception as exc:
            _GDELT["fails"] += 1
            _GDELT["last_error"] = f"{type(exc).__name__}: {exc}"[:240]
            res = DataResult("failed", "GDELT", None, error=_GDELT["last_error"])
    res.seconds = round(time.perf_counter() - t0, 3)
    rec.update({"status": "success" if res.ok else "failed", "error": res.error})
    if span:
        span.__exit__(None, None, None)
    return res


# -------------------------------------------------------------------- air ---
def fetch_openaq_city(city: str, tracker=None, stage: str = "data_acquisition") -> DataResult:
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        return DataResult("failed", "OpenAQ", None, error="OPENAQ_API_KEY not set (optional)")
    t0 = time.perf_counter()
    try:
        import requests
        r = requests.get("https://api.openaq.org/v3/locations", headers={"X-API-Key": key},
                         params={"limit": 20, "iso": "IN", "parameters_id": 2},   # 2 = PM2.5
                         timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        rows = [x for x in r.json().get("results", []) if city.lower() in json.dumps(x).lower()]
        return DataResult("success", "OpenAQ v3", rows, seconds=round(time.perf_counter() - t0, 3))
    except Exception as exc:
        return DataResult("failed", "OpenAQ", None, error=f"{type(exc).__name__}: {exc}")
