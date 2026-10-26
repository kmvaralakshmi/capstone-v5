"""Self-check for the external pieces:   python -m esg_agents.diagnostics

Tests Ollama (server, model installed, plain answer, tool calling), Yahoo Finance and GDELT,
and prints the exact error plus a hint. Run this BEFORE a long --dynamic run."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def line(ok, name, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def main() -> int:
    import requests
    from dotenv import load_dotenv
    load_dotenv()
    base = (os.getenv("LLM_BASE_URL") or "http://localhost:11434/v1").replace("/v1", "").rstrip("/")
    model = os.getenv("LLM_MODEL") or "qwen2.5:7b-instruct"
    bad = 0
    # --- Ollama
    try:
        tags = requests.get(base + "/api/tags", timeout=4).json().get("models", [])
        names = [m["name"] for m in tags]
        line(True, "Ollama server reachable", base)
        have = model in names or any(n.split(":")[0] == model.split(":")[0] and model in n for n in names)
        line(have, f"model '{model}' installed", f"installed: {names}" if names else "no models installed")
        if not have:
            print(f"       -> run:  ollama pull {model}   (wait for the word 'success'; re-run if it ends with an error)")
            bad += 1
        else:
            from esg_agents.framework import LLMBrain, Tool
            b = LLMBrain(enabled=True, provider="ollama", model=model)
            t = time.perf_counter()
            ok, msg = b.healthcheck(timeout=180)
            line(ok, "model answers a plain prompt", f"{time.perf_counter() - t:.1f}s  {msg}")
            bad += not ok
            if ok:
                b2 = LLMBrain(enabled=True, provider="ollama", model=model)
                d = b2.decide("Pick the tool.", {"todo": "call ping"}, [], [Tool("ping", "ping the server", lambda: 1)])
                line(d is not None and d.tool in ("ping", "finish"), "model can call a tool (function calling)",
                     f"-> {d.tool if d else 'NO TOOL CALL: ' + str(b2.last_error)}")
                bad += d is None
    except Exception as exc:
        line(False, "Ollama server reachable", f"{type(exc).__name__}: {exc}")
        print("       -> start Ollama (app in the system tray, or `ollama serve`)")
        bad += 1
    # --- Yahoo
    try:
        from esg_agents.data_providers import fetch_market
        t = time.perf_counter()
        r = fetch_market("INFY.NS", dynamic=True)
        line(r.ok and "static" not in r.source, "Yahoo Finance (INFY.NS)", f"{time.perf_counter() - t:.1f}s via {r.source}; {r.error or ''}")
        bad += not (r.ok and "static" not in r.source)
    except Exception as exc:
        line(False, "Yahoo Finance", str(exc)); bad += 1
    # --- GDELT
    for ts in ("3m", "12w", "90d"):
        try:
            t = time.perf_counter()
            r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", timeout=15, params={
                "query": '"Infosys" sustainability', "mode": "ArtList", "format": "json", "maxrecords": 5, "timespan": ts})
            try:
                n = len(r.json().get("articles", []))
                line(True, f"GDELT timespan={ts}", f"{time.perf_counter() - t:.1f}s, {n} articles")
                break
            except ValueError:
                line(False, f"GDELT timespan={ts}", f"HTTP {r.status_code}, not JSON: {r.text[:100]!r}")
        except Exception as exc:
            line(False, f"GDELT timespan={ts}", f"{type(exc).__name__}: {exc}")
            print("       -> GDELT unreachable from this network (timeout). The pipeline will fall back to the bundled news file.")
            break
    print("\\nDone." if not bad else f"\\n{bad} problem(s) found - fix the FAIL lines above first.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
