"""LLM "brain" used by agents to choose tools and (Agent 7) to explain results.

PROVIDERS (set in .env; no paid key required)
    LLM_PROVIDER=auto        pick OpenAI if OPENAI_API_KEY is set, else a local
                             Ollama server if one is running, else no LLM
    LLM_PROVIDER=ollama      free, unlimited, runs on your PC  (recommended now)
    LLM_PROVIDER=openai      OpenAI Responses API
    LLM_PROVIDER=compatible  any OpenAI-compatible endpoint (Groq, Gemini's
                             OpenAI endpoint, LM Studio, vLLM ...) via
                             LLM_BASE_URL + LLM_API_KEY
    LLM_PROVIDER=none        deterministic policies only

    LLM_MODEL=...            model name (default depends on provider)

Design rules
* The LLM only *chooses among tools* an agent exposes. All arithmetic stays in
  deterministic code. Agents work (and are tested) with no LLM at all.
* Each call is stateless: compact state + recent history sent as plain text.
* Small local models sometimes pick a wrong tool; the BaseAgent guardrails then
  substitute the agent's policy action, so runs stay correct.
"""
from __future__ import annotations

import json
import os
from typing import Tuple
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

DEFAULT_MODELS = {"openai": "gpt-4.1-mini", "ollama": "qwen2.5:7b-instruct",
                  "compatible": "llama-3.1-8b-instant"}


@dataclass
class Decision:
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    source: str = "policy"          # "llm" | "policy" | "guardrail"


FINISH_SPEC = {
    "type": "function", "name": "finish",
    "description": "Declare the task complete. Only when every completion criterion in the state is satisfied.",
    "parameters": {"type": "object", "properties": {"summary": {"type": "string"}},
                   "required": ["summary"], "additionalProperties": False},
}


def _ollama_up(base: str) -> bool:
    try:
        import requests
        root = base.replace("/v1", "").rstrip("/")
        return requests.get(root + "/api/tags", timeout=1.5).ok
    except Exception:
        return False


class LLMBrain:
    def __init__(self, model: Optional[str] = None, enabled: bool = True, client: Any = None,
                 provider: Optional[str] = None):
        self.enabled = enabled
        self._client = client
        self.last_error: Optional[str] = None
        self.fail_streak = 0
        self.disabled_reason: Optional[str] = None
        self.provider = "injected" if client is not None else "none"
        self.api_style = "responses"
        self.model = model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or ""
        if client is not None or not enabled or OpenAI is None:
            self.model = self.model or "fake"
            return
        want = (provider or os.getenv("LLM_PROVIDER") or "auto").lower()
        key = os.getenv("OPENAI_API_KEY")
        base = os.getenv("LLM_BASE_URL") or "http://localhost:11434/v1"
        if want == "auto":
            want = "openai" if key else ("ollama" if _ollama_up(base) else "none")
        timeout = float(os.getenv("LLM_TIMEOUT", "120"))
        try:
            if want == "openai" and key:
                self._client = OpenAI(api_key=key, timeout=timeout)
            elif want == "ollama":
                self._client = OpenAI(api_key="ollama", base_url=base, timeout=timeout)
                self.api_style = "chat"
            elif want == "compatible":
                self._client = OpenAI(api_key=os.getenv("LLM_API_KEY", "none"),
                                      base_url=os.getenv("LLM_BASE_URL"), timeout=timeout)
                self.api_style = "chat"
            if self._client is not None:
                self.provider = want
                self.model = self.model or DEFAULT_MODELS.get(want, "")
        except Exception as exc:  # pragma: no cover
            self.last_error = str(exc)

    @property
    def available(self) -> bool:
        return self.enabled and self._client is not None

    def _fail(self, msg: str) -> None:
        """Record a failure; after 3 in a row switch the LLM off for this run (circuit breaker)."""
        self.last_error = msg
        self.fail_streak += 1
        if self.fail_streak >= 3 and self._client is not None:
            self.disabled_reason = f"disabled after {self.fail_streak} consecutive failures; last error: {msg}"
            self._client = None

    def healthcheck(self, timeout: float = 90.0) -> Tuple[bool, str]:
        """Really call the model once. Without this, 'LLM mode' only meant a client object existed."""
        if not self.available:
            return False, self.disabled_reason or "no LLM client configured"
        try:
            c = self._client.with_options(timeout=timeout) if hasattr(self._client, "with_options") else self._client
            if self.api_style == "chat":
                r = c.chat.completions.create(model=self.model, max_tokens=8, temperature=0,
                                              messages=[{"role": "user", "content": "Reply with the single word OK"}])
                text = (r.choices[0].message.content or "").strip()
            else:
                r = c.responses.create(model=self.model, input="Reply with the single word OK")
                text = (getattr(r, "output_text", "") or "").strip()
            self.fail_streak = 0
            return True, f"model answered: {text[:20]!r}"
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            hint = ""
            low = msg.lower()
            if self.provider == "ollama" and ("404" in low or "not found" in low):
                hint = f"  -> model '{self.model}' is not installed. Run: ollama list   then: ollama pull {self.model}"
            elif "connect" in low or "refused" in low:
                hint = "  -> Ollama is not running. Start it (Ollama app / `ollama serve`)."
            elif "timed out" in low or "timeout" in low:
                hint = "  -> first call loads the model into memory; retry, or raise LLM_TIMEOUT / use a smaller model."
            self.disabled_reason = msg[:300] + hint
            self.last_error = msg
            self._client = None            # do not pretend to be in LLM mode
            return False, self.disabled_reason

    def describe(self) -> str:
        return f"{self.provider}:{self.model}" if self.available else "none (deterministic policies)"

    # ------------------------------------------------------------------
    def decide(self, system: str, state: Dict[str, Any], history: List[Dict[str, Any]],
               tools: List[Any]) -> Optional[Decision]:
        if not self.available:
            return None
        prompt = ("CURRENT STATE (JSON):\n" + json.dumps(state, default=str)[:2500]
                  + "\n\nRECENT ACTIONS AND OBSERVATIONS:\n"
                  + json.dumps([{k: h.get(k) for k in ("tool", "args", "observation")} for h in history[-4:]], default=str)[:1800]
                  + "\n\nChoose exactly one tool to call next. Do not repeat a call "
                    "that already failed with the same arguments.")
        specs = [t.spec() for t in tools] + [FINISH_SPEC]
        try:
            if hasattr(self._client, "with_options"):
                client_full = self._client
                self._client = client_full.with_options(timeout=float(os.getenv("LLM_DECISION_TIMEOUT", "90")))
            if self.api_style == "chat":
                return self._decide_chat(system, prompt, specs)
            return self._decide_responses(system, prompt, specs)
        except Exception as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            return None
        finally:
            if "client_full" in locals():
                self._client = client_full if self._client is not None else None

    def _decide_responses(self, system, prompt, specs) -> Optional[Decision]:
        resp = self._client.responses.create(model=self.model, instructions=system, input=prompt,
                                             tools=specs, tool_choice="required")
        thought, call = "", None
        for item in getattr(resp, "output", []) or []:
            kind = getattr(item, "type", None)
            if kind == "function_call" and call is None:
                call = item
            elif kind == "message":
                for part in getattr(item, "content", []) or []:
                    thought += getattr(part, "text", "") or ""
        if call is None:
            self.last_error = "model returned no function call"
            return None
        return self._mk(call.name, getattr(call, "arguments", ""), thought)

    def _decide_chat(self, system, prompt, specs) -> Optional[Decision]:
        tools = [{"type": "function", "function": {"name": s["name"], "description": s["description"],
                                                   "parameters": s["parameters"]}} for s in specs]
        kw = dict(model=self.model, tools=tools, temperature=0,
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        try:
            resp = self._client.chat.completions.create(tool_choice="required", **kw)
        except Exception as exc:                      # some servers reject tool_choice
            if "tool_choice" not in str(exc).lower():
                raise
            resp = self._client.chat.completions.create(**kw)
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            self._fail("model answered in text instead of calling a tool")
            return None
        fn = calls[0].function
        return self._mk(fn.name, fn.arguments, getattr(msg, "content", "") or "")

    def _mk(self, name, raw_args, thought) -> Optional[Decision]:
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            self.last_error = "model returned invalid JSON arguments"
            return None
        self.fail_streak = 0
        return Decision(tool=name, args=args, thought=(thought or "").strip(), source="llm")

    def write(self, system: str, prompt: str, max_chars: int = 12000) -> Optional[str]:
        """Free-text generation (evidence-grounded narration)."""
        if not self.available:
            return None
        try:
            if self.api_style == "chat":
                r = self._client.chat.completions.create(
                    model=self.model, temperature=0.2,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": prompt[:max_chars]}])
                return (r.choices[0].message.content or "").strip() or None
            r = self._client.responses.create(model=self.model, instructions=system, input=prompt[:max_chars])
            return (getattr(r, "output_text", "") or "").strip() or None
        except Exception as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            return None
