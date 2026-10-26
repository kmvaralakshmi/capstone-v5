"""Agent 3 - ESG News Sentiment Agent.

ROLE
    Turn a raw stock-market news corpus into a *reliable* per-company ESG
    reputation signal.

PROBLEMS THIS AGENT DETECTS AND FIXES (found in the earlier engine)
    1. Substring attribution: short codes such as "SIS", "NIIT" or "TITAN" were
       matched inside unrelated words ("analysis", "Titan..."), and every
       "Stocks to Watch: A, B, C ..." round-up was credited to all companies.
       -> word-boundary alias matching; title-level evidence required;
          round-up articles (many tracked companies in the title) are rejected.
    2. Score saturation: VADER run on title+description+full body returns
       compound ~ +/-0.99 for almost any long text, so everything was
       "Positive". -> sentences are scored separately and averaged, using only
       the title and the first sentences of the description.
    3. Over-confidence on tiny samples: 3 articles said "perfect 10/10".
       -> company score is shrunk toward neutral by sample size and a
          reliability value is published with it.

WHAT MAKES IT AN AGENT
    It inspects the corpus, measures how bad each problem is, picks cleaning
    steps in an order driven by those measurements, audits the *distribution*
    of the result (saturation check) and reports coverage gaps to the board.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from .common import (COMPANIES, PROCESSED, ROOT, OUTPUT_FILES, clean, np, pd)
from .data_providers import fetch_news, reset_gdelt_circuit, gdelt_last_error
from .framework import BaseAgent, Decision, Tool
import time

from utils.config import ESG_INCLUDE_KEYWORDS, ESG_EXCLUDE_KEYWORDS

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover
    SentimentIntensityAnalyzer = None

# Extra finance/ESG-flavoured terms VADER does not know (small, auditable).
ESG_LEXICON = {
    "greenwashing": -3.0, "fine": -1.5, "penalty": -2.0, "violation": -2.5,
    "lawsuit": -2.0, "fraud": -3.2, "scam": -3.0, "probe": -1.5, "layoffs": -2.0,
    "layoff": -2.0, "pollution": -2.2, "emission": -0.8, "breach": -2.2,
    "net zero": 2.0, "renewable": 1.8, "sustainability": 1.5, "sustainable": 1.5,
    "carbon neutral": 2.2, "diversity": 1.2, "inclusion": 1.2, "csr": 1.0,
    "governance": 0.8, "esg": 0.6, "award": 1.8, "recognised": 1.5, "recognized": 1.5,
}
ROUNDUP_PAT = re.compile(r"(?:stocks? to (?:watch|buy)|top (?:gainers|losers)|market (?:wrap|"
                         r"live|today)|nifty|sensex|buzzing stocks|stocks in focus|"
                         r"shares? (?:to|that) )", re.I)

CATEGORY_WORDS = {
    "Environmental": ["emission", "carbon", "climate", "pollution", "renewable", "solar", "water", "waste",
                      "energy", "net zero", "green", "environment", "biodiversity", "sustainab"],
    "Social": ["employee", "layoff", "diversity", "inclusion", "gender", "women", "safety", "labour",
               "labor", "csr", "community", "wage", "attrition", "hiring", "union", "harass"],
    "Governance": ["board", "audit", "fraud", "governance", "bribery", "corruption", "penalty", "fine",
                   "lawsuit", "sebi", "probe", "ethic", "compliance", "whistle", "scam", "insider"],
}


def categorize(text: str) -> str:
    t = text.lower()
    best = max(CATEGORY_WORDS, key=lambda k: sum(w in t for w in CATEGORY_WORDS[k]))
    return best if any(w in t for w in CATEGORY_WORDS[best]) else "General"


NAME_STOP = {"limited", "ltd", "technologies", "technology", "services", "systems",
             "software", "india", "solutions", "global", "company", "corporation",
             "and", "the", "of", "formerly", "legacy", "symbol"}


def aliases_for(code: str, full_name: str) -> List[str]:
    """Distinctive, word-boundary-safe aliases for a company."""
    out = {full_name.lower()}
    base = re.sub(r"\(.*?\)", "", full_name).strip().lower()
    out.add(base)
    tokens = [t for t in re.split(r"[^a-z0-9&]+", base) if t and t not in NAME_STOP]
    if tokens:
        out.add(" ".join(tokens[:2]) if len(tokens) > 1 else tokens[0])
    if len(code) >= 4 and code.isalpha():
        out.add(code.lower())
    if len(code) <= 3 and code.isalpha() and code in ("TCS",):
        out.add(code.lower())  # TCS is unambiguous in practice
    return sorted(a for a in out if len(a) >= 3)


class NewsSentimentAgent(BaseAgent):
    agent_id = "agent3"
    name = "Agent 3 - ESG News Sentiment Agent"
    role = "Build a defensible per-company ESG news sentiment signal."
    goal = ("Corpus inspected, attribution cleaned, articles ESG-filtered and "
            "scored sentence-wise, distribution audited, outputs written.")
    rules = [
        "Inspect the corpus before choosing cleaning steps.",
        "Attribute an article to a company only with title-level or lead-text "
        "evidence using word boundaries; reject market round-ups.",
        "Never report a high score from very few articles - shrink toward neutral.",
        "Audit the final sentiment distribution; if it is saturated, say so.",
    ]

    def __init__(self, *args, companies: Optional[List[str]] = None, **kw):
        self.codes = [c.upper() for c in companies] if companies else list(COMPANIES)
        super().__init__(*args, **kw)
        self.vader = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        if self.vader:
            self.vader.lexicon.update(ESG_LEXICON)
        self.memory.update({"corpus": None, "inspected": False, "attributed": None,
                            "scored": None, "audited": False, "finalized": False,
                            "diag": {}})

    # ------------------------------------------------------------------ tools
    def build_tools(self) -> List[Tool]:
        return [
            Tool("load_corpus", "Load the news corpus and report its shape and "
                 "how many articles look like multi-company round-ups.", self.t_load),
            Tool("attribute_articles",
                 "Assign articles to companies with word-boundary aliases; "
                 "reject round-ups. strict=true requires the alias in the title.",
                 self.t_attribute,
                 {"strict": {"type": "boolean"}}, []),
            Tool("score_articles", "ESG-filter attributed articles and score "
                 "sentiment sentence-by-sentence.", self.t_score),
            Tool("audit_distribution", "Check the sentiment distribution for "
                 "saturation and thin coverage.", self.t_audit),
            Tool("finalize", "Write the article-level and company-level CSVs "
                 "and publish coverage warnings.", self.t_finalize),
        ]

    # ----------------------------------------------------------------- stages
    def _path(self):
        for f in (ROOT / "raw-datasets").glob("*ESG News*"):
            return f
        return None

    def _load_dynamic(self):
        """GDELT per company (rate-limited ~1 request / 5 s; cached 3 h)."""
        frames, ok_n, failed = [], 0, []
        reset_gdelt_circuit()
        for i, c in enumerate(self.codes):
            res = fetch_news(COMPANIES[c]["full_name"], tracker=self.tracker)
            if not res.ok and "circuit open" in (res.error or ""):
                failed.extend(self.codes[i:])
                break
            if res.ok and res.payload:
                ok_n += 1
                d = pd.DataFrame(res.payload)
                for col in ("title", "url", "seendate", "domain"):
                    if col not in d:
                        d[col] = ""
                frames.append(pd.DataFrame({"Title": d["title"].fillna("").astype(str), "Description": "",
                                            "Content": "", "Date": pd.to_datetime(d["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce"),
                                            "URL": d["url"], "Domain": d["domain"], "Query_Company": c}))
            else:
                failed.append(c)
            if res.status == "success" and i < len(self.codes) - 1:
                time.sleep(5.2)          # respect GDELT's rate limit
        if not frames:
            return None, {"failed": failed}
        df = pd.concat(frames).drop_duplicates(["Title", "Query_Company"]).reset_index(drop=True)
        return df, {"companies_ok": ok_n, "failed": failed}

    def t_load(self) -> Dict[str, Any]:
        if self.dynamic:
            df, info = self._load_dynamic()
            if df is not None and len(df):
                self.memory.update({"corpus": df, "corpus_source": "GDELT"})
                self.memory["diag"]["roundup_like_titles"] = int(df["Title"].str.contains(ROUNDUP_PAT).sum())
                if info["failed"]:
                    self.add_finding("gdelt_gaps", "medium", f"GDELT returned nothing for {len(info['failed'])} companies",
                                     companies=info["failed"])
                return {"ok": True, "source": "GDELT", "articles": len(df), **info}
            self.add_finding("gdelt_unavailable", "high",
                             f"GDELT fetch failed; using bundled corpus. Reason: {gdelt_last_error()}")
            if self.tracker:
                with self.tracker.span("news:fallback_to_bundled_csv", "data_acquisition", "bundled csv") as rec:
                    rec["fallback_used"] = True
        self.memory["corpus_source"] = "bundled csv" if not self.dynamic else "bundled csv (GDELT fallback)"
        p = self._path()
        if p is None:
            return {"ok": False, "error": "news corpus not found in raw-datasets/"}
        df = pd.read_csv(p)
        for c in ("Title", "Description", "Content"):
            if c not in df.columns:
                df[c] = ""
        df["Title"] = df["Title"].fillna("").astype(str)
        df["Description"] = df["Description"].fillna("").astype(str)
        df["Content"] = df["Content"].fillna("").astype(str)
        df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce")
        roundup = df["Title"].str.contains(ROUNDUP_PAT)
        self.memory["corpus"] = df
        self.memory["diag"]["roundup_like_titles"] = int(roundup.sum())
        return {"ok": True, "articles": len(df),
                "date_min": str(df["Date"].min().date()) if df["Date"].notna().any() else None,
                "date_max": str(df["Date"].max().date()) if df["Date"].notna().any() else None,
                "roundup_like_titles": int(roundup.sum())}

    def t_attribute(self, strict: bool = True) -> Dict[str, Any]:
        df = self.memory["corpus"]
        pats = {c: re.compile(r"\b(" + "|".join(re.escape(a) for a in
                              aliases_for(c, COMPANIES[c]["full_name"])) + r")\b", re.I)
                for c in self.codes}
        rows, rejected_roundup, rejected_multi = [], 0, 0
        for i, r in df.iterrows():
            title = r["Title"]
            lead = (r["Description"] + " " + r["Content"][:300])
            hits_title = [c for c, p in pats.items() if p.search(title)]
            # market round-ups ("Stocks to Watch Today: A, B, C ...") are rejected whatever the
            # number of tracked companies in the title (a single-company run sees only 1 hit)
            if len(hits_title) > 3 or ROUNDUP_PAT.search(title) or title.count(",") >= 4:
                rejected_roundup += 1
                continue
            if hits_title:
                cands = hits_title
            elif not strict:
                cands = [c for c, p in pats.items() if p.search(lead)]
                if len(cands) != 1:
                    rejected_multi += int(len(cands) > 1)
                    continue
            else:
                continue
            for c in cands:
                rows.append({"idx": i, "Company_Code": c, "Title": title, "URL": r.get("URL", ""),
                             "Domain": r.get("Domain", ""),
                             "Description": r["Description"], "Date": r["Date"],
                             "evidence": "title" if hits_title else "lead"})
        att = pd.DataFrame(rows)
        self.memory["attributed"] = att
        per = att.groupby("Company_Code").size() if not att.empty else pd.Series(dtype=int)
        return {"ok": True, "attributed_rows": len(att), "rejected_roundups": rejected_roundup,
                "rejected_ambiguous": rejected_multi,
                "companies_with_articles": int(per.shape[0]),
                "companies_with_5plus": int((per >= 5).sum())}

    def _relevance(self, text: str) -> Tuple[int, bool, List[str], List[str]]:
        t = text.lower()
        inc = [k for k in ESG_INCLUDE_KEYWORDS if re.search(r"\b" + re.escape(k) + r"\b", t)]
        exc = [k for k in ESG_EXCLUDE_KEYWORDS if re.search(r"\b" + re.escape(k) + r"\b", t)]
        score = len(inc) - len(exc)
        return score, (len(inc) >= 1 and score >= 1), inc, exc

    def _sentiment(self, title: str, description: str) -> float:
        sents = [title] + re.split(r"(?<=[.!?])\s+", description)[:3]
        vals = [self.vader.polarity_scores(s)["compound"] for s in sents if s.strip()]
        return float(np.mean(vals)) if vals else 0.0

    def t_score(self) -> Dict[str, Any]:
        if self.vader is None:
            return {"ok": False, "error": "vaderSentiment is not installed"}
        att = self.memory["attributed"]
        keep, drop, general = [], [], []
        for _, r in att.iterrows():
            text = f"{r['Title']}. {r['Description']}"
            sc, ok, inc, exc = self._relevance(text)
            if self.memory.get("corpus_source") == "GDELT":
                ok = ok or sc >= 0        # GDELT query already filtered for ESG vocabulary
            base = {"Company_Code": r["Company_Code"],
                    "Company_Name": COMPANIES[r["Company_Code"]]["full_name"],
                    "Article_Title": r["Title"][:200], "Article_Date": r["Date"],
                    "ESG_Relevance_Score": sc, "Is_ESG_Relevant": ok,
                    "Matched_ESG_Keywords": ", ".join(inc),
                    "Matched_Non_ESG_Keywords": ", ".join(exc),
                    "Attribution_Evidence": r["evidence"], "ESG_Category": categorize(text),
                    "URL": r.get("URL", ""), "Domain": r.get("Domain", ""),
                    "Analysis_Date": date.today().isoformat()}
            comp = self._sentiment(r["Title"], r["Description"])
            general.append({"Company_Code": r["Company_Code"], "score": comp})
            if not ok:
                drop.append({**base, "Rejection_Reason": "Low ESG relevance or finance-only context"})
                continue
            vs = self.vader.polarity_scores(r["Title"])
            keep.append({**base, "Sentiment_Score": round(comp, 4),
                         "Sentiment_Class": "Positive" if comp >= 0.15 else
                         "Negative" if comp <= -0.15 else "Neutral",
                         "Positive_Score": vs["pos"], "Negative_Score": vs["neg"],
                         "Neutral_Score": vs["neu"]})
        self.memory["scored"] = pd.DataFrame(keep)
        self.memory["rejected"] = pd.DataFrame(drop)
        self.memory["general"] = pd.DataFrame(general)
        return {"ok": True, "esg_relevant": len(keep), "rejected_not_esg": len(drop)}

    def t_audit(self) -> Dict[str, Any]:
        s = self.memory["scored"]
        if s is None or s.empty:
            self.memory["audited"] = True
            return {"ok": True, "warning": "no scored articles"}
        extreme = float((s["Sentiment_Score"].abs() > 0.9).mean())
        classes = s["Sentiment_Class"].value_counts(normalize=True).round(3).to_dict()
        per = s.groupby("Company_Code").size()
        thin = sorted(set(self.codes) - set(per[per >= 5].index))
        self.memory["diag"].update({"extreme_share": extreme, "classes": classes,
                                    "thin_companies": thin})
        self.memory["audited"] = True
        return {"ok": True, "share_abs_gt_0.9": round(extreme, 3), "class_mix": classes,
                "mean": round(float(s["Sentiment_Score"].mean()), 3),
                "std": round(float(s["Sentiment_Score"].std()), 3),
                "companies_below_5_articles": len(thin)}

    def t_finalize(self) -> Dict[str, Any]:
        s, rej = self.memory["scored"], self.memory["rejected"]
        rows, summary = [], []
        for c in self.codes:
            sub = s[s["Company_Code"] == c] if s is not None and not s.empty else pd.DataFrame()
            gen = self.memory.get("general")
            g = gen[gen["Company_Code"] == c]["score"] if gen is not None and not gen.empty else pd.Series(dtype=float)
            g_n = len(g)
            g_shrunk = float(g.mean()) * g_n / (g_n + 5.0) if g_n else None
            if sub.empty:
                summary.append({"Company_Code": c, "Articles": 0, "Raw_Sentiment": None,
                                "Shrunk_Sentiment": None, "Sentiment_Reliability": 0.0,
                                "News_Sentiment_Score_10": None,
                                "General_Articles": g_n,
                                "General_Sentiment_Shrunk": None if g_shrunk is None else round(g_shrunk, 4)})
                continue
            n = len(sub)
            raw = float(sub["Sentiment_Score"].mean())
            rel = n / (n + 5.0)
            shrunk = raw * rel                       # shrink toward neutral (0)
            summary.append({"Company_Code": c, "Articles": n, "Raw_Sentiment": round(raw, 4),
                            "Shrunk_Sentiment": round(shrunk, 4),
                            "Sentiment_Reliability": round(rel, 3),
                            "Positive_Share": round(float((sub["Sentiment_Class"] == "Positive").mean()), 3),
                            "Negative_Share": round(float((sub["Sentiment_Class"] == "Negative").mean()), 3),
                            "News_Sentiment_Score_10": round(5 + 5 * shrunk, 2),
                            "General_Articles": g_n,
                            "General_Sentiment_Shrunk": None if g_shrunk is None else round(g_shrunk, 4),
                            "Environmental_Events": int((sub["ESG_Category"] == "Environmental").sum()),
                            "Social_Events": int((sub["ESG_Category"] == "Social").sum()),
                            "Governance_Events": int((sub["ESG_Category"] == "Governance").sum()),
                            "Negative_Events": int((sub["Sentiment_Class"] == "Negative").sum()),
                            "Corpus_Source": self.memory.get("corpus_source")})
        ps = PROCESSED / OUTPUT_FILES["news_sentiment"]
        (s if s is not None else pd.DataFrame()).to_csv(ps, index=False)
        (rej if rej is not None else pd.DataFrame()).to_csv(PROCESSED / OUTPUT_FILES["news_rejected"], index=False)
        pd.DataFrame(summary).to_csv(PROCESSED / "esg_news_company_summary.csv", index=False)
        thin = [r["Company_Code"] for r in summary if r["Articles"] < 5]
        self.board.post(self.agent_id, "news_coverage",
                        {"companies_below_5_articles": thin,
                         "extreme_share": self.memory["diag"].get("extreme_share")})
        self.memory["finalized"] = True
        self.artifacts.append(str(ps.relative_to(ROOT)))
        return {"ok": True, "articles_scored": 0 if s is None else len(s),
                "companies_with_5plus_articles": len(self.codes) - len(thin)}

    # --------------------------------------------------------------- reasoning
    def observe(self) -> Dict[str, Any]:
        m = self.memory
        return {"corpus_loaded": m["corpus"] is not None,
                "attributed": m["attributed"] is not None,
                "scored": m["scored"] is not None, "audited": m["audited"],
                "finalized": m["finalized"], "diagnostics": m["diag"]}

    def is_done(self, state) -> Tuple[bool, str]:
        if state["finalized"]:
            return True, "news sentiment written with reliability and shrinkage"
        return False, ""

    def policy(self, state) -> Optional[Decision]:
        if not state["corpus_loaded"]:
            return Decision("load_corpus", {}, "inspect the corpus first")
        if not state["attributed"]:
            return Decision("attribute_articles", {"strict": True},
                            "title-level attribution avoids false credit")
        if not state["scored"]:
            return Decision("score_articles", {}, "score relevant attributed articles")
        if not state["audited"]:
            return Decision("audit_distribution", {}, "verify the score distribution")
        return Decision("finalize", {}, "write outputs")

    def reflect(self, decision: Decision, result: Dict[str, Any]) -> None:
        if decision.tool == "attribute_articles" and result.get("ok") \
                and result.get("companies_with_5plus", 0) < len(self.codes) * 0.4 \
                and not self.memory.get("relaxed"):
            # Too few companies have usable evidence: relax to lead-text
            # attribution once and re-run, recording that it was relaxed.
            self.memory["relaxed"] = True
            self.memory["attributed"] = None
            self.add_finding("relaxed_attribution", "low",
                             "strict title attribution left <40% of companies with 5+ "
                             "articles; relaxed to lead-text attribution")
            self._relax_next = True
        if decision.tool == "audit_distribution" and result.get("share_abs_gt_0.9", 0) > 0.5:
            self.add_finding("saturated_sentiment", "high",
                             "over half of article scores are extreme (|s|>0.9)")

    def _choose(self, state):
        d, e = super()._choose(state)
        if d and d.tool == "attribute_articles" and getattr(self, "_relax_next", False):
            d.args = {"strict": False}
            self._relax_next = False
        return d, e
