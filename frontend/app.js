/* ESG Intelligence System - single-page app (vanilla JS, no build step). */
const $ = (s, r = document) => r.querySelector(s);
const API = "";
const PAGES = [
  ["analyze", "Analyze Company"], ["dashboard", "Dashboard"], ["performance", "ESG Performance"],
  ["market", "Market Analysis"], ["news", "News Intelligence"], ["env", "Environmental Risk"],
  ["greenwashing", "Greenwashing Detection"], ["agents", "Agent Activity"], ["timing", "Performance"],
  ["sources", "Sources & Evidence"], ["validation", "Validation & Research"], ["cohort", "Cohort Ranking"]];
const S = { page: "analyze", report: null, job: null, timer: null };

const esc = v => String(v ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v, d = 1) => (v === null || v === undefined || v === "" || isNaN(v)) ? "—" : Number(v).toFixed(d);
const badge = (t, k) => `<span class="badge b-${k}">${esc(t)}</span>`;
const riskBadge = r => badge(r || "n/a", { Low: "good", Medium: "warn", High: "bad" }[r] || "blue");
const bar = (v, max = 100) => `<div class="bar"><i style="width:${Math.max(0, Math.min(100, (v / max) * 100))}%"></i></div>`;
const cite = t => esc(t).replace(/\[([A-Z]\d+)\]/g, '<span class="cite">[$1]</span>');
async function get(u) { const r = await fetch(API + u); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); return r.json(); }
async function post(u, b) { const r = await fetch(API + u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) }); if (!r.ok) throw new Error(await r.text()); return r.json(); }

function nav() {
  $("#nav").innerHTML = PAGES.map(([k, t]) => `<a href="#${k}" class="${S.page === k ? "on" : ""}">${t}</a>`).join("");
  const r = S.report;
  $("#ctxTitle").innerHTML = r ? `<h2>${esc(r.scores.Company_Name)} <small class="muted">${esc(r.company)}</small></h2><p class="muted">FY 2024-25 · run: ${esc(r.run)}</p>`
    : `<h2>${esc(PAGES.find(p => p[0] === S.page)[1])}</h2><p class="muted">BRSR · XBRL · Market · News · Air quality · Agents</p>`;
}
function route() { S.page = (location.hash || "#analyze").slice(1); if (!PAGES.find(p => p[0] === S.page)) S.page = "analyze"; nav(); render(); }
window.addEventListener("hashchange", route);

async function openCompany(code, run) {
  try { S.report = run ? await get(`/api/report/${run}?company=${code}`) : await get(`/api/company/${encodeURIComponent(code)}`); location.hash = "#dashboard"; route(); }
  catch (e) { alert("Could not open " + code + ": " + e.message); }
}
$("#quickBtn").onclick = () => $("#quick").value && openCompany($("#quick").value.trim().toUpperCase());

/* ------------------------------------------------------------ pages ----- */
const need = () => `<div class="card empty">No company loaded yet. <a class="lnk" href="#analyze">Analyze a company</a> or pick one from <a class="lnk" href="#cohort">Cohort Ranking</a>.</div>`;

async function pAnalyze() {
  const runs = (await get("/api/runs")).runs;
  $("#view").innerHTML = `
  <div class="card"><h3>AI-powered ESG analysis</h3>
    <p class="muted">Enter a company name or NSE symbol. The agents discover the BRSR/XBRL filings, fetch market, news and air-quality data, validate, score, check consistency and explain the result.</p>
    <div class="row" style="margin-top:12px"><input id="co" style="min-width:300px" placeholder="Infosys / INFY / HDFCBANK">
    <input id="hq" placeholder="HQ city (only if outside the 50-company cohort)">
    <label><input type="checkbox" id="dyn" checked> live market &amp; news data</label>
    <label><input type="checkbox" id="llm" checked> use LLM (if configured)</label>
    <button class="primary" id="go">Analyze company</button></div></div>
  <div id="prog"></div>
  <div class="card" style="margin-top:14px"><h3>Previous analyses</h3>${runs.length ? `<table><tr><th>Run</th><th>Company</th><th>Seconds</th><th></th></tr>${runs.map(r =>
    `<tr><td>${esc(r.run)}</td><td>${esc((r.companies || []).join(", "))}</td><td>${num(r.total_seconds)}</td><td><a class="lnk" data-run="${esc(r.run)}" data-code="${esc((r.companies || [""])[0])}">open</a></td></tr>`).join("")}</table>` : `<p class="muted">None yet.</p>`}</div>`;
  document.querySelectorAll("a[data-run]").forEach(a => a.onclick = () => openCompany(a.dataset.code, a.dataset.run));
  $("#go").onclick = async () => {
    const company = $("#co").value.trim(); if (!company) return;
    $("#go").disabled = true;
    try { const j = await post("/api/analyze", { company, hq_city: $("#hq").value, dynamic: $("#dyn").checked, use_llm: $("#llm").checked }); S.job = j.job_id; poll(); }
    catch (e) { alert(e.message); $("#go").disabled = false; }
  };
  if (S.job) poll();
}
const AGENTS = [["agent0", "Filing discovery"], ["agent1", "BRSR/XBRL extraction"], ["agent2", "Environmental context"], ["agent3", "News sentiment"], ["agent4", "Market performance"], ["agent5", "Consistency / greenwashing"], ["agent6", "ESG scoring"], ["agent7", "Explanation"]];
function poll() {
  clearTimeout(S.timer);
  get(`/api/jobs/${S.job}`).then(j => {
    const done = {}; (j.agents_finished || []).forEach(l => { const m = l.match(/\[(agent\d)\] (\w+)/); if (m) done[m[1]] = m[2]; });
    const spans = (j.metrics && j.metrics.spans) || [];
    const dur = id => { const s = spans.find(x => x.source === id && x.kind === "agent"); return s ? num(s.duration, 2) + " s" : ""; };
    const el = $("#prog"); if (!el) return;
    el.innerHTML = `<div class="card" style="margin-top:14px"><h3>Analyzing ${esc(j.company)} — ${esc(j.status)}</h3>
      ${AGENTS.map(([id, t]) => { const st = done[id]; const cls = st === "completed" ? "ok" : st ? "bad" : (j.status === "running" ? "run" : ""); return `<div class="step"><span class="dot ${cls}"></span><b>${t}</b><span class="muted">${st || (j.status === "running" ? "waiting / running" : "")}</span><span style="margin-left:auto">${dur(id)}</span></div>`; }).join("")}
      <details open><summary>Live log</summary><div class="log">${esc((j.log_tail || []).join("\n"))}</div></details>
      ${j.status === "completed" ? `<p><button class="primary" id="openRes">Open results</button></p>` : ""}
      ${j.status === "failed" ? `<div class="warnbox">The run failed. See the log above. ${esc(j.error || "")}</div>` : ""}</div>`;
    if (j.status === "running") S.timer = setTimeout(poll, 1500);
    else { $("#go") && ($("#go").disabled = false); const b = $("#openRes"); if (b) b.onclick = async () => { const rep = await get(`/api/report/${j.run}`); S.report = rep; location.hash = "#dashboard"; route(); }; }
  }).catch(() => { });
}

function ring(v) {
  const r = 58, c = 2 * Math.PI * r, p = v == null ? 0 : v / 100, col = v == null ? "var(--line)" : v >= 70 ? "var(--good)" : v >= 50 ? "var(--warn)" : "var(--bad)";
  return `<svg class="ring" viewBox="0 0 150 150"><circle cx="75" cy="75" r="${r}" fill="none" stroke="var(--line)" stroke-width="14"/><circle cx="75" cy="75" r="${r}" fill="none" stroke="${col}" stroke-width="14" stroke-linecap="round" stroke-dasharray="${c * p} ${c}" transform="rotate(-90 75 75)"/><text x="75" y="82" text-anchor="middle" font-size="28" font-weight="700">${v == null ? "—" : num(v, 1)}</text><text x="75" y="102" text-anchor="middle" font-size="11" fill="var(--muted)">/ 100</text></svg>`;
}
function pDashboard() {
  const r = S.report; if (!r) return $("#view").innerHTML = need();
  const s = r.scores, ex = (r.explanation || [])[0] || {};
  const pil = [["Environmental", s.Environmental_Score], ["Social", s.Social_Score], ["Governance", s.Governance_Score]];
  $("#view").innerHTML = `
  <div class="grid g2"><div class="card"><div class="row">${ring(s.Master_ESG_Score)}<div>
    <div class="kpi">${s.Master_ESG_Score == null ? "Score withheld" : num(s.Master_ESG_Score, 1)}<small> master ESG</small></div>
    <p>${riskBadge(s.Risk_Level)} ${badge("evidence: " + s.Quality_Gate_Status, s.Quality_Gate_Status === "Pass" ? "good" : "warn")} ${s.Rank ? badge("rank " + s.Rank, "blue") : ""}</p>
    <p class="muted">${s.Usable_Metrics}/18 BRSR metrics trusted</p></div></div></div>
  <div class="card"><h3>Pillars</h3>${pil.map(([n, v]) => `<div style="margin:8px 0"><div class="row" style="justify-content:space-between"><b>${n}</b><span>${v == null ? "no reliable BRSR evidence" : num(v, 1)}</span></div>${bar(v || 0)}</div>`).join("")}</div></div>
  <div class="card" style="margin-top:14px"><h3>Why this score? ${ex.Narrative_Source ? badge(ex.Narrative_Source, String(ex.Narrative_Source).startsWith("llm") ? "good" : "blue") : ""} ${ex.Grounding_Check ? badge("grounding: " + ex.Grounding_Check, ex.Grounding_Check === "pass" ? "good" : "warn") : ""}</h3>
    <div class="expl">${cite(ex.Score_Explanation || "No explanation available.")}</div>
    ${ex.Recommendations ? `<h3 style="margin-top:14px">Suggested actions</h3><ul>${ex.Recommendations.split(" | ").map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : ""}
    <p class="muted">Bracketed IDs (e.g. [M1]) point to rows in <a class="lnk" href="#sources">Sources &amp; Evidence</a>.</p></div>`;
}
function pPerformance() {
  const r = S.report; if (!r) return $("#view").innerHTML = need();
  const tr = Object.fromEntries((r.trends || []).map(t => [t.Metric, t]));
  const arrow = m => { const t = tr[m]; if (!t) return ""; const a = t.Direction === "improving" ? "▲" : t.Direction === "worsening" ? "▼" : "→"; return `<span style="color:var(--${t.Direction === "improving" ? "good" : t.Direction === "worsening" ? "bad" : "muted"})">${a} ${num(t.Change, 1)}${esc(t.Change_Unit)}</span>`; };
  const block = (p, title) => { const rows = (r.metrics || []).filter(m => m.Pillar === p); return `<div class="card"><h3>${title}</h3>${rows.length ? `<table><tr><th>Metric</th><th>Value</th><th>Score</th><th>YoY</th><th>Source</th></tr>${rows.map(m => `<tr><td>${esc(m.Metric)}</td><td>${num(m.Value, 2)}</td><td>${num(m.Score_0_100, 0)}${bar(m.Score_0_100)}</td><td>${arrow(m.Metric)}</td><td class="muted">${esc(m.Source)}${m.Page ? " · p." + m.Page : ""}</td></tr>`).join("")}</table>` : `<p class="muted">No trusted BRSR evidence for this pillar yet (see Sources &amp; Evidence for what was quarantined).</p>`}</div>`; };
  $("#view").innerHTML = `<div class="grid">${block("E", "Environmental")}${block("S", "Social")}${block("G", "Governance")}</div>
  <div class="card" style="margin-top:14px"><h3>Year-over-year trends (from prior-year values in the same XBRL)</h3>${(r.trends || []).length ? `<table><tr><th>Metric</th><th>Prior</th><th>Current</th><th>Change</th><th></th></tr>${r.trends.map(t => `<tr><td>${esc(t.Metric)}</td><td>${num(t.Prior, 2)}</td><td>${num(t.Current, 2)}</td><td>${num(t.Change, 1)} ${esc(t.Change_Unit)}</td><td>${badge(t.Direction, t.Direction === "improving" ? "good" : t.Direction === "worsening" ? "bad" : "blue")}</td></tr>`).join("")}</table>` : `<p class="muted">No prior-year XBRL values available.</p>`}</div>`;
}
function pMarket() {
  const r = S.report; if (!r) return $("#view").innerHTML = need();
  const m = (r.market || [])[0];
  if (!m || m.Coverage !== "price_data") return $("#view").innerHTML = `<div class="card empty">No price history for this company. ${esc(m && m.Error || "")}<br>Run with <code>--dynamic</code> to fetch live data.</div>`;
  const items = [["1M", m.Return_1M_Pct], ["3M", m.Return_3M_Pct], ["6M", m.Return_6M_Pct], ["1Y", m.Return_1Y_Pct], ["3Y", m.Return_3Y_Pct], ["FY24-25", m.FY_Return_Pct]];
  $("#view").innerHTML = `<div class="grid g4">${items.map(([k, v]) => `<div class="card"><span class="muted">${k} return</span><div class="kpi" style="color:var(--${v >= 0 ? "good" : "bad"})">${num(v, 1)}%</div></div>`).join("")}
  <div class="card"><span class="muted">Volatility</span><div class="kpi">${num(m.Volatility_Pct, 1)}%</div></div><div class="card"><span class="muted">Max drawdown</span><div class="kpi">${num(m.Max_Drawdown_Pct, 1)}%</div></div>
  <div class="card"><span class="muted">Market score (vs cohort)</span><div class="kpi">${num(m.Stock_Performance_Score, 1)}<small>/10</small></div></div></div>
  <div class="card" style="margin-top:14px"><h3>Return profile</h3>${items.map(([k, v]) => `<div class="row"><span style="width:60px">${k}</span><div style="flex:1">${bar(Math.abs(v || 0), 60)}</div><span style="width:70px;text-align:right">${num(v, 1)}%</span></div>`).join("")}
  <p class="muted">Source: ${esc(m.Data_Source)} · window end ${esc(m.Window_End)} · ${m.Data_Points} data points. ESG-vs-market relationships are statistical associations, not causation (see Validation &amp; Research).</p></div>`;
}
function pNews() {
  const r = S.report; if (!r) return $("#view").innerHTML = need();
  const n = (r.news_summary || [])[0] || {}, arts = r.news_articles || [];
  if (!arts.length && !(n.General_Articles > 0)) return $("#view").innerHTML = `<div class="card empty">No ESG news found for this company in the corpus. Run with <code>--dynamic</code> for live GDELT news.</div>`;
  const pos = arts.filter(a => a.Sentiment_Class === "Positive").length, neg = arts.filter(a => a.Sentiment_Class === "Negative").length, neu = arts.length - pos - neg, t = Math.max(arts.length, 1);
  $("#view").innerHTML = `<div class="grid g4"><div class="card"><span class="muted">ESG articles</span><div class="kpi">${n.Articles ?? 0}</div></div>
   <div class="card"><span class="muted">Positive / Neutral / Negative</span><div class="kpi" style="font-size:20px">${Math.round(100 * pos / t)}% · ${Math.round(100 * neu / t)}% · ${Math.round(100 * neg / t)}%</div></div>
   <div class="card"><span class="muted">E / S / G events</span><div class="kpi" style="font-size:20px">${n.Environmental_Events ?? 0} / ${n.Social_Events ?? 0} / ${n.Governance_Events ?? 0}</div></div>
   <div class="card"><span class="muted">Reliability</span><div class="kpi">${num(n.Sentiment_Reliability, 2)}</div><span class="muted">${esc(n.Corpus_Source || "")}</span></div></div>
  <div class="card" style="margin-top:14px"><h3>Articles</h3>${arts.length ? `<table><tr><th>Date</th><th>Headline</th><th>Category</th><th>Sentiment</th></tr>${arts.map(a => `<tr><td>${esc(String(a.Article_Date || "").slice(0, 10))}</td><td>${a.URL ? `<a class="lnk" href="${esc(a.URL)}" target="_blank" rel="noopener">${esc(a.Article_Title)}</a>` : esc(a.Article_Title)}<div class="muted">${esc(a.Domain || "")}</div></td><td>${esc(a.ESG_Category || "")}</td><td>${badge(a.Sentiment_Class, a.Sentiment_Class === "Positive" ? "good" : a.Sentiment_Class === "Negative" ? "bad" : "blue")} ${num(a.Sentiment_Score, 2)}</td></tr>`).join("")}</table>` : `<p class="muted">No ESG-relevant articles retained.</p>`}
   <p class="muted">Scores use sentence-level sentiment shrunk toward neutral by sample size; small samples are not trusted.</p></div>`;
}
function pEnv() {
  const r = S.report; if (!r) return $("#view").innerHTML = need();
  const a = r.air || [];
  $("#view").innerHTML = `<div class="card"><h3>Operating locations — air-quality context</h3>${a.length ? `<table><tr><th>Location</th><th>Type</th><th>Avg AQI</th><th>Category</th><th>Evidence</th><th>Confidence</th></tr>${a.map(x => `<tr><td>${esc(x.City)}${x.Area ? " · " + esc(x.Area) : ""}</td><td>${esc(x.Location_Type)}</td><td>${num(x.Avg_AQI, 0)}</td><td>${esc(x.Risk_Level || "unresolved")}</td><td class="muted">${esc(x.Match_Method)}${x.Matched_City ? " → " + esc(x.Matched_City) : ""}<br>${esc(x.AQI_Source || "")}</td><td>${num(x.Evidence_Confidence, 2)}</td></tr>`).join("")}</table>` : `<p class="muted">No locations.</p>`}
  <div class="warnbox">Air quality reflects the whole location (traffic, industry, weather). It is shown as <b>context</b> for the company's environmental performance, never as proof of it.</div></div>`;
}
function pGreen() {
  const r = S.report; if (!r) return $("#view").innerHTML = need();
  const c = (r.consistency || [])[0] || {}, claims = r.claims || [], sig = r.signals || [];
  const vb = v => badge(v, v === "Supported" ? "good" : v === "Partly offset" ? "warn" : "bad");
  $("#view").innerHTML = `<div class="card"><div class="row"><div class="kpi">${esc(c.Risk_Level || "n/a")}<small> consistency risk</small></div>${c.Trust_Score != null ? badge("trust " + num(c.Trust_Score, 1) + "/10", "blue") : ""}</div>
  <p class="muted">Signals indicate potential inconsistencies between disclosures and independent evidence. They are not accusations.</p></div>
  <div class="card" style="margin-top:14px"><h3>Claim → evidence → verdict</h3>${claims.length ? `<table><tr><th>Claim</th><th>Verdict</th><th>Evidence</th></tr>${claims.map(x => `<tr><td>${esc(x.Claim)}</td><td>${vb(x.Verdict)}</td><td class="muted">${esc(x.Evidence)}</td></tr>`).join("")}</table>` : `<p class="muted">No improvement claims to test (no prior-year data).</p>`}</div>
  <div class="card" style="margin-top:14px"><h3>Signals</h3>${sig.length ? `<table>${sig.map(x => `<tr><td>${esc(x.signal)}</td><td>${num(x.strength, 2)}</td><td class="muted">${esc(x.evidence)}</td></tr>`).join("")}</table>` : `<p class="muted">No inconsistency signals.</p>`}</div>`;
}
function pAgents() {
  const r = S.report; if (!r) return $("#view").innerHTML = need();
  const ag = r.agents || {};
  $("#view").innerHTML = `<div class="grid g2">${Object.entries(ag).map(([id, a]) => `<div class="card"><h3>${esc(a.name)}</h3>
    <p>${badge(a.status, a.status === "completed" ? "good" : "bad")} ${num(a.seconds, 2)} s · ${a.steps} steps · ${a.llm ? "LLM" : "policy"} mode</p>
    <p class="muted">${esc(a.summary)}</p><p>Decisions: ${Object.entries(a.decisions || {}).map(([k, v]) => `${k}: ${v}`).join(" · ")}</p>
    ${(a.findings || []).slice(0, 3).map(f => `<div class="warnbox">${esc(f.message)}</div>`).join("")}
    <details><summary>Decision trace (last ${(a.history || []).length})</summary><table>${(a.history || []).map(h => `<tr><td>${h.step}</td><td><b>${esc(h.tool)}</b> <span class="muted">${esc(JSON.stringify(h.args)).slice(0, 80)}</span><br><span class="muted">[${esc(h.source)}] ${esc(String(h.observation).slice(0, 160))}</span></td></tr>`).join("")}</table></details></div>`).join("") || `<div class="card empty">No agent traces in this run.</div>`}</div>`;
}
function pTiming() {
  const r = S.report; if (!r) return $("#view").innerHTML = need();
  const m = r.run_metrics; if (!m) return $("#view").innerHTML = `<div class="card empty">No timing data for this run.</div>`;
  const s = m.summary, spans = m.spans || [];
  $("#view").innerHTML = `<div class="grid g4"><div class="card"><span class="muted">End-to-end</span><div class="kpi">${num(s.total_seconds, 2)} s</div></div>
  <div class="card"><span class="muted">Bottleneck</span><div class="kpi" style="font-size:16px">${s.bottleneck ? esc(s.bottleneck.name) + "<br>" + num(s.bottleneck.seconds, 2) + " s" : "—"}</div></div>
  <div class="card"><span class="muted">Fallbacks used</span><div class="kpi">${(s.fallbacks_used || []).length}</div></div>
  <div class="card"><span class="muted">Failures</span><div class="kpi">${(s.failures || []).length}</div></div></div>
  <div class="card" style="margin-top:14px"><h3>Stages (wall-clock vs total work)</h3><table><tr><th>Stage</th><th>Wall</th><th>Sum of parts</th><th>Parallel gain</th></tr>${Object.entries(s.stages || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${num(v.stage_wall_seconds, 2)} s</td><td>${num(v.sum_seconds, 2)} s</td><td>${v.stage_wall_seconds != null ? num(v.sum_seconds - v.stage_wall_seconds, 2) + " s" : "—"}</td></tr>`).join("")}</table></div>
  <div class="card" style="margin-top:14px"><h3>Every step</h3><table><tr><th>Step</th><th>Source</th><th>Status</th><th>Seconds</th><th>Fallback</th></tr>${spans.map(x => `<tr><td>${esc(x.name)}</td><td>${esc(x.source)}</td><td>${badge(x.status, x.status === "success" ? "good" : "bad")}</td><td>${num(x.duration, 2)}</td><td>${x.fallback_used ? "yes" : ""}</td></tr>`).join("")}</table></div>`;
}
function pSources() {
  const r = S.report; if (!r) return $("#view").innerHTML = need();
  const ev = r.evidence_package || {}, f = r.filings;
  const src = [["BRSR PDF", f && Object.values(f.companies || {})[0]?.valid?.pdf], ["XBRL", f && Object.values(f.companies || {})[0]?.valid?.xbrl], ["Market", (r.market || [])[0]?.Coverage === "price_data"], ["News", (r.news_summary || [])[0]?.Articles > 0], ["Air quality", (r.air || []).some(a => a.Avg_AQI != null)]];
  $("#view").innerHTML = `<div class="card"><h3>Data sources</h3><div class="row">${src.map(([n, ok]) => `<span>${badge(n + (ok === undefined ? " ?" : ok ? " ✓" : " ✗"), ok ? "good" : ok === undefined ? "blue" : "warn")}</span>`).join("")}</div>
  <p class="muted">Market: ${esc((r.market || [])[0]?.Data_Source || "n/a")} · News corpus: ${esc((r.news_summary || [])[0]?.Corpus_Source || "n/a")}</p></div>
  <div class="card" style="margin-top:14px"><h3>BRSR metrics — provenance</h3><table><tr><th>Metric</th><th>Value</th><th>Unit</th><th>Source</th><th>Status</th><th>Confidence</th><th>Audit note</th></tr>${(r.brsr || []).map(x => `<tr><td>${esc(x.Metric_Name)}</td><td>${x.Metric_Value == null ? "—" : num(x.Metric_Value, 2)}</td><td>${esc(x.Unit)}</td><td class="muted">${esc(x.Final_Source)}</td><td>${badge(x.Validation_Status || "n/a", x.Validation_Status === "VALID" ? "good" : x.Validation_Status === "QUARANTINED" ? "bad" : "warn")}</td><td>${num(x.Confidence_Score, 2)}</td><td class="muted">${esc(String(x.Audit_Note || "").slice(0, 120))}</td></tr>`).join("")}</table></div>
  <div class="card" style="margin-top:14px"><h3>Evidence package given to the Explanation Agent</h3><details><summary>show JSON</summary><div class="log">${esc(JSON.stringify(ev, null, 1))}</div></details></div>`;
}
async function pValidation() {
  const v = await get("/api/validation"), a = (v.summary || {}).analyses || {};
  const names = { nse_top200: "Our ESG vs NSE Top 200", nse_cfa: "Our BRSR metrics vs NSE/CFA research", esg_stock: "ESG ↔ Stock", esg_news: "ESG ↔ News", esg_air: "ESG ↔ Air quality" };
  const tbl = rows => rows && rows.length ? `<div style="overflow:auto"><table><tr>${Object.keys(rows[0]).map(k => `<th>${esc(k)}</th>`).join("")}</tr>${rows.map(r => `<tr>${Object.values(r).map(x => `<td>${esc(typeof x === "number" ? num(x, 3) : x)}</td>`).join("")}</tr>`).join("")}</table></div>` : "";
  const files = { nse_top200: "validation_nse_top200.csv", nse_cfa: "validation_nse_cfa.csv", esg_stock: "validation_esg_stock.csv", esg_news: "validation_esg_news.csv", esg_air: "validation_esg_air.csv" };
  $("#view").innerHTML = `<div class="warnbox">${esc((v.summary || {}).caveat || "Run the batch pipeline first: python run_agents.py")}</div><div class="grid g2">${Object.entries(names).map(([k, t]) => { const x = a[k] || {}; return `<div class="card"><h3>${t}</h3><p>${badge(x.status || "not run", x.status === "ok" ? "good" : "warn")}</p>${x.how ? `<p class="muted">${esc(x.how)}</p>` : ""}${tbl(v.tables[files[k]])}</div>`; }).join("")}</div>
  <div class="card" style="margin-top:14px"><h3>Benchmark files</h3><p>nse_top200_esg.csv: ${v.benchmark_files_present["nse_top200_esg.csv"] ? badge("present", "good") : badge("missing", "warn")} · nse_cfa_reference.csv: ${v.benchmark_files_present["nse_cfa_reference.csv"] ? badge("present", "good") : badge("missing", "warn")}</p><p class="muted">See data/benchmark/README.txt for the file formats, then run <code>python run_agents.py --agents agent8</code>.</p></div>`;
}
async function pCohort() {
  const rows = (await get("/api/cohort")).companies;
  $("#view").innerHTML = `<div class="card"><div class="row" style="justify-content:space-between"><h3>Batch results (${rows.length} companies)</h3><button id="runBatch">Re-run batch (offline)</button></div>
  <table><tr><th>Rank</th><th>Company</th><th>Master</th><th>E</th><th>S</th><th>G</th><th>Risk</th><th>Evidence</th></tr>${rows.map(r => `<tr><td>${r.Rank ?? "—"}</td><td><a class="lnk" data-c="${esc(r.Company_Code)}">${esc(r.Company_Name)}</a></td><td>${num(r.Master_ESG_Score, 1)}</td><td>${num(r.Environmental_Score, 0)}</td><td>${num(r.Social_Score, 0)}</td><td>${num(r.Governance_Score, 0)}</td><td>${riskBadge(r.Risk_Level)}</td><td>${esc(r.Quality_Gate_Status)}</td></tr>`).join("")}</table></div>`;
  document.querySelectorAll("a[data-c]").forEach(a => a.onclick = () => openCompany(a.dataset.c, "batch"));
  $("#runBatch").onclick = async () => { const j = await post("/api/batch/run?dynamic=false&use_llm=false"); alert("Batch started (job " + j.job_id + "). Refresh in a few minutes."); };
  $("#codes").innerHTML = rows.map(r => `<option value="${esc(r.Company_Code)}">${esc(r.Company_Name)}</option>`).join("");
}
function render() {
  const f = { analyze: pAnalyze, dashboard: pDashboard, performance: pPerformance, market: pMarket, news: pNews, env: pEnv, greenwashing: pGreen, agents: pAgents, timing: pTiming, sources: pSources, validation: pValidation, cohort: pCohort }[S.page];
  Promise.resolve(f()).catch(e => $("#view").innerHTML = `<div class="card empty">Error: ${esc(e.message)}</div>`);
}
get("/api/health").then(h => { $("#llmInfo").textContent = "LLM: " + h.llm; }).catch(() => { });
get("/api/cohort").then(c => { $("#codes").innerHTML = c.companies.map(r => `<option value="${esc(r.Company_Code)}">${esc(r.Company_Name)}</option>`).join(""); }).catch(() => { });
route();
