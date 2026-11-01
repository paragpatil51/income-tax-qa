// Frontend for the Income-tax Act QA backend (dependency-free).
// Served by the backend (http://localhost:8000) the API is same-origin; served
// separately (e.g. `python -m http.server 5500`) it calls the backend on :8000.
const API_BASE = window.location.port === "8000" ? "" : "http://localhost:8000";
const REQUEST_TIMEOUT_MS = 180000;

const SAMPLES = [
  "I own a residential property that I have rented out. Can I claim a 30% deduction on the annual value of the property? I also have a housing loan on this property. Can I claim a deduction for the interest paid on the loan? Which sections of the Income-tax Act apply?",
  "What deductions are available for payments towards life insurance premia and provident fund contributions?",
  "Is agricultural income taxable?",
  "How is the residential status of an individual determined?",
  "What is the GST rate on restaurant services?",
];

const STATUS_TEXT = {
  answered: "Answered",
  partial: "Partially answered",
  insufficient: "Insufficient support",
  out_of_scope: "Out of scope",
  error: "Error",
};

// How a provision reached the answer, in words (the API's "via" codes).
const VIA_TEXT = {
  bm25: "keyword search",
  dense: "meaning search",
  explicit: "named in the question",
  xref: "cross-reference",
  neighbour: "adjacent text",
};

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => Number(n).toLocaleString("en-IN");

let health = null;
let lastAnswer = null;

const EMPTY_SHEET = `<p class="empty">Provisions retrieved for your question appear here, with the words each citation quotes marked in the text of the Act.</p>`;

// ---------- API ----------

async function api(path, options = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), REQUEST_TIMEOUT_MS);
  try {
    const r = await fetch(API_BASE + path, { ...options, signal: ctrl.signal });
    const body = await r.text();
    let data = null;
    try { data = JSON.parse(body); } catch { /* non-JSON error page */ }
    if (!r.ok) {
      // Backend errors are {"error": {"type", "message"}}; FastAPI's own are {"detail": ...}.
      const detail = data && ((data.error && data.error.message) || data.detail);
      const msg = Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail || body.slice(0, 300);
      throw new Error(`HTTP ${r.status}: ${msg}`);
    }
    return data;
  } catch (e) {
    if (e.name === "AbortError") throw new Error("The request timed out. The backend may be loading models; try again.");
    if (e instanceof TypeError) throw new Error(`Cannot reach the backend at ${API_BASE || window.location.origin}.`);
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

function systemAlert(kind, html) {
  $("#system-alert").innerHTML = html ? `<div class="alert ${kind}">${html}</div>` : "";
}

function setAsking(disabled) {
  $("#ask").disabled = disabled;
  document.querySelectorAll(".sample").forEach((b) => (b.disabled = disabled));
}

async function loadHealth() {
  try {
    health = await api("/api/health");
  } catch (e) {
    health = null;
    $("#edition").textContent = "Backend not reachable";
    systemAlert("bad", `${esc(e.message)} Start it from <code>backend/</code> with
      <code>uvicorn taxqa.api.main:app --port 8000</code>.
      <button type="button" class="btn" id="retry-health">Retry</button>`);
    $("#retry-health").onclick = loadHealth;
    setAsking(true);
    return;
  }
  if (!health.ready) {
    $("#edition").textContent = "Index not available";
    systemAlert("bad", `${esc(health.detail)} Place the official PDF in <code>backend/data/raw/</code> and run
      <code>python scripts/ingest.py</code> from <code>backend/</code>.
      <button type="button" class="btn" id="retry-health">Retry</button>`);
    $("#retry-health").onclick = loadHealth;
    setAsking(true);
    return;
  }
  setAsking(false);
  $("#edition").innerHTML = `<dl>
    <div><dt>Sections</dt><dd class="num">${esc(fmt(health.sections))}</dd></div>
    <div><dt>Indexed passages</dt><dd class="num">${esc(fmt(health.chunks))}</dd></div>
    <div><dt>Source</dt><dd>${esc(health.source_document)}</dd></div>
  </dl>`;
  systemAlert(
    health.llm_configured ? "" : "warn",
    health.llm_configured ? "" : `No LLM API key is configured, so the app can retrieve provisions but cannot write answers.
      Set <code>GEMINI_API_KEY</code> in <code>backend/.env</code> and restart the backend.`,
  );
}

function renderLoading() {
  $("#result").innerHTML = `<div class="skeleton" aria-hidden="true"><span></span><span></span><span></span><span></span></div>`;
  $("#sources").innerHTML = `<div class="skeleton" aria-hidden="true"><span></span><span></span><span></span><span></span></div>`;
}

async function ask(event) {
  if (event) event.preventDefault();
  const q = $("#q").value.trim();
  if (q.length < 5) {
    $("#status").textContent = "Please enter a question of at least 5 characters.";
    $("#q").focus();
    return;
  }
  setAsking(true);
  $("#result").setAttribute("aria-busy", "true");
  $("#status").innerHTML = '<span class="spinner keep-motion" aria-hidden="true"></span>Retrieving provisions, drafting and verifying the answer…';
  renderLoading();
  try {
    const answer = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    lastAnswer = answer;
    renderAnswer(answer);
    $("#status").textContent = `Done: ${STATUS_TEXT[answer.status] || answer.status}.`;
  } catch (e) {
    $("#status").textContent = "";
    $("#result").innerHTML = `<div class="alert bad">The request failed: ${esc(e.message)} Check that the backend is running, then ask again.</div>`;
    $("#sources").innerHTML = EMPTY_SHEET;
  } finally {
    setAsking(!!(health && !health.ready));
    $("#result").setAttribute("aria-busy", "false");
  }
}

// ---------- Citation helpers ----------

function pages(c) {
  if (!c.page_start) return "";
  return c.page_start === c.page_end ? `p. ${c.page_start}` : `pp. ${c.page_start}-${c.page_end}`;
}

function isVerified(c) {
  return c.found_in_sources && c.pinpoint_ok && c.quote_ok;
}

function checkState(c) {
  if (!c.found_in_sources) return { cls: "failed", text: "not verified" };
  return isVerified(c) ? { cls: "verified", text: "verified" } : { cls: "partial", text: "partly verified" };
}

function checkItems(c) {
  return [
    [c.found_in_sources, "Provision found in retrieved text of the Act"],
    [c.pinpoint_ok, "Sub-section or clause present in that text"],
    [c.quote_ok, `Quote matches the statute text (${Math.round(c.quote_score || 0)}/100)`],
  ];
}

function checksText(c) {
  return checkItems(c).map(([ok, label]) => `${ok ? "Yes" : "No"}: ${label}`).join("\n") + (c.note ? `\nNote: ${c.note}` : "");
}

function checksHtml(c) {
  return `<ul class="checks">${checkItems(c).map(([ok, label]) =>
    `<li class="${ok ? "yes" : "no"}"><b>${ok ? "Yes" : "No"}</b><span>${esc(label)}</span></li>`).join("")}</ul>`
    + (c.note ? `<p class="point__note">${esc(c.note)}</p>` : "");
}

function citeButton(c) {
  const st = checkState(c);
  const label = `${c.provision || c.source_id}${c.page_start ? ", PDF " + pages(c) : ""}, ${st.text}. Show it in the Act.`;
  return `<button type="button" class="cite cite--${st.cls}" title="${esc(checksText(c))}" aria-label="${esc(label)}"
    data-chunk="${esc(c.chunk_id || "")}"><span class="cite__prov">${esc(c.provision || c.source_id)}</span>
    ${c.page_start ? `<span class="cite__page num">${esc(pages(c))}</span>` : ""}
    <span class="cite__check">${esc(st.text)}</span></button>`;
}

// Highlight verified quotes in the raw statute text, tolerant to line breaks; all text is escaped.
function highlight(text, quotes) {
  const ranges = [];
  for (const q of quotes) {
    const words = String(q).trim().split(/\s+/).filter(Boolean);
    if (words.length < 4) continue;
    const re = new RegExp(words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\s+"), "i");
    const m = re.exec(text);
    if (m) ranges.push([m.index, m.index + m[0].length]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  let out = "", pos = 0;
  for (const [s, e] of ranges) {
    if (s < pos) continue;
    out += esc(text.slice(pos, s)) + "<mark>" + esc(text.slice(s, e)) + "</mark>";
    pos = e;
  }
  return out + esc(text.slice(pos));
}

// ---------- Rendering ----------

function renderStatusNote(a) {
  const notes = {
    insufficient: "No provision retrieved from the Act supports an answer strongly enough, so none is given. The closest provisions are on the sheet for reference only; they do not answer the question.",
    out_of_scope: "This tool only answers questions about the Income-tax Act, 1961.",
    error: health && !health.llm_configured
      ? "Answer generation is unavailable because no LLM API key is configured. The retrieved provisions are still on the sheet."
      : "Answer generation failed. The retrieved provisions are still on the sheet.",
  };
  return notes[a.status] ? `<div class="alert ${a.status === "error" ? "bad" : ""}">${esc(notes[a.status])}</div>` : "";
}

function renderPoints(kept) {
  return `<div class="block"><h3>What the Act provides</h3><ol class="points">${kept.map((p) => `
    <li class="point"><div>
      ${p.heading ? `<h4>${esc(p.heading)}</h4>` : ""}
      <p>${esc(p.statement)}</p>
      <div class="point__cites">${p.citations.map(citeButton).join("")}</div>
      ${p.support === "partial" && p.support_reason ? `<p class="point__note">Verifier: partly supported. ${esc(p.support_reason)}</p>` : ""}
      ${(p.flags || []).map((f) => `<p class="point__note">Check: ${esc(f)}</p>`).join("")}
    </div></li>`).join("")}</ol></div>`;
}

function renderCitationTable(kept, sourceDocument) {
  const rows = [];
  kept.forEach((p) => p.citations.forEach((c, i) => {
    rows.push(`<tr>
      <td data-label="Statement">${i === 0 ? esc(p.heading || p.statement.slice(0, 90)) : ""}</td>
      <td data-label="Provision"><strong>${esc(c.provision || c.source_id)}</strong>
        ${c.quote ? `<p class="quote">“${esc(c.quote)}”</p>` : ""}</td>
      <td data-label="Document">${esc(c.source_document || sourceDocument)}</td>
      <td data-label="PDF page" class="num">${esc(pages(c) || "n/a")}</td>
      <td data-label="Checks">${checksHtml(c)}</td></tr>`);
  }));
  return `<div class="block"><h3>Citations</h3><div class="table-wrap"><table class="stack">
    <caption>Each statement, the provision it rests on, where it is printed and how it was checked</caption>
    <thead><tr><th scope="col">Statement</th><th scope="col">Provision</th><th scope="col">Document</th>
    <th scope="col" class="num">PDF page</th><th scope="col">Checks</th></tr></thead>
    <tbody>${rows.join("")}</tbody></table></div></div>`;
}

function renderConfidence(c) {
  if (!c || c.score === null || c.score === undefined) return "";
  const rows = Object.entries(c.components || {}).map(([k, v]) => {
    const pct = Math.round(Number(v) * 100);
    return `<tr><th scope="row">${esc(k[0].toUpperCase() + k.slice(1))}</th>
      <td class="num">${pct}%</td>
      <td><span class="meter" role="img" aria-label="${esc(k)} ${pct}%"><span style="transform: scaleX(${pct / 100})"></span></span></td></tr>`;
  }).join("");
  return `<div class="block"><h3>How well the answer is supported</h3>
    <div class="confidence">
      <div class="confidence__score"><span class="confidence__figure">${esc(c.score)}%</span>
        <span class="confidence__band">${esc(c.band)} confidence</span></div>
      <div class="table-wrap"><table><thead><tr><th scope="col">Measurement</th><th scope="col" class="num">Score</th><th scope="col"><span class="visually-hidden">Bar</span></th></tr></thead>
        <tbody>${rows}</tbody></table></div>
    </div>
    <details><summary>How this score is computed</summary>
      <ul>${(c.explanation || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
      <p class="point__note">Weights: support 35%, citations 25%, retrieval relevance 25%, coverage 15%. High is 75 or more,
      medium is 50 or more, otherwise low. Removed statements count as unsupported, and the score is capped at 70
      when the entailment check could not run. The score comes from verification and retrieval measurements;
      the language model is never asked to rate itself.</p></details></div>`;
}

function renderSources(sources, quotesByChunk) {
  if (!sources.length) return `<p class="empty">No provisions were retrieved for this question.</p>`;
  return sources.map((s) => {
    const sec = s.label.startsWith("Section ") ? s.label.slice(8).split("(")[0] : "";
    const via = (s.via || []).map((v) => VIA_TEXT[v] || v).join(", ");
    return `<article class="source" id="src-${esc(s.chunk_id)}" tabindex="-1" aria-labelledby="h-${esc(s.chunk_id)}">
      <header class="source__head">
        <h3 id="h-${esc(s.chunk_id)}">${esc(s.label)}</h3>
        ${s.section_title ? `<p class="source__title">${esc(s.section_title)}</p>` : ""}
        <dl class="source__meta">
          <div><dt>PDF</dt><dd class="num">${esc(String(s.pages).replace("–", "-"))}</dd></div>
          ${s.chapter ? `<div><dt>Chapter</dt><dd>${esc(s.chapter)}</dd></div>` : ""}
          <div><dt>Relevance</dt><dd class="num">${esc((Number(s.relevance) * 100).toFixed(0))}%</dd></div>
          ${via ? `<div><dt>Found by</dt><dd>${esc(via)}</dd></div>` : ""}
        </dl>
      </header>
      ${s.context ? `<p class="source__context">${esc(s.context)}</p>` : ""}
      <div class="statute">${highlight(s.text, quotesByChunk[s.chunk_id] || [])}</div>
      ${sec ? `<details data-sec="${esc(sec)}"><summary>The whole of section ${esc(sec)}</summary><div class="statute">Loading…</div></details>` : ""}
    </article>`;
  }).join("");
}

function renderAnswer(a) {
  const kept = a.points.filter((p) => !p.removed);
  const removed = a.points.filter((p) => p.removed);
  const quotesByChunk = {};
  kept.forEach((p) => p.citations.forEach((c) => {
    if (c.chunk_id && c.quote_ok) (quotesByChunk[c.chunk_id] ||= []).push(c.quote);
  }));
  const secs = a.timings_ms && a.timings_ms.total ? (a.timings_ms.total / 1000).toFixed(1) : null;

  let h = `<div class="answer__head">
      <h2>The answer</h2>
      <div class="answer__facts">
        <span class="state state--${esc(a.status)}" role="status">${esc(STATUS_TEXT[a.status] || a.status)}</span>
        ${secs ? `<span><span class="num">${esc(secs)}</span> s</span>` : ""}
        ${kept.length ? '<button type="button" class="btn copy" id="copy-md">Copy as Markdown</button>' : ""}
      </div>
    </div>
    ${a.summary ? `<p class="summary">${esc(a.summary)}</p>` : ""}
    ${renderStatusNote(a)}`;
  (a.warnings || []).forEach((w) => (h += `<div class="alert warn">${esc(w)}</div>`));

  if (kept.length) h += renderPoints(kept) + renderCitationTable(kept, a.source_document);
  if (a.not_covered.length) h += `<div class="block"><h3>Not covered by the retrieved provisions</h3><ul class="list">${a.not_covered.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>`;
  if (a.caveats.length) h += `<div class="block"><h3>Assumptions</h3><ul class="list">${a.caveats.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>`;
  h += renderConfidence(a.confidence);

  const droppedSummary = a.summary_removed || [];
  if (removed.length || droppedSummary.length) {
    h += `<details><summary>${removed.length + droppedSummary.length} unverified statement(s) removed</summary>`;
    removed.forEach((p) => (h += `<div class="removed"><p><strong>${esc(p.heading)}</strong> ${esc(p.statement)}</p><p>Reason: ${esc(p.removal_reason)}</p></div>`));
    droppedSummary.forEach((s) => (h += `<div class="removed"><p>${esc(s)}</p><p>Reason: summary sentence not backed by a verified statement</p></div>`));
    h += `</details>`;
  }
  if (a.sub_questions.length > 1) {
    h += `<details><summary>How the question was broken down</summary><ul>${a.sub_questions.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></details>`;
  }
  h += `<dl class="colophon"><dt>Edition</dt><dd>${esc(a.edition)}</dd>
    ${Object.entries(a.models || {}).map(([k, v]) => `<dt>${esc(k[0].toUpperCase() + k.slice(1))}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`;

  $("#result").innerHTML = h;
  $("#sources").innerHTML = renderSources(a.sources || [], quotesByChunk);
  bindResultEvents();
}

function toMarkdown(a) {
  const kept = a.points.filter((p) => !p.removed);
  const cell = (s) => String(s ?? "").replace(/\|/g, "\\|").replace(/\s+/g, " ").trim();
  const lines = [
    `**Question:** ${a.question}`, "",
    `**Status:** ${STATUS_TEXT[a.status] || a.status}` + (a.confidence && a.confidence.score !== null ? `, **confidence:** ${a.confidence.score}% (${a.confidence.band})` : ""), "",
    `**Answer:** ${a.summary}`, "",
  ];
  kept.forEach((p) => lines.push(`- **${p.heading}**: ${p.statement}`));
  lines.push("", "| Answer statement | Statutory reference | Source document | PDF page | Verified |", "|---|---|---|---|---|");
  kept.forEach((p) => p.citations.forEach((c) => lines.push(
    `| ${cell(p.heading || p.statement.slice(0, 80))} | ${cell(c.provision)} | ${cell(c.source_document || a.source_document)} | ${cell(pages(c))} | ${isVerified(c) ? "Yes" : "Partly or no"} |`)));
  if (a.not_covered.length) lines.push("", "**Not covered:** " + a.not_covered.join("; "));
  if (a.caveats.length) lines.push("", "**Assumptions:** " + a.caveats.join("; "));
  lines.push("", `_Source: ${a.source_document} (${a.edition}). Generated ${new Date().toISOString().slice(0, 10)}._`);
  return lines.join("\n");
}

function bindResultEvents() {
  document.querySelectorAll(".cite[data-chunk]").forEach((b) => (b.onclick = () => {
    const el = document.getElementById("src-" + b.dataset.chunk);
    if (!el) return;
    document.querySelectorAll(".source.is-current").forEach((s) => s.classList.remove("is-current"));
    el.classList.add("is-current");
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    el.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    el.focus({ preventScroll: true });
  }));
  document.querySelectorAll("details[data-sec]").forEach((d) => d.addEventListener("toggle", async () => {
    if (!d.open || d.dataset.loaded) return;
    d.dataset.loaded = "1";
    try {
      const s = await api("/api/section/" + encodeURIComponent(d.dataset.sec));
      d.querySelector(".statute").textContent = (s.lines || []).map((l) => l[1]).join("\n");
    } catch (e) {
      d.querySelector(".statute").textContent = "Could not load the section: " + e.message;
    }
  }));
  const copy = $("#copy-md");
  if (copy) copy.onclick = async () => {
    const md = toMarkdown(lastAnswer);
    try {
      await navigator.clipboard.writeText(md);
      copy.textContent = "Copied";
    } catch {
      copy.insertAdjacentHTML("afterend", `<textarea aria-label="Answer as Markdown" rows="8">${esc(md)}</textarea>`);
    }
    setTimeout(() => (copy.textContent = "Copy as Markdown"), 2000);
  };
}

function renderSamples() {
  $("#samples").innerHTML = `<p class="samples__label" id="samples-label">Or try one of these</p>
    <ul aria-labelledby="samples-label">${SAMPLES.map((s, i) =>
      `<li><button type="button" class="sample" data-i="${i}" title="${esc(s)}">${esc(s.length > 110 ? s.slice(0, 107) + "…" : s)}</button></li>`).join("")}</ul>`;
  $("#samples").querySelectorAll(".sample").forEach((b) => (b.onclick = () => {
    $("#q").value = SAMPLES[Number(b.dataset.i)];
    ask();
  }));
}

// ---------- Init ----------

$("#ask-form").addEventListener("submit", ask);
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask(e); });
renderSamples();
loadHealth().then(() => {
  // Shareable / scriptable links: /?q=<question> fills the box and asks it (used for the demo screenshots).
  const q = new URLSearchParams(window.location.search).get("q");
  if (q && health && health.ready) {
    $("#q").value = q;
    ask();
  }
});
