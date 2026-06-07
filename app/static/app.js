"use strict";

/* ───────────── constants ───────────── */
const RANK_COLORS = ["#10b981", "#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444", "#06b6d4"];
const QUALITY_LABELS = {
  concurrency: "并发量", realtime: "实时性", reliability: "可靠性",
  scalability: "可扩展性", data_intensity: "数据密度", ai_reasoning: "AI 推理",
};
const QUALITY_COLORS = {
  concurrency: "#10b981", realtime: "#3b82f6", reliability: "#8b5cf6",
  scalability: "#f59e0b", data_intensity: "#ef4444", ai_reasoning: "#06b6d4",
};
const DIM_ORDER = ["scalability", "performance", "reliability", "modifiability", "complexity", "realtime"];
const DIM_LABELS = {
  scalability: "可扩展性", performance: "性能", reliability: "可靠性",
  modifiability: "可维护性", complexity: "复杂度友好度", realtime: "实时性",
};
const AGENT_STEPS = [
  { icon: "🔍", name: "需求解析 Agent", action: "正在提取需求关键特征…" },
  { icon: "🧠", name: "架构匹配 Agent", action: "正在从知识库匹配候选架构…" },
  { icon: "📊", name: "评估生成 Agent", action: "正在生成多维度评估报告…" },
];
const EXAMPLE_REQUIREMENT = "开发一个跨平台的即时通讯系统，要求支持万人同时在线，需要保证消息的实时性和可靠性，后期可能需要快速扩展视频通话功能";
const HINTS = [
  {
    label: "高并发秒杀",
    text: "构建一个面向全国的电商秒杀平台，平时日活千万、大促瞬时并发每秒数十万笔下单，要求商品浏览、加购、下单、库存扣减、优惠计算与支付在 200 毫秒内完成；库存与订单需保证最终一致性，杜绝超卖，支付超时自动释放库存；活动期间要削峰限流、热点缓存与异步落单，订单状态变化实时推送给用户和商家，后续还要接入风控、积分、物流和实时经营大盘，核心服务要求独立部署、弹性扩缩容与灰度发布。",
  },
  {
    label: "即时通讯",
    text: "开发一个企业级即时通讯与协作平台，支持百万级用户、十万人同时在线，单聊/群聊/频道消息要求毫秒级送达且不丢不重，离线消息可靠同步，支持已读回执、消息撤回、文件传输与音视频通话；需要在线状态实时广播、消息全文检索、敏感内容风控审核，后期快速扩展聊天机器人、工作流审批与开放平台，要求多端一致、按服务独立扩缩容、单数据中心故障不影响其他区域。",
  },
  {
    label: "电商平台",
    text: "建设一个多商家入驻的综合电商平台，覆盖商品中心、店铺管理、搜索推荐、购物车、订单、支付结算、营销促销、库存履约、评价售后与商家结算。支持千万级商品和百万级日订单，大促高并发，价格与库存频繁变更需读写分离与缓存一致；订单履约跨仓配送，支付、库存、物流状态需异步解耦、最终一致并可补偿；要求灰度发布、按业务域独立扩缩容，并接入实时风控与经营分析大盘。",
  },
  {
    label: "物联网",
    text: "开发一个工业物联网设备管理与监控平台，接入百万级传感器与设备，每秒采集数十万条遥测数据，要求实时清洗、聚合、规则告警与时序存储；支持设备远程控制下发、固件 OTA 升级、离线缓存补传；告警事件需实时推送到监控大屏和运维，海量历史数据要支持离线分析与可视化报表，平台要能弹性扩展接入规模并隔离多租户。",
  },
  {
    label: "数据分析",
    text: "构建一个企业级实时数据分析平台，从业务库、日志、埋点和第三方接口汇聚 TB 级数据，支持流式实时计算与离线批处理混合，完成清洗、转换、聚合、特征加工与多维 OLAP 查询；要求秒级实时大盘、灵活的自助分析与机器学习特征供给，数据管道可编排、可回溯、可监控，计算与存储分离并按需弹性扩缩容。",
  },
];
const MIN_PROC_MS = 2600;

/* ───────────── state ───────────── */
const state = {
  phase: "input",          // input | processing | results | error
  requirement: "",
  features: null,
  candidates: [],
  finalRec: null,
  composition: {},
  matrix: [],
  decisionTrace: {},
  trace: [],
  report: "",
  topologyDiagrams: null,
  topologyGraphs: {},
  activeTab: "overview",
  procStep: 0,
  coreReady: false,
  procStart: 0,
  errorMessage: "",
};

let topoAbort = null;
let topoState = { source: "", svg: "", scale: 1, panX: 0, panY: 0 };
let modalState = { scale: 1, panX: 0, panY: 0 };
let kg = { style: null, topo: null, cases: null, status: null, styles: null };
let visNetwork = null;
const seenToastKeys = new Set();

const appEl = document.querySelector("#app");
const toastRootEl = document.querySelector("#toastRoot");
const modalEl = document.querySelector("#topologyModal");
const modalCanvasEl = document.querySelector("#topologyModalCanvas");

if (window.mermaid) {
  mermaid.initialize({
    startOnLoad: false,
    theme: "base",
    flowchart: { useMaxWidth: false, htmlLabels: true, padding: 14, nodeSpacing: 46, rankSpacing: 64 },
    themeVariables: {
      background: "#0f0f11", primaryColor: "#26262e", primaryTextColor: "#fafafa",
      primaryBorderColor: "#10b981", lineColor: "#9aa0ab", secondaryColor: "#1f2937",
      tertiaryColor: "#181820", fontFamily: "Noto Sans SC, sans-serif", fontSize: "16px",
      clusterBkg: "rgba(16,185,129,0.06)", clusterBorder: "#3f3f46", edgeLabelBackground: "#141416",
      titleColor: "#fafafa", nodeTextColor: "#fafafa",
    },
  });
}

/* modal controls (static in index.html) */
document.querySelector("#modalClose").addEventListener("click", closeModal);
document.querySelector("#modalZoomIn").addEventListener("click", () => setModalScale(modalState.scale * 1.2));
document.querySelector("#modalZoomOut").addEventListener("click", () => setModalScale(modalState.scale / 1.2));
document.querySelector("#modalZoomReset").addEventListener("click", fitModal);
modalEl.addEventListener("click", (e) => { if (e.target === modalEl) closeModal(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && modalEl.classList.contains("open")) closeModal(); });

/* ───────────── utils ───────────── */
function escapeHtml(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[c]));
}
function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }
function colorFor(i) { return RANK_COLORS[i % RANK_COLORS.length]; }
function pct(v) { return `${Math.round((Number(v) || 0) * 100)}`; }

function showToast(title, message, type = "error") {
  if (!toastRootEl) return;
  const t = document.createElement("div");
  t.className = `toast ${type === "info" ? "info" : ""}`;
  t.innerHTML = `<strong>${escapeHtml(title)}</strong><span>${escapeHtml(message)}</span>`;
  toastRootEl.appendChild(t);
  setTimeout(() => t.remove(), 3600);
}
function showToastOnce(key, title, message, type = "error") {
  if (seenToastKeys.has(key)) return;
  seenToastKeys.add(key);
  showToast(title, message, type);
}

/* ───────────── root render ───────────── */
function renderApp() {
  if (state.phase === "input") renderInput();
  else if (state.phase === "processing") renderProcessing();
  else if (state.phase === "results") renderResults();
  else if (state.phase === "error") renderError();
  else if (state.phase === "settings") renderSettings();
}

/* ───────────── input phase ───────────── */
function renderInput() {
  appEl.innerHTML = `
    <div class="input-phase">
      <button class="kg-entry" id="kgEntry">◎ 知识中心</button>
      <div class="input-inner">
        <div class="brand-mark">◇</div>
        <h1 class="brand-title">ArchWise</h1>
        <p class="brand-sub">描述你的软件需求，AI 为你推荐最合适的架构方案</p>
        <div class="input-box">
          <textarea id="reqInput" placeholder="描述你的软件系统需求，例如：用户规模、核心业务流程、性能与一致性要求、部署约束…"></textarea>
          <div class="input-bar">
            <button class="example-btn" id="exampleBtn">💡 试试示例需求</button>
            <button class="submit-btn" id="submitBtn" disabled>开始分析 →</button>
          </div>
        </div>
        <div class="hints">
          ${HINTS.map((h, i) => `<button class="hint-chip" data-hint="${i}">${escapeHtml(h.label)}</button>`).join("")}
        </div>
      </div>
    </div>`;

  const input = document.querySelector("#reqInput");
  const submit = document.querySelector("#submitBtn");
  const sync = () => { submit.disabled = !input.value.trim(); };
  input.addEventListener("input", sync);
  document.querySelector("#exampleBtn").addEventListener("click", () => { input.value = EXAMPLE_REQUIREMENT; sync(); input.focus(); });
  document.querySelectorAll(".hint-chip").forEach((c) => c.addEventListener("click", () => {
    const h = HINTS[Number(c.dataset.hint)];
    if (!h) return;
    input.value = h.text;
    sync();
    input.focus();
    input.scrollTop = 0;
  }));
  submit.addEventListener("click", () => { if (input.value.trim()) startAnalysis(input.value.trim()); });
  document.querySelector("#kgEntry").addEventListener("click", openSettings);
  input.value = "";
  sync();
}

/* ───────────── processing phase ───────────── */
function renderProcessing() {
  appEl.innerHTML = `
    <div class="proc-phase">
      <div class="proc-inner">
        <div class="spinner"></div>
        <h2 class="proc-title">正在分析你的需求</h2>
        <p class="proc-sub">三个智能体正在协同工作</p>
        <div class="agent-steps" id="agentSteps"></div>
        <div class="proc-bar"><div class="proc-bar-fill" id="procBar"></div></div>
      </div>
    </div>`;
  updateProcessing();
}
function updateProcessing() {
  const stepsEl = document.querySelector("#agentSteps");
  if (!stepsEl) return;
  stepsEl.innerHTML = AGENT_STEPS.map((s, i) => {
    const done = i < state.procStep;
    const active = i === state.procStep;
    return `
      <div class="agent-step ${done ? "done" : active ? "active" : ""}">
        <div class="step-icon">${done ? "✓" : s.icon}</div>
        <div>
          <div class="step-name">${escapeHtml(s.name)}</div>
          <div class="step-action">${done ? "完成" : active ? escapeHtml(s.action) : "等待中"}</div>
        </div>
      </div>`;
  }).join("");
  const bar = document.querySelector("#procBar");
  if (bar) bar.style.width = `${Math.min(100, (state.procStep / AGENT_STEPS.length) * 100 + 8)}%`;
}

/* ───────────── analysis flow (SSE) ───────────── */
async function startAnalysis(requirement) {
  const runId = (state.runId || 0) + 1;
  Object.assign(state, {
    phase: "processing", requirement, features: null, candidates: [], finalRec: null,
    composition: {}, matrix: [], decisionTrace: {}, trace: [], report: "",
    topologyDiagrams: null, topologyGraphs: {}, activeTab: "overview",
    procStep: 0, coreReady: false, animComplete: false, runId, errorMessage: "",
  });
  seenToastKeys.clear();
  if (topoAbort) { topoAbort.abort(); topoAbort = null; }
  renderApp();
  // Drive the 3-agent animation on a timeline; backend streaming order is not guaranteed,
  // so steps advance by time and we only enter results once data is ready AND the animation played.
  setTimeout(() => { if (state.phase === "processing" && state.runId === runId && state.procStep < 1) { state.procStep = 1; updateProcessing(); } }, 1300);
  setTimeout(() => { if (state.phase === "processing" && state.runId === runId && state.procStep < 2) { state.procStep = 2; updateProcessing(); } }, 2700);
  setTimeout(() => { if (state.runId === runId) { state.animComplete = true; maybeEnterResults(); } }, 3700);

  try {
    const res = await fetch("/api/recommend/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({ requirement, top_k: 12 }),
    });
    if (!res.ok || !res.body) {
      const p = await res.json().catch(() => ({}));
      throw new Error(p.detail || "推荐接口返回异常");
    }
    await consumeStream(res.body, handleRecEvent);
  } catch (err) {
    failAnalysis(err.message);
  }
}

async function consumeStream(body, handler) {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const raw of parts) await handler(parseSse(raw));
  }
  if (buffer.trim()) await handler(parseSse(buffer));
}
function parseSse(raw) {
  // The backend emits standard SSE blocks with one event line and one JSON data payload.
  const lines = raw.split("\n");
  const event = lines.find((l) => l.startsWith("event:"))?.slice(6).trim() || "message";
  const data = lines.filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trim()).join("\n");
  let payload = {};
  if (data) { try { payload = JSON.parse(data); } catch { payload = {}; } }
  return { event, payload };
}

async function handleRecEvent({ event, payload }) {
  if (event === "features") {
    state.features = payload.features || state.features;
    state.trace = payload.trace || state.trace;
  } else if (event === "recommendation" || event === "initial") {
    if (payload.features) state.features = payload.features;
    state.candidates = payload.candidates || [];
    state.finalRec = payload.final_recommendation || state.candidates[0] || null;
    state.composition = payload.composition_recommendation || {};
    state.matrix = payload.comparison_matrix || [];
    state.decisionTrace = payload.decision_trace || {};
    state.trace = payload.trace || state.trace;
    state.coreReady = true;
    startTopologyStream();
    maybeEnterResults();
  } else if (event === "report_delta") {
    state.report += payload.delta || "";
    patchReport();
  } else if (event === "error") {
    failAnalysis(payload.message || "DeepSeek 需求解析失败，请检查模型服务或输入更完整的需求。");
  } else if (event === "done") {
    if (payload.ok !== false) { state.procStep = AGENT_STEPS.length; updateProcessing(); patchReport(); }
  }
}

function maybeEnterResults() {
  if (state.phase !== "processing") return;
  if (!state.coreReady || !state.animComplete) return;
  state.procStep = AGENT_STEPS.length;
  state.phase = "results";
  renderApp();
}

function failAnalysis(message) {
  if (state.phase === "results") { showToast("服务异常", message); return; }
  state.phase = "error";
  state.errorMessage = message || "系统调用异常。";
  renderApp();
}

async function startTopologyStream() {
  if (!state.requirement || !state.features || !state.finalRec) return;
  if (topoAbort) topoAbort.abort();
  const controller = new AbortController();
  topoAbort = controller;
  try {
    const res = await fetch("/api/topology/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({
        requirement: state.requirement,
        features: state.features,
        final_recommendation: state.finalRec,
        composition_recommendation: state.composition || {},
        decision_trace: state.decisionTrace || {},
        topology_fast_mode: true,
        topology_llm_timeout_seconds: 12,
        topology_repair_max_rounds: 1,
      }),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) {
      const p = await res.json().catch(() => ({}));
      throw new Error(p.detail || "拓扑接口返回异常");
    }
    await consumeStream(res.body, handleTopoEvent);
  } catch (err) {
    if (err.name === "AbortError") return;
    if (!state.topologyDiagrams) state.topologyDiagrams = {};
    if (state.activeTab === "topology") renderActiveTab();
  } finally {
    if (topoAbort === controller) topoAbort = null;
  }
}
async function handleTopoEvent({ event, payload }) {
  if (event === "topology") {
    state.topologyDiagrams = payload.topology_diagrams || {};
    state.topologyGraphs = payload.topology_graphs || {};
    if (payload.decision_trace) state.decisionTrace = payload.decision_trace;
    if (state.phase === "results" && state.activeTab === "topology") renderActiveTab();
    if (state.phase === "results" && state.activeTab === "trace") renderActiveTab();
  } else if (event === "error") {
    state.topologyDiagrams = state.topologyDiagrams || {};
    if (state.phase === "results" && state.activeTab === "topology") renderActiveTab();
  }
}

/* ───────────── error view ───────────── */
function renderError() {
  appEl.innerHTML = `
    <div class="proc-phase">
      <div class="proc-inner">
        <div class="brand-mark" style="background:linear-gradient(135deg,#ef4444,#f59e0b)">!</div>
        <h2 class="proc-title">需求分析未完成</h2>
        <p class="proc-sub" style="margin-bottom:28px">${escapeHtml(state.errorMessage)}</p>
        <button class="submit-btn" id="retryBtn">重新输入需求</button>
      </div>
    </div>`;
  document.querySelector("#retryBtn").addEventListener("click", () => { state.phase = "input"; renderApp(); });
}

/* ───────────── results shell ───────────── */
const TABS = [
  { key: "overview", label: "推荐总览" },
  { key: "compare", label: "对比分析" },
  { key: "topology", label: "架构拓扑" },
  { key: "trace", label: "决策溯源" },
];
function renderResults() {
  appEl.innerHTML = `
    <div class="results">
      <header class="r-header">
        <div class="r-header-inner">
          <div class="brand-row"><div class="mini-mark">◇</div><span class="brand-name">ArchWise</span></div>
          <div class="header-actions"><button class="new-btn" id="kgBtn">知识中心</button><button class="new-btn" id="newBtn">+ 新的分析</button></div>
        </div>
        <div class="tabs" id="tabs">
          ${TABS.map((t) => `<button class="tab-btn ${t.key === state.activeTab ? "active" : ""}" data-tab="${t.key}">${t.label}</button>`).join("")}
        </div>
      </header>
      <div class="tab-wrap" id="tabWrap"></div>
    </div>`;
  document.querySelector("#newBtn").addEventListener("click", () => { state.phase = "input"; renderApp(); });
  document.querySelector("#kgBtn").addEventListener("click", openSettings);
  document.querySelectorAll(".tab-btn").forEach((b) => b.addEventListener("click", () => {
    state.activeTab = b.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach((x) => x.classList.toggle("active", x.dataset.tab === state.activeTab));
    renderActiveTab();
  }));
  renderActiveTab();
}
function renderActiveTab() {
  const wrap = document.querySelector("#tabWrap");
  if (!wrap) return;
  wrap.innerHTML = "";
  if (state.activeTab === "overview") renderOverview(wrap);
  else if (state.activeTab === "compare") renderCompare(wrap);
  else if (state.activeTab === "topology") renderTopology(wrap);
  else if (state.activeTab === "trace") renderTrace(wrap);
}

/* ───────────── charts ───────────── */
function scoreRing(score, size, color) {
  const r = (size - 8) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - clamp(score, 0, 100) / 100);
  return `
    <div class="score-ring" style="width:${size}px;height:${size}px">
      <svg width="${size}" height="${size}" style="transform:rotate(-90deg)">
        <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="#27272a" stroke-width="4"/>
        <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="${color}" stroke-width="4"
          stroke-dasharray="${circ.toFixed(1)}" stroke-dashoffset="${offset.toFixed(1)}" stroke-linecap="round"
          style="transition:stroke-dashoffset 1s ease"/>
      </svg>
      <div class="ring-num" style="font-size:${(size * 0.3).toFixed(0)}px;color:#fafafa">${Math.round(score)}</div>
    </div>`;
}

function radarSVG(cands, colors) {
  const size = 280, c = size / 2, R = 96, axes = DIM_ORDER.length;
  const ang = (i) => (-90 + i * (360 / axes)) * Math.PI / 180;
  const pt = (i, rad) => ({ x: c + rad * Math.cos(ang(i)), y: c + rad * Math.sin(ang(i)) });
  let svg = `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" style="font-family:var(--mono)">`;
  for (let ring = 1; ring <= 4; ring++) {
    const pts = DIM_ORDER.map((_, i) => { const p = pt(i, (R * ring) / 4); return `${p.x.toFixed(1)},${p.y.toFixed(1)}`; }).join(" ");
    svg += `<polygon points="${pts}" fill="none" stroke="#27272a" stroke-width="1"/>`;
  }
  DIM_ORDER.forEach((_, i) => { const p = pt(i, R); svg += `<line x1="${c}" y1="${c}" x2="${p.x.toFixed(1)}" y2="${p.y.toFixed(1)}" stroke="#27272a" stroke-width="1"/>`; });
  cands.forEach((cand, ci) => {
    const pts = DIM_ORDER.map((dim, i) => { const v = clamp(cand.quality_scores?.[dim] ?? 0, 0, 1); const p = pt(i, R * v); return `${p.x.toFixed(1)},${p.y.toFixed(1)}`; }).join(" ");
    svg += `<polygon points="${pts}" fill="${colors[ci]}" fill-opacity="0.12" stroke="${colors[ci]}" stroke-width="2"/>`;
  });
  DIM_ORDER.forEach((dim, i) => {
    const p = pt(i, R + 16);
    const anchor = Math.abs(p.x - c) < 6 ? "middle" : p.x > c ? "start" : "end";
    svg += `<text x="${p.x.toFixed(1)}" y="${(p.y + 3).toFixed(1)}" text-anchor="${anchor}" font-size="10" fill="#71717a">${escapeHtml(DIM_LABELS[dim])}</text>`;
  });
  return svg + "</svg>";
}

/* ───────────── Tab 1: overview ───────────── */
function candSummary(cand) {
  return (cand?.matched_reasons && cand.matched_reasons[0]) || cand?.recommendation_role || "候选架构";
}
function renderOverview(wrap) {
  const f = state.features || {};
  const tags = Object.entries(f.quality_attributes || {})
    .filter(([, v]) => Number(v) > 0.05)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => {
      const color = QUALITY_COLORS[k] || "#10b981";
      return `<div class="feature-tag" style="background:${color}15;border:1px solid ${color}30">
        <span class="dot" style="background:${color};box-shadow:0 0 8px ${color}66"></span>
        <span>${escapeHtml(QUALITY_LABELS[k] || k)}</span>
        <span class="ft-val" style="color:${color}">${pct(v)}%</span></div>`;
    }).join("");
  const summaryText = [f.domain, f.data_flow ? `数据流 ${f.data_flow}` : "", ...(f.keywords || []).slice(0, 6)].filter(Boolean).join(" · ");

  const winner = state.finalRec || state.candidates[0];
  const others = state.candidates.slice(1, 3);

  let html = `
    <div class="card req-summary">
      <div class="req-text">
        <div class="section-label" style="margin:0 0 8px">需求摘要</div>
        <p>${escapeHtml(summaryText || state.requirement.slice(0, 120))}</p>
      </div>
      <div class="feature-tags">${tags || '<span class="empty-hint">暂无特征</span>'}</div>
    </div>
    <div class="section-label">推荐结果</div>`;

  if (winner) {
    const pros = (winner.matched_reasons || []).slice(0, 3);
    html += `
      <div class="hero-card">
        ${scoreRing(winner.score, 82, "#10b981")}
        <div class="hero-body">
          <div class="hero-title-row">
            <h2>${escapeHtml(winner.name)}</h2>
            <span class="badge" style="background:#10b98120;color:#10b981;border:1px solid #10b98140">${escapeHtml(winner.recommendation_role || "核心推荐")}</span>
            <span class="badge" style="background:#1c1c1f;color:#a1a1aa;border:1px solid #27272a">置信度 ${escapeHtml(winner.confidence || "中")}</span>
          </div>
          <p class="hero-summary">${escapeHtml(candSummary(winner))}</p>
          <div class="pro-chips">${pros.map((p) => `<span class="pro-chip">✓ ${escapeHtml(p)}</span>`).join("")}</div>
        </div>
        <button class="hero-detail-btn" id="heroDetail">详细对比 →</button>
      </div>`;
  }

  if (others.length) {
    html += `<div class="cand-grid">${others.map((cand, idx) => {
      const color = colorFor(idx + 1);
      return `<div class="card hoverable cand-card" data-cand="${idx + 1}">
        ${scoreRing(cand.score, 56, color)}
        <div class="cand-meta">
          <div class="cand-name-row">
            <span class="cand-name">${escapeHtml(cand.name)}</span>
            <span class="badge" style="background:${color}20;color:${color};border:1px solid ${color}40">${escapeHtml(cand.recommendation_role || "备选方案")}</span>
          </div>
          <div class="cand-summary">${escapeHtml(candSummary(cand))}</div>
        </div>
      </div>`;
    }).join("")}</div>`;
  }

  html += renderCompositionCard();
  html += `
    <details class="fold report-fold" style="margin-top:24px">
      <summary>完整评估报告</summary>
      <div class="fold-body"><div class="markdown-body" id="reportBody">${state.report ? markdownToHtml(state.report) : '<p class="empty-hint">评估报告生成中…</p>'}</div></div>
    </details>`;

  wrap.innerHTML = html;
  const hd = document.querySelector("#heroDetail");
  if (hd) hd.addEventListener("click", () => switchTab("compare"));
  document.querySelectorAll(".cand-card").forEach((c) => c.addEventListener("click", () => switchTab("compare")));
}

function renderCompositionCard() {
  const comp = state.composition || {};
  if (!Object.keys(comp).length) return "";
  const needed = Boolean(comp.composition_needed);
  if (!needed) {
    return `
      <div class="card" style="margin-top:24px">
        <div class="section-label" style="margin:0 0 6px">组合策略</div>
        <h3 style="font-size:17px;margin-bottom:6px">不建议采用复杂组合架构</h3>
        <p style="font-size:14px;color:var(--text-2)">核心架构：${escapeHtml(comp.primary_style || "—")}。${escapeHtml(comp.reason || "单一架构即可满足当前需求。")}</p>
        ${(comp.overengineering_warnings || []).length ? `<div class="combo-warn">⚠ ${escapeHtml((comp.overengineering_warnings || []).join("；"))}</div>` : ""}
      </div>`;
  }
  const layers = buildComboLayers(comp);
  return `
    <div class="card" style="margin-top:24px">
      <div class="section-label" style="margin:0 0 6px">组合策略</div>
      <h3 style="font-size:17px;margin-bottom:6px">以 ${escapeHtml(comp.primary_style || "核心架构")} 为基座的组合方案</h3>
      <p style="font-size:14px;color:var(--text-2)">${escapeHtml(comp.reason || "")}</p>
      <div class="combo-layers">
        ${layers.map((l, i) => `
          <div class="combo-layer" style="background:${l.color}0d;border-left:3px solid ${l.color};
            border-radius:${i === 0 ? "12px 12px 0 0" : i === layers.length - 1 ? "0 0 12px 12px" : "0"}">
            <span class="layer-name" style="color:${l.color}">${escapeHtml(l.name)}</span>
            <span class="layer-tech">${escapeHtml(l.tech)}</span>
          </div>`).join("")}
      </div>
    </div>`;
}
function buildComboLayers(comp) {
  const layers = [{ name: "主架构", tech: comp.primary_style || "核心架构", color: "#10b981" }];
  (comp.supporting_styles || []).slice(0, 4).forEach((s, i) => {
    layers.push({
      name: s.role ? String(s.role).slice(0, 6) : `辅助 ${i + 1}`,
      tech: `${s.style || s.style_id || "辅助架构"}${s.apply_to && s.apply_to.length ? "（" + s.apply_to.slice(0, 3).join("、") + "）" : ""}`,
      color: colorFor(i + 1),
    });
  });
  return layers;
}

function patchReport() {
  const el = document.querySelector("#reportBody");
  if (el && state.report) el.innerHTML = markdownToHtml(state.report);
}
function switchTab(key) {
  state.activeTab = key;
  document.querySelectorAll(".tab-btn").forEach((x) => x.classList.toggle("active", x.dataset.tab === key));
  renderActiveTab();
}

/* ───────────── Tab 2: compare ───────────── */
function renderCompare(wrap) {
  const top = state.candidates.slice(0, 3);
  const colors = top.map((_, i) => colorFor(i));
  let html = `<div class="section-label">多维度对比分析</div>
    <div class="chart-grid">
      <div class="card">
        <div class="chart-title">能力雷达（6 维）</div>
        <div class="radar-wrap">${radarSVG(top, colors)}</div>
        <div class="legend">${top.map((a, i) => `<div class="legend-item"><span class="swatch" style="background:${colors[i]}"></span>${escapeHtml(a.name)}</div>`).join("")}</div>
      </div>
      <div class="card">
        <div class="chart-title">综合评分</div>
        <div class="bars">${top.map((a, i) => `
          <div class="bar-row">
            <span class="bar-label">${escapeHtml(a.name)}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${clamp(a.score, 0, 100)}%;background:${colors[i]}"></div></div>
            <span class="bar-score" style="color:${colors[i]}">${Math.round(a.score)}</span>
          </div>`).join("")}</div>
      </div>
    </div>
    <div class="section-label">逐项分析</div>
    <div id="accList">${top.map((a, i) => accordionHtml(a, i)).join("")}</div>`;

  if (state.matrix && state.matrix.length) {
    html += `
      <details class="fold" style="margin-top:20px">
        <summary>显示全部候选架构对比矩阵（${state.matrix.length} 种）</summary>
        <div class="fold-body" style="overflow-x:auto">${matrixTable(state.matrix)}</div>
      </details>`;
  }
  wrap.innerHTML = html;
  bindAccordions();
}
function accordionHtml(cand, i) {
  const color = colorFor(i);
  const pros = cand.matched_reasons || [];
  const cons = [...(cand.risks || []), ...(cand.deductions || [])];
  return `
    <div class="acc ${i === 0 ? "open" : ""}">
      <button class="acc-head">
        ${scoreRing(cand.score, 50, color)}
        <div class="acc-titles">
          <div class="acc-name">${escapeHtml(cand.name)}
            <span class="badge" style="background:${color}20;color:${color};border:1px solid ${color}40">${escapeHtml(cand.recommendation_role || "候选")}</span>
          </div>
          <div class="acc-sub">${escapeHtml(candSummary(cand))}</div>
        </div>
        <span class="acc-chev">▾</span>
      </button>
      <div class="acc-body" ${i === 0 ? "" : 'style="display:none"'}>
        <div class="pc-grid">
          <div>
            <div class="pc-head pro">✓ 优势 / 匹配理由</div>
            ${pros.length ? pros.map((p) => `<div class="pc-item"><span class="pc-dot" style="color:${color}">•</span>${escapeHtml(p)}</div>`).join("") : '<div class="pc-item empty-hint">暂无</div>'}
          </div>
          <div>
            <div class="pc-head con">✗ 风险 / 扣分</div>
            ${cons.length ? cons.map((c) => `<div class="pc-item"><span class="pc-dot" style="color:var(--red)">•</span>${escapeHtml(c)}</div>`).join("") : '<div class="pc-item empty-hint">无明显风险</div>'}
          </div>
        </div>
        <div class="dims">${DIM_ORDER.map((dim) => {
          const v = clamp(cand.quality_scores?.[dim] ?? 0, 0, 1);
          return `<div class="dim"><div class="dim-name">${escapeHtml(DIM_LABELS[dim])}</div>
            <div class="dim-bar-row"><div class="dim-track"><div class="dim-fill" style="width:${(v * 100).toFixed(0)}%;background:${color}"></div></div>
            <span class="dim-val" style="color:${color}">${(v * 100).toFixed(0)}</span></div></div>`;
        }).join("")}</div>
      </div>
    </div>`;
}
function bindAccordions() {
  document.querySelectorAll(".acc").forEach((acc) => {
    const head = acc.querySelector(".acc-head");
    const body = acc.querySelector(".acc-body");
    head.addEventListener("click", () => {
      const open = acc.classList.toggle("open");
      body.style.display = open ? "" : "none";
    });
  });
}
function matrixTable(rows) {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  return `<table class="markdown-body" style="min-width:760px"><thead><tr>${headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${headers.map((h) => `<td>${escapeHtml(String(r[h]))}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

/* ───────────── Tab 3: topology ───────────── */
function renderTopology(wrap) {
  const diagrams = state.topologyDiagrams;
  if (!diagrams) {
    wrap.innerHTML = `<div class="section-label">架构拓扑图</div>
      <div class="card topo-card"><div class="topo-stage"><div class="topo-loading"><div class="spinner"></div><strong>架构图生成中</strong><span>正在结合需求特征、知识图谱与规则校验生成定制拓扑</span></div></div></div>`;
    return;
  }
  const entries = Object.entries(diagrams).filter(([, src]) => src && src.trim());
  if (!entries.length) {
    wrap.innerHTML = `<div class="section-label">架构拓扑图</div>
      <div class="card topo-card"><div class="topo-stage"><div class="topo-loading"><strong>架构图未生成</strong><span>后端未返回可渲染的 Mermaid 拓扑源码。</span></div></div></div>`;
    return;
  }
  const complete = entries.find(([n]) => n.includes("完整")) || entries[0];
  const [name, source] = complete;
  let html = `<div class="section-label">架构拓扑图</div>
    <div class="card topo-card">
      <div class="topo-toolbar">
        <span class="topo-name">${escapeHtml(name)}　·　拖拽平移 · 滚轮缩放</span>
        <div class="topo-actions">
          <button class="ghost-btn" id="tZoomOut">缩小</button>
          <button class="ghost-btn" id="tZoomReset">适配</button>
          <button class="ghost-btn" id="tZoomIn">放大</button>
          <button class="ghost-btn" id="tFull">大图查看</button>
          <button class="ghost-btn" id="tCopySvg">复制 SVG</button>
        </div>
      </div>
      <div class="topo-stage" id="topoStage"><div class="topo-loading"><div class="spinner"></div></div></div>
    </div>`;
  if (state.composition && state.composition.composition_needed) {
    const layers = buildComboLayers(state.composition);
    html += `<div class="card" style="margin-top:24px"><div class="chart-title">推荐组合架构分层</div>
      <div class="combo-layers">${layers.map((l, i) => `<div class="combo-layer" style="background:${l.color}0d;border-left:3px solid ${l.color};border-radius:${i === 0 ? "12px 12px 0 0" : i === layers.length - 1 ? "0 0 12px 12px" : "0"}">
        <span class="layer-name" style="color:${l.color}">${escapeHtml(l.name)}</span><span class="layer-tech">${escapeHtml(l.tech)}</span></div>`).join("")}</div></div>`;
  }
  wrap.innerHTML = html;

  document.querySelector("#tZoomIn").addEventListener("click", () => setTopoScale(topoState.scale * 1.2));
  document.querySelector("#tZoomOut").addEventListener("click", () => setTopoScale(topoState.scale / 1.2));
  document.querySelector("#tZoomReset").addEventListener("click", fitTopo);
  document.querySelector("#tFull").addEventListener("click", openModal);
  document.querySelector("#tCopySvg").addEventListener("click", () => copyText(topoState.svg, "已复制 SVG"));
  drawTopology(source);
}

async function drawTopology(source) {
  const stage = document.querySelector("#topoStage");
  if (!stage) return;
  topoState.source = source; topoState.svg = "";
  topoState.scale = 1; topoState.panX = 0; topoState.panY = 0;
  // Rendered SVG is cached so the modal can reuse the exact diagram without a second Mermaid pass.
  if (!window.mermaid) { stage.innerHTML = `<div class="topo-loading"><strong>Mermaid 未加载</strong><span>请检查网络后刷新页面。</span></div>`; return; }
  try {
    const { svg } = await mermaid.render(`topo-${Date.now()}`, source);
    topoState.svg = svg;
    stage.innerHTML = `<div class="topo-canvas" id="topoCanvas">${svg}</div>`;
    bindStagePanZoom(stage, "topo");
    fitTopo();
  } catch {
    stage.innerHTML = `<pre class="diagram-error">${escapeHtml(source)}</pre>`;
    showToastOnce("mermaid-fail", "架构图渲染失败", "Mermaid 无法渲染当前拓扑源码。");
  }
}
function applyTopoTransform() {
  const c = document.querySelector("#topoCanvas");
  if (c) c.style.transform = `translate(${topoState.panX}px, ${topoState.panY}px) scale(${topoState.scale})`;
}
function setTopoScale(v) { topoState.scale = clamp(v, 0.3, 5); applyTopoTransform(); }
function fitTopo() {
  const stage = document.querySelector("#topoStage");
  const svg = stage?.querySelector("svg");
  if (!svg) return;
  const s = svgSize(svg);
  const availW = Math.max(320, stage.clientWidth - 40);
  const availH = Math.max(360, stage.clientHeight - 40);
  topoState.scale = clamp(Math.min(availW / s.w, availH / s.h) * 0.98, 0.3, 4);
  topoState.panX = 0; topoState.panY = 0;
  applyTopoTransform();
}
function svgSize(svg) {
  const vb = svg.getAttribute("viewBox");
  if (vb) { const p = vb.split(/\s+/).map(Number); if (p.length === 4 && p[2] > 0 && p[3] > 0) return { w: p[2], h: p[3] }; }
  return { w: parseFloat(svg.getAttribute("width")) || 900, h: parseFloat(svg.getAttribute("height")) || 600 };
}
function bindStagePanZoom(stage, which) {
  let dragging = false, sx = 0, sy = 0, px = 0, py = 0;
  const st = () => (which === "topo" ? topoState : modalState);
  const apply = which === "topo" ? applyTopoTransform : applyModalTransform;
  stage.style.cursor = "grab";
  stage.addEventListener("pointerdown", (e) => {
    if (e.target.closest("button")) return;
    dragging = true; stage.style.cursor = "grabbing";
    sx = e.clientX; sy = e.clientY; px = st().panX; py = st().panY;
    try { stage.setPointerCapture(e.pointerId); } catch {}
  });
  stage.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    st().panX = px + (e.clientX - sx);
    st().panY = py + (e.clientY - sy);
    apply();
  });
  const end = (e) => { dragging = false; stage.style.cursor = "grab"; try { stage.releasePointerCapture(e.pointerId); } catch {} };
  stage.addEventListener("pointerup", end);
  stage.addEventListener("pointercancel", end);
  stage.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    if (which === "topo") setTopoScale(topoState.scale * factor);
    else setModalScale(modalState.scale * factor);
  }, { passive: false });
}
function openModal() {
  if (!topoState.svg) { showToast("架构图未生成", "当前没有可放大的拓扑图。", "info"); return; }
  modalState = { scale: 1, panX: 0, panY: 0 };
  modalCanvasEl.innerHTML = `<div class="topo-canvas" id="modalCanvas">${topoState.svg}</div>`;
  modalEl.classList.add("open"); modalEl.setAttribute("aria-hidden", "false");
  bindStagePanZoom(modalCanvasEl, "modal");
  setTimeout(fitModal, 30);
}
function fitModal() {
  const svg = modalCanvasEl.querySelector("svg");
  if (!svg) return;
  const s = svgSize(svg);
  const availW = Math.max(480, modalCanvasEl.clientWidth - 60);
  const availH = Math.max(480, modalCanvasEl.clientHeight - 60);
  modalState.scale = clamp(Math.min(availW / s.w, availH / s.h) * 0.98, 0.3, 5);
  modalState.panX = 0; modalState.panY = 0;
  applyModalTransform();
}
function closeModal() { modalEl.classList.remove("open"); modalEl.setAttribute("aria-hidden", "true"); modalCanvasEl.innerHTML = ""; }
function applyModalTransform() { const c = document.querySelector("#modalCanvas"); if (c) c.style.transform = `translate(${modalState.panX}px, ${modalState.panY}px) scale(${modalState.scale})`; }
function setModalScale(v) { modalState.scale = clamp(v, 0.3, 6); applyModalTransform(); }

/* ───────────── Tab 4: trace (human-readable) ───────────── */
function renderTrace(wrap) {
  const t = state.decisionTrace || {};
  const f = t.requirement_features || state.features || {};
  const rules = t.rule_evidence || {};
  const caseMem = t.case_memory_evidence || {};
  const scores = (t.score_evidence && t.score_evidence.length ? t.score_evidence : state.candidates) || [];
  const localMatcher = t.local_matcher_evidence || [];

  const qTags = Object.entries(f.quality_attributes || {}).filter(([, v]) => Number(v) > 0.05)
    .map(([k, v]) => { const color = QUALITY_COLORS[k] || "#10b981"; return `<div class="feature-tag" style="background:${color}15;border:1px solid ${color}30"><span class="dot" style="background:${color}"></span><span>${escapeHtml(QUALITY_LABELS[k] || k)}</span><span class="ft-val" style="color:${color}">${pct(v)}%</span></div>`; }).join("");

  // rules
  const ruleReasons = rules.reasons || [];
  const rejected = rules.rejected_style_ids || [];
  let ruleRows = ruleReasons.map((r) => {
    const m = String(r).match(/^(R\d+)\s*[:：]?\s*(.*)$/);
    return `<div class="rule-row hit"><span class="rule-id">${escapeHtml(m ? m[1] : "规则")}</span><span class="rule-text">${escapeHtml(m ? m[2] : r)}</span><span class="rule-flag">命中</span></div>`;
  }).join("");
  if (rejected.length) ruleRows += `<div class="rule-row exclude"><span class="rule-id">排除</span><span class="rule-text">硬约束降权：${escapeHtml(rejected.join("、"))}</span><span class="rule-flag">排除</span></div>`;
  if (!ruleRows) ruleRows = `<div class="empty-hint">未命中特定硬约束规则，推荐主要由大模型匹配 Agent 与本地评分共同得出。</div>`;

  // case memory
  const retrieved = caseMem.retrieved || [];
  let caseHtml = retrieved.length
    ? retrieved.slice(0, 4).map((c) => `<div class="case-row"><div class="case-title">${escapeHtml(c.title || "历史案例")}</div><div class="case-meta">相似度 ${escapeHtml(String(c.similarity ?? "—"))} · ${escapeHtml((c.expected_styles || []).join("、"))}</div></div>`).join("")
    : `<div class="empty-hint">本次为新需求，未命中可信历史案例（不注入，由大模型独立判断）。</div>`;
  caseHtml += `<div class="kv" style="margin-top:8px">策略：${escapeHtml(caseMem.policy || "仅可信案例参与检索注入")}</div>`;

  // per-candidate signals
  const sigByStyle = {};
  localMatcher.forEach((m) => { sigByStyle[m.name] = m; });
  let accHtml = scores.slice(0, 3).map((s, i) => {
    const signals = (s.matched_reasons || []).slice(0, 4);
    const color = colorFor(i);
    return `<div class="acc ${i === 0 ? "open" : ""}">
      <button class="acc-head"><div class="acc-titles"><div class="acc-name">${escapeHtml(s.name)} <span class="badge" style="background:${color}20;color:${color};border:1px solid ${color}40">${Math.round(s.score)}/100</span></div></div><span class="acc-chev">▾</span></button>
      <div class="acc-body" ${i === 0 ? "" : 'style="display:none"'}>
        ${signals.length ? signals.map((sg) => `<div class="signal-row"><span class="sig-arrow">↔</span><span class="sig-text">${escapeHtml(sg)}</span></div>`).join("") : '<div class="empty-hint">暂无匹配信号</div>'}
        ${(s.deductions && s.deductions.length) ? `<div class="kv" style="margin-top:8px;color:var(--red)">扣分：${escapeHtml(s.deductions.slice(0, 2).join("；"))}</div>` : ""}
        <details class="fold" style="margin-top:10px"><summary>技术细节（原始分 / 本地匹配）</summary><div class="fold-body">
          <div class="kv">综合分 ${escapeHtml(String(s.score))} · 原始分 ${escapeHtml(String(s.raw_score ?? "—"))} · 置信度 ${escapeHtml(s.confidence || "中")}</div>
          ${sigByStyle[s.name] ? `<div class="kv">本地匹配器：${escapeHtml(String(sigByStyle[s.name].score))}/100（原始 ${escapeHtml(String(sigByStyle[s.name].raw_score ?? "—"))}）</div>` : ""}
        </div></details>
      </div></div>`;
  }).join("");

  let html = `
    <div class="section-label">决策溯源</div>
    <p class="trace-intro">以下记录了 AI 推荐的完整决策依据：需求特征提取 → 规则引擎与案例记忆 → 架构匹配信号，逐层可展开。</p>
    <div class="card" style="margin-bottom:16px">
      <div class="trace-step-title"><span class="step-no">1</span>需求特征提取</div>
      <div class="feature-tags">${qTags || '<span class="empty-hint">暂无特征</span>'}</div>
      <div class="agent-note"><div class="note-label">解析 Agent 输出</div>
        <p>领域：${escapeHtml(f.domain || "未知")} · 数据流：${escapeHtml(f.data_flow || "未知")}。关键词：${escapeHtml((f.keywords || []).join("、") || "无")}。${(f.ambiguity_notes && f.ambiguity_notes.length) ? "模糊点：" + escapeHtml(f.ambiguity_notes.join("；")) : "需求信号清晰。"}</p></div>
    </div>
    <div class="card" style="margin-bottom:16px">
      <div class="trace-step-title"><span class="step-no">2</span>规则引擎 + 案例记忆</div>
      ${ruleRows}
      <div style="margin-top:14px"><div class="pc-head" style="color:var(--text-2);margin-bottom:8px">案例记忆检索</div>${caseHtml}</div>
    </div>
    <div class="card" style="margin-bottom:16px">
      <div class="trace-step-title"><span class="step-no">3</span>架构匹配信号</div>
      ${accHtml || '<div class="empty-hint">暂无匹配信号</div>'}
    </div>
    ${renderTraceExtras(t)}`;
  wrap.innerHTML = html;
  bindAccordions();
  bindFolds();
}
function renderTraceExtras(t) {
  const llm = t.llm_review || [];
  const cache = t.recommendation_cache_evidence || {};
  const topo = t.topology_evidence || {};
  const items = [];
  if (llm.length) items.push(`<div class="pc-head" style="color:var(--text-2)">DeepSeek 复核意见</div><ul class="muted-list">${llm.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>`);
  if (t.final_reason) items.push(`<div class="kv" style="margin-top:8px">最终说明：${escapeHtml(t.final_reason)}</div>`);
  if (cache.hit) items.push(`<div class="kv">推荐缓存命中：${escapeHtml(cache.hit_type || "")} · 相似度 ${escapeHtml(String(cache.similarity ?? ""))}</div>`);
  if (topo.scenarios && topo.scenarios.length) items.push(`<div class="kv">拓扑场景：${escapeHtml(topo.scenarios.join("、"))}</div>`);
  if (topo.capabilities && topo.capabilities.length) items.push(`<div class="kv">业务能力：${escapeHtml(topo.capabilities.slice(0, 12).join("、"))}</div>`);
  const traceLog = state.trace || [];
  return `
    <details class="fold"><summary>更多技术证据（LLM 复核 / 缓存 / 拓扑覆盖）</summary><div class="fold-body">${items.join("") || '<div class="empty-hint">暂无</div>'}</div></details>
    <details class="fold"><summary>Agent 协作日志（${traceLog.length} 步）</summary><div class="fold-body"><ul class="muted-list">${traceLog.map((x) => `<li>${escapeHtml(x)}</li>`).join("") || '<li>暂无</li>'}</ul></div></details>`;
}
function bindFolds() { /* native <details>, no-op kept for clarity */ }

/* ───────────── markdown (report) ───────────── */
function markdownToHtml(md) {
  const lines = String(md).split("\n");
  const out = [];
  let inList = false, inTable = false;
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { if (inList) { out.push("</ul>"); inList = false; } if (inTable) { out.push("</tbody></table>"); inTable = false; } continue; }
    if (line.startsWith("|") && line.endsWith("|")) {
      const cells = line.split("|").slice(1, -1).map((c) => inlineMd(c.trim()));
      if (cells.every((c) => /^:?-{3,}:?$/.test(c))) continue;
      if (!inTable) { out.push("<table><tbody>"); inTable = true; }
      const tag = out[out.length - 1] === "<table><tbody>" ? "th" : "td";
      out.push(`<tr>${cells.map((c) => `<${tag}>${c}</${tag}>`).join("")}</tr>`); continue;
    }
    if (inTable) { out.push("</tbody></table>"); inTable = false; }
    if (line.startsWith("## ")) { out.push(`<h2>${inlineMd(line.slice(3))}</h2>`); continue; }
    if (line.startsWith("# ")) { out.push(`<h1>${inlineMd(line.slice(2))}</h1>`); continue; }
    if (line.startsWith("> ")) { out.push(`<blockquote>${inlineMd(line.slice(2))}</blockquote>`); continue; }
    if (line.startsWith("- ")) { if (!inList) { out.push("<ul>"); inList = true; } out.push(`<li>${inlineMd(line.slice(2))}</li>`); continue; }
    if (line.startsWith("√ ") || line.startsWith("× ")) { const good = line.startsWith("√ "); out.push(`<p class="${good ? "pro-line" : "con-line"}"><strong>${line.slice(0, 1)}</strong> ${inlineMd(line.slice(2))}</p>`); continue; }
    if (inList) { out.push("</ul>"); inList = false; }
    out.push(`<p>${inlineMd(line)}</p>`);
  }
  if (inList) out.push("</ul>");
  if (inTable) out.push("</tbody></table>");
  return out.join("");
}
function inlineMd(text) { return escapeHtml(text).replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>"); }

/* ───────────── clipboard ───────────── */
async function copyText(value, ok) {
  if (!value) { showToast("复制失败", "当前没有可复制的内容。"); return; }
  try { await navigator.clipboard.writeText(value); showToast("复制成功", ok, "info"); }
  catch { showToast("复制失败", "浏览器剪贴板不可用。"); }
}

/* ───────────── knowledge center (settings) ───────────── */
const NODE_TYPE_STYLE = {
  architecture_style: { color: "#10b981", label: "架构风格", size: 22 },
  category: { color: "#f59e0b", label: "类别", size: 16 },
  scenario: { color: "#3b82f6", label: "适用场景", size: 14 },
  quality_attribute: { color: "#8b5cf6", label: "质量属性", size: 14 },
  qualityattribute: { color: "#8b5cf6", label: "质量属性", size: 14 },
  domainscenario: { color: "#3b82f6", label: "领域场景", size: 22 },
  businesscapability: { color: "#10b981", label: "业务能力", size: 18 },
  architecturecomponent: { color: "#06b6d4", label: "架构组件", size: 14 },
  datastore: { color: "#f59e0b", label: "数据存储", size: 13 },
};
const REL_LABELS = {
  BELONGS_TO: "属于", SUITABLE_FOR: "适用", HAS_SCORE: "",
  REQUIRES: "需要", IMPLEMENTED_BY: "实现", USES_STORE: "用存储", STORES_IN: "写入", DEPENDS_ON: "依赖",
};

async function fetchJson(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) { const p = await res.json().catch(() => ({})); throw new Error(p.detail || `${url} ${res.status}`); }
  return res.json();
}

function openSettings() {
  state.phase = "settings";
  state.settingsTab = state.settingsTab || "overview";
  renderApp();
}

const KG_TABS = [
  { key: "overview", label: "概览" },
  { key: "style", label: "风格知识图谱" },
  { key: "topo", label: "领域拓扑图谱" },
  { key: "cases", label: "案例数据库" },
];
function renderSettings() {
  appEl.innerHTML = `
    <div class="results">
      <header class="r-header">
        <div class="r-header-inner">
          <div class="brand-row"><div class="mini-mark">◎</div><span class="brand-name">ArchWise · 知识中心</span></div>
          <button class="new-btn" id="kgBack">← 返回</button>
        </div>
        <div class="tabs">${KG_TABS.map((t) => `<button class="tab-btn ${t.key === state.settingsTab ? "active" : ""}" data-kgtab="${t.key}">${t.label}</button>`).join("")}</div>
      </header>
      <div class="tab-wrap" id="kgBody"></div>
    </div>`;
  document.querySelector("#kgBack").addEventListener("click", () => { state.phase = state.finalRec ? "results" : "input"; renderApp(); });
  document.querySelectorAll("[data-kgtab]").forEach((b) => b.addEventListener("click", () => {
    state.settingsTab = b.dataset.kgtab;
    document.querySelectorAll("[data-kgtab]").forEach((x) => x.classList.toggle("active", x.dataset.kgtab === state.settingsTab));
    renderSettingsTab();
  }));
  renderSettingsTab();
}
function renderSettingsTab() {
  const body = document.querySelector("#kgBody");
  if (!body) return;
  if (visNetwork) { try { visNetwork.destroy(); } catch {} visNetwork = null; }
  if (state.settingsTab === "overview") renderKgOverview(body);
  else if (state.settingsTab === "style") renderKgGraph(body, "style");
  else if (state.settingsTab === "topo") renderKgGraph(body, "topo");
  else if (state.settingsTab === "cases") renderKgCases(body);
}

/* overview dashboard */
function statCard(label, value, unit, color) {
  return `<div class="kg-stat"><div class="kg-stat-val" style="color:${color}">${value}<span class="kg-stat-unit">${unit}</span></div><div class="kg-stat-label">${escapeHtml(label)}</div></div>`;
}
function sourceBar(label, n, total, color) {
  const w = total ? Math.round((n / total) * 100) : 0;
  return `<div class="bar-row" style="margin:8px 0"><span class="bar-label" style="width:88px">${escapeHtml(label)}</span><div class="bar-track" style="height:18px"><div class="bar-fill" style="width:${w}%;background:${color}"></div></div><span class="bar-score" style="color:${color}">${n}</span></div>`;
}
async function renderKgOverview(body) {
  body.innerHTML = `<div class="section-label">系统知识概览</div><div class="topo-loading"><div class="spinner"></div></div>`;
  try {
    const [status, styles, cases] = await Promise.all([
      fetchJson("/api/knowledge/neo4j/status").catch(() => ({ configured: false })),
      fetchJson("/api/styles").catch(() => []),
      fetchJson("/api/cases").catch(() => []),
    ]);
    kg.status = status; kg.styles = styles; kg.cases = cases;
    const trusted = cases.filter((c) => c.status === "trusted").length;
    const candidate = cases.filter((c) => c.status === "candidate").length;
    const seed = cases.filter((c) => c.source === "seed").length;
    const runtime = cases.filter((c) => c.source === "runtime").length;
    const manual = cases.filter((c) => c.source === "manual").length;
    const neoOk = status.configured && status.ok;
    body.innerHTML = `
      <div class="section-label">系统知识概览</div>
      <div class="kg-stats">
        ${statCard("架构风格", styles.length, " 种", "#10b981")}
        ${statCard("案例总数", cases.length, " 条", "#3b82f6")}
        ${statCard("可信案例", trusted, " 条", "#10b981")}
        ${statCard("候选案例", candidate, " 条", "#f59e0b")}
      </div>
      <div class="grid-2" style="margin-top:16px">
        <div class="card">
          <div class="chart-title">案例来源构成</div>
          ${sourceBar("种子", seed, cases.length, "#8b5cf6")}
          ${sourceBar("运行时捕获", runtime, cases.length, "#3b82f6")}
          ${sourceBar("人工录入", manual, cases.length, "#10b981")}
          <p class="kg-hint">「运行时捕获」是系统从推荐里自己沉淀的案例，是自我进化的直接证据。</p>
        </div>
        <div class="card">
          <div class="chart-title">Neo4j 知识图谱状态</div>
          <div class="kg-status ${neoOk ? "ok" : "off"}"><span class="status-dot"></span><span>${neoOk ? "已连接" : status.configured ? "已配置 · 连接失败" : "未配置"}</span></div>
          <div class="kv">URI：${escapeHtml(status.uri || "—")}</div>
          <div class="kv">数据库：${escapeHtml(status.database || "—")}</div>
          ${status.error ? `<div class="kv" style="color:var(--red)">${escapeHtml(status.error)}</div>` : ""}
          <div style="margin-top:12px;display:flex;align-items:center;gap:10px">
            <button class="ghost-btn" id="kgSync">同步知识到 Neo4j</button>
            <span class="kv" id="kgSyncStatus"></span>
          </div>
        </div>
      </div>`;
    const sync = document.querySelector("#kgSync");
    if (sync) sync.addEventListener("click", triggerNeo4jSync);
  } catch (e) {
    body.innerHTML = `<div class="section-label">系统知识概览</div><div class="empty-hint">概览加载失败：${escapeHtml(e.message)}</div>`;
  }
}
async function triggerNeo4jSync() {
  const el = document.querySelector("#kgSyncStatus");
  try {
    el.textContent = "同步触发中…";
    await fetchJson("/api/knowledge/neo4j/sync", { method: "POST" });
    const poll = async () => {
      try {
        const s = await fetchJson("/api/knowledge/neo4j/sync/status");
        if (s.running) { el.textContent = "后台同步进行中…"; setTimeout(poll, 1600); }
        else { el.textContent = s.ok ? "同步完成，可刷新图谱" : `同步失败：${s.error || "未知"}`; kg.style = null; kg.topo = null; }
      } catch { el.textContent = "状态查询失败"; }
    };
    setTimeout(poll, 1200);
  } catch (e) { el.textContent = `触发失败：${e.message}`; }
}

/* graph viz */
async function renderKgGraph(body, which) {
  const url = which === "style" ? "/api/knowledge/graph" : "/api/knowledge/topology-graph";
  const title = which === "style" ? "架构风格知识图谱" : "领域拓扑知识图谱";
  body.innerHTML = `
    <div class="section-label">${title}</div>
    <div class="card kg-graph-card">
      <div class="kg-graph-toolbar">
        <div class="kg-legend" id="kgLegend"></div>
        <div class="topo-actions"><span class="kv" id="kgGraphMeta"></span><button class="ghost-btn" id="kgRefresh">刷新</button></div>
      </div>
      <div class="kg-graph" id="kgGraph"><div class="topo-loading"><div class="spinner"></div><span>正在读取知识图谱…</span></div></div>
    </div>`;
  document.querySelector("#kgRefresh").addEventListener("click", () => { kg[which] = null; renderKgGraph(body, which); });
  try {
    if (!kg[which]) kg[which] = await fetchJson(url);
    const graph = kg[which];
    const gEl = document.querySelector("#kgGraph");
    if (!graph || !graph.nodes || !graph.nodes.length) {
      gEl.innerHTML = `<div class="topo-loading"><strong>暂无图谱数据</strong><span>${which === "topo" ? "领域拓扑图谱为空。请先在概览页同步/进化拓扑知识，或确认 Neo4j 已连接。" : "未取到风格图谱数据。"}</span></div>`;
      return;
    }
    document.querySelector("#kgGraphMeta").textContent = `${graph.nodes.length} 节点 · ${graph.edges.length} 关系`;
    renderVisNetwork(gEl, graph, document.querySelector("#kgLegend"));
  } catch (e) {
    document.querySelector("#kgGraph").innerHTML = `<div class="topo-loading"><strong>图谱加载失败</strong><span>${escapeHtml(e.message)}</span></div>`;
  }
}
function edgeLabel(e) {
  const rel = String(e.relation || "");
  const base = rel.split(":")[0];
  if (base in REL_LABELS) return REL_LABELS[base];
  return e.label || base;
}
function renderVisNetwork(container, graph, legendEl) {
  if (!window.vis) { container.innerHTML = `<div class="topo-loading"><strong>图库未加载</strong><span>vis-network 未加载，请检查网络后刷新。</span></div>`; return; }
  const typesPresent = new Set();
  // Node type metadata drives both the graph styling and the legend.
  const nodes = graph.nodes.map((n) => {
    typesPresent.add(n.type);
    const st = NODE_TYPE_STYLE[n.type] || { color: "#a1a1aa", size: 14 };
    return { id: n.id, label: n.label, shape: "dot", size: st.size,
      color: { background: st.color, border: st.color, highlight: { background: st.color, border: "#fafafa" } },
      font: { color: "#e4e4e7", size: 13, face: "Noto Sans SC", strokeWidth: 3, strokeColor: "#0a0a0b" } };
  });
  const edges = graph.edges.map((e) => ({ from: e.source, to: e.target, label: edgeLabel(e), arrows: "to",
    color: { color: "#3f3f46", highlight: "#10b981", opacity: 0.9 }, width: 1,
    font: { color: "#71717a", size: 10, strokeWidth: 0, background: "#0a0a0b" },
    dashes: e.kind === "event", smooth: { enabled: true, type: "dynamic" } }));
  container.innerHTML = "";
  if (legendEl) legendEl.innerHTML = [...typesPresent].map((t) => { const st = NODE_TYPE_STYLE[t] || { color: "#a1a1aa", label: t }; return `<span class="kg-legend-item"><span class="kg-dot" style="background:${st.color}"></span>${escapeHtml(st.label)}</span>`; }).join("");
  visNetwork = new vis.Network(container,
    { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) },
    { physics: { solver: "forceAtlas2Based", forceAtlas2Based: { gravitationalConstant: -48, springLength: 120, springConstant: 0.045, avoidOverlap: 0.4 }, stabilization: { iterations: 200 } },
      interaction: { hover: true, tooltipDelay: 120, dragNodes: true, navigationButtons: false },
      nodes: { borderWidth: 0 } });
}

/* case database */
async function renderKgCases(body) {
  body.innerHTML = `<div class="section-label">案例数据库</div><div class="topo-loading"><div class="spinner"></div></div>`;
  try {
    kg.cases = await fetchJson("/api/cases");
    renderCaseList(body, kg.cases, "all");
  } catch (e) {
    body.innerHTML = `<div class="section-label">案例数据库</div><div class="empty-hint">案例库加载失败：${escapeHtml(e.message)}（可能 Chroma / embedding 未连接）</div>`;
  }
}
function renderCaseList(body, cases, filter) {
  const filters = [
    { key: "all", label: `全部 ${cases.length}` },
    { key: "trusted", label: `可信 ${cases.filter((c) => c.status === "trusted").length}` },
    { key: "candidate", label: `候选 ${cases.filter((c) => c.status === "candidate").length}` },
    { key: "runtime", label: `运行时 ${cases.filter((c) => c.source === "runtime").length}` },
    { key: "seed", label: `种子 ${cases.filter((c) => c.source === "seed").length}` },
  ];
  const filtered = cases.filter((c) => (filter === "all" ? true : c.status === filter || c.source === filter));
  body.innerHTML = `
    <div class="kg-cases-head"><div class="section-label" style="margin:0">案例数据库</div><button class="new-btn" id="kgAddCase">+ 新增案例</button></div>
    <p class="trace-intro">系统内置 + 运行时学到的案例。<b>运行时 / 候选</b>是自我进化的证据；「提升为可信」纳入检索注入，「删除」可清理重复或错误案例。</p>
    <div class="kg-filters">${filters.map((f) => `<button class="kg-filter ${f.key === filter ? "active" : ""}" data-filter="${f.key}">${escapeHtml(f.label)}</button>`).join("")}</div>
    <div class="kg-cases">${filtered.length ? filtered.map(caseCard).join("") : '<div class="empty-hint">该筛选下暂无案例。</div>'}</div>`;
  document.querySelector("#kgAddCase").addEventListener("click", () => openAddCaseModal(body, filter));
  body.querySelectorAll("[data-filter]").forEach((b) => b.addEventListener("click", () => renderCaseList(body, cases, b.dataset.filter)));
  body.querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", async () => {
    if (!window.confirm("确定删除这条案例吗？此操作不可撤销。")) return;
    b.disabled = true; b.textContent = "删除中…";
    try {
      await fetchJson("/api/knowledge/cases/delete", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ case_id: b.dataset.del }) });
      kg.cases = await fetchJson("/api/cases");
      renderCaseList(body, kg.cases, filter);
      showToast("已删除", "案例已从案例库移除。", "info");
    } catch (e) { b.disabled = false; b.textContent = "删除"; showToast("删除失败", e.message); }
  }));
  body.querySelectorAll("[data-trust]").forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true; b.textContent = "提升中…";
    try {
      await fetchJson("/api/knowledge/cases/trust", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ case_id: b.dataset.trust }) });
      kg.cases = await fetchJson("/api/cases");
      renderCaseList(body, kg.cases, filter);
      showToast("已提升为可信", "该案例将参与后续相似检索注入。", "info");
    } catch (e) { b.disabled = false; b.textContent = "提升为可信"; showToast("提升失败", e.message); }
  }));
}
function caseCard(c) {
  const trusted = c.status === "trusted";
  const statusColor = trusted ? "#10b981" : "#f59e0b";
  const sourceMap = { seed: "种子", manual: "人工", runtime: "运行时" };
  const req = String(c.requirement || "");
  return `<div class="card kg-case">
    <div class="kg-case-head">
      <span class="kg-case-title">${escapeHtml(c.title || "未命名案例")}</span>
      <span class="badge" style="background:${statusColor}20;color:${statusColor};border:1px solid ${statusColor}40">${trusted ? "可信" : "候选"}</span>
      <span class="badge" style="background:#1c1c1f;color:#a1a1aa;border:1px solid #27272a">${escapeHtml(sourceMap[c.source] || c.source || "")}</span>
    </div>
    <p class="kg-case-req">${escapeHtml(req.slice(0, 150))}${req.length > 150 ? "…" : ""}</p>
    <div class="kg-case-styles">
      ${(c.expected_styles || []).length ? `<span class="kv">期望：${escapeHtml((c.expected_styles || []).join("、"))}</span>` : ""}
      ${(c.recommended_styles || []).length ? `<span class="kv">推荐：${escapeHtml((c.recommended_styles || []).join("、"))}</span>` : ""}
    </div>
    <div class="kg-case-foot">
      <span class="kv">置信度 ${escapeHtml(String(c.confidence ?? "—"))} · ${escapeHtml(String(c.updated_at || c.created_at || "").slice(0, 10))}</span>
      <div class="kg-case-actions">
        ${trusted ? "" : `<button class="ghost-btn" data-trust="${escapeHtml(c.id)}">提升为可信</button>`}
        <button class="ghost-btn danger" data-del="${escapeHtml(c.id)}">删除</button>
      </div>
    </div>
  </div>`;
}

function openAddCaseModal(body, filter) {
  const overlay = document.createElement("div");
  overlay.className = "kg-modal-overlay";
  overlay.innerHTML = `
    <div class="kg-modal">
      <div class="kg-modal-head"><strong>新增案例</strong><button class="ghost-btn" data-close>关闭</button></div>
      <div class="kg-modal-body">
        <label class="kg-field"><span>标题</span><input id="acTitle" type="text" placeholder="例如：跨境电商交易中台"></label>
        <label class="kg-field"><span>需求描述</span><textarea id="acReq" rows="4" placeholder="描述这个系统的需求、规模、并发、一致性与部署约束…"></textarea></label>
        <label class="kg-field"><span>推荐 / 期望架构（用、或逗号分隔）</span><input id="acStyles" type="text" placeholder="微服务架构、事件驱动架构、CQRS 架构"></label>
        <label class="kg-field"><span>备注（可选）</span><input id="acNotes" type="text" placeholder="补充说明"></label>
        <label class="kg-check"><input id="acTrusted" type="checkbox"> 我确定，直接加为可信（默认仅作候选，需再确认才进入检索）</label>
        <div id="acAdvisory" class="kg-advisory"></div>
      </div>
      <div class="kg-modal-foot">
        <button class="ghost-btn" data-close>取消</button>
        <button class="submit-btn" id="acSubmit" style="padding:9px 22px">检查并新增</button>
      </div>
    </div>`;
  document.body.appendChild(overlay);
  const close = () => overlay.remove();
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
  overlay.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", close));

  const parseStyles = (s) => s.split(/[，,、]/).map((x) => x.trim()).filter(Boolean);
  const submit = overlay.querySelector("#acSubmit");
  const advisory = overlay.querySelector("#acAdvisory");
  let confirmedDespiteWarning = false;

  submit.addEventListener("click", async () => {
    const title = overlay.querySelector("#acTitle").value.trim();
    const requirement = overlay.querySelector("#acReq").value.trim();
    const expected_styles = parseStyles(overlay.querySelector("#acStyles").value);
    const notes = overlay.querySelector("#acNotes").value.trim();
    const as_trusted = overlay.querySelector("#acTrusted").checked;
    if (title.length < 1 || requirement.length < 5) {
      advisory.className = "kg-advisory warn";
      advisory.textContent = "请填写标题，且需求描述至少 5 个字。";
      return;
    }
    submit.disabled = true;
    submit.textContent = "检查中…";
    try {
      if (!confirmedDespiteWarning) {
        const check = await fetchJson("/api/knowledge/cases/check", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title, requirement, expected_styles, notes }),
        });
        if (check.verdict && check.verdict !== "new") {
          advisory.className = `kg-advisory ${check.verdict === "conflict" ? "warn" : "info"}`;
          advisory.textContent = `${check.message || "检测到相似案例。"} 仍要新增请再次点击。`;
          confirmedDespiteWarning = true;
          submit.disabled = false;
          submit.textContent = "仍然新增";
          return;
        }
      }
      await fetchJson("/api/knowledge/cases", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, requirement, expected_styles, notes, as_trusted }),
      });
      close();
      kg.cases = await fetchJson("/api/cases");
      renderCaseList(body, kg.cases, filter);
      showToast("已新增", as_trusted ? "已作为可信案例加入。" : "已作为候选案例加入，待提升为可信。", "info");
    } catch (e) {
      advisory.className = "kg-advisory warn";
      advisory.textContent = `新增失败：${e.message}`;
      submit.disabled = false;
      submit.textContent = confirmedDespiteWarning ? "仍然新增" : "检查并新增";
    }
  });
}

/* ───────────── boot ───────────── */
renderApp();
