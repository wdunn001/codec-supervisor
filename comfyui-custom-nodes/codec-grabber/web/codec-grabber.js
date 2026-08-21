/**
 * ComfyUI-CodecGrabber — frontend.
 *
 * v8 — corruption surfacing + per-entry re-download for any on-disk model.
 *
 * Adds:
 *   - "Verify all" button in the Downloads panel: runs server-side
 *     safetensors header-coverage check across every "done" entry, marks
 *     corrupt ones with a ⚠ badge.
 *   - Auto-verify on panel open (debounced); corrupt entries get a "Re-
 *     download" button right there with the right URL already populated.
 *   - Self-heal toast on SafetensorError now uses the v6 verify endpoint
 *     to confirm corruption before offering re-download, and accepts
 *     explicit URL via the v6 /redownload path so files that never
 *     reached history (e.g. failed first-attempt downloads) can still be
 *     repaired by URL.
 *   - URL capture: per-call queue instead of a shared captureActive flag,
 *     so concurrent Download clicks don't overwrite each other's URL.
 *
 * v7 baseline: downloads run as server-side background asyncio tasks.
 * The browser just observes via the ComfyUI WebSocket. Closing the tab,
 * reloading, or navigating away does not stop any download.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const log = (...a) => console.log("[codec-grabber]", ...a);
const AUGMENTED_ATTR = "data-codec-grabber-augmented";

// ─── URL capture for the Missing Models panel ───────────────────────────
//
// The Frontend's downloadModel() handler synthesises an <a download href=URL>,
// programmatically clicks it, then removes it. We intercept by monkey-patching
// document.createElement: any <a> created while at least one capture call is
// pending has its .click() swallowed and the (href, download) recorded into
// the OLDEST pending capture's queue.
//
// Earlier versions used a single shared captureActive flag + captured
// variable, which meant concurrent Download clicks (e.g. Download All)
// would all write to the same slot — and the last write wins. Two rows
// could end up with the same URL, and the corrupt-on-disk file looked
// indistinguishable from a real download. v8 fixes this by giving every
// captureNextDownload call its own queue.

const origCreateElement = document.createElement.bind(document);
const pendingCaptures = []; // FIFO of {urls: []}; each captureNextDownload call gets one.
document.createElement = function patchedCreateElement(tag, opts) {
  const el = origCreateElement(tag, opts);
  if (pendingCaptures.length && typeof tag === "string" && tag.toLowerCase() === "a") {
    const origClick = el.click.bind(el);
    el.click = function patchedAnchorClick() {
      if (pendingCaptures.length) {
        // Route to the oldest pending capture so concurrent calls don't collide.
        pendingCaptures[0].urls.push({ url: el.href, name: el.download });
        return;
      }
      origClick();
    };
  }
  return el;
};
async function captureNextDownload(action) {
  const ticket = { urls: [] };
  pendingCaptures.push(ticket);
  try {
    await action();
    await new Promise(r => setTimeout(r, 50));
  } finally {
    const i = pendingCaptures.indexOf(ticket);
    if (i >= 0) pendingCaptures.splice(i, 1);
  }
  return ticket.urls[0] || null;
}

// ─── Pretty formatters ──────────────────────────────────────────────────

function fmtBytes(n) {
  if (n == null) return "?";
  if (n < 1024) return `${n} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let v = n / 1024, u = 0;
  while (v >= 1024 && u < units.length - 1) { v /= 1024; u++; }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[u]}`;
}

// ─── Backend helpers ────────────────────────────────────────────────────

async function backendStart(meta) {
  const r = await api.fetchApi("/codec/grab-model/start", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(meta),
  });
  return { status: r.status, body: await r.json().catch(() => ({})) };
}
async function backendPause(meta) {
  await api.fetchApi("/codec/grab-model/pause", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(meta),
  });
}
async function backendCancel(meta) {
  await api.fetchApi("/codec/grab-model/cancel", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(meta),
  });
}

// ─── Subscriber registry — multiple UI elements per download ────────────
//
// Every UI element that wants progress updates for a (filename, directory)
// pair registers a subscriber. When a WS event arrives we fan out to all
// subscribers matching that pair (or by entry id).

const subscribers = new Set();
function subscribe(predicate, handler) {
  const s = { predicate, handler };
  subscribers.add(s);
  return () => subscribers.delete(s);
}

api.addEventListener("codec_grabber_progress", (ev) => {
  const d = ev.detail || {};
  for (const s of subscribers) {
    try { if (s.predicate(d)) s.handler(d); } catch (e) { console.warn(e); }
  }
});

// ─── DOM helpers for the Missing Models panel ──────────────────────────

function rowDirectory(rowEl) {
  let cur = rowEl;
  for (let depth = 0; depth < 8 && cur; depth++) {
    const header = cur.querySelector?.(":scope > div > p > span");
    if (header) {
      const m = (header.textContent || "").match(/^\s*([a-z0-9_]+)\s*\(\d+\)\s*$/i);
      if (m) return m[1].toLowerCase();
    }
    cur = cur.parentElement;
  }
  return null;
}
function rowFilename(rowEl) {
  return rowEl.querySelector("p[title]")?.getAttribute("title") || null;
}
function findRow(btn) {
  let row = btn;
  while (row && !row.querySelector("p[title]")) row = row.parentElement;
  return row;
}

// ─── Missing Models panel: per-row swap-in-place ───────────────────────

function controlRow(origBtn, rowEl) {
  const clone = origBtn.cloneNode(true);
  clone.removeAttribute("data-testid");
  clone.removeAttribute("aria-label");
  origBtn.style.display = "none";
  origBtn.parentElement.insertBefore(clone, origBtn.nextSibling);
  const labelSpan = clone.querySelector("span.truncate") || clone.querySelector("span");
  const origLabel = labelSpan?.textContent?.trim() || "Download";
  const setLabel = (t, ico) => {
    if (labelSpan) labelSpan.textContent = t;
    const i = clone.querySelector("i[class*='icon-[lucide']");
    if (i && ico) i.className = i.className.replace(/icon-\[lucide--[a-z0-9-]+\]/, `icon-[lucide--${ico}]`);
  };

  const bar = document.createElement("div");
  bar.style.cssText = "margin-top:6px;display:none;flex-direction:column;gap:2px;width:100%;";
  const track = document.createElement("div");
  track.style.cssText = "width:100%;height:6px;background:rgba(255,255,255,0.08);border-radius:3px;overflow:hidden;";
  const fill = document.createElement("div");
  fill.style.cssText = "width:0%;height:100%;background:linear-gradient(90deg,#4a8,#6fc);transition:width 0.15s ease-out;";
  track.appendChild(fill);
  const status = document.createElement("small");
  status.style.cssText = "font-size:10px;color:#aaa;";
  bar.appendChild(track); bar.appendChild(status);
  (clone.closest("div.flex.w-full") || clone.parentElement).appendChild(bar);

  const cancelBtn = document.createElement("button");
  cancelBtn.textContent = "×"; cancelBtn.title = "Cancel + delete partial";
  cancelBtn.style.cssText = "display:none;margin-left:4px;padding:0 6px;border:1px solid #844;background:#400;color:#eee;border-radius:6px;cursor:pointer;font-size:12px;line-height:1.6;";
  clone.parentElement.insertBefore(cancelBtn, clone.nextSibling);

  const rowState = { meta: null /* {url, filename, directory} */ };

  function applyState(d) {
    bar.style.display = "flex";
    cancelBtn.style.display = (d.status === "done") ? "none" : "inline-block";
    const total = d.expected_bytes;
    const bytes = d.downloaded_bytes || 0;
    const frac = total ? bytes / total : null;
    fill.style.width = `${frac == null ? (d.status === "done" ? 100 : 0) : frac * 100}%`;
    if (d.status === "done") {
      fill.style.background = "limegreen";
      status.textContent = `✓ saved ${fmtBytes(bytes)}`;
      status.style.color = "limegreen";
      setLabel("✓ Downloaded", "check");
      clone.disabled = true;
    } else if (d.status === "error") {
      fill.style.background = "tomato";
      status.textContent = `✗ ${d.error || "error"}`;
      status.style.color = "tomato";
      setLabel("Retry", "rotate-cw");
      clone.disabled = false;
    } else if (d.status === "paused") {
      fill.style.background = "khaki";
      status.textContent = `paused at ${fmtBytes(bytes)}${total ? ` / ${fmtBytes(total)} (${(frac * 100).toFixed(1)}%)` : ""}${d.error ? ` — ${d.error}` : ""}`;
      status.style.color = "khaki";
      setLabel("Resume", "play");
      clone.disabled = false;
    } else if (d.status === "downloading") {
      fill.style.background = "linear-gradient(90deg,#4a8,#6fc)";
      status.textContent = total
        ? `${fmtBytes(bytes)} / ${fmtBytes(total)} (${(frac * 100).toFixed(1)}%)`
        : `${fmtBytes(bytes)}`;
      status.style.color = "deepskyblue";
      setLabel("Pause", "pause");
      clone.disabled = false;
    } else if (d.status === "cancelled") {
      bar.style.display = "none";
      cancelBtn.style.display = "none";
      setLabel(origLabel, "download");
      clone.disabled = false;
    }
  }

  // Listen for WS updates for any entry matching this row's filename+dir.
  subscribe(
    (d) => {
      const meta = rowState.meta;
      if (!meta) return false;
      return d.filename === meta.filename && d.directory === meta.directory;
    },
    applyState,
  );

  async function startOrResume() {
    if (!rowState.meta) {
      const filename = rowFilename(rowEl);
      const directory = rowDirectory(rowEl);
      if (!filename || !directory) {
        bar.style.display = "flex";
        status.textContent = "✗ row parse failed";
        status.style.color = "tomato";
        return;
      }
      bar.style.display = "flex";
      status.textContent = "capturing URL…";
      status.style.color = "deepskyblue";
      const cap = await captureNextDownload(() => origBtn.click());
      if (!cap?.url) {
        status.textContent = "✗ couldn't capture URL";
        status.style.color = "tomato";
        return;
      }
      rowState.meta = { url: cap.url, filename, directory };
    }
    const { status: code, body } = await backendStart(rowState.meta);
    if (code === 409) {
      applyState({ status: "done", downloaded_bytes: 0, expected_bytes: null });
      return;
    }
    if (code >= 400) {
      bar.style.display = "flex";
      status.textContent = `✗ ${body.error || "start failed"}`;
      status.style.color = "tomato";
      return;
    }
    // The first WS event will update the UI; show a placeholder.
    applyState({ status: "downloading", downloaded_bytes: body.downloaded_bytes || 0,
                 expected_bytes: body.expected_bytes });
  }

  async function pause() {
    if (!rowState.meta) return;
    await backendPause(rowState.meta);
    // WS event will follow with status=paused.
  }
  async function cancel() {
    if (rowState.meta) {
      await backendCancel(rowState.meta);
    }
    rowState.meta = null;
    bar.style.display = "none";
    cancelBtn.style.display = "none";
    setLabel(origLabel, "download");
    fill.style.width = "0%";
    fill.style.background = "linear-gradient(90deg,#4a8,#6fc)";
    clone.disabled = false;
  }

  clone.addEventListener("click", (ev) => {
    ev.preventDefault(); ev.stopPropagation();
    const label = (labelSpan?.textContent || "").trim();
    // Drive entirely off the visible label — the source of truth is
    // the server, and the label reflects what the server last told us.
    if (label === "Pause") pause();
    else if (label === "Resume") startOrResume();
    else if (label === "✓ Downloaded") {} // no-op
    else startOrResume(); // Download / Retry / anything else
  });
  cancelBtn.addEventListener("click", (ev) => { ev.preventDefault(); ev.stopPropagation(); cancel(); });

  // On panel open, check if there's already a known download for this
  // row (e.g., the user reloaded the page while a download was running).
  hydrateRow(rowEl, rowState, applyState);
}

async function hydrateRow(rowEl, rowState, applyState) {
  const filename = rowFilename(rowEl);
  const directory = rowDirectory(rowEl);
  if (!filename || !directory) return;
  try {
    const r = await api.fetchApi("/codec/grab-model/history");
    const body = await r.json();
    const match = (body.entries || []).find(
      (e) => e.filename === filename && e.directory === directory,
    );
    if (match) {
      rowState.meta = { url: match.url, filename, directory };
      applyState(match);
    }
  } catch {}
}

function controlAll(origAllBtn) {
  const clone = origAllBtn.cloneNode(true);
  clone.removeAttribute("data-testid");
  origAllBtn.style.display = "none";
  origAllBtn.parentElement.insertBefore(clone, origAllBtn.nextSibling);
  clone.addEventListener("click", (ev) => {
    ev.preventDefault(); ev.stopPropagation();
    const rowClones = document.querySelectorAll('[data-testid="missing-model-download"] + button');
    let fired = 0;
    rowClones.forEach((b) => { if (!b.disabled) { b.click(); fired++; } });
    log(`Download all → fired ${fired} of ${rowClones.length}`);
  });
}

function augmentMissingModelsPanel() {
  for (const dlBtn of document.querySelectorAll('[data-testid="missing-model-download"]')) {
    if (dlBtn.hasAttribute(AUGMENTED_ATTR)) continue;
    const row = findRow(dlBtn);
    if (!row) continue;
    controlRow(dlBtn, row);
    dlBtn.setAttribute(AUGMENTED_ATTR, "1");
  }
  const allBtn = document.querySelector('[data-testid="missing-model-download-all"]');
  if (allBtn && !allBtn.hasAttribute(AUGMENTED_ATTR)) {
    controlAll(allBtn);
    allBtn.setAttribute(AUGMENTED_ATTR, "1");
  }
}

// ─── Self-heal toast on SafetensorError ─────────────────────────────────

const CORRUPT_PATTERNS = [
  /safetensors_rust\.SafetensorError/i,
  /incomplete metadata/i,
  /file not fully covered/i,
  /HeaderTooSmall/i,
  /Error while deserializing header/i,
];

function findNodeFilename(nodeId) {
  const g = app.graph;
  if (!g) return null;
  const nodes = g._nodes || g.nodes || [];
  const node = nodes.find((n) => String(n.id) === String(nodeId));
  if (!node) return null;
  const candidates = [];
  for (const w of (node.widgets || [])) {
    if (typeof w.value === "string" && /\.(safetensors|gguf|ckpt|pt|pth|bin|sft)$/i.test(w.value)) {
      candidates.push(w.value);
    }
  }
  for (const [, v] of Object.entries(node.properties || {})) {
    if (typeof v === "string" && /\.(safetensors|gguf|ckpt|pt|pth|bin|sft)$/i.test(v)) {
      candidates.push(v);
    }
  }
  return candidates[0] || null;
}

function showSelfHealToast(filename, nodeType) {
  const existing = document.querySelector("[data-codec-grabber-toast]");
  if (existing) existing.remove();
  const banner = document.createElement("div");
  banner.setAttribute("data-codec-grabber-toast", "1");
  banner.style.cssText = `
    position:fixed;top:20px;right:20px;z-index:99999;
    background:#1a0;border:1px solid #4a8;border-radius:8px;padding:12px 16px;
    color:#fff;font-family:sans-serif;font-size:13px;max-width:380px;
    box-shadow:0 4px 16px rgba(0,0,0,0.5);
  `;
  banner.innerHTML = `
    <div style="font-weight:bold;margin-bottom:6px;">⚠ Model file looks corrupt</div>
    <div style="font-size:11px;margin-bottom:4px;"><code>${filename}</code></div>
    <div style="font-size:11px;color:#cfc;margin-bottom:10px;opacity:.85;">${nodeType || ""}</div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;">
      <button data-act="redl"    style="padding:4px 10px;border:1px solid #4a8;background:#062;color:#eee;border-radius:6px;cursor:pointer;font-size:12px;">Delete + Re-download</button>
      <button data-act="dismiss" style="padding:4px 10px;border:1px solid #555;background:#222;color:#ccc;border-radius:6px;cursor:pointer;font-size:12px;">Dismiss</button>
    </div>
    <div data-status style="margin-top:8px;font-size:11px;color:#aaa;"></div>
  `;
  document.body.appendChild(banner);
  const statusEl = banner.querySelector("[data-status]");
  banner.querySelector('[data-act="dismiss"]').addEventListener("click", () => banner.remove());
  banner.querySelector('[data-act="redl"]').addEventListener("click", async () => {
    statusEl.textContent = "looking up in history…"; statusEl.style.color = "deepskyblue";
    let lookup;
    try {
      const r = await api.fetchApi("/codec/grab-model/redownload", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename }),
      });
      const body = await r.json();
      if (!r.ok) { statusEl.textContent = `✗ ${body.error}`; statusEl.style.color = "tomato"; return; }
      lookup = body;
    } catch (e) { statusEl.textContent = `✗ ${e.message}`; statusEl.style.color = "tomato"; return; }
    statusEl.textContent = `started server-side download from ${new URL(lookup.url).hostname}; see panel for progress`;
    statusEl.style.color = "limegreen";
    await backendStart({ url: lookup.url, filename: lookup.filename, directory: lookup.directory });
  });
}

api.addEventListener("execution_error", (ev) => {
  const d = ev.detail || {};
  const tb = (d.traceback || []).join(" ") || "";
  const msg = `${d.exception_message || ""} ${d.exception_type || ""} ${tb}`;
  if (!CORRUPT_PATTERNS.some((re) => re.test(msg))) return;
  const filename = findNodeFilename(d.node_id);
  if (!filename) { log("corrupt-model error but no filename:", d); return; }
  log("corrupt-model:", filename, "node:", d.node_type);
  showSelfHealToast(filename, d.node_type);
});

// ─── Floating Downloads panel ──────────────────────────────────────────

function makeDownloadsPanel() {
  const fab = document.createElement("button");
  fab.textContent = "⤓";
  fab.title = "Codec downloads";
  fab.style.cssText = `
    position:fixed;bottom:16px;right:16px;z-index:9998;
    width:42px;height:42px;border-radius:50%;
    background:#062;color:#fff;border:1px solid #4a8;
    cursor:pointer;font-size:20px;line-height:42px;text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,0.4);
  `;
  document.body.appendChild(fab);
  const badge = document.createElement("span");
  badge.style.cssText = `
    position:fixed;bottom:46px;right:10px;z-index:9998;
    background:#c00;color:#fff;font-size:10px;border-radius:8px;
    padding:1px 5px;display:none;font-family:sans-serif;
  `;
  document.body.appendChild(badge);

  const panel = document.createElement("div");
  panel.style.cssText = `
    position:fixed;bottom:68px;right:16px;z-index:9999;
    width:480px;max-height:60vh;
    background:#181818;border:1px solid #4a8;border-radius:8px;
    display:none;flex-direction:column;
    box-shadow:0 4px 24px rgba(0,0,0,0.6);
    font-family:sans-serif;color:#ddd;font-size:12px;
  `;
  panel.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 12px;border-bottom:1px solid #333;gap:6px;">
      <strong>Codec Downloads</strong>
      <div style="display:flex;gap:4px;">
        <button data-act="verify-all" title="Re-check every downloaded model's safetensors header" style="background:transparent;border:1px solid #4a8;color:#cfc;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:11px;">Verify all</button>
        <button data-act="refresh" style="background:transparent;border:1px solid #555;color:#ccc;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:11px;">Refresh</button>
      </div>
    </div>
    <div data-verify-summary style="display:none;padding:6px 12px;background:#220;color:#fdb;font-size:11px;border-bottom:1px solid #441;"></div>
    <div data-section="active"  style="padding:8px 12px;border-bottom:1px solid #333;"></div>
    <div data-section="orphans" style="padding:8px 12px;border-bottom:1px solid #333;"></div>
    <div data-section="history" style="padding:8px 12px;overflow-y:auto;flex:1;"></div>
    <div style="padding:8px 12px;border-top:1px solid #333;">
      <details>
        <summary style="cursor:pointer;color:#aaa;font-size:11px;">Manual URL download / re-download by URL</summary>
        <div style="display:flex;flex-direction:column;gap:4px;margin-top:6px;">
          <input data-f="url"      placeholder="https://huggingface.co/…/model.safetensors" style="background:#0a0a0a;color:#ddd;border:1px solid #444;padding:4px 6px;font-size:11px;border-radius:4px;">
          <input data-f="filename" placeholder="model.safetensors" style="background:#0a0a0a;color:#ddd;border:1px solid #444;padding:4px 6px;font-size:11px;border-radius:4px;">
          <select data-f="directory" style="background:#0a0a0a;color:#ddd;border:1px solid #444;padding:4px 6px;font-size:11px;border-radius:4px;"></select>
          <div style="display:flex;gap:4px;">
            <button data-act="manual-add" style="flex:1;background:#062;border:1px solid #4a8;color:#eee;padding:4px 8px;border-radius:4px;cursor:pointer;font-size:11px;">Download</button>
            <button data-act="manual-redl" title="Delete any existing file at filename + download fresh" style="flex:1;background:#600;border:1px solid #a44;color:#eee;padding:4px 8px;border-radius:4px;cursor:pointer;font-size:11px;">Force re-download</button>
          </div>
          <div data-manual-status style="font-size:10px;color:#aaa;"></div>
        </div>
      </details>
    </div>
  `;
  document.body.appendChild(panel);

  fab.addEventListener("click", () => {
    panel.style.display = panel.style.display === "flex" ? "none" : "flex";
    if (panel.style.display === "flex") render();
  });
  panel.querySelector('[data-act="refresh"]').addEventListener("click", () => render());

  async function render() {
    const sections = {
      active: panel.querySelector('[data-section="active"]'),
      orphans: panel.querySelector('[data-section="orphans"]'),
      history: panel.querySelector('[data-section="history"]'),
    };
    // First paint only: show a placeholder. On live WS-driven refreshes keep
    // the existing content visible until the new data arrives — otherwise the
    // panel flashes loading↔content every 500ms during an active download.
    if (!sections.active.innerHTML.trim()) {
      sections.active.innerHTML = `<div style="color:#888;">loading…</div>`;
    }
    const [histRes, orphRes, dirsRes] = await Promise.all([
      api.fetchApi("/codec/grab-model/history").then((r) => r.json()).catch(() => ({entries: []})),
      api.fetchApi("/codec/grab-model/orphans").then((r) => r.json()).catch(() => ({orphans: []})),
      api.fetchApi("/codec/grab-model/allowed-directories").then((r) => r.json()).catch(() => ({directories: []})),
    ]);
    const entries = histRes.entries || [];
    const active = entries.filter((e) => e.status === "downloading" || e.status === "paused");
    const done = entries.filter((e) => e.status === "done");
    const errored = entries.filter((e) => e.status === "error" || e.status === "cancelled");

    badge.textContent = active.length;
    badge.style.display = active.length ? "block" : "none";

    const renderEntry = (e, opts = {}) => {
      const stateColor = { downloading: "deepskyblue", paused: "khaki", done: "limegreen", error: "tomato", cancelled: "#888" }[e.status] || "#aaa";
      const sz = e.expected_bytes
        ? `${fmtBytes(e.downloaded_bytes || 0)} / ${fmtBytes(e.expected_bytes)}`
        : fmtBytes(e.downloaded_bytes || 0);
      let actions = "";
      if (e.status === "downloading") {
        actions = `<button data-pause="${e.id}" style="background:#622;border:1px solid #844;color:#eee;border-radius:4px;padding:1px 6px;cursor:pointer;font-size:10px;">Pause</button>
                   <button data-cancel="${e.id}" style="background:#400;border:1px solid #844;color:#eee;border-radius:4px;padding:1px 6px;cursor:pointer;font-size:10px;">Cancel</button>`;
      } else if (e.status === "paused") {
        actions = `<button data-resume="${encodeURIComponent(JSON.stringify({url:e.url,filename:e.filename,directory:e.directory}))}" style="background:#062;border:1px solid #4a8;color:#eee;border-radius:4px;padding:1px 6px;cursor:pointer;font-size:10px;">Resume</button>
                   <button data-cancel="${e.id}" style="background:#400;border:1px solid #844;color:#eee;border-radius:4px;padding:1px 6px;cursor:pointer;font-size:10px;">Cancel</button>`;
      } else if (opts.actions) {
        // Re-download is always offered for done/errored entries — it uses
        // the URL recorded in history, deletes the on-disk file and any
        // tempfile, then kicks a fresh background download.
        actions = `<button data-redl="${encodeURIComponent(e.filename)}|${encodeURIComponent(e.directory)}" style="background:#062;border:1px solid #4a8;color:#eee;border-radius:4px;padding:1px 6px;cursor:pointer;font-size:10px;">Re-download</button>`;
      }
      // A done entry whose on-disk file fails verification gets a ⚠ flag
      // and a more prominent Re-download. Verify state is attached after
      // the initial render by a background /verify-all sweep.
      const corruptFlag = e._corrupt
        ? `<span data-corrupt="${e.id}" style="display:inline-block;margin-left:6px;padding:0 5px;background:#a00;color:#fff;border-radius:8px;font-size:10px;" title="${(e._corrupt.error || '').replace(/"/g, '&quot;')}">⚠ corrupt</span>`
        : "";
      return `
        <div data-entry="${e.id}" data-filename="${e.filename}" data-directory="${e.directory}" style="padding:4px 0;border-top:1px solid #2a2a2a;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:240px;" title="${e.filename}">${e.filename}${corruptFlag}</span>
            <span style="color:${stateColor};font-size:10px;">${e.status}</span>
          </div>
          <div style="color:#888;font-size:10px;">${e.directory} · ${sz}${e.error ? ` · ${e.error.slice(0, 80)}` : ""}</div>
          ${actions ? `<div style="margin-top:3px;display:flex;gap:4px;flex-wrap:wrap;">${actions}</div>` : ""}
        </div>`;
    };

    sections.active.innerHTML = `<div style="font-weight:bold;color:#4a8;margin-bottom:4px;">Active (${active.length})</div>${active.map((e) => renderEntry(e)).join("") || `<div style="color:#666;font-size:11px;">none</div>`}`;
    sections.orphans.innerHTML = `<div style="font-weight:bold;color:#fa8;margin-bottom:4px;">Orphan tempfiles (${(orphRes.orphans || []).length})</div>` +
      ((orphRes.orphans || []).map((o) => `<div style="font-size:11px;color:#888;">${o.filename} (${o.directory}, ${fmtBytes(o.bytes)})</div>`).join("")
       || `<div style="color:#666;font-size:11px;">none</div>`);
    sections.history.innerHTML = `<div style="font-weight:bold;color:#aaa;margin-bottom:4px;">Recent (${done.length + errored.length})</div>${[...done, ...errored].slice(0, 50).map((e) => renderEntry(e, { actions: true })).join("") || `<div style="color:#666;font-size:11px;">none</div>`}`;

    panel.querySelectorAll("[data-pause]").forEach((b) =>
      b.addEventListener("click", async () => { await api.fetchApi("/codec/grab-model/pause", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id: b.getAttribute("data-pause")})}); render(); }));
    panel.querySelectorAll("[data-cancel]").forEach((b) =>
      b.addEventListener("click", async () => { await api.fetchApi("/codec/grab-model/cancel", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id: b.getAttribute("data-cancel")})}); render(); }));
    panel.querySelectorAll("[data-resume]").forEach((b) =>
      b.addEventListener("click", async () => {
        const meta = JSON.parse(decodeURIComponent(b.getAttribute("data-resume")));
        await backendStart(meta);
        render();
      }));
    panel.querySelectorAll("[data-redl]").forEach((b) =>
      b.addEventListener("click", async () => {
        const [fn, dir] = b.getAttribute("data-redl").split("|").map(decodeURIComponent);
        b.disabled = true; b.textContent = "…";
        try {
          const r = await api.fetchApi("/codec/grab-model/redownload", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({filename: fn, directory: dir}),
          });
          const body = await r.json();
          if (!r.ok) { b.textContent = "✗"; return; }
          await backendStart(body);
          render();
        } catch { b.textContent = "✗"; }
      }));

    const dirSel = panel.querySelector('[data-f="directory"]');
    if (!dirSel.options.length) {
      for (const d of dirsRes.directories || []) {
        const o = document.createElement("option"); o.value = d; o.textContent = d; dirSel.appendChild(o);
      }
    }
  }

  panel.querySelector('[data-act="manual-add"]').addEventListener("click", async () => {
    const url = panel.querySelector('[data-f="url"]').value.trim();
    const filename = panel.querySelector('[data-f="filename"]').value.trim();
    const directory = panel.querySelector('[data-f="directory"]').value;
    const s = panel.querySelector('[data-manual-status]');
    if (!url || !filename || !directory) { s.textContent = "url + filename + directory required"; s.style.color = "tomato"; return; }
    const { status, body } = await backendStart({ url, filename, directory });
    if (status >= 400) { s.textContent = `✗ ${body.error || "failed"}`; s.style.color = "tomato"; return; }
    s.textContent = `✓ started — see Active section`;
    s.style.color = "limegreen";
    render();
  });

  // Force re-download — delete any existing file at the target path and
  // start fresh. Routes through v6's explicit-URL /redownload path so it
  // works even when no history entry exists (e.g. a partial earlier
  // attempt that errored before history could record it).
  panel.querySelector('[data-act="manual-redl"]').addEventListener("click", async () => {
    const url = panel.querySelector('[data-f="url"]').value.trim();
    const filename = panel.querySelector('[data-f="filename"]').value.trim();
    const directory = panel.querySelector('[data-f="directory"]').value;
    const s = panel.querySelector('[data-manual-status]');
    if (!url || !filename || !directory) { s.textContent = "url + filename + directory required"; s.style.color = "tomato"; return; }
    s.textContent = "deleting old file…"; s.style.color = "khaki";
    try {
      const r = await api.fetchApi("/codec/grab-model/redownload", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, filename, directory }),
      });
      const body = await r.json();
      if (!r.ok) { s.textContent = `✗ ${body.error || "redownload failed"}`; s.style.color = "tomato"; return; }
      const { status } = await backendStart({ url, filename, directory });
      s.textContent = status < 400
        ? `✓ deleted ${body.deleted.length} file(s); fresh download started`
        : `✓ deleted ${body.deleted.length} file(s); but start returned ${status}`;
      s.style.color = "limegreen";
      render();
    } catch (e) { s.textContent = `✗ ${e.message}`; s.style.color = "tomato"; }
  });

  // Verify all — runs server-side safetensors header-coverage check
  // across every "done" entry. Bad files get a ⚠ corrupt badge in the
  // history section and the Re-download button becomes the primary cure.
  async function runVerifyAll(silent = false) {
    const sum = panel.querySelector('[data-verify-summary]');
    if (!silent) {
      sum.style.display = "block"; sum.textContent = "verifying…";
      sum.style.background = "#220"; sum.style.color = "#fdb";
    }
    try {
      const r = await api.fetchApi("/codec/grab-model/verify-all");
      const body = await r.json();
      const badByFn = new Map();
      for (const e of (body.entries || [])) {
        if (!e.verify?.ok) badByFn.set(`${e.filename}|${e.directory}`, e.verify);
      }
      // Stamp ⚠ badges + flip Re-download styling for any matched row.
      for (const row of panel.querySelectorAll("[data-entry]")) {
        const fn = row.getAttribute("data-filename");
        const dir = row.getAttribute("data-directory");
        const v = badByFn.get(`${fn}|${dir}`);
        if (!v) continue;
        if (!row.querySelector("[data-corrupt]")) {
          const span = document.createElement("span");
          span.style.cssText = "display:inline-block;margin-left:6px;padding:0 5px;background:#a00;color:#fff;border-radius:8px;font-size:10px;";
          span.title = v.error || "corrupt";
          span.textContent = "⚠ corrupt";
          row.querySelector("span[title]")?.appendChild(span);
        }
        const redl = row.querySelector("[data-redl]");
        if (redl) {
          redl.style.background = "#600"; redl.style.borderColor = "#a44";
          redl.textContent = "Re-download (corrupt)";
        }
      }
      sum.style.display = body.bad > 0 ? "block" : (silent ? "none" : "block");
      if (body.bad > 0) {
        sum.style.background = "#400"; sum.style.color = "#fee";
        sum.textContent = `⚠ ${body.bad} of ${body.checked} on-disk file(s) failed safetensors verify — see ⚠ corrupt below.`;
      } else if (!silent) {
        sum.style.background = "#020"; sum.style.color = "#cfc";
        sum.textContent = `✓ all ${body.checked} on-disk file(s) verified OK`;
      }
    } catch (e) {
      sum.style.display = "block";
      sum.style.background = "#400"; sum.style.color = "#fcc";
      sum.textContent = `✗ verify failed: ${e.message}`;
    }
  }
  panel.querySelector('[data-act="verify-all"]').addEventListener("click", () => runVerifyAll(false));

  // Auto-verify on panel open, then at most every 30s while open. Avoids
  // hammering the disk on every WS event (which the debounced re-render
  // already covers for the entry list).
  let lastVerify = 0;
  fab.addEventListener("click", () => {
    if (panel.style.display === "flex" && Date.now() - lastVerify > 30000) {
      lastVerify = Date.now();
      runVerifyAll(true);
    }
  });

  // Live update via WS — re-render only when something interesting changed.
  let renderQueued = false;
  subscribe(() => true, () => {
    if (renderQueued) return;
    renderQueued = true;
    setTimeout(() => { renderQueued = false; if (panel.style.display === "flex") render(); }, 500);
    // Also keep the badge count fresh even when panel is closed.
    api.fetchApi("/codec/grab-model/active").then(r => r.json()).then(d => {
      const n = (d.active || []).filter(e => e.status === "downloading" || e.status === "paused").length;
      badge.textContent = n;
      badge.style.display = n ? "block" : "none";
    }).catch(() => {});
  });

  // Initial badge state.
  api.fetchApi("/codec/grab-model/active").then(r => r.json()).then(d => {
    const n = (d.active || []).length;
    badge.textContent = n;
    badge.style.display = n ? "block" : "none";
  }).catch(() => {});
}

// ─── Wiring ─────────────────────────────────────────────────────────────

app.registerExtension({
  name: "codec.grabber",
  async setup() {
    log("v8: corruption surfacing + force-redownload + per-call URL capture");
    const observer = new MutationObserver(() => augmentMissingModelsPanel());
    observer.observe(document.body, { childList: true, subtree: true });
    augmentMissingModelsPanel();
    makeDownloadsPanel();
  },
});
