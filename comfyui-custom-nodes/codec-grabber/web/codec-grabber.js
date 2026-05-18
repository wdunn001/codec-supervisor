/**
 * ComfyUI-CodecGrabber — frontend hook for the Missing Models panel.
 *
 * Adds a server-side "Install ⤓" button next to each per-row Download
 * button in the Workflow Overview → Errors → Missing Models panel,
 * and a server-side "Install all" button next to "Download all".
 *
 * The existing browser-download buttons stay working (some users want the
 * file local). Server-side install POSTs to /codec/grab-model with the
 * row's url + filename + directory and the container does the wget into
 * the mounted models volume.
 *
 * The frontend panel is part of the compiled ComfyUI-Frontend bundle, so
 * we can't import its components directly. We attach a MutationObserver
 * to the document body and rewrite each Missing Models row whenever it
 * appears or re-renders. Lightweight: O(rows) per mutation.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const log = (...args) => console.log("[codec-grabber]", ...args);

// Marker we stamp on rows we've already augmented so we don't double-add
// buttons on re-renders.
const AUGMENTED_ATTR = "data-codec-grabber-augmented";

// Heuristic selectors for the Missing Models panel. The Frontend
// classnames are stable across recent v1.x releases but might evolve;
// keep these grouped at the top so future-us can fix in one place.
const SELECTORS = {
  // Each per-row "Download" button — text content is the discriminator.
  rowDownloadBtns: () =>
    Array.from(document.querySelectorAll("button"))
      .filter(b => /^Download(\s*\([\d.]+\s*[GMK]B\))?$/i.test((b.textContent || "").trim())),
  // The "Download all (XX GB)" button at the top of the Missing Models section.
  downloadAllBtn: () =>
    Array.from(document.querySelectorAll("button"))
      .find(b => /^Download all\b/i.test((b.textContent || "").trim())),
};

/** Walk up from a button to find the row container with url/filename/directory. */
function extractRowMeta(btn) {
  // The row structure (as of ComfyUI-Frontend ~1.4 inspected via the
  // user's screenshot) puts the filename and category as sibling text
  // inside a parent div. We look for any nearby element that has a
  // data-url or text we can parse. Fall back to scanning DOM ancestors.
  let cur = btn;
  for (let i = 0; i < 8 && cur; i++) {
    // Look for an explicit data-* hint first (some Frontend rows expose it).
    const url = cur.dataset?.modelUrl || cur.dataset?.url;
    if (url) {
      return {
        url,
        filename: cur.dataset?.modelFilename || cur.dataset?.filename || null,
        directory: cur.dataset?.modelDirectory || cur.dataset?.directory || null,
      };
    }
    cur = cur.parentElement;
  }
  // No data attrs — fall back to clicking the Copy URL sibling to read the URL,
  // and parse the filename from a nearby text node. This is fragile; we log
  // and bail if it doesn't yield a workable triple.
  const row = btn.closest("[role='listitem'], li, .model-row, .missing-model-row, div");
  if (!row) return null;
  const copyBtn = Array.from(row.querySelectorAll("button"))
    .find(b => /^Copy URL$/i.test((b.textContent || "").trim()));
  // We can't directly read the clipboard text without user gesture, so
  // we rely on the row's text content for the filename + a nearby anchor
  // for the URL.
  const anchor = row.querySelector("a[href^='http']");
  if (!anchor) return null;
  // Filename heuristic: a code/span/text element that ends in .safetensors/.gguf/.ckpt
  const filenameNode = Array.from(row.querySelectorAll("*"))
    .find(el => {
      const t = (el.textContent || "").trim();
      return /\.(safetensors|gguf|ckpt|pt|pth|bin|onnx|sft)$/i.test(t) && t.length < 200;
    });
  const filename = filenameNode ? filenameNode.textContent.trim() : null;
  // Directory heuristic: a heading text above this row matches a known
  // category (e.g. "checkpoints (1)"). Walk up parents looking for a
  // header with one of the known categories.
  let directory = null;
  let p = row.parentElement;
  while (p && !directory) {
    const headerMatch = (p.textContent || "").match(/\b(checkpoints|loras|vae|unet|diffusion_models|text_encoders|clip|clip_vision|controlnet|upscale_models|latent_upscale_models|embeddings|style_models|hypernetworks|gligen|diffusers|photomaker|model_patches|audio_encoders)\b\s*\(\d+\)/i);
    if (headerMatch) directory = headerMatch[1].toLowerCase();
    p = p.parentElement;
  }
  return { url: anchor.href, filename, directory };
}

async function grabServerSide(meta, statusEl) {
  if (!meta?.url || !meta?.filename || !meta?.directory) {
    statusEl.textContent = "missing url/filename/directory";
    statusEl.style.color = "tomato";
    return;
  }
  statusEl.textContent = `installing → /models/${meta.directory}/${meta.filename} ...`;
  statusEl.style.color = "deepskyblue";
  try {
    const res = await api.fetchApi("/codec/grab-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(meta),
    });
    const body = await res.json().catch(() => ({}));
    if (res.ok) {
      const mb = body.bytes ? (body.bytes / 1048576).toFixed(1) : "?";
      statusEl.textContent = `✓ installed (${mb} MB)`;
      statusEl.style.color = "limegreen";
    } else {
      statusEl.textContent = `✗ ${body.error || res.statusText}`;
      statusEl.style.color = "tomato";
    }
  } catch (e) {
    statusEl.textContent = `✗ ${e.message}`;
    statusEl.style.color = "tomato";
  }
}

function buildInstallBtn(meta) {
  const wrap = document.createElement("span");
  wrap.style.cssText = "display:inline-flex;align-items:center;gap:6px;margin-left:6px;";
  const btn = document.createElement("button");
  btn.textContent = "Install ⤓";
  btn.title = `Server-side download: ${meta?.url || "?"} → /models/${meta?.directory || "?"}/${meta?.filename || "?"}`;
  btn.style.cssText = "padding:2px 8px;border:1px solid #4a8;background:#062;color:#eee;border-radius:4px;cursor:pointer;font-size:11px;";
  const status = document.createElement("small");
  status.style.cssText = "font-size:10px;color:#aaa;max-width:280px;overflow:hidden;text-overflow:ellipsis;";
  btn.addEventListener("click", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    grabServerSide(meta, status);
  });
  wrap.appendChild(btn);
  wrap.appendChild(status);
  return wrap;
}

function augmentRows() {
  for (const dlBtn of SELECTORS.rowDownloadBtns()) {
    if (dlBtn.hasAttribute(AUGMENTED_ATTR)) continue;
    const meta = extractRowMeta(dlBtn);
    if (!meta) continue;
    const installBtn = buildInstallBtn(meta);
    dlBtn.parentElement?.insertBefore(installBtn, dlBtn.nextSibling);
    dlBtn.setAttribute(AUGMENTED_ATTR, "1");
  }
  // "Download all" → "Install all (server-side)"
  const all = SELECTORS.downloadAllBtn();
  if (all && !all.hasAttribute(AUGMENTED_ATTR)) {
    const wrap = document.createElement("span");
    wrap.style.cssText = "display:inline-flex;align-items:center;gap:6px;margin-left:8px;";
    const btn = document.createElement("button");
    btn.textContent = "Install all (server-side)";
    btn.style.cssText = "padding:2px 8px;border:1px solid #4a8;background:#062;color:#eee;border-radius:4px;cursor:pointer;font-size:11px;";
    const status = document.createElement("small");
    status.style.cssText = "font-size:10px;color:#aaa;";
    btn.addEventListener("click", async (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      // Aggregate every per-row meta by re-scanning, then sequentially install.
      const targets = SELECTORS.rowDownloadBtns()
        .map(b => extractRowMeta(b))
        .filter(m => m && m.url && m.filename && m.directory);
      if (!targets.length) {
        status.textContent = "no targets resolved";
        status.style.color = "tomato";
        return;
      }
      let ok = 0, fail = 0;
      for (let i = 0; i < targets.length; i++) {
        status.textContent = `installing ${i + 1}/${targets.length}: ${targets[i].filename} ...`;
        status.style.color = "deepskyblue";
        try {
          const r = await api.fetchApi("/codec/grab-model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(targets[i]),
          });
          if (r.ok) { ok++; } else { fail++; }
        } catch { fail++; }
      }
      status.textContent = `done: ${ok} ok, ${fail} failed`;
      status.style.color = fail ? "tomato" : "limegreen";
    });
    wrap.appendChild(btn);
    wrap.appendChild(status);
    all.parentElement?.insertBefore(wrap, all.nextSibling);
    all.setAttribute(AUGMENTED_ATTR, "1");
  }
}

app.registerExtension({
  name: "codec.grabber",
  async setup() {
    log("setup: server-side install hooks active");
    // Observe the whole document — the Workflow Overview panel mounts /
    // unmounts dynamically when the user opens the right-side sidebar.
    const observer = new MutationObserver(() => augmentRows());
    observer.observe(document.body, { childList: true, subtree: true });
    // Initial pass in case the panel is already mounted on extension load.
    augmentRows();
  },
});
