/**
 * ComfyUI-CodecOutputCleaner — frontend.
 *
 * Adds a 🗑 floating button (bottom-left) that opens a small panel showing
 * how much is in the ComfyUI output folder, with a deliberate two-step
 * "delete all permanently" action. Built to be impossible to trigger by a
 * single accidental click: you open the panel, see the exact file count +
 * size, then click the labelled red button.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const log = (...a) => console.log("[codec-output-cleaner]", ...a);

function makeCleaner() {
  const fab = document.createElement("button");
  fab.textContent = "🗑";
  fab.title = "Empty ComfyUI output folder";
  fab.style.cssText = `
    position:fixed;bottom:16px;left:16px;z-index:9998;
    width:42px;height:42px;border-radius:50%;
    background:#400;color:#fff;border:1px solid #a44;
    cursor:pointer;font-size:18px;line-height:42px;text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,0.4);`;
  document.body.appendChild(fab);

  const panel = document.createElement("div");
  panel.style.cssText = `
    position:fixed;bottom:68px;left:16px;z-index:9999;width:320px;
    background:#181818;border:1px solid #a44;border-radius:8px;
    display:none;flex-direction:column;padding:12px;
    box-shadow:0 4px 24px rgba(0,0,0,0.6);
    font-family:sans-serif;color:#ddd;font-size:12px;`;
  panel.innerHTML = `
    <div style="font-weight:bold;color:#f88;margin-bottom:8px;">Empty output folder</div>
    <div data-stats style="color:#aaa;margin-bottom:10px;word-break:break-all;">scanning…</div>
    <div style="display:flex;gap:6px;">
      <button data-act="empty" disabled
        style="flex:1;background:#600;border:1px solid #a44;color:#eee;padding:6px;border-radius:6px;cursor:pointer;">…</button>
      <button data-act="cancel"
        style="background:#222;border:1px solid #555;color:#ccc;padding:6px 10px;border-radius:6px;cursor:pointer;">Cancel</button>
    </div>
    <div data-result style="margin-top:8px;"></div>
    <div style="margin-top:6px;color:#777;font-size:10px;">Permanent — no recycle bin.</div>`;
  document.body.appendChild(panel);

  const statsEl = panel.querySelector("[data-stats]");
  const emptyBtn = panel.querySelector('[data-act="empty"]');
  const resultEl = panel.querySelector("[data-result]");

  async function refresh() {
    statsEl.textContent = "scanning…";
    emptyBtn.disabled = true;
    emptyBtn.textContent = "…";
    resultEl.textContent = "";
    try {
      const r = await api.fetchApi("/codec/output/stats");
      const d = await r.json();
      statsEl.innerHTML = `<code style="color:#bbb;">${d.dir}</code><br><b>${d.files}</b> files · <b>${d.human}</b>`;
      if (d.files === 0) {
        emptyBtn.disabled = true;
        emptyBtn.textContent = "Already empty";
      } else {
        emptyBtn.disabled = false;
        emptyBtn.textContent = `Delete all ${d.files} files (${d.human})`;
      }
    } catch (e) {
      statsEl.textContent = "✗ " + e.message;
    }
  }

  fab.addEventListener("click", () => {
    const show = panel.style.display !== "flex";
    panel.style.display = show ? "flex" : "none";
    if (show) refresh();
  });
  panel.querySelector('[data-act="cancel"]').addEventListener("click", () => {
    panel.style.display = "none";
  });

  emptyBtn.addEventListener("click", async () => {
    emptyBtn.disabled = true;
    emptyBtn.textContent = "deleting…";
    try {
      const r = await api.fetchApi("/codec/output/empty", { method: "POST" });
      const d = await r.json();
      if (!r.ok) {
        resultEl.style.color = "#f88";
        resultEl.textContent = "✗ " + (d.error || "failed");
        return;
      }
      resultEl.style.color = "#8c8";
      resultEl.textContent = `✓ Freed ${d.human} — deleted ${d.deleted_files} files`;
      if (d.errors && d.errors.length) {
        resultEl.textContent += ` · ${d.errors.length} error(s)`;
      }
      await refresh();
    } catch (e) {
      resultEl.style.color = "#f88";
      resultEl.textContent = "✗ " + e.message;
    }
  });
}

app.registerExtension({
  name: "codec.output_cleaner",
  async setup() {
    log("v1: empty-output button");
    makeCleaner();
  },
});
