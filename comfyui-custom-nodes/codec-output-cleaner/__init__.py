"""
ComfyUI-CodecOutputCleaner — one-click "empty the output folder" button.

Space-tight machines accumulate gigabytes of generated images/videos in
ComfyUI's output directory. This node adds a small UI button that reports
how much is sitting there and lets the operator permanently delete all of
it in one (deliberate, two-step) click.

Endpoints (registered on the ComfyUI server, reachable via the supervisor):
  GET  /codec/output/stats  -> {dir, files, dirs, bytes, human}
  POST /codec/output/empty  -> {deleted_files, deleted_dirs, freed_bytes, human, errors}

Safety:
  - Only ever operates inside folder_paths.get_output_directory().
  - Clears the *contents* of that directory; never removes the dir itself.
  - Iterates direct children only and uses shutil.rmtree, which removes
    symlinks as links (does NOT follow them), so a stray symlink in output
    can't be used to delete data elsewhere on disk.
  - Deletion is permanent (no recycle bin) — that's the intent.

Frontend: web/codec-output-cleaner.js injects the 🗑 button + confirm panel.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import folder_paths
from aiohttp import web
from server import PromptServer

log = logging.getLogger("codec-output-cleaner")

WEB_DIRECTORY = "./web"
# We register routes only — no graph nodes.
NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}


def _output_root() -> Path:
    return Path(folder_paths.get_output_directory()).resolve()


def _human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def _scan(root: Path) -> tuple[int, int, int]:
    files = dirs = total = 0
    for dp, dnames, fnames in os.walk(root):
        dirs += len(dnames)
        for fn in fnames:
            p = Path(dp) / fn
            try:
                if not p.is_symlink():
                    total += p.stat().st_size
                files += 1
            except OSError:
                pass
    return files, dirs, total


@PromptServer.instance.routes.get("/codec/output/stats")
async def output_stats(request):
    root = _output_root()
    if not root.is_dir():
        return web.json_response({"dir": str(root), "files": 0, "dirs": 0, "bytes": 0, "human": "0.0 B"})
    files, dirs, total = _scan(root)
    return web.json_response({
        "dir": str(root), "files": files, "dirs": dirs,
        "bytes": total, "human": _human(total),
    })


@PromptServer.instance.routes.post("/codec/output/empty")
async def output_empty(request):
    root = _output_root()
    if not root.is_dir():
        return web.json_response({"error": f"output dir not found: {root}"}, status=400)

    deleted_files = deleted_dirs = freed = 0
    errors: list[str] = []

    for entry in list(root.iterdir()):
        try:
            if entry.is_symlink() or entry.is_file():
                try:
                    freed += entry.stat().st_size if (entry.is_file() and not entry.is_symlink()) else 0
                except OSError:
                    pass
                entry.unlink()
                deleted_files += 1
            elif entry.is_dir():
                f, d, b = _scan(entry)
                shutil.rmtree(entry)  # removes symlinks as links, does not follow
                deleted_files += f
                deleted_dirs += d + 1
                freed += b
        except Exception as e:  # noqa: BLE001 — report, keep going
            errors.append(f"{entry.name}: {e}")

    log.info("[codec-output-cleaner] emptied %s: %d files, %s freed (%d errors)",
             root, deleted_files, _human(freed), len(errors))
    return web.json_response({
        "deleted_files": deleted_files, "deleted_dirs": deleted_dirs,
        "freed_bytes": freed, "human": _human(freed), "errors": errors,
    })
