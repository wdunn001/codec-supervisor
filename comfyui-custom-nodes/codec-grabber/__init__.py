"""
ComfyUI-CodecGrabber — server-side model installer for the Missing Models panel.

Modern ComfyUI Frontend (v1.x) shows a "Missing Models" overlay on the
Workflow Overview → Errors tab with per-model `Download` and `Download all`
buttons. Those buttons trigger BROWSER downloads, which is wrong for any
headless / containerised / remote ComfyUI deployment because the file
lands in the user's local Downloads folder, not inside the container's
mounted models volume.

This custom node bolts a server-side download path onto the same UI:

  - Backend: `POST /codec/grab-model` accepts `{url, filename, directory}`
    (the three fields the existing UI already has per row), downloads the
    URL inside the container using urllib (no extra deps), atomically
    moves the file into `folder_paths.folder_names_and_paths[directory][0][0]`
    — i.e. the first registered directory for the model category — which
    on the codec-comfyui image is `/opt/codec/comfyui/models/<directory>/`
    and is bind-mounted to a Docker volume so the file persists.

  - Frontend: a small extension script (web/codec-grabber.js) injects an
    "Install ⤓" button next to each existing Download button + replaces
    the "Download all" with a server-side variant. Plain DOM hooks; no
    dependency on Manager.

  - Security: requests are only honoured for `directory` values that
    appear in `folder_paths.folder_names_and_paths` (no arbitrary path
    traversal); requests for files that already exist return 409 instead
    of overwriting. URL scheme is restricted to https / http.

Bakes cleanly into Dockerfile.comfyui via `COPY` into custom_nodes/.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import folder_paths
from aiohttp import web
from server import PromptServer

log = logging.getLogger("codec-grabber")

WEB_DIRECTORY = "./web"

# Empty NODE_CLASS_MAPPINGS so ComfyUI sees us as a "custom node pack"
# without us actually registering any graph nodes — we're a server +
# frontend extension, not a workflow node.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _is_safe_filename(name: str) -> bool:
    # No path separators, no traversal, no NUL, must be a recognisable filename.
    if not name or "/" in name or "\\" in name or "\x00" in name or ".." in name:
        return False
    return bool(re.match(r"^[\w\-. ]+\.(safetensors|gguf|ckpt|pt|pth|bin|onnx|gguf2|sft)$", name, re.I))


def _resolve_target_dir(directory: str) -> Path | None:
    # Only allow categories ComfyUI itself knows about. This is the
    # security boundary — no arbitrary path access.
    if directory not in folder_paths.folder_names_and_paths:
        return None
    paths = folder_paths.folder_names_and_paths[directory][0]
    if not paths:
        return None
    return Path(paths[0])


def _download_to(url: str, dest: Path) -> tuple[int, str]:
    """Stream `url` to `dest` atomically. Returns (bytes_written, sha256)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("https", "http"):
        raise ValueError(f"unsupported scheme: {parsed.scheme!r}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    bytes_written = 0
    # Write to a sibling tempfile then atomic rename so a partial download
    # never gets picked up by a workflow.
    with tempfile.NamedTemporaryFile(
        dir=dest.parent, prefix=".grabbing-", suffix=dest.suffix, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ComfyUI-CodecGrabber/1.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                # 64 KiB chunks; fine for the few-GB model files we expect.
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    sha.update(chunk)
                    bytes_written += len(chunk)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    # Atomic move; same-filesystem rename = a single inode update.
    shutil.move(str(tmp_path), str(dest))
    return bytes_written, sha.hexdigest()


@PromptServer.instance.routes.post("/codec/grab-model")
async def grab_model(request):
    """
    Server-side download for the Missing Models panel.

    Body: { "url": str, "filename": str, "directory": str, "sha256"?: str }
    Returns:
      200 {"ok": true, "path": str, "bytes": int, "sha256": str}
      400 on bad input
      403 on disallowed directory
      409 if file already exists
      502 on download failure
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    url = body.get("url")
    filename = body.get("filename")
    directory = body.get("directory")
    expected_sha = body.get("sha256")  # optional

    if not (isinstance(url, str) and isinstance(filename, str) and isinstance(directory, str)):
        return web.json_response({"error": "url, filename, directory required as strings"}, status=400)
    if not _is_safe_filename(filename):
        return web.json_response({"error": f"unsafe filename: {filename!r}"}, status=400)

    target_dir = _resolve_target_dir(directory)
    if target_dir is None:
        return web.json_response(
            {"error": f"unknown model directory: {directory!r}",
             "allowed": sorted(folder_paths.folder_names_and_paths.keys())},
            status=403,
        )

    dest = target_dir / filename
    if dest.exists():
        return web.json_response(
            {"error": "file already exists", "path": str(dest)},
            status=409,
        )

    log.info("[codec-grabber] grabbing %s -> %s", url, dest)
    try:
        bytes_written, sha256 = await request.loop.run_in_executor(
            None, _download_to, url, dest
        )
    except Exception as e:
        log.exception("[codec-grabber] download failed")
        return web.json_response({"error": f"download failed: {e}"}, status=502)

    if expected_sha and expected_sha.lower() != sha256.lower():
        # Reject silently-corrupt mirrors; leave the downloaded bytes for
        # forensics but mark the request failed.
        return web.json_response(
            {"error": "sha256 mismatch", "expected": expected_sha, "actual": sha256, "path": str(dest)},
            status=502,
        )

    log.info("[codec-grabber] done %s (%d bytes, sha256=%s)", dest.name, bytes_written, sha256[:12])
    return web.json_response(
        {"ok": True, "path": str(dest), "bytes": bytes_written, "sha256": sha256},
        status=200,
    )


@PromptServer.instance.routes.get("/codec/grab-model/allowed-directories")
async def list_directories(request):
    """List the model categories /codec/grab-model accepts. Used by the
    frontend extension to map a missing-model entry's `directory` field
    to a known good target."""
    return web.json_response(
        {"directories": sorted(folder_paths.folder_names_and_paths.keys())},
        status=200,
    )


log.info("[codec-grabber] registered POST /codec/grab-model")
