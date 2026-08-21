"""
ComfyUI-CodecGrabber — server-side model installer for the Missing Models panel.

v6 — adds safetensors header-coverage verification on download finalize
plus a /verify endpoint the frontend can call against any on-disk entry.
Catches the "Content-Length-matches-but-file-is-actually-truncated" class
of failures (upstream silently sent a wrong-size body, or a partial+resume
mixed bytes from two different upstreams). Also detects URL mismatch on
resume — if the current request URL differs from the URL in history for
that tempfile, the tempfile is discarded and the download starts fresh.

v5 — downloads run as server-side background asyncio tasks. The frontend
just observes via ComfyUI's existing WebSocket (`codec_grabber_progress`
events). Closing the browser, reloading the page, navigating to a
different workflow — none of these stop or even pause a download. The
download keeps running until it completes, errors, the user explicitly
pauses/cancels, or the container restarts.

Endpoints:
  POST /codec/grab-model/start    body={url,filename,directory}
                                  → starts a background task (or returns
                                    the running one if same eid).
                                    Returns {id, status}. Idempotent.
  POST /codec/grab-model/pause    body={url|id, filename?, directory?}
                                  → sets pause flag; task exits its
                                    download loop, keeps the tempfile.
  POST /codec/grab-model/cancel   body=same → cancels + deletes tempfile.
  GET  /codec/grab-model/history  → unchanged. Source of truth for state.
  GET  /codec/grab-model/active   → currently-running tasks (for the UI).
  GET  /codec/grab-model/orphans  → unchanged.
  POST /codec/grab-model/redownload {filename, directory?} → deletes
                                    final + tmp, returns {url, filename,
                                    directory} to feed into /start.
  POST /codec/grab-model/stream   legacy: starts a task, also streams
                                    progress as NDJSON to this client.
                                    Closing the connection does NOT
                                    stop the task (v4 bug fixed).
  POST /codec/grab-model           synchronous (curl-friendly). Still
                                    blocks until done, but uses the
                                    same background machinery so it's
                                    cancellable from another client.

On container startup, any history entry that was "downloading" when the
container stopped gets demoted to "paused" (it can't actually be running
since the task is gone) so the UI shows the right state.

History at <user_dir>/codec-grabber/history.json — persists via the
mounted user volume.

WebSocket event shape:
  type: "codec_grabber_progress"
  data: {
    id: "<entry_id>",
    filename, directory, url,
    status: "downloading" | "paused" | "done" | "error" | "cancelled",
    downloaded_bytes, expected_bytes,
    error?: "<str>",
  }
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import re
import shutil
import struct
import urllib.parse
from pathlib import Path
from typing import Any

import aiohttp
import folder_paths
from aiohttp import web
from server import PromptServer

log = logging.getLogger("codec-grabber")

WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ─── History storage (same as v4) ───────────────────────────────────────

def _user_dir() -> Path:
    try:
        from folder_paths import get_user_directory  # type: ignore
        return Path(get_user_directory())
    except Exception:
        return Path("/opt/codec/comfyui/user")


_HISTORY_DIR = _user_dir() / "codec-grabber"
_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
_HISTORY_PATH = _HISTORY_DIR / "history.json"
_HISTORY_LOCK = asyncio.Lock()


def _entry_id(url: str, filename: str, directory: str) -> str:
    return hashlib.sha256(f"{url}\x00{filename}\x00{directory}".encode()).hexdigest()[:16]


def _load_history_sync() -> dict[str, Any]:
    if not _HISTORY_PATH.exists():
        return {"version": 1, "entries": {}}
    try:
        d = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
        d.setdefault("entries", {})
        return d
    except Exception as e:
        log.warning(f"history file unreadable ({e}); starting fresh")
        return {"version": 1, "entries": {}}


def _save_history_sync(data: dict[str, Any]) -> None:
    tmp = _HISTORY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(_HISTORY_PATH)


async def _history_update(eid: str, patch: dict[str, Any]) -> dict[str, Any]:
    async with _HISTORY_LOCK:
        d = _load_history_sync()
        entry = d["entries"].get(eid, {})
        entry.update(patch)
        entry["id"] = eid
        d["entries"][eid] = entry
        _save_history_sync(d)
        return entry


async def _history_get_all() -> list[dict[str, Any]]:
    async with _HISTORY_LOCK:
        d = _load_history_sync()
        items = list(d["entries"].values())
        items.sort(key=lambda e: e.get("started_at") or "", reverse=True)
        return items


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ─── Validation ─────────────────────────────────────────────────────────

def _is_safe_filename(name: str) -> bool:
    if not name or "/" in name or "\\" in name or "\x00" in name or ".." in name:
        return False
    return bool(re.match(r"^[\w\-. ]+\.(safetensors|gguf|ckpt|pt|pth|bin|onnx|gguf2|sft)$", name, re.I))


def _resolve_target_dir(directory: str) -> Path | None:
    if directory not in folder_paths.folder_names_and_paths:
        return None
    paths = folder_paths.folder_names_and_paths[directory][0]
    return Path(paths[0]) if paths else None


def _validate(body: dict):
    url = body.get("url")
    filename = body.get("filename")
    directory = body.get("directory")
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
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("https", "http"):
        return web.json_response({"error": f"unsupported scheme: {parsed.scheme!r}"}, status=400)
    return url, filename, target_dir / filename


# ─── Background-task registry ───────────────────────────────────────────

class DownloadTask:
    """One background download. Survives client disconnects."""
    def __init__(self, eid: str, url: str, filename: str, directory: str, dest: Path):
        self.eid = eid
        self.url = url
        self.filename = filename
        self.directory = directory
        self.dest = dest
        self.tmp_path = dest.parent / f".grabbing-{dest.name}"
        self.pause_event = asyncio.Event()
        self.cancel_event = asyncio.Event()
        self.bytes_written = 0
        self.total: int | None = None
        self.status = "starting"  # downloading | paused | done | error | cancelled
        self.error: str | None = None
        self.task: asyncio.Task | None = None
        # Per-subscriber event queues for /stream observers. Each queue
        # gets every progress event; subscribers drop in on connect,
        # drop out on disconnect.
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try: self._subscribers.remove(q)
        except ValueError: pass

    async def _emit(self, kind: str, **extra) -> None:
        msg = {
            "id": self.eid, "filename": self.filename, "directory": self.directory,
            "url": self.url, "status": self.status,
            "downloaded_bytes": self.bytes_written, "expected_bytes": self.total,
            "error": self.error,
        }
        msg.update(extra)
        # Broadcast to ComfyUI WebSocket (all browser tabs).
        try:
            PromptServer.instance.send_sync("codec_grabber_progress", msg)
        except Exception:
            pass
        # Fan out to any /stream subscribers.
        for q in list(self._subscribers):
            try: q.put_nowait({"phase": kind, **msg})
            except Exception: pass


_TASKS: dict[str, DownloadTask] = {}
_TASKS_LOCK = asyncio.Lock()
_DISCONNECT_ERRORS = (
    asyncio.CancelledError, ConnectionResetError, ConnectionAbortedError,
    BrokenPipeError, aiohttp.ClientConnectionResetError,
)


def _verify_safetensors(path: Path) -> dict[str, Any]:
    """Parse the safetensors header and verify it covers all declared tensor
    bytes. Returns {ok, kind, file_bytes, needed_bytes, short_by?, error?}.

    A valid safetensors file is: <header_len: u64 LE> <header JSON> <tensor data>.
    Every tensor's data_offsets[1] must be <= len(tensor_data). If file is
    shorter than `8 + header_len + max(data_offsets[1])`, the load will fail
    with "incomplete metadata, file not fully covered" — exactly the class
    of corruption we want to catch BEFORE promoting a tempfile.
    """
    try:
        fs = path.stat().st_size
    except OSError as e:
        return {"ok": False, "kind": "missing", "error": str(e)}

    if path.suffix.lower() != ".safetensors":
        return {"ok": True, "kind": "unknown", "file_bytes": fs,
                "note": "non-safetensors; size-only check"}

    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return {"ok": False, "kind": "safetensors", "file_bytes": fs,
                        "error": "file too small to hold header length"}
            hl = struct.unpack("<Q", raw)[0]
            if hl <= 0 or hl > 100 * 1024 * 1024:
                return {"ok": False, "kind": "safetensors", "file_bytes": fs,
                        "error": f"implausible header length: {hl}"}
            hdr_raw = f.read(hl)
            if len(hdr_raw) < hl:
                return {"ok": False, "kind": "safetensors", "file_bytes": fs,
                        "error": f"header truncated ({len(hdr_raw)}/{hl})"}
            hdr = json.loads(hdr_raw)
    except (OSError, json.JSONDecodeError, struct.error) as e:
        return {"ok": False, "kind": "safetensors", "file_bytes": fs,
                "error": f"header parse: {e}"}

    max_end = 0
    for k, v in hdr.items():
        if not isinstance(v, dict): continue
        off = v.get("data_offsets")
        if isinstance(off, list) and len(off) == 2 and isinstance(off[1], int):
            if off[1] > max_end:
                max_end = off[1]
    needed = 8 + hl + max_end
    if fs < needed:
        return {"ok": False, "kind": "safetensors", "file_bytes": fs,
                "needed_bytes": needed, "short_by": needed - fs,
                "error": f"truncated: header expects {needed:,} bytes; file is {fs:,} (short by {needed - fs:,})"}
    return {"ok": True, "kind": "safetensors", "file_bytes": fs,
            "needed_bytes": needed, "tensor_count": sum(1 for v in hdr.values() if isinstance(v, dict) and "data_offsets" in v)}


def _hash_existing_partial(tmp_path: Path) -> tuple[hashlib._Hash, int]:
    sha = hashlib.sha256()
    n = 0
    with open(tmp_path, "rb") as f:
        while True:
            chunk = f.read(1048576)
            if not chunk: break
            sha.update(chunk); n += len(chunk)
    return sha, n


async def _run_download(t: DownloadTask) -> None:
    """The background download body. Never tied to any HTTP request."""
    sha = hashlib.sha256()
    start_byte = 0
    if t.tmp_path.exists():
        # URL-consistency guard: a tempfile is only a valid resume target if
        # the prior download was from the SAME url. If the recorded history
        # url differs from the current request url, mixing the bytes would
        # produce a Frankenstein file (e.g. dev-fp8 header + lora payload —
        # the exact failure mode that triggered v6).
        prior_url = None
        try:
            hist = _load_history_sync().get("entries", {}).get(t.eid, {})
            prior_url = hist.get("url")
        except Exception:
            pass
        if prior_url and prior_url != t.url:
            log.warning(
                f"[codec-grabber] tempfile {t.tmp_path.name} was for a "
                f"different URL; discarding to avoid mixed-source corruption "
                f"(prior={prior_url!r} now={t.url!r})"
            )
            t.tmp_path.unlink(missing_ok=True)
        else:
            try:
                sha, start_byte = _hash_existing_partial(t.tmp_path)
                log.info(f"[codec-grabber] resuming {t.tmp_path.name} from byte {start_byte}")
            except OSError as e:
                log.warning(f"[codec-grabber] can't read existing partial: {e}; starting fresh")
                t.tmp_path.unlink(missing_ok=True)
                start_byte = 0
                sha = hashlib.sha256()
    t.bytes_written = start_byte
    last_emit = t.bytes_written
    t.status = "downloading"
    t.error = None
    await _history_update(t.eid, {
        "url": t.url, "filename": t.filename, "directory": t.directory,
        "started_at": _now_iso(), "status": "downloading",
        "downloaded_bytes": t.bytes_written,
    })
    await t._emit("start", resumed_from=start_byte)

    headers = {"User-Agent": "ComfyUI-CodecGrabber/5.0"}
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"

    try:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(t.url, headers=headers) as r:
                if r.status >= 400:
                    t.status = "error"; t.error = f"upstream {r.status}: {r.reason}"
                    await _history_update(t.eid, {"status": "error", "error": t.error})
                    await t._emit("error")
                    return

                restarted = False
                if start_byte > 0 and r.status == 200:
                    log.warning(f"[codec-grabber] upstream ignored Range for {t.filename}; restarting from 0")
                    t.tmp_path.unlink(missing_ok=True)
                    t.bytes_written = 0; last_emit = 0
                    sha = hashlib.sha256(); restarted = True

                t.total = None
                if r.status == 206 and "/" in (r.headers.get("Content-Range") or ""):
                    try: t.total = int(r.headers["Content-Range"].split("/")[-1])
                    except Exception: t.total = None
                else:
                    cl = r.headers.get("Content-Length")
                    if cl and cl.isdigit():
                        t.total = int(cl) + (t.bytes_written if not restarted else 0)
                await _history_update(t.eid, {"expected_bytes": t.total})

                mode = "ab" if t.bytes_written > 0 else "wb"
                with open(t.tmp_path, mode) as f:
                    async for chunk in r.content.iter_chunked(1048576):
                        # Cooperative pause/cancel check.
                        if t.cancel_event.is_set():
                            t.status = "cancelled"
                            t.tmp_path.unlink(missing_ok=True)
                            await _history_update(t.eid, {
                                "status": "cancelled", "downloaded_bytes": t.bytes_written,
                                "completed_at": _now_iso(),
                            })
                            await t._emit("cancelled")
                            return
                        if t.pause_event.is_set():
                            t.status = "paused"
                            await _history_update(t.eid, {
                                "status": "paused", "downloaded_bytes": t.bytes_written,
                            })
                            await t._emit("paused")
                            return
                        f.write(chunk)
                        sha.update(chunk)
                        t.bytes_written += len(chunk)
                        if t.bytes_written - last_emit >= 2097152:
                            await _history_update(t.eid, {"downloaded_bytes": t.bytes_written})
                            await t._emit("progress")
                            last_emit = t.bytes_written

        # Loop exited. Verify completeness.
        if t.total is not None and t.bytes_written < t.total:
            short = t.total - t.bytes_written
            log.warning(
                f"[codec-grabber] {t.tmp_path.name} short {short} bytes "
                f"({t.bytes_written}/{t.total}); keeping tempfile, marking paused"
            )
            t.status = "paused"; t.error = f"truncated mid-stream ({t.bytes_written}/{t.total}); resume to continue"
            await _history_update(t.eid, {
                "status": "paused", "downloaded_bytes": t.bytes_written, "error": t.error,
            })
            await t._emit("paused")
            return

        # Integrity gate: for .safetensors files, parse the header and
        # verify it covers all declared tensor bytes. Catches the case where
        # upstream Content-Length matches what we received but the file is
        # actually a truncated copy of something larger (e.g. CDN cut the
        # connection mid-stream and reported a smaller size; or a partial
        # resume mixed bytes from a different upstream).
        v = _verify_safetensors(t.tmp_path)
        if not v.get("ok"):
            log.warning(f"[codec-grabber] {t.tmp_path.name} fails safetensors verify: {v.get('error')}")
            t.status = "error"
            t.error = f"corrupt safetensors: {v.get('error')}"
            await _history_update(t.eid, {
                "status": "error", "downloaded_bytes": t.bytes_written,
                "error": t.error, "verify": v,
                "completed_at": _now_iso(),
            })
            await t._emit("error", verify=v)
            return

        # Promote.
        shutil.move(str(t.tmp_path), str(t.dest))
        t.status = "done"
        await _history_update(t.eid, {
            "status": "done", "downloaded_bytes": t.bytes_written,
            "sha256": sha.hexdigest(), "path": str(t.dest),
            "verify": v, "completed_at": _now_iso(),
        })
        await t._emit("done", sha256=sha.hexdigest(), path=str(t.dest), verify=v)

    except _DISCONNECT_ERRORS:
        # Should be rare for a background task (we don't hold a client
        # connection). But if the asyncio.Task gets cancelled, treat as
        # pause to preserve work.
        log.info(f"[codec-grabber] background task interrupted ({t.tmp_path.name}); preserving partial")
        t.status = "paused"
        await _history_update(t.eid, {"status": "paused", "downloaded_bytes": t.bytes_written})
        await t._emit("paused")
    except Exception as e:
        # A mid-stream failure (network drop, upstream reset, ClientPayloadError,
        # ServerDisconnectedError, …) must NOT destroy the partial — that
        # partial is exactly the work worth resuming. Keep the tempfile and mark
        # the entry "paused" so POST /start resumes it from byte N via Range.
        # Only an explicit Cancel deletes a partial. (Previously this handler
        # unlink()'d the tempfile, throwing away multi-GB downloads on a blip.)
        log.warning(
            "[codec-grabber] download interrupted (%s) at %d bytes: %s; "
            "preserving partial for resume", t.tmp_path.name, t.bytes_written, e,
        )
        t.status = "paused"
        t.error = f"interrupted: {e}; resume to continue"
        await _history_update(t.eid, {
            "status": "paused", "downloaded_bytes": t.bytes_written, "error": t.error,
        })
        await t._emit("paused")


async def _get_or_start(url: str, filename: str, directory: str, dest: Path) -> DownloadTask:
    eid = _entry_id(url, filename, directory)
    async with _TASKS_LOCK:
        existing = _TASKS.get(eid)
        if existing and existing.task and not existing.task.done():
            return existing
        # Stale entry — reset & restart.
        t = DownloadTask(eid, url, filename, directory, dest)
        _TASKS[eid] = t
        # Don't await — fire and forget. The task lives until completion
        # or pause/cancel.
        t.task = asyncio.create_task(_run_download(t))
        return t


# ─── HTTP routes ────────────────────────────────────────────────────────

@PromptServer.instance.routes.post("/codec/grab-model/start")
async def start_download(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    v = _validate(body)
    if isinstance(v, web.Response): return v
    url, filename, dest = v
    if dest.exists():
        return web.json_response({"error": "file already exists", "path": str(dest)}, status=409)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # If a previously-cancelled/errored task exists for the same eid,
    # clear its flags so the new run is clean.
    eid = _entry_id(url, filename, body["directory"])
    async with _TASKS_LOCK:
        old = _TASKS.get(eid)
        if old and old.task and old.task.done():
            old.pause_event.clear()
            old.cancel_event.clear()
            del _TASKS[eid]
    t = await _get_or_start(url, filename, body["directory"], dest)
    return web.json_response({"id": t.eid, "status": t.status,
                              "downloaded_bytes": t.bytes_written,
                              "expected_bytes": t.total})


@PromptServer.instance.routes.post("/codec/grab-model/pause")
async def pause_download(request):
    try: body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    # Accept either {id} or {url,filename,directory}.
    eid = body.get("id")
    if not eid:
        v = _validate(body)
        if isinstance(v, web.Response): return v
        eid = _entry_id(body["url"], body["filename"], body["directory"])
    t = _TASKS.get(eid)
    if not t:
        return web.json_response({"error": "no active task for that id"}, status=404)
    t.pause_event.set()
    return web.json_response({"ok": True, "id": eid})


@PromptServer.instance.routes.post("/codec/grab-model/cancel")
async def cancel_download(request):
    try: body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    # Accept either {id} or {url,filename,directory}. The latter also
    # lets you cancel a download that has no live task (just delete the
    # tempfile + mark history).
    eid = body.get("id")
    filename = body.get("filename")
    directory = body.get("directory")
    deleted = []
    if eid and eid in _TASKS:
        t = _TASKS[eid]
        t.cancel_event.set()
        filename = filename or t.filename
        directory = directory or t.directory
    elif body.get("url"):
        v = _validate(body)
        if isinstance(v, web.Response): return v
        eid = _entry_id(body["url"], body["filename"], body["directory"])
        if eid in _TASKS:
            _TASKS[eid].cancel_event.set()
    # Also nuke any leftover tempfile on disk (in case there's no live task).
    if filename and directory and _is_safe_filename(filename):
        target_dir = _resolve_target_dir(directory)
        if target_dir is not None:
            tmp = target_dir / f".grabbing-{filename}"
            if tmp.exists():
                tmp.unlink(); deleted.append(str(tmp))
            if eid:
                await _history_update(eid, {"status": "cancelled", "completed_at": _now_iso()})
    return web.json_response({"ok": True, "id": eid, "deleted": deleted})


@PromptServer.instance.routes.get("/codec/grab-model/active")
async def list_active(request):
    """Currently-running tasks (downloading or paused)."""
    out = []
    for eid, t in list(_TASKS.items()):
        out.append({
            "id": eid, "filename": t.filename, "directory": t.directory,
            "url": t.url, "status": t.status,
            "downloaded_bytes": t.bytes_written, "expected_bytes": t.total,
            "error": t.error,
        })
    return web.json_response({"active": out})


@PromptServer.instance.routes.post("/codec/grab-model/stream")
async def stream_download(request):
    """Legacy: start a task (or attach to running one) + tail its events
    as NDJSON. Client disconnect no longer stops the task — the task
    keeps running in the background; we just stop streaming events to
    *this* client."""
    try: body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    v = _validate(body)
    if isinstance(v, web.Response): return v
    url, filename, dest = v
    if dest.exists():
        return web.json_response({"error": "file already exists", "path": str(dest)}, status=409)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Reset stale entry like /start does.
    eid = _entry_id(url, filename, body["directory"])
    async with _TASKS_LOCK:
        old = _TASKS.get(eid)
        if old and old.task and old.task.done():
            old.pause_event.clear()
            old.cancel_event.clear()
            del _TASKS[eid]
    t = await _get_or_start(url, filename, body["directory"], dest)

    response = web.StreamResponse(
        status=200,
        headers={"Content-Type": "application/x-ndjson", "Cache-Control": "no-cache"},
    )
    await response.prepare(request)
    q = t.subscribe()
    # Send an initial state snapshot so a late subscriber sees current bytes.
    try:
        await response.write((json.dumps({
            "phase": "start",
            "id": t.eid, "filename": t.filename, "directory": t.directory,
            "url": t.url, "status": t.status,
            "downloaded_bytes": t.bytes_written, "expected_bytes": t.total,
            "resumed_from": t.bytes_written,
        }) + "\n").encode())
        while not t.task.done():
            try:
                msg = await asyncio.wait_for(q.get(), timeout=10.0)
                await response.write((json.dumps(msg) + "\n").encode())
                if msg.get("phase") in ("done", "error", "cancelled"):
                    break
            except asyncio.TimeoutError:
                # Heartbeat — keeps proxies/CDNs from closing the connection.
                try:
                    await response.write(b'{"phase":"heartbeat"}\n')
                except _DISCONNECT_ERRORS:
                    break
    except _DISCONNECT_ERRORS:
        # Client gone. Task keeps running.
        pass
    finally:
        t.unsubscribe(q)
    return response


@PromptServer.instance.routes.post("/codec/grab-model")
async def grab_model_blocking(request):
    """Synchronous, blocks until done. Useful for curl/scripts."""
    try: body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    v = _validate(body)
    if isinstance(v, web.Response): return v
    url, filename, dest = v
    if dest.exists():
        return web.json_response({"error": "file already exists", "path": str(dest)}, status=409)
    dest.parent.mkdir(parents=True, exist_ok=True)
    t = await _get_or_start(url, filename, body["directory"], dest)
    try:
        await t.task
    except _DISCONNECT_ERRORS:
        pass
    if t.status == "done":
        return web.json_response({"ok": True, "path": str(t.dest), "bytes": t.bytes_written})
    return web.json_response({"error": t.error or t.status, "bytes": t.bytes_written}, status=502)


@PromptServer.instance.routes.get("/codec/grab-model/status")
async def status_model(request):
    directory = request.query.get("directory")
    filename = request.query.get("filename")
    if not directory or not filename:
        return web.json_response({"error": "directory + filename required"}, status=400)
    if not _is_safe_filename(filename):
        return web.json_response({"error": "unsafe filename"}, status=400)
    target_dir = _resolve_target_dir(directory)
    if target_dir is None:
        return web.json_response({"error": "unknown directory"}, status=403)
    final = target_dir / filename
    tmp = target_dir / f".grabbing-{filename}"
    if final.exists():
        return web.json_response({"state": "complete", "bytes": final.stat().st_size, "path": str(final)})
    if tmp.exists():
        return web.json_response({"state": "partial", "bytes": tmp.stat().st_size, "path": str(tmp)})
    return web.json_response({"state": "absent"})


@PromptServer.instance.routes.get("/codec/grab-model/verify")
async def verify_file(request):
    """Verify a single on-disk model. For .safetensors, parses the header
    and checks byte coverage. Returns {ok, file_bytes, needed_bytes, ...}."""
    directory = request.query.get("directory")
    filename = request.query.get("filename")
    if not directory or not filename:
        return web.json_response({"error": "directory + filename required"}, status=400)
    if not _is_safe_filename(filename):
        return web.json_response({"error": "unsafe filename"}, status=400)
    target_dir = _resolve_target_dir(directory)
    if target_dir is None:
        return web.json_response({"error": "unknown directory"}, status=403)
    path = target_dir / filename
    if not path.exists():
        return web.json_response({"ok": False, "error": "file not on disk", "kind": "missing"}, status=404)
    return web.json_response(_verify_safetensors(path))


@PromptServer.instance.routes.get("/codec/grab-model/verify-all")
async def verify_all(request):
    """Verify every history entry whose status is 'done'. Returns a list of
    {entry_id, filename, directory, verify}. UI uses this to surface
    corrupt-on-disk files in one pass."""
    items = await _history_get_all()
    out = []
    for it in items:
        if it.get("status") != "done": continue
        directory = it.get("directory")
        filename = it.get("filename")
        if not (directory and filename and _is_safe_filename(filename)): continue
        target_dir = _resolve_target_dir(directory)
        if target_dir is None: continue
        path = target_dir / filename
        if not path.exists():
            v = {"ok": False, "kind": "missing", "error": "history claims done but file is gone"}
        else:
            v = _verify_safetensors(path)
        out.append({
            "id": it.get("id"), "filename": filename, "directory": directory,
            "url": it.get("url"), "verify": v,
        })
    bad = [o for o in out if not o["verify"].get("ok")]
    return web.json_response({"checked": len(out), "bad": len(bad), "entries": out})


@PromptServer.instance.routes.get("/codec/grab-model/allowed-directories")
async def list_directories(request):
    return web.json_response(
        {"directories": sorted(folder_paths.folder_names_and_paths.keys())},
        status=200,
    )


@PromptServer.instance.routes.get("/codec/grab-model/history")
async def get_history(request):
    items = await _history_get_all()
    for it in items:
        try:
            tgt = _resolve_target_dir(it.get("directory") or "")
            if tgt:
                final = tgt / it.get("filename", "")
                tmp = tgt / f".grabbing-{it.get('filename', '')}"
                if final.exists():
                    it["on_disk"] = {"path": str(final), "bytes": final.stat().st_size, "kind": "final"}
                elif tmp.exists():
                    it["on_disk"] = {"path": str(tmp), "bytes": tmp.stat().st_size, "kind": "partial"}
                else:
                    it["on_disk"] = None
        except Exception:
            it["on_disk"] = None
    return web.json_response({"entries": items})


@PromptServer.instance.routes.get("/codec/grab-model/orphans")
async def get_orphans(request):
    items = await _history_get_all()
    active_names = {i.get("filename") for i in items
                    if i.get("status") in ("downloading", "paused")}
    orphans = []
    for category, (paths, _exts) in folder_paths.folder_names_and_paths.items():
        for p in paths:
            try:
                for f in Path(p).glob(".grabbing-*"):
                    name = f.name[len(".grabbing-"):]
                    if name in active_names: continue
                    orphans.append({
                        "path": str(f), "bytes": f.stat().st_size,
                        "filename": name, "directory": category,
                    })
            except Exception: continue
    return web.json_response({"orphans": orphans})


@PromptServer.instance.routes.post("/codec/grab-model/redownload")
async def redownload(request):
    """Delete an on-disk file (+ any tempfile) and return the metadata
    needed to start a fresh download. Accepts either:
      - {filename, directory?}     → look up URL from history (or fail)
      - {url, filename, directory} → explicit; works even with no history
                                     (use for files written by other tools,
                                     or by failed earlier attempts that
                                     never made it into history)
    """
    try: body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    filename = body.get("filename")
    if not isinstance(filename, str) or not _is_safe_filename(filename):
        return web.json_response({"error": "filename required"}, status=400)
    directory = body.get("directory")
    url = body.get("url")

    # Explicit-URL path: skip history lookup entirely.
    if isinstance(url, str) and isinstance(directory, str):
        target_dir = _resolve_target_dir(directory)
        if not target_dir:
            return web.json_response({"error": f"unknown directory {directory!r}"}, status=403)
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("https", "http"):
            return web.json_response({"error": f"unsupported scheme: {parsed.scheme!r}"}, status=400)
        deleted = []
        for p in (target_dir / filename, target_dir / f".grabbing-{filename}"):
            if p.exists():
                p.unlink(); deleted.append(str(p))
        eid = _entry_id(url, filename, directory)
        await _history_update(eid, {
            "url": url, "filename": filename, "directory": directory,
            "status": "cancelled", "error": "redownload requested (explicit URL)",
        })
        return web.json_response({
            "ok": True, "deleted": deleted, "source": "explicit",
            "url": url, "filename": filename, "directory": directory,
        })

    # History-lookup path.
    items = await _history_get_all()
    matches = [
        i for i in items
        if i.get("filename") == filename
        and (not directory or i.get("directory") == directory)
    ]
    matches.sort(key=lambda i: 0 if i.get("status") == "done" else 1)
    if not matches:
        return web.json_response(
            {"error": f"no history entry for {filename!r}; pass {{url, filename, directory}} explicitly to force"},
            status=404,
        )
    entry = matches[0]
    directory = entry["directory"]
    url = entry["url"]
    target_dir = _resolve_target_dir(directory)
    if not target_dir:
        return web.json_response({"error": f"unknown directory {directory!r}"}, status=403)

    deleted = []
    for p in (target_dir / filename, target_dir / f".grabbing-{filename}"):
        if p.exists():
            p.unlink(); deleted.append(str(p))
    eid = _entry_id(url, filename, directory)
    await _history_update(eid, {"status": "cancelled", "error": "redownload requested"})

    return web.json_response({
        "ok": True, "deleted": deleted, "source": "history",
        "url": url, "filename": filename, "directory": directory,
        "sha256": entry.get("sha256"),
    })


# ─── Startup: demote stale "downloading" entries ────────────────────────

async def _startup_repair():
    """Anything marked downloading in history that we don't have a
    matching live task for is a leftover from the prior container
    process. Mark it paused so the UI doesn't lie about it."""
    async with _HISTORY_LOCK:
        d = _load_history_sync()
        changed = False
        for eid, entry in d["entries"].items():
            if entry.get("status") == "downloading":
                entry["status"] = "paused"
                entry["error"] = (entry.get("error") or "") + " (interrupted by container restart)"
                changed = True
        if changed:
            _save_history_sync(d)
    log.info("[codec-grabber] startup repair pass complete")


# Schedule the repair pass to run on the event loop once it's available.
def _arm_startup_repair():
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(_startup_repair())
    except RuntimeError:
        # No loop yet; ComfyUI will start one. Defer.
        import threading
        def deferred():
            import time; time.sleep(2)
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(_startup_repair())
            except Exception as e:
                log.warning(f"[codec-grabber] startup repair couldn't schedule: {e}")
        threading.Thread(target=deferred, daemon=True).start()


_arm_startup_repair()

log.info(
    "[codec-grabber] v6 — adds safetensors verify on finalize + /verify "
    "+ /verify-all endpoints; tempfile discarded if URL changed on resume; "
    "/redownload accepts explicit {url,filename,directory} for files "
    f"without history. history at {_HISTORY_PATH}"
)
