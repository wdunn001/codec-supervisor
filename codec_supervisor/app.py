"""FastAPI app: admin endpoints + reverse-proxy catch-all.

Routes:
  GET    /health                                — supervisor liveness
  GET    /admin/status                          — backend state, current model, uptime
  GET    /admin/models                          — list models in /models
  POST   /admin/models/pull                     — snapshot_download from HF into /models
  POST   /admin/models/upload                   — multipart tarball upload into /models/{name}
  DELETE /admin/models/{name}                   — remove from /models
  POST   /admin/load                            — restart backend with the named model
  POST   /admin/stop                            — stop the backend (supervisor stays up)

  GET    /admin/policies                        — list safety policies
  GET    /admin/policies/{id}                   — read internal policy config
  PUT    /admin/policies/{id}                   — write internal policy config + snapshot
  DELETE /admin/policies/{id}                   — drop internal policy config
  POST   /admin/policies/{id}/sanitize          — emit publishable descriptor + hash
  GET    /admin/policies/_versions              — list archived descriptor hashes
  GET    /admin/policies/_versions/{hex}        — read an archived descriptor
  *      /{path:path}                           — proxy everything else to the backend

The catch-all is registered last so admin routes win.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from . import __version__
from .backend import get_backend
from .config import Config
from .manager import BackendCrashed, BackendNotReady, ProcessManager
from .models import (
    delete_model,
    extract_tarball,
    list_models,
    model_path,
    pull_from_hf,
)
from .admin_safety import create_safety_router
from .proxy import proxy_request
from .schemas import (
    LoadRequest,
    ModelsListResponse,
    PullRequest,
    StatusResponse,
)

logger = logging.getLogger(__name__)


def create_app(config: Config) -> FastAPI:
    backend = get_backend(config.backend)
    manager = ProcessManager(
        backend=backend,
        host=config.backend_host,
        port=config.backend_port,
        startup_timeout_s=config.startup_timeout_s,
        shutdown_grace_s=config.shutdown_grace_s,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        config.models_dir.mkdir(parents=True, exist_ok=True)
        config.policies_dir.mkdir(parents=True, exist_ok=True)
        if config.initial_model:
            target = _resolve_local_or_passthrough(
                config.initial_model, config.models_dir
            )
            try:
                await manager.start(target, config.backend_args)
            except (BackendCrashed, BackendNotReady) as e:
                logger.error("initial model load failed: %s", e)
        yield
        await manager.stop()

    app = FastAPI(
        title="codec-supervisor",
        description="Backend-agnostic supervisor / control plane for inference servers.",
        version=__version__,
        lifespan=lifespan,
    )

    # ---------- liveness ----------
    @app.get("/health")
    async def health():
        return {
            "supervisor": "ok",
            "version": __version__,
            "backend_running": manager.is_running,
        }

    # ---------- status ----------
    @app.get("/admin/status", response_model=StatusResponse)
    async def status():
        return StatusResponse(
            backend=backend.name,
            running=manager.is_running,
            current_model=manager.current_model,
            started_at=manager.started_at.isoformat() if manager.started_at else None,
            backend_url=manager.base_url,
            models_dir=str(config.models_dir),
        )

    # ---------- model registry ----------
    @app.get("/admin/models", response_model=ModelsListResponse)
    async def models_list():
        return ModelsListResponse(models=list_models(config.models_dir))

    @app.post("/admin/models/pull")
    async def models_pull(body: PullRequest):
        token = body.token or os.environ.get("HF_TOKEN")
        try:
            target = await pull_from_hf(
                body.repo_id,
                config.models_dir,
                revision=body.revision,
                token=token,
            )
        except Exception as e:  # huggingface_hub raises a zoo of exception types
            logger.exception("HF pull failed")
            raise HTTPException(502, f"pull failed: {e}") from e
        return {"name": target.name, "path": str(target)}

    @app.post("/admin/models/upload")
    async def models_upload(name: str, file: UploadFile):
        async def chunks():
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk

        try:
            target = await extract_tarball(name, chunks(), config.models_dir)
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e
        except (ValueError, OSError) as e:
            raise HTTPException(400, str(e)) from e
        return {"name": target.name, "path": str(target)}

    @app.delete("/admin/models/{name}")
    async def models_delete(name: str):
        # Safety: refuse to delete the model that's currently loaded.
        local = model_path(config.models_dir, name)
        if local is not None and manager.current_model == str(local):
            raise HTTPException(409, f"model {name!r} is currently loaded; stop first")
        try:
            target = delete_model(config.models_dir, name)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        return {"deleted": target.name}

    # ---------- backend control ----------
    @app.post("/admin/load")
    async def load(body: LoadRequest):
        local = model_path(config.models_dir, body.name)
        if local is not None:
            target = str(local)
        elif body.allow_remote and config.allow_remote_load:
            target = body.name  # let the backend resolve (HF id or path)
        else:
            raise HTTPException(
                404,
                f"unknown local model {body.name!r}. "
                "Set allow_remote=true to pass through to the backend.",
            )
        extra = body.extra_args if body.extra_args is not None else config.backend_args
        try:
            await manager.start(target, extra)
        except (BackendCrashed, BackendNotReady) as e:
            raise HTTPException(500, str(e)) from e
        return {"loaded": target}

    @app.post("/admin/stop")
    async def stop():
        await manager.stop()
        return {"stopped": True}

    # ---------- safety policy admin ----------
    app.include_router(create_safety_router(config.policies_dir))

    # ---------- catch-all proxy (must be last) ----------
    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
    )
    async def catchall(path: str, request: Request):
        if not manager.is_running:
            return JSONResponse(
                status_code=503,
                content={"error": "no backend loaded; POST /admin/load first"},
            )
        return await proxy_request(request, manager.base_url)

    return app


def _resolve_local_or_passthrough(spec: str, models_dir: Path) -> str:
    """If spec resolves to a local model dir, return its path; else return as-is."""
    p = Path(spec)
    if p.is_absolute() and p.is_dir():
        return str(p)
    local = model_path(models_dir, spec)
    if local is not None:
        return str(local)
    return spec  # let the backend handle HF id / its own cache
