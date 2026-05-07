from __future__ import annotations

import argparse
import logging
import os
import shlex
import sys
from pathlib import Path

import uvicorn

from . import __version__
from .app import create_app
from .backend import BACKENDS
from .config import Config


def _env(key: str, default: str | None = None) -> str | None:
    """Read CODEC_* env var with a fallback default."""
    return os.environ.get(key, default)


def _parse_backend_args(items: list[str] | None) -> list[str]:
    """Each --backend-arg may be one token; CODEC_BACKEND_ARGS may be a single
    shell-quoted string (e.g. "--mem-fraction-static 0.85 --tp 2"). Both are
    flattened into a single argv list."""
    out: list[str] = []
    if items:
        for item in items:
            out.extend(shlex.split(item))
    env_args = _env("CODEC_BACKEND_ARGS")
    if env_args:
        out.extend(shlex.split(env_args))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="codec-supervisor",
        description="Backend-agnostic supervisor / control plane for inference servers.",
    )
    p.add_argument("--version", action="version", version=f"codec-supervisor {__version__}")
    p.add_argument("--host", default=_env("CODEC_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(_env("CODEC_PORT", "8080")))
    p.add_argument(
        "--backend",
        default=_env("CODEC_BACKEND", "sglang"),
        choices=list(BACKENDS),
    )
    p.add_argument("--backend-host", default=_env("CODEC_BACKEND_HOST", "127.0.0.1"))
    p.add_argument(
        "--backend-port",
        type=int,
        default=int(_env("CODEC_BACKEND_PORT", "30000")),
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=Path(_env("CODEC_MODELS_DIR", "/models")),
    )
    p.add_argument(
        "--initial-model",
        default=_env("CODEC_INITIAL_MODEL"),
        help="Local model name, absolute path, or HF repo id to load on startup.",
    )
    p.add_argument(
        "--backend-arg",
        action="append",
        dest="backend_args",
        default=None,
        help="Extra arg passed to the backend, repeatable. Also picked up from CODEC_BACKEND_ARGS.",
    )
    p.add_argument(
        "--startup-timeout",
        type=int,
        default=int(_env("CODEC_STARTUP_TIMEOUT_S", "1800")),
    )
    p.add_argument(
        "--shutdown-grace",
        type=int,
        default=int(_env("CODEC_SHUTDOWN_GRACE_S", "30")),
    )
    p.add_argument(
        "--log-level",
        default=_env("CODEC_LOG_LEVEL", "INFO"),
    )
    p.add_argument(
        "--no-allow-remote-load",
        action="store_true",
        help="Refuse /admin/load with allow_remote=true (lock the supervisor to local /models only).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    config = Config(
        host=args.host,
        port=args.port,
        backend=args.backend,
        backend_host=args.backend_host,
        backend_port=args.backend_port,
        models_dir=args.models_dir,
        initial_model=args.initial_model,
        backend_args=_parse_backend_args(args.backend_args),
        startup_timeout_s=args.startup_timeout,
        shutdown_grace_s=args.shutdown_grace,
        log_level=args.log_level,
        allow_remote_load=not args.no_allow_remote_load,
    )

    logger = logging.getLogger("codec_supervisor")
    logger.info(
        "codec-supervisor %s — supervising %s on %s:%d, listening on %s:%d",
        __version__,
        config.backend,
        config.backend_host,
        config.backend_port,
        config.host,
        config.port,
    )

    app = create_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
