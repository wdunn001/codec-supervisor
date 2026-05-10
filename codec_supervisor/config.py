from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8080
    backend: str = "sglang"
    backend_port: int = 30000
    backend_host: str = "127.0.0.1"
    models_dir: Path = Path("/models")
    # Operator-side safety policy storage. One JSON per policy id; archived
    # descriptor snapshots under `<policies_dir>/.versions/<sha256>.json`.
    # See `codec_supervisor/safety.py` for the disclosure-boundary contract
    # (internal config vs publishable descriptor) and `admin_safety.py` for
    # the `/admin/policies/*` REST surface.
    policies_dir: Path = Path("/policies")
    initial_model: str | None = None
    backend_args: list[str] = field(default_factory=list)
    startup_timeout_s: int = 1800
    shutdown_grace_s: int = 30
    log_level: str = "INFO"
    allow_remote_load: bool = True
