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
    initial_model: str | None = None
    backend_args: list[str] = field(default_factory=list)
    startup_timeout_s: int = 1800
    shutdown_grace_s: int = 30
    log_level: str = "INFO"
    allow_remote_load: bool = True
