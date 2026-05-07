from __future__ import annotations

from pydantic import BaseModel, Field


class PullRequest(BaseModel):
    repo_id: str = Field(..., description="Hugging Face repo id, e.g. 'Qwen/Qwen2.5-0.5B-Instruct'")
    revision: str | None = Field(None, description="Optional branch / tag / commit")
    token: str | None = Field(
        None,
        description="HF access token. If omitted, falls back to HF_TOKEN env.",
    )


class LoadRequest(BaseModel):
    name: str = Field(
        ...,
        description="Local model name (in models_dir) OR an HF id when allow_remote=true.",
    )
    extra_args: list[str] | None = Field(
        None,
        description="Per-load backend args. If omitted, uses the supervisor's default backend_args.",
    )
    allow_remote: bool = Field(
        False,
        description="If true and `name` isn't a local model, treat it as an HF id and pass through to the backend.",
    )


class StatusResponse(BaseModel):
    backend: str
    running: bool
    current_model: str | None
    started_at: str | None
    backend_url: str
    models_dir: str


class ModelEntry(BaseModel):
    name: str
    path: str
    size_bytes: int


class ModelsListResponse(BaseModel):
    models: list[ModelEntry]
