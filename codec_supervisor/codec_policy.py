"""Read-only view of the codec v0.4 capability state.

Mirrors the env-var contract that the engine fork (sglang) uses in
`python/sglang/srt/entrypoints/codec_version.py`. The supervisor doesn't
mutate this state — the values are baked at backend boot via env vars
on the docker-compose / systemd unit and a hot-swap is a backend
restart. This endpoint exists so an operator can ask "what wire surface
is this deployment actually offering?" without inspecting the
container env.

Spec: `spec/versions/v0.4.md § Capabilities are opt-on at the server`.
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field


class CapabilityState(BaseModel):
    """Per-capability enable + enforce status."""

    name: str
    enabled: bool
    enforced: bool
    config: dict = Field(default_factory=dict, description="Capability-specific knobs.")


class CodecPolicyResponse(BaseModel):
    """The deployment's codec-policy view.

    A v0.4 server with no env vars set returns all capabilities as
    enabled=False, enforced=False — the default-OFF ship state.
    """

    minimum_version: str = Field(
        default="0.4",
        description="The lowest codec minor version this deployment requires, IF "
        "any capability is enforced. When no capability is enforced, this "
        "field is advisory — the deployment will serve any client version.",
    )
    any_mandatory: bool = Field(
        description="True iff at least one capability is set to ENFORCE — "
        "i.e. the deployment will return 426 to clients below the floor.",
    )
    capabilities: list[CapabilityState]
    deployment_id: Optional[str] = None
    docs_url: str = "https://codecai.net/docs/version-negotiation/"


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "").strip() in ("1", "true", "True", "yes")


def snapshot() -> CodecPolicyResponse:
    """Read the current v0.4 capability state from env."""
    safety_id = os.environ.get("CODEC_SAFETY_POLICY", "").strip()
    safety_enabled = bool(safety_id)
    safety_required = safety_enabled and _bool_env("CODEC_SAFETY_POLICY_REQUIRED")

    vp_raw = os.environ.get("CODEC_VERSION_POLICY", "off").strip().lower()
    vp_mode = vp_raw if vp_raw in ("off", "advisory", "strict") else "off"
    vp_enabled = vp_mode in ("advisory", "strict")
    vp_required = vp_mode == "strict"

    caps = [
        CapabilityState(
            name="safety-policy",
            enabled=safety_enabled,
            enforced=safety_required,
            config={"policy_id": safety_id} if safety_id else {},
        ),
        CapabilityState(
            name="version-policy",
            enabled=vp_enabled,
            enforced=vp_required,
            config={"mode": vp_mode},
        ),
    ]
    any_mandatory = any(c.enforced for c in caps)
    deployment_id = os.environ.get("CODEC_DEPLOYMENT_ID", "").strip() or None

    return CodecPolicyResponse(
        any_mandatory=any_mandatory,
        capabilities=caps,
        deployment_id=deployment_id,
    )
