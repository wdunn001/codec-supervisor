"""Endpoint-surface regression tests.

Added after the v0.4.1 post-mortem: the codec-sglang:v0.4.1 image was built
from a stale codec-supervisor tree (pre-1228cbd safety-admin merge, pre-40c0838
codec-policy endpoint). The image started cleanly, the /openapi.json silently
omitted the v0.4 admin surface, and operators discovered the gap by hitting 404
on /admin/codec-policy. These tests assert every v0.4-mandated admin endpoint is
registered at app-construction time, so a future merge that drops a router
import fails CI loudly instead of shipping a hollow image.

Spec floor: the production codec-supervisor build MUST expose all endpoints in
EXPECTED_PATHS below at every release tag in the v0.4 line. Add new endpoints
to the list as v0.5+ ships them.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codec_supervisor.app import create_app
from codec_supervisor.config import Config


# Endpoints introduced through v0.4 that MUST be present on every release-tagged
# codec-supervisor build. Path templates use FastAPI's {placeholder} syntax —
# they match the keys in app.openapi()["paths"].
EXPECTED_PATHS: set[str] = {
    # Liveness + state
    "/health",
    "/admin/status",
    # Model lifecycle (v0.3+)
    "/admin/models",
    "/admin/models/pull",
    "/admin/models/upload",
    "/admin/models/{name}",
    "/admin/load",
    "/admin/stop",
    # v0.4 safety policy admin (introduced via codec-supervisor 1228cbd)
    "/admin/policies",
    "/admin/policies/_classifiers",
    "/admin/policies/_versions",
    "/admin/policies/_versions/{hex_hash}",
    "/admin/policies/{policy_id}",
    "/admin/policies/{policy_id}/sanitize",
    # v0.4 capability-state snapshot (introduced via codec-supervisor 40c0838)
    "/admin/codec-policy",
}


@pytest.fixture
def app(tmp_path: Path):
    config = Config(
        host="127.0.0.1",
        port=0,
        backend="sglang",
        backend_port=0,
        models_dir=tmp_path / "models",
        initial_model=None,
    )
    return create_app(config)


def test_v04_endpoint_surface_present(app):
    """Every v0.4-mandated endpoint must appear in the live OpenAPI surface.

    If this fails, the build is regressed — a router import was dropped, an
    endpoint was renamed/removed, or app.include_router(...) is missing.
    Don't suppress this test; fix the regression.
    """
    paths = set(app.openapi().get("paths", {}).keys())
    missing = EXPECTED_PATHS - paths
    assert not missing, (
        f"v0.4-mandated endpoints missing from supervisor surface: {sorted(missing)}. "
        f"This is a release-blocker per the v0.4 spec — check that "
        f"admin_safety.create_safety_router and codec_policy.snapshot are still "
        f"imported and mounted in codec_supervisor/app.py."
    )


def test_no_unexpected_admin_paths(app):
    """Catch the inverse regression: an admin path appeared that we didn't
    expect. Forces a deliberate update to EXPECTED_PATHS whenever the admin
    surface grows — keeps the list authoritative."""
    paths = {p for p in app.openapi().get("paths", {}) if p.startswith("/admin/")}
    extras = paths - EXPECTED_PATHS
    assert not extras, (
        f"New /admin/* endpoint(s) appeared that aren't in EXPECTED_PATHS: "
        f"{sorted(extras)}. Add them to EXPECTED_PATHS in tests/test_endpoint_surface.py "
        f"and bump the codec_supervisor version if this is a v0.X+1 cap."
    )
