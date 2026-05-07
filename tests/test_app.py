"""Smoke tests for codec_supervisor.

These don't spawn a real backend — the ProcessManager is exercised lightly via
admin endpoints that don't require a running child.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codec_supervisor.app import create_app
from codec_supervisor.config import Config


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    config = Config(
        host="127.0.0.1",
        port=0,
        backend="sglang",
        backend_port=0,
        models_dir=tmp_path / "models",
        initial_model=None,  # don't try to launch a real backend
    )
    # Bypass the lifespan event so we don't actually start sglang.
    app = create_app(config)
    return TestClient(app)


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["supervisor"] == "ok"
    assert body["backend_running"] is False


def test_status(client: TestClient):
    r = client.get("/admin/status")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "sglang"
    assert body["running"] is False
    assert body["current_model"] is None


def test_models_list_empty(client: TestClient):
    r = client.get("/admin/models")
    assert r.status_code == 200
    assert r.json() == {"models": []}


def test_models_list_finds_local(client: TestClient, tmp_path: Path):
    (tmp_path / "models" / "fake-model").mkdir(parents=True)
    (tmp_path / "models" / "fake-model" / "config.json").write_text("{}")
    r = client.get("/admin/models")
    assert r.status_code == 200
    body = r.json()
    assert len(body["models"]) == 1
    assert body["models"][0]["name"] == "fake-model"


def test_load_unknown_local_rejected(client: TestClient):
    r = client.post("/admin/load", json={"name": "does-not-exist"})
    assert r.status_code == 404


def test_proxy_returns_503_when_no_backend(client: TestClient):
    r = client.get("/v1/models")
    assert r.status_code == 503
    assert "no backend loaded" in r.json()["error"]


def test_delete_unknown_model_404(client: TestClient):
    r = client.delete("/admin/models/nope")
    assert r.status_code == 404
