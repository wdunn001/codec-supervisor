"""Tests for codec_supervisor.safety + the /admin/policies/* endpoints.

These exercise:
  - InternalPolicy validation (Pydantic + policy-id rules)
  - sanitize() — strips internal-only fields, derives summary counts
  - descriptor_canonical_bytes / descriptor_sha256 (canonical form pinned)
  - storage layer: list / load / save / list_versions / load_version
  - REST surface: GET list / GET item / PUT / DELETE / POST sanitize
                  GET _versions, GET _versions/{hex}, body/url id mismatch

Backend lifecycle is bypassed — same pattern as test_app.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codec_supervisor.app import create_app
from codec_supervisor.config import Config
from codec_supervisor.safety import (
    INTERNAL_ONLY_FIELDS,
    InternalPolicy,
    PolicyIdError,
    PolicyNotFoundError,
    descriptor_canonical_bytes,
    descriptor_sha256,
    list_policies,
    list_versions,
    load_policy,
    load_version,
    policy_filename,
    sanitize,
    save_policy,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def policies_dir(tmp_path: Path) -> Path:
    d = tmp_path / "policies"
    d.mkdir()
    return d


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    config = Config(
        host="127.0.0.1",
        port=0,
        backend="sglang",
        backend_port=0,
        models_dir=tmp_path / "models",
        policies_dir=tmp_path / "policies",
        initial_model=None,
    )
    return TestClient(create_app(config))


def make_internal(**overrides) -> dict:
    base: dict = {
        "id": "acme/strict-v3",
        "version": "1",
        "tokenizers": ["meta-llama/llama-3"],
        "categories": [
            {"name": "secrets", "action": "stop"},
            {"name": "pii", "action": "redact", "description": "Email and phone."},
        ],
        "classifier": {
            "family": "llama-guard-3-1b",
            "host": "server",
            "requires_engine_features": ["logits_processor", "sampling_chain"],
        },
        "banned_token_ids": [4218, 5544, 9012, 12345, 67890],
        "regex_patterns": ["(?i)AKIA[0-9A-Z]{16}", "ghp_[A-Za-z0-9]{36}"],
        "grammar_constraints": [{"name": "no_urls"}],
        "multi_token_patterns": [{"literal": "secret-1"}, {"literal": "secret-2"}],
        "classifier_internal": {"thresholds": {"secrets": 0.9, "pii": 0.7}},
    }
    base.update(overrides)
    return base


# ── Validation ───────────────────────────────────────────────────────────────


def test_internal_policy_validates_a_realistic_payload():
    InternalPolicy.model_validate(make_internal())


def test_internal_policy_rejects_bad_category_name():
    bad = make_internal(categories=[{"name": "BadCaps", "action": "stop"}])
    with pytest.raises(Exception):
        InternalPolicy.model_validate(bad)


def test_internal_policy_rejects_unknown_action():
    bad = make_internal(categories=[{"name": "x", "action": "banhammer"}])
    with pytest.raises(Exception):
        InternalPolicy.model_validate(bad)


def test_internal_policy_rejects_extra_top_level_keys():
    bad = make_internal(unknown_field="oops")
    with pytest.raises(Exception):
        InternalPolicy.model_validate(bad)


# ── Sanitize ─────────────────────────────────────────────────────────────────


def test_sanitize_strips_internal_only_fields_and_derives_counts():
    internal = InternalPolicy.model_validate(make_internal())
    descriptor = sanitize(internal)
    payload = descriptor.model_dump(mode="json", exclude_none=True)
    for k in INTERNAL_ONLY_FIELDS:
        assert k not in payload, f"sanitized descriptor MUST NOT contain {k!r}"
    rs = payload["rules_summary"]
    assert rs["banned_token_id_count"] == 5
    assert rs["regex_pattern_count"] == 2
    assert rs["grammar_constraint_count"] == 1
    assert rs["multi_token_pattern_count"] == 2


def test_sanitize_preserves_disclosed_fields():
    internal = InternalPolicy.model_validate(
        make_internal(category_registry="mlcommons/ailuminate-v0.5"),
    )
    descriptor = sanitize(internal)
    assert descriptor.id == internal.id
    assert descriptor.tokenizers == internal.tokenizers
    assert descriptor.classifier.family == internal.classifier.family
    assert descriptor.category_registry == "mlcommons/ailuminate-v0.5"


def test_sanitize_omits_rules_summary_when_no_internal_fields_present():
    rules_free = make_internal()
    for k in INTERNAL_ONLY_FIELDS:
        rules_free.pop(k, None)
    internal = InternalPolicy.model_validate(rules_free)
    descriptor = sanitize(internal)
    assert descriptor.rules_summary is None


# ── Hashing ──────────────────────────────────────────────────────────────────


def test_descriptor_hash_is_deterministic_and_sha256_format():
    internal = InternalPolicy.model_validate(make_internal())
    h1 = descriptor_sha256(sanitize(internal))
    h2 = descriptor_sha256(sanitize(internal))
    assert h1 == h2
    assert h1.startswith("sha256:")
    assert len(h1.split(":", 1)[1]) == 64


def test_descriptor_canonical_bytes_match_2_space_indent_with_trailing_newline():
    internal = InternalPolicy.model_validate(make_internal())
    raw = descriptor_canonical_bytes(sanitize(internal))
    text = raw.decode("utf-8")
    assert text.endswith("\n")
    # Must round-trip through json.loads.
    json.loads(text)
    # Pretty-printed: contains a newline-then-two-spaces indent.
    assert "\n  " in text


def test_descriptor_hash_unaffected_by_internal_only_churn():
    """A regex change is invisible at the wire — same descriptor hash."""
    internal_a = InternalPolicy.model_validate(
        make_internal(regex_patterns=["one"]),
    )
    internal_b = InternalPolicy.model_validate(
        make_internal(regex_patterns=["one", "two"]),
    )
    # Different rules_summary → different hashes (counts ARE disclosed).
    assert descriptor_sha256(sanitize(internal_a)) != descriptor_sha256(
        sanitize(internal_b),
    )

    # But changing the *content* without changing the count yields the same hash.
    internal_c = InternalPolicy.model_validate(
        make_internal(regex_patterns=["one"]),
    )
    internal_d = InternalPolicy.model_validate(
        make_internal(regex_patterns=["different-but-still-one-pattern"]),
    )
    assert descriptor_sha256(sanitize(internal_c)) == descriptor_sha256(
        sanitize(internal_d),
    )


# ── Storage / IDs ────────────────────────────────────────────────────────────


def test_policy_filename_encodes_slashes():
    assert policy_filename("acme/strict-v3") == "acme%2Fstrict-v3.json"
    assert policy_filename("plain") == "plain.json"


def test_policy_filename_rejects_traversal_and_charset():
    for bad in ("../etc", "/abs", "trailing/", "Acme/Strict", "acme//strict"):
        with pytest.raises(PolicyIdError):
            policy_filename(bad)


def test_save_then_load_roundtrip(policies_dir: Path):
    internal = InternalPolicy.model_validate(make_internal())
    path, hash_ = save_policy(policies_dir, internal)
    assert path.is_file()
    assert hash_.startswith("sha256:")

    loaded = load_policy(policies_dir, internal.id)
    assert loaded.model_dump(mode="json", exclude_none=True) == internal.model_dump(
        mode="json", exclude_none=True,
    )


def test_save_archives_a_descriptor_snapshot(policies_dir: Path):
    internal = InternalPolicy.model_validate(make_internal())
    _, hash_ = save_policy(policies_dir, internal)
    hex_ = hash_.split(":", 1)[1]
    versions = list_versions(policies_dir)
    assert hex_ in versions
    descriptor = load_version(policies_dir, hex_)
    # A round-trip via load_version yields a publishable descriptor whose
    # internal fields are NOT present.
    payload = descriptor.model_dump(mode="json", exclude_none=True)
    for k in INTERNAL_ONLY_FIELDS:
        assert k not in payload


def test_list_policies_alphabetical(policies_dir: Path):
    for pid in ("zacme/v1", "acme/v2", "midco/v1"):
        save_policy(
            policies_dir,
            InternalPolicy.model_validate(make_internal(id=pid)),
        )
    listed = list_policies(policies_dir)
    assert listed == ["acme/v2", "midco/v1", "zacme/v1"]


def test_load_unknown_policy_raises(policies_dir: Path):
    with pytest.raises(PolicyNotFoundError):
        load_policy(policies_dir, "no-such/v1")


# ── REST API ─────────────────────────────────────────────────────────────────


def test_get_policies_empty(client: TestClient):
    r = client.get("/admin/policies")
    assert r.status_code == 200
    assert r.json() == {"policies": []}


def test_put_then_get_roundtrip(client: TestClient):
    body = make_internal()
    r = client.put(f"/admin/policies/{body['id']}", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["hash"].startswith("sha256:")
    # The descriptor in the response MUST already be sanitized.
    desc = out["descriptor"]
    for k in INTERNAL_ONLY_FIELDS:
        assert k not in desc

    r = client.get(f"/admin/policies/{body['id']}")
    assert r.status_code == 200
    got = r.json()
    # The internal endpoint DOES disclose the internal fields to operators.
    assert got["banned_token_ids"] == body["banned_token_ids"]


def test_put_url_id_must_match_body_id(client: TestClient):
    body = make_internal(id="acme/v1")
    r = client.put("/admin/policies/midco/v2", json=body)
    assert r.status_code == 400
    assert "does not match" in r.json()["detail"]


def test_put_rejects_invalid_id(client: TestClient):
    body = make_internal(id="Bad/Caps")
    r = client.put("/admin/policies/Bad/Caps", json=body)
    # Pydantic field-validation may fire first; both 400 and 422 are acceptable.
    assert r.status_code in (400, 422)


def test_post_sanitize_endpoint(client: TestClient):
    body = make_internal()
    client.put(f"/admin/policies/{body['id']}", json=body)
    r = client.post(f"/admin/policies/{body['id']}/sanitize")
    assert r.status_code == 200, r.text
    out = r.json()
    desc = out["descriptor"]
    for k in INTERNAL_ONLY_FIELDS:
        assert k not in desc
    assert desc["rules_summary"]["banned_token_id_count"] == 5


def test_post_sanitize_unknown_id_returns_404(client: TestClient):
    r = client.post("/admin/policies/no-such/v1/sanitize")
    assert r.status_code == 404


def test_delete_policy(client: TestClient):
    body = make_internal()
    client.put(f"/admin/policies/{body['id']}", json=body)
    r = client.delete(f"/admin/policies/{body['id']}")
    assert r.status_code == 200
    assert r.json()["deleted"] == body["id"]
    # Subsequent GET → 404.
    r = client.get(f"/admin/policies/{body['id']}")
    assert r.status_code == 404


def test_versions_list_and_fetch(client: TestClient):
    body = make_internal()
    r = client.put(f"/admin/policies/{body['id']}", json=body)
    hash_ = r.json()["hash"].split(":", 1)[1]

    r = client.get("/admin/policies/_versions")
    assert r.status_code == 200
    assert hash_ in r.json()["versions"]

    r = client.get(f"/admin/policies/_versions/{hash_}")
    assert r.status_code == 200
    desc = r.json()
    for k in INTERNAL_ONLY_FIELDS:
        assert k not in desc


def test_versions_fetch_invalid_hex(client: TestClient):
    r = client.get("/admin/policies/_versions/not-hex")
    assert r.status_code == 400


def test_get_unknown_policy_returns_404(client: TestClient):
    r = client.get("/admin/policies/no-such/v1")
    assert r.status_code == 404
