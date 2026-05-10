"""Operator-side safety-policy storage, validation, and sanitization.

This is the *internal* full-detail policy file an operator authors and
codec-supervisor reads to drive enforcement. It is a strict superset of the
publishable descriptor at `Project Codec/Codec/spec/safety-policy.schema.json`:
the descriptor is what the world sees at `.well-known/codec/policies/<id>.json`,
this file is what the operator's safety subsystem actually consumes.

Disclosure boundary (load-bearing):

- **Internal-only fields**, NEVER published in the descriptor:
    - `banned_token_ids: list[int]`
    - `regex_patterns: list[str]`
    - `grammar_constraints: list[dict]`
    - `multi_token_patterns: list[dict]`
    - `classifier_internal: dict` (thresholds, weights URLs, fine-tuned cuts)

- **Disclosed fields** (passed through verbatim into the descriptor):
    `id`, `version`, `tokenizers`, `categories`, `category_registry`,
    `classifier.{family, host, requires_engine_features}`, `client_hooks`,
    `published_at`, `publisher`.

- **Derived fields** (computed from internal-only counts):
    `rules_summary.{banned_token_id_count, regex_pattern_count,
    grammar_constraint_count, multi_token_pattern_count}`.

The `sanitize(internal)` function turns one into the other and is the ONLY
sanctioned path from internal to publishable.

Storage: one JSON file per policy id under `<policies_dir>/<safe-id>.json`,
where `<safe-id>` URL-encodes the slash separator (e.g. `acme/strict-v3`
→ `acme%2Fstrict-v3.json`). Versions are content-addressed snapshots
mirroring `<policies_dir>/.versions/<sha256>.json`; the mutable per-id file
is updated atomically (write-temp + rename). This matches the way the
spec's `tokenizer-map.schema.json` thinks about content-addressing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Pydantic models ──────────────────────────────────────────────────────────

ActionLiteral = Literal["stop", "redact", "regenerate", "flag"]
HostLiteral = Literal["server", "client", "both"]
EngineFeatureLiteral = Literal[
    "logits_processor", "hidden_states", "sampling_chain"
]


CATEGORY_NAME_RE = re.compile(r"^[a-z0-9_-]+$")


class Category(BaseModel):
    name: str
    action: ActionLiteral
    description: str | None = None

    @field_validator("name")
    @classmethod
    def _category_name(cls, v: str) -> str:
        if not CATEGORY_NAME_RE.fullmatch(v):
            raise ValueError(
                f"category.name must match {CATEGORY_NAME_RE.pattern!r}; got {v!r}"
            )
        return v


class ClassifierBlock(BaseModel):
    """The disclosed half of the classifier spec."""

    model_config = ConfigDict(extra="forbid")

    family: str = Field(..., min_length=1)
    host: HostLiteral | None = None
    requires_engine_features: list[EngineFeatureLiteral] | None = None


class ClassifierInternalBlock(BaseModel):
    """Operator-internal classifier metadata stripped during sanitize()."""

    model_config = ConfigDict(extra="allow")

    thresholds: dict[str, float] | None = None
    weights_url: str | None = None


class ClientHooksBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prefilter_categories: list[str] | None = None
    client_classifier_family: str | None = None


class PublisherBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    url: str | None = None
    contact: str | None = None


class InternalPolicy(BaseModel):
    """Operator-side full-detail policy. Never published.

    The internal config carries the descriptor's fields plus the
    enforcement contents. `sanitize()` strips the contents to produce
    the publishable descriptor.
    """

    model_config = ConfigDict(extra="forbid")

    # ── Disclosed fields (pass through to the descriptor) ────────────────────
    id: str = Field(..., min_length=1)
    version: str
    tokenizers: list[str] = Field(..., min_length=1)
    categories: list[Category] = Field(..., min_length=1)
    category_registry: str | None = None
    classifier: ClassifierBlock
    client_hooks: ClientHooksBlock | None = None
    published_at: str | None = None
    publisher: PublisherBlock | None = None

    # ── Internal-only fields (stripped during sanitize()) ────────────────────
    banned_token_ids: list[int] | None = None
    regex_patterns: list[str] | None = None
    grammar_constraints: list[dict] | None = None
    multi_token_patterns: list[dict] | None = None
    classifier_internal: ClassifierInternalBlock | None = None


class RulesSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    banned_token_id_count: int | None = None
    regex_pattern_count: int | None = None
    grammar_constraint_count: int | None = None
    multi_token_pattern_count: int | None = None


class PublishedDescriptor(BaseModel):
    """Sanitized descriptor — matches `spec/safety-policy.schema.json` v1.

    This is what gets emitted at `.well-known/codec/policies/<id>.json`.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    tokenizers: list[str]
    categories: list[Category]
    category_registry: str | None = None
    classifier: ClassifierBlock
    rules_summary: RulesSummary | None = None
    client_hooks: ClientHooksBlock | None = None
    published_at: str | None = None
    publisher: PublisherBlock | None = None


# ── Sanitize: internal → publishable ──────────────────────────────────────────

INTERNAL_ONLY_FIELDS = (
    "banned_token_ids",
    "regex_patterns",
    "grammar_constraints",
    "multi_token_patterns",
    "classifier_internal",
)


def sanitize(internal: InternalPolicy) -> PublishedDescriptor:
    """Strip internal-only fields and emit the publishable descriptor.

    - `banned_token_ids`, `regex_patterns`, `grammar_constraints`,
      `multi_token_patterns`, and `classifier_internal` are dropped.
    - Their `len(...)` becomes the corresponding `rules_summary.*_count`.
    - Everything else is passed through verbatim.

    This is the ONLY sanctioned path from internal config to publishable
    descriptor — `admin_api`'s sanitize endpoint and any future CLI invocation
    must funnel through here.
    """
    summary_fields: dict[str, int] = {}
    if internal.banned_token_ids is not None:
        summary_fields["banned_token_id_count"] = len(internal.banned_token_ids)
    if internal.regex_patterns is not None:
        summary_fields["regex_pattern_count"] = len(internal.regex_patterns)
    if internal.grammar_constraints is not None:
        summary_fields["grammar_constraint_count"] = len(internal.grammar_constraints)
    if internal.multi_token_patterns is not None:
        summary_fields["multi_token_pattern_count"] = len(internal.multi_token_patterns)

    summary = RulesSummary(**summary_fields) if summary_fields else None

    return PublishedDescriptor(
        id=internal.id,
        version=internal.version,
        tokenizers=internal.tokenizers,
        categories=internal.categories,
        category_registry=internal.category_registry,
        classifier=internal.classifier,
        rules_summary=summary,
        client_hooks=internal.client_hooks,
        published_at=internal.published_at,
        publisher=internal.publisher,
    )


def descriptor_canonical_bytes(descriptor: PublishedDescriptor) -> bytes:
    """Canonical JSON serialization used for hashing and well-known publish.

    Matches the @codecai/maps-cli format: 2-space indent + trailing newline.
    The bytes that hash to `safety_policy_hash` MUST be exactly these bytes.
    """
    payload = descriptor.model_dump(mode="json", exclude_none=True)
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def descriptor_sha256(descriptor: PublishedDescriptor) -> str:
    return f"sha256:{hashlib.sha256(descriptor_canonical_bytes(descriptor)).hexdigest()}"


# ── Storage layer ────────────────────────────────────────────────────────────

POLICY_ID_RE = re.compile(r"^[a-z0-9._/-]+$")


class PolicyIdError(ValueError):
    pass


def _validate_policy_id(policy_id: str) -> None:
    """Reject policy ids that would either fail well-known publishing or
    let a path-traversal escape `policies_dir`."""
    if not POLICY_ID_RE.fullmatch(policy_id):
        raise PolicyIdError(
            f"policy id {policy_id!r} must match {POLICY_ID_RE.pattern!r}"
        )
    if (
        ".." in policy_id
        or policy_id.startswith("/")
        or policy_id.endswith("/")
        or "//" in policy_id
    ):
        raise PolicyIdError(
            f"policy id {policy_id!r} contains a path-traversal or empty segment"
        )


def policy_filename(policy_id: str) -> str:
    """Map a policy id to its on-disk filename.

    `acme/strict-v3` → `acme%2Fstrict-v3.json`. Encoding the slash means
    every policy lives directly under `<policies_dir>/` — no nested
    directories to traverse, simpler atomicity, and a hash collision in
    the encoded form remains unique because `%2F` is the only character
    we encode.
    """
    _validate_policy_id(policy_id)
    return policy_id.replace("/", "%2F") + ".json"


def policy_path(policies_dir: Path, policy_id: str) -> Path:
    return policies_dir / policy_filename(policy_id)


def version_path(policies_dir: Path, hash_hex: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", hash_hex):
        raise ValueError(f"invalid policy hash hex: {hash_hex!r}")
    return policies_dir / ".versions" / f"{hash_hex}.json"


class PolicyNotFoundError(LookupError):
    pass


def list_policies(policies_dir: Path) -> list[str]:
    """Enumerate policy ids that exist on disk, in alphabetical order."""
    if not policies_dir.is_dir():
        return []
    out: list[str] = []
    for entry in sorted(policies_dir.iterdir()):
        if entry.name.startswith("."):
            continue  # `.versions/` and friends
        if not entry.is_file() or entry.suffix != ".json":
            continue
        decoded = entry.stem.replace("%2F", "/")
        try:
            _validate_policy_id(decoded)
        except PolicyIdError:
            continue
        out.append(decoded)
    return out


def load_policy(policies_dir: Path, policy_id: str) -> InternalPolicy:
    p = policy_path(policies_dir, policy_id)
    if not p.is_file():
        raise PolicyNotFoundError(policy_id)
    raw = json.loads(p.read_text(encoding="utf-8"))
    return InternalPolicy.model_validate(raw)


def save_policy(
    policies_dir: Path,
    internal: InternalPolicy,
) -> tuple[Path, str]:
    """Atomically write an internal policy and snapshot its descriptor hash.

    Returns `(written_path, content_hash)`. The content hash is computed on
    the *sanitized descriptor* (what would be published) so the version
    history is keyed to publishable identity, not operator-internal churn —
    two saves that change a regex but not a category produce different
    on-disk files but identical descriptor hashes, which is what auditors
    and the wire's `safety_policy_hash` care about.
    """
    if internal.id != internal.id.strip() or not internal.id:
        raise ValueError("internal.id must be non-empty and untrimmed")
    _validate_policy_id(internal.id)

    policies_dir.mkdir(parents=True, exist_ok=True)
    versions_dir = policies_dir / ".versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    target = policy_path(policies_dir, internal.id)
    payload = internal.model_dump(mode="json", exclude_none=True)
    raw = (json.dumps(payload, indent=2) + "\n").encode("utf-8")

    descriptor = sanitize(internal)
    desc_hash = descriptor_sha256(descriptor)
    desc_hash_hex = desc_hash.split(":", 1)[1]

    # Atomic write of the mutable per-id file.
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, target)

    # Snapshot the *descriptor* (publishable shape) under .versions/<hash>.json.
    # The descriptor is what `safety_policy_hash` pins on the wire, so the
    # version archive is keyed by descriptor hash, not internal-blob hash.
    snapshot = versions_dir / f"{desc_hash_hex}.json"
    if not snapshot.exists():
        snapshot.write_bytes(descriptor_canonical_bytes(descriptor))

    return target, desc_hash


def list_versions(policies_dir: Path) -> list[str]:
    """Return all archived descriptor hashes, newest-first by mtime."""
    versions_dir = policies_dir / ".versions"
    if not versions_dir.is_dir():
        return []
    entries = [
        (entry.name.removesuffix(".json"), entry.stat().st_mtime)
        for entry in versions_dir.iterdir()
        if entry.is_file() and entry.suffix == ".json"
    ]
    entries.sort(key=lambda t: t[1], reverse=True)
    return [hex_ for hex_, _ in entries]


def load_version(policies_dir: Path, hash_hex: str) -> PublishedDescriptor:
    p = version_path(policies_dir, hash_hex)
    if not p.is_file():
        raise PolicyNotFoundError(f"sha256:{hash_hex}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return PublishedDescriptor.model_validate(raw)


def assert_no_internal_fields_in(payload: dict) -> Iterable[str]:
    """Yield internal-only field names present in `payload`. Empty iterable
    when the payload is already sanitized. Used by the descriptor-validation
    endpoint and by tests to assert sanitize() is the only path to publishable.
    """
    for k in INTERNAL_ONLY_FIELDS:
        if k in payload:
            yield k
