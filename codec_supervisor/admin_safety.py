"""FastAPI router for `/admin/policies/*` — the operator-side safety admin REST API.

The React admin app (`codec-supervisor/admin/`) hits these endpoints. Every
write goes through `safety.save_policy()`, which:
  1. Validates the input as `InternalPolicy` (Pydantic + `policy_id` rules).
  2. Atomically writes the internal config (write-temp + os.replace).
  3. Snapshots the *sanitized descriptor* under `.versions/<sha256>.json`
     so version history is keyed to publishable identity.

The sanitize endpoint mirrors `@codecai/maps-cli`'s `policies-sanitize`
subcommand bit-for-bit: same field-stripping, same summary-count derivation,
same canonical JSON shape (2-space indent + trailing newline).

Routes:
  GET    /admin/policies                 — list policy ids
  GET    /admin/policies/{id}            — read internal config
  PUT    /admin/policies/{id}            — replace internal config + snapshot
  DELETE /admin/policies/{id}            — drop internal config (versions kept)
  POST   /admin/policies/{id}/sanitize   — emit publishable descriptor + hash
  GET    /admin/policies/_versions       — list archived descriptor hashes
  GET    /admin/policies/_versions/{hex} — read an archived descriptor

Slice 5 ships these endpoints + the editor/categories tabs in the admin
app. The classifier-picker, test-bench, and well-known-generator tabs land
in subsequent slices and reuse this same router.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .safety import (
    InternalPolicy,
    PolicyIdError,
    PolicyNotFoundError,
    PublishedDescriptor,
    descriptor_sha256,
    list_policies,
    list_versions,
    load_policy,
    load_version,
    policy_path,
    sanitize,
    save_policy,
)


class _PoliciesListResponse(BaseModel):
    policies: list[str]


class _SanitizeResponse(BaseModel):
    descriptor: PublishedDescriptor
    hash: str


class _VersionsListResponse(BaseModel):
    versions: list[str]


class _DeleteResponse(BaseModel):
    deleted: str


def create_safety_router(policies_dir: Path) -> APIRouter:
    """Build the `/admin/policies/*` router rooted at `policies_dir`.

    Wired into the FastAPI app by `app.create_app(...)` after the model
    admin endpoints and before the catch-all proxy. The catch-all only
    forwards paths that don't match a registered route, so this stays
    out of the proxy's way.
    """
    router = APIRouter(prefix="/admin/policies", tags=["safety"])

    @router.get("", response_model=_PoliciesListResponse)
    async def get_policies():
        return _PoliciesListResponse(policies=list_policies(policies_dir))

    @router.get("/_versions", response_model=_VersionsListResponse)
    async def get_versions():
        return _VersionsListResponse(versions=list_versions(policies_dir))

    @router.get("/_versions/{hex_hash}", response_model=PublishedDescriptor)
    async def get_version(hex_hash: str):
        try:
            return load_version(policies_dir, hex_hash)
        except PolicyNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @router.get("/{policy_id:path}", response_model=InternalPolicy)
    async def get_policy(policy_id: str):
        try:
            return load_policy(policies_dir, policy_id)
        except PolicyIdError as e:
            raise HTTPException(400, str(e)) from e
        except PolicyNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    @router.put("/{policy_id:path}", response_model=_SanitizeResponse)
    async def put_policy(policy_id: str, body: InternalPolicy):
        # Refuse a body whose embedded id contradicts the URL — surfacing the
        # mismatch at the boundary catches editor bugs before they corrupt
        # storage. (`save_policy` would also reject it via id-validation,
        # but the message here is clearer.)
        if body.id != policy_id:
            raise HTTPException(
                400,
                f"body.id {body.id!r} does not match URL policy id {policy_id!r}",
            )
        try:
            _written, hash_ = save_policy(policies_dir, body)
        except PolicyIdError as e:
            raise HTTPException(400, str(e)) from e
        descriptor = sanitize(body)
        return _SanitizeResponse(descriptor=descriptor, hash=hash_)

    @router.delete("/{policy_id:path}", response_model=_DeleteResponse)
    async def delete_policy(policy_id: str):
        try:
            target = policy_path(policies_dir, policy_id)
        except PolicyIdError as e:
            raise HTTPException(400, str(e)) from e
        if not target.is_file():
            raise HTTPException(404, f"policy {policy_id!r} not found")
        # Note: we keep `.versions/` archives — auditors can still reach
        # the descriptor history of a deleted policy by hash.
        target.unlink()
        return _DeleteResponse(deleted=policy_id)

    @router.post("/{policy_id:path}/sanitize", response_model=_SanitizeResponse)
    async def post_sanitize(policy_id: str):
        try:
            internal = load_policy(policies_dir, policy_id)
        except PolicyIdError as e:
            raise HTTPException(400, str(e)) from e
        except PolicyNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        descriptor = sanitize(internal)
        return _SanitizeResponse(descriptor=descriptor, hash=descriptor_sha256(descriptor))

    return router
