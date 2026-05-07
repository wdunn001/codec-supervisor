"""Model registry — manages the on-disk ``models_dir``.

Two ways a model lands here:
  - ``pull_from_hf``: snapshot_download from Hugging Face.
  - ``extract_tarball``: user uploads a .tar / .tar.gz of a model directory.

Sglang's own HF cache (``~/.cache/huggingface``) is separate and untouched —
``initial_model`` and any HF id passed straight to the backend live there.
``models_dir`` is the *explicit* user-managed registry surfaced via /admin.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tarfile
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)

UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024  # 8 MiB


def list_models(models_dir: Path) -> list[dict]:
    if not models_dir.exists():
        return []
    out: list[dict] = []
    for entry in sorted(models_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        try:
            size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
        except OSError:
            size = -1
        out.append(
            {
                "name": entry.name,
                "path": str(entry),
                "size_bytes": size,
            }
        )
    return out


def model_path(models_dir: Path, name: str) -> Path | None:
    """Return the path for a registered local model, or None."""
    if "/" in name or name.startswith(".."):
        return None
    p = models_dir / name
    return p if p.is_dir() else None


async def pull_from_hf(
    repo_id: str,
    models_dir: Path,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download a HF repo snapshot into models_dir/<safe-name>."""
    from huggingface_hub import snapshot_download

    name = repo_id.replace("/", "--")
    if revision:
        name = f"{name}@{revision}"
    target = models_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)

    logger.info("snapshot_download %s -> %s", repo_id, target)

    def _do() -> str:
        return snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(target),
            token=token,
        )

    await asyncio.to_thread(_do)
    return target


async def extract_tarball(
    name: str,
    chunks: AsyncIterator[bytes],
    models_dir: Path,
) -> Path:
    """Stream a tarball into a temp file then extract into models_dir/<name>.

    Path-traversal-safe: rejects members that escape ``target``.
    """
    if "/" in name or name.startswith(".") or name == "":
        raise ValueError("invalid model name (must be a single path component)")

    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / name
    if target.exists():
        raise FileExistsError(f"model {name!r} already exists at {target}")

    tmp = models_dir / f".upload-{name}.tar"
    bytes_written = 0
    try:
        with open(tmp, "wb") as f:
            async for chunk in chunks:
                f.write(chunk)
                bytes_written += len(chunk)
        logger.info("uploaded %d bytes to %s, extracting", bytes_written, tmp)
        await asyncio.to_thread(_safe_extract, tmp, target)
    except Exception:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        if tmp.exists():
            tmp.unlink()
    return target


def _safe_extract(tar_path: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    target_resolved = target.resolve()
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            full = (target / member.name).resolve()
            try:
                full.relative_to(target_resolved)
            except ValueError as e:
                raise ValueError(f"unsafe tar member: {member.name!r}") from e
            if member.issym() or member.islnk():
                # Disallow links to avoid pointing outside target via symlink.
                raise ValueError(f"links not allowed in upload: {member.name!r}")
        tar.extractall(target)


def delete_model(models_dir: Path, name: str) -> Path:
    target = model_path(models_dir, name)
    if target is None:
        raise FileNotFoundError(f"unknown model {name!r}")
    shutil.rmtree(target)
    return target
