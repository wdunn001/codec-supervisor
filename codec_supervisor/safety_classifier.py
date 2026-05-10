"""Server-side SafetyClassifier Protocol + registry.

Python twin of the browser-side `@codecai/web-safety` interface
(`packages/web-safety/src/base.ts` + `registry.ts`). Same shape so a
policy descriptor's `classifier.family` value resolves to compatible
implementations on either side; admin UIs and integration code think
of "the classifier picker" as one concept.

Design notes (matching the browser side intentionally):

- **Lazy-loaded.** Each implementation defers heavy work (transformers,
  torch, GPU init) until first `load()` or `score()`. The registry
  caches one instance per model_id.
- **Generator-DI for tests.** Implementations accept an optional
  injected `generator` callable so tests stub model output without
  pulling a multi-GB checkpoint.
- **Capability-driven fallback.** `resolve_classifier(id)` calls
  `capability()` first; on failure (no torch, no GPU, missing weights)
  it falls back to the lowest-tier capable alternative and reports
  `downgraded=True` so the host can surface a "downgraded
  enforcement" badge.
- **Tiers**: 1 = always-on / cheap (CPU-OK, small); 2 = opt-in / heavy
  (GPU-needed, large). Mirrors the browser's tier-1 (Prompt Guard 86M)
  vs tier-2 (Llama Guard 3 1B) split — slightly different model
  families since the server doesn't have WebGPU constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, Mapping, Protocol, runtime_checkable


# ── Protocol ─────────────────────────────────────────────────────────────────


ClassifierInputForm = Literal["text", "embeddings"]


@dataclass(frozen=True)
class ClassificationInput:
    """What a classifier consumes per scoring call.

    `form` MUST equal the classifier's `requires`. For `form="text"`
    `payload` is a string (a prompt or detok'd window); for
    `form="embeddings"` it's a flat list/sequence of floats from the
    engine's hidden states.
    """

    form: ClassifierInputForm
    payload: object  # str | Sequence[float] — runtime-checked per impl


@dataclass(frozen=True)
class ClassificationResult:
    """Per-category scores in [0, 1]. Categories absent from `scores`
    SHOULD be treated as 0 by the host. `raw` is implementation-specific
    and exposed for debug/telemetry only — host code MUST NOT depend on
    its shape.
    """

    scores: Mapping[str, float]
    raw: object | None = None


@runtime_checkable
class SafetyClassifier(Protocol):
    """The contract every server-side classifier honors. Same shape as
    the browser's `SafetyClassifier` interface.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def requires(self) -> ClassifierInputForm: ...

    @property
    def categories(self) -> tuple[str, ...]: ...

    async def score(self, input: ClassificationInput) -> ClassificationResult: ...

    async def load(self) -> None: ...

    async def unload(self) -> None: ...

    async def capability(self) -> str | None:
        """Return None if this classifier can run here; a human-readable
        reason string when it can't (missing torch, no CUDA on a model
        that needs it, etc.). The registry uses the return value for
        fallback decisions in `resolve_classifier`.
        """
        ...


# ── Registry ─────────────────────────────────────────────────────────────────


ClassifierFactory = Callable[[], SafetyClassifier]


@dataclass(frozen=True)
class RegistryEntry:
    model_id: str
    factory: ClassifierFactory
    tier: int
    description: str | None = None
    # Categories surfaced for admin-UI picker without instantiating; see
    # `list_registered()`. Implementations MUST keep this in sync with
    # their instance's `categories` property.
    advertised_categories: tuple[str, ...] = field(default_factory=tuple)
    advertised_requires: ClassifierInputForm = "text"


_REGISTRY: dict[str, RegistryEntry] = {}
_INSTANCE_CACHE: dict[str, SafetyClassifier] = {}


class RegistryError(ValueError):
    pass


def register(entry: RegistryEntry) -> None:
    """Append-only registration. Re-registering the same `model_id`
    raises — replicate the browser-side semantics so accidental
    duplicate-registration in long-running processes is loud.
    """
    if entry.model_id in _REGISTRY:
        raise RegistryError(
            f"classifier {entry.model_id!r} is already registered",
        )
    _REGISTRY[entry.model_id] = entry


def unregister(model_id: str) -> bool:
    """Drop a classifier and its cached instance. Returns True when
    something was removed. Hosts that want to release weights / GPU
    memory should `await classifier.unload()` first; the registry
    can't because unregistration must stay synchronous for predictable
    UI toggling.
    """
    had = model_id in _REGISTRY
    _REGISTRY.pop(model_id, None)
    _INSTANCE_CACHE.pop(model_id, None)
    return had


def _reset_for_test() -> None:
    """Test-only: drop every registration."""
    _REGISTRY.clear()
    _INSTANCE_CACHE.clear()


def has_classifier(model_id: str) -> bool:
    return model_id in _REGISTRY


def list_registered() -> list[RegistryEntry]:
    """Return registry entries sorted by tier ascending, then model_id.
    Stable ordering — admin UI listings don't reshuffle on re-render.
    """
    return sorted(
        _REGISTRY.values(),
        key=lambda e: (e.tier, e.model_id),
    )


@dataclass(frozen=True)
class ResolveResult:
    classifier: SafetyClassifier
    downgraded: bool
    requested_model_id: str | None = None
    downgrade_reason: str | None = None


async def resolve_classifier(
    model_id: str,
    *,
    allow_fallback: bool = True,
) -> ResolveResult:
    """Return a ready-to-use classifier instance, falling back to a
    lower-tier alternative when the requested one isn't capable.

    Mirrors `@codecai/web-safety`'s `resolveClassifier` exactly: load on
    first capable hit, surface `downgraded=True` when fallback fires
    (so the host can render a badge), refuse with `RegistryError` when
    nothing in the registry is capable.
    """
    requested = _REGISTRY.get(model_id)
    if requested is None:
        if not allow_fallback:
            raise RegistryError(f"no classifier registered for {model_id!r}")
        return await _fallback_or_raise(
            model_id,
            f"no classifier registered for {model_id!r}",
        )

    candidate = await _instantiate(requested)
    reason = await _safe_capability(candidate)
    if reason is None:
        await candidate.load()
        return ResolveResult(classifier=candidate, downgraded=False)

    if not allow_fallback:
        raise RegistryError(
            f"{model_id!r} not supported here: {reason}",
        )
    return await _fallback_or_raise(model_id, reason)


async def _fallback_or_raise(
    requested_id: str,
    reason: str,
) -> ResolveResult:
    candidates = [e for e in list_registered() if e.model_id != requested_id]
    for entry in candidates:
        inst = await _instantiate(entry)
        r = await _safe_capability(inst)
        if r is None:
            await inst.load()
            return ResolveResult(
                classifier=inst,
                downgraded=True,
                requested_model_id=requested_id,
                downgrade_reason=reason,
            )
    raise RegistryError(
        f"no capable classifier available "
        f"(requested {requested_id!r}: {reason})",
    )


async def _instantiate(entry: RegistryEntry) -> SafetyClassifier:
    inst = _INSTANCE_CACHE.get(entry.model_id)
    if inst is None:
        inst = entry.factory()
        _INSTANCE_CACHE[entry.model_id] = inst
    return inst


async def _safe_capability(c: SafetyClassifier) -> str | None:
    """Wrap `capability()` so that a buggy implementation that throws
    is treated as 'not capable here' rather than crashing the resolve
    path. Bugs upstream still surface in logs (capability() exceptions
    are re-raised in tests via direct call), but at runtime a misbehaving
    classifier degrades gracefully.
    """
    try:
        return await c.capability()
    except Exception as e:  # noqa: BLE001 — see docstring
        return f"capability check raised: {e}"


# ── Async-vs-sync helper ─────────────────────────────────────────────────────
#
# Implementations are expected async. `Generator` callables (used in tests
# for DI) MAY be sync or async — the helper below normalizes them.


SyncOrAsync = Callable[..., object] | Callable[..., Awaitable[object]]


async def maybe_await(value: object) -> object:
    """Await `value` if it's awaitable, else return it directly."""
    if hasattr(value, "__await__"):
        return await value  # type: ignore[no-any-return]
    return value
