"""Token-space safety enforcement primitives.

This is the engine-side companion to `safety.py` (operator-side storage)
and `admin_safety.py` (REST surface). The classes here run *inside the
inference loop*: vLLM (or any other engine that accepts a logits-processor
callable) invokes them on each generation step, the processor masks banned
token IDs to -inf, and sampling is forced away from them.

Design:

- **No torch dependency on codec-supervisor.** Processors are duck-typed
  against the `scores` argument's `__setitem__` and `__len__`. They work
  with torch tensors, numpy arrays, plain Python lists — anything that
  satisfies the protocol. This keeps the supervisor's pyproject.toml
  free of multi-GB ML deps; the actual numeric work happens in the engine.
- **vLLM signature first.** The standard per-request shape vLLM passes
  is `(token_ids: List[int], scores: <tensor>) -> <tensor>`. Our
  processors match that. HF Transformers' `LogitsProcessor` ABC has a
  similar shape and these classes can be wrapped trivially for either.
- **Pure functions over instances.** Each processor instance is
  constructed once per policy load and reused across requests. Mutation
  happens in-place on `scores`; the processor returns the same tensor
  reference (vLLM expects this).

Slice 6 ships the banned-token processor. Slice 8 extends with an
Aho-Corasick token-trie matcher over enumerated tokenizations (multi-
token patterns); slice 9 adds the embedding-space classifier hook.
Grammar / FSM masks are delegated to the engine's existing guided-decoding
machinery (vLLM `GuidedDecodingParams`, llama.cpp grammars) — see
`safety_enforcement.py` for the passthrough that exposes a policy's
`grammar_constraints` for engine-specific wiring.
"""

from __future__ import annotations

import math
from typing import Iterable, Protocol, Sequence


# ── Score-tensor protocol ────────────────────────────────────────────────────


class _MutableScores(Protocol):
    """Minimal interface a processor needs from `scores`.

    torch.Tensor, numpy.ndarray, and `list[float]` all satisfy this.
    """

    def __len__(self) -> int: ...

    def __setitem__(self, index: int, value: float) -> None: ...


# ── Banned-token processor ───────────────────────────────────────────────────


class BannedTokenLogitsProcessor:
    """Mask a fixed set of token IDs by writing -inf into `scores`.

    Compatible with vLLM's per-request logits-processor signature:
    `(token_ids: list[int], scores: <tensor>) -> <tensor>`. The
    `token_ids` argument (the prefix the engine has produced so far) is
    ignored — the banned set is constant per policy.

    Out-of-range IDs are silently skipped. This makes the processor
    safe against a tokenizer-version mismatch (the policy might have
    been authored against a vocab of size 32k while the loaded engine
    uses a 50k vocab; banning ID 35000 there is a no-op rather than a
    crash). Operators detect drift via the `tokenizers` field of the
    policy descriptor — the supervisor refuses to load a policy whose
    tokenizer-id doesn't match the engine's loaded model.

    Constructor accepts any iterable of ints; duplicates are folded.
    Empty bans produce a no-op processor (still valid; vLLM accepts it).
    """

    __slots__ = ("_banned",)

    def __init__(self, banned_ids: Iterable[int]) -> None:
        # Sorted tuple → cache-friendly iteration, deterministic ordering
        # for telemetry, and immutability so the same processor can be
        # shared across requests without re-validation.
        self._banned: tuple[int, ...] = tuple(sorted({int(i) for i in banned_ids}))

    @property
    def banned_ids(self) -> tuple[int, ...]:
        return self._banned

    def __len__(self) -> int:
        return len(self._banned)

    def __repr__(self) -> str:  # pragma: no cover — debug aid only
        return f"BannedTokenLogitsProcessor(n={len(self._banned)})"

    def __call__(
        self,
        token_ids: Sequence[int],  # noqa: ARG002 — vLLM signature
        scores: _MutableScores,
    ) -> _MutableScores:
        if not self._banned:
            return scores
        vocab_size = len(scores)
        neg_inf = -math.inf
        for tid in self._banned:
            if 0 <= tid < vocab_size:
                scores[tid] = neg_inf
        return scores


# ── Composer ─────────────────────────────────────────────────────────────────


def compose(*processors: object) -> object:
    """Chain logits processors. Returns a callable that runs each in order.

    Each processor's output is fed to the next as the `scores` argument.
    Use to combine `BannedTokenLogitsProcessor` with future processors
    (Aho-Corasick token-trie from slice 8, embedding-space scorer from
    slice 9 — once those land).

    Returns the original processor when only one is supplied (cheap
    no-op composition).
    """
    if len(processors) == 1:
        return processors[0]

    def composed(token_ids: Sequence[int], scores: _MutableScores) -> _MutableScores:
        out = scores
        for p in processors:
            out = p(token_ids, out)  # type: ignore[operator]
        return out

    return composed
