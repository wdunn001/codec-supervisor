"""`EmbeddingSpaceClassifier` — token-embedding-space safety scoring.

The novel piece of the safety architecture (Layer 3 done right). Where
Llama Guard / ShieldGemma operate on detokenized text inside a private
buffer, this classifier reads the engine's hidden states *directly* and
scores per-token without touching UTF-8 anywhere in the pipeline. Pure
token-space enforcement.

Two integration patterns:

1. **Per-request (single-shot).** Host calls `score()` with a complete
   sequence of token embeddings (one vector per generated token). Used
   when an offline / batch path can grab embeddings for a finished
   completion before deciding whether to deliver it. Returns a single
   `ClassificationResult` — average across the sequence.

2. **Streaming (per-token).** Host calls `make_streaming_state()` to
   get a `DelayKDecisionState`, then pushes per-token scores as the
   engine emits hidden states. The state machine's delay-k logic
   (slice 8 / SCM paper) decides when to fire a stop. Highest fidelity
   path; needs the engine to emit hidden states per generation step.

Engine-side hookup is per-fork. vLLM doesn't expose hidden states on
its public API today; an operator who wants this layer wires up a
callback inside their fork (see plan slice 8 — "engine hidden-state
hook (vLLM first)") and feeds the resulting tensor to `score()` /
`push()`. Until that hookup exists, `capability()` reports "no
embedding provider configured" and the registry transparently falls
back to text-based classifiers on resolve.

The classifier head itself (BERT-tier, ~100M params) is trained from
the corpus produced by `training/bootstrap_corpus.py` against the
existing text-space classifiers — a distillation step. This module
ships the runtime; the trainer is a separate offline process.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Mapping, Sequence

from ..safety_classifier import (
    ClassificationInput,
    ClassificationResult,
    ClassifierInputForm,
    RegistryEntry,
    maybe_await,
    register,
)
from ..safety_streaming import DelayKDecisionState, make_delay_k_state


DEFAULT_CATEGORIES: tuple[str, ...] = (
    # Mirror the Llama Guard taxonomy by default — the most likely
    # distillation target. Operators training a head against a different
    # taxonomy override this at construction.
    "violent_crimes",
    "non_violent_crimes",
    "sex_crimes",
    "child_exploitation",
    "defamation",
    "specialized_advice",
    "privacy",
    "intellectual_property",
    "indiscriminate_weapons",
    "hate",
    "self_harm",
    "sexual_content",
    "elections",
    "code_interpreter_abuse",
)


# Scorer signature: a single token's embedding (Sequence[float]) →
# per-category scores (Mapping[str, float] in [0, 1]). MAY be sync or
# async; the classifier normalizes via `maybe_await`.
EmbeddingScorer = Callable[
    [Sequence[float]],
    Awaitable[Mapping[str, float]] | Mapping[str, float],
]


class EmbeddingSpaceClassifier:
    """Token-embedding-space safety classifier.

    Capability fails by default: instantiating without an injected
    `scorer` means the runtime can't score, so `capability()` returns
    a non-None reason and the registry falls back to a text classifier.
    Operators who've wired up an engine hidden-state hook re-register
    with `scorer=` to enable the layer.

    Per-token scores are aggregated into a single ClassificationResult
    via max-over-tokens (any token crossing threshold is "the
    classification" for the sequence) — this matches the streaming
    semantics where a single token spike is what fires the stop. For
    streaming consumers, prefer `make_streaming_state()` and feed
    `push()` per token directly.
    """

    MODEL_ID: str = "embedding-space-v1"

    def __init__(
        self,
        *,
        scorer: EmbeddingScorer | None = None,
        categories: tuple[str, ...] = DEFAULT_CATEGORIES,
        delay_k: int = 3,
        threshold: float = 0.5,
        head_weights_path: str | None = None,
    ) -> None:
        self._scorer = scorer
        self._categories = categories
        self._delay_k = delay_k
        self._threshold = threshold
        self._head_weights_path = head_weights_path
        self._loaded = scorer is not None
        self._load_lock = asyncio.Lock()

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    @property
    def requires(self) -> ClassifierInputForm:
        return "embeddings"

    @property
    def categories(self) -> tuple[str, ...]:
        return self._categories

    @property
    def delay_k(self) -> int:
        return self._delay_k

    @property
    def threshold(self) -> float:
        return self._threshold

    async def capability(self) -> str | None:
        # Without an injected scorer AND without trained head weights
        # on disk, this classifier can't score. The fallback is the
        # explicit signal — registry's resolve() falls through to a
        # text classifier. Operators flip the bit by re-registering
        # with `scorer=` once the engine hook is wired.
        if self._scorer is not None:
            return None
        if self._head_weights_path is None:
            return (
                "no embedding provider configured "
                "(re-register with scorer= or head_weights_path=)"
            )
        # Path provided but no scorer — the default factory needs torch
        # + transformers to load the head. Capability reflects what's
        # importable.
        try:
            import torch  # noqa: F401
        except ImportError as e:
            return f"torch not installed: {e}"
        return None

    async def load(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            self._scorer = await self._build_default_scorer()
            self._loaded = True

    async def unload(self) -> None:
        self._scorer = None
        self._loaded = False

    async def score(self, input: ClassificationInput) -> ClassificationResult:
        if input.form != "embeddings":
            raise ValueError(
                f"EmbeddingSpaceClassifier only accepts form='embeddings'; "
                f"got form={input.form!r}",
            )
        if not isinstance(input.payload, (list, tuple)):
            raise ValueError(
                "EmbeddingSpaceClassifier: embeddings payload must be a "
                "sequence (1D vector for a single token, or 2D for a "
                "token sequence)",
            )

        await self.load()
        assert self._scorer is not None

        # Detect single-token vs multi-token. A nested sequence (list of
        # vectors) → multi-token; a flat list of floats → single token.
        is_multi = (
            len(input.payload) > 0
            and isinstance(input.payload[0], (list, tuple))
        )

        if is_multi:
            per_token = [
                await maybe_await(self._scorer(tok))  # type: ignore[arg-type]
                for tok in input.payload
            ]
            # Aggregate via max-over-tokens. Streaming consumers prefer
            # per-token decisions via DelayKDecisionState; this
            # aggregation is the single-shot fallback.
            scores: dict[str, float] = {c: 0.0 for c in self._categories}
            for tok_scores in per_token:
                tok_scores_typed = self._validate_scores(tok_scores)
                for cat, val in tok_scores_typed.items():
                    if cat in scores and val > scores[cat]:
                        scores[cat] = val
            return ClassificationResult(
                scores=scores,
                raw={"per_token": [dict(s) for s in per_token]},
            )

        # Single token.
        result = await maybe_await(self._scorer(input.payload))  # type: ignore[arg-type]
        scores = dict(self._validate_scores(result))
        # Fill in zeros for unscored categories so the shape is uniform.
        for cat in self._categories:
            scores.setdefault(cat, 0.0)
        return ClassificationResult(scores=scores)

    def make_streaming_state(
        self,
        *,
        actions: Mapping[str, str] | None = None,
    ) -> DelayKDecisionState:
        """Build a fresh `DelayKDecisionState` pre-bound to this
        classifier's threshold, k, and categories. Use it to drive a
        per-token streaming decision over the engine's hidden states.
        """
        return make_delay_k_state(
            threshold=self._threshold,
            k=self._delay_k,
            categories=self._categories,
            actions=actions,
        )

    # -------------------------------------------------------------- helpers

    def _validate_scores(self, raw: object) -> Mapping[str, float]:
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"EmbeddingSpaceClassifier scorer returned "
                f"{type(raw).__name__}; expected Mapping[str, float]",
            )
        out: dict[str, float] = {}
        for k, v in raw.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"scorer key {k!r} is not a string",
                )
            f = float(v)
            if not 0.0 <= f <= 1.0:
                raise ValueError(
                    f"scorer score for {k!r} is {f!r}; expected [0, 1]",
                )
            out[k] = f
        return out

    async def _build_default_scorer(self) -> EmbeddingScorer:
        # Default factory loads a small classification head from
        # `head_weights_path` and runs forward inference per token.
        # The actual head architecture is intentionally unspecified
        # here — operators training their own head pass it in via the
        # weights path; this module's job is the runtime, not a fixed
        # model design.
        if self._head_weights_path is None:
            raise RuntimeError(
                "EmbeddingSpaceClassifier default scorer requires "
                "head_weights_path. Either pass scorer= directly or "
                "supply trained head weights from "
                "`training/bootstrap_corpus.py` + a downstream trainer.",
            )
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                f"EmbeddingSpaceClassifier default scorer needs torch: {e}",
            ) from e

        # Loading the head is operator-defined — this is a stub that
        # raises a clear "not yet implemented" so an operator who only
        # has a path but no loader knows where to plug in.
        path = self._head_weights_path
        raise NotImplementedError(
            f"Load your head implementation here for {path!r}. "
            "EmbeddingSpaceClassifier ships a runtime; the head architecture "
            "is operator-chosen. Pass scorer= directly to bypass.",
        )


# ── Registration ─────────────────────────────────────────────────────────────


def register_embedding_space(
    *,
    scorer: EmbeddingScorer | None = None,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    delay_k: int = 3,
    threshold: float = 0.5,
    head_weights_path: str | None = None,
) -> None:
    """Add the embedding-space classifier to the registry. Idempotent.

    Default registration ships a capability-failing instance — the
    registry's `resolve_classifier()` then falls back to text
    classifiers automatically. Operators who've wired up an engine
    hidden-state hook re-register with `scorer=` to enable real
    enforcement; calling this with a `scorer` after the no-op
    registration is a no-op (the first registration wins, mirroring
    the browser-side semantics).
    """
    from ..safety_classifier import has_classifier

    if has_classifier(EmbeddingSpaceClassifier.MODEL_ID):
        return

    register(
        RegistryEntry(
            model_id=EmbeddingSpaceClassifier.MODEL_ID,
            factory=lambda: EmbeddingSpaceClassifier(
                scorer=scorer,
                categories=categories,
                delay_k=delay_k,
                threshold=threshold,
                head_weights_path=head_weights_path,
            ),
            tier=0,  # tier 0 — lowest cost when wired (per-token, no detok)
            description=(
                "Embedding-space classifier — pure token-space safety "
                "scoring via engine hidden states. Tier 0 when configured; "
                "capability-fails by default and falls back to text "
                "classifiers on resolve."
            ),
            advertised_categories=categories,
            advertised_requires="embeddings",
        ),
    )
