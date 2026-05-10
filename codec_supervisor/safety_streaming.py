"""Streaming safety decision state — SCM-style delay-k token-level scoring.

The Streaming Content Monitor (SCM) paper observed that token-level
safety classifiers with a single-step trigger fire too eagerly: a
classifier's score on token T may spike on a benign word, then the
score on T+1 brings it back down. Hard-stopping on the first crossing
produces too many false positives.

The fix — `delay-k` — is to require **k consecutive tokens** above the
per-category threshold before deciding the stream is unsafe. Lower k
trades latency for false positives; higher k means more confidence
before triggering, at the cost of letting more unsafe tokens through
before the stop.

This module implements the state machine. It's pure-logic — no torch
dependency, no model loading. `EmbeddingSpaceClassifier` (slice 8) and
any future streaming consumer plug into this.

Recommended defaults from the SCM paper for sub-percent FPR at
reasonable detection latency:
  - threshold ≈ 0.5
  - k = 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping


# ── Decisions ────────────────────────────────────────────────────────────────


DecisionKind = Literal["continue", "stop", "flag"]


@dataclass(frozen=True)
class StreamDecision:
    """Result of a single push to the streaming state."""

    kind: DecisionKind
    """`continue` — keep generating; `stop` — terminate the stream now;
    `flag` — k consecutive tokens crossed but the policy bound this
    category to the `flag` action (annotate, don't terminate)."""

    fired_categories: tuple[str, ...] = ()
    """Category names whose score crossed threshold on the triggering
    token. Empty on `continue`."""

    consecutive_unsafe: int = 0
    """How many of the last consecutive tokens have been over threshold
    on at least one fired category. Resets on a clean token."""

    confidence: float = 0.0
    """Max score across the fired categories on the triggering token.
    Useful for telemetry and threshold tuning."""


# ── State machine ────────────────────────────────────────────────────────────


@dataclass
class DelayKDecisionState:
    """Per-stream state for delay-k decisioning.

    Construct one instance per generation request — the state is
    *stream-local* (a previous request's unsafe streak doesn't carry
    over to the next). `push(scores)` is called once per generated
    token; it returns a `StreamDecision`.

    Per-category actions decide what `kind` to emit when the threshold
    is crossed for k consecutive tokens. Common bindings (matching the
    safety-policy schema):
      - `stop`         — kind="stop" (terminate)
      - `redact`       — kind="stop" + caller emits placeholder IDs
      - `regenerate`   — kind="stop" + caller resamples
      - `flag`         — kind="flag" (annotate, keep generating)

    `redact` and `regenerate` map to "stop" at this layer because both
    require terminating the current decode pass; the caller (proxy /
    engine integration) implements the placeholder / resample loop.
    """

    threshold: float = 0.5
    k: int = 3
    """Threshold and consecutive-token requirement. k=1 is single-step
    triggering (eager); k=∞ disables the stop entirely."""

    categories: tuple[str, ...] = field(default_factory=tuple)
    """Categories to monitor. Scores for category names not in this
    tuple are ignored."""

    actions: Mapping[str, str] = field(default_factory=dict)
    """Category name → action (`stop` / `redact` / `regenerate` / `flag`).
    Categories absent from the map default to `stop`."""

    # ── Mutable state (per-stream) ──────────────────────────────────────────
    _consecutive_unsafe: int = 0
    _last_fired: tuple[str, ...] = ()
    _last_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.threshold < 0.0 or self.threshold > 1.0:
            raise ValueError(
                f"threshold must be in [0, 1]; got {self.threshold!r}",
            )
        if self.k < 1:
            raise ValueError(f"k must be ≥ 1; got {self.k!r}")

    def push(self, scores: Mapping[str, float]) -> StreamDecision:
        """Feed one token's per-category scores; return the decision.

        After a `stop`/`flag` decision, the state machine continues to
        track new tokens — callers MAY keep pushing if they want
        post-stop telemetry, but the kind on each subsequent push is
        recomputed independently. Most callers stop pushing once they
        get a stop.
        """
        fired: list[str] = []
        max_score = 0.0
        if self.categories:
            for cat in self.categories:
                s = scores.get(cat, 0.0)
                if s >= self.threshold:
                    fired.append(cat)
                    if s > max_score:
                        max_score = s
        else:
            # No filter list — monitor every category in the score map.
            for cat, s in scores.items():
                if s >= self.threshold:
                    fired.append(cat)
                    if s > max_score:
                        max_score = s

        if not fired:
            self._consecutive_unsafe = 0
            self._last_fired = ()
            self._last_confidence = 0.0
            return StreamDecision(kind="continue")

        self._consecutive_unsafe += 1
        self._last_fired = tuple(fired)
        self._last_confidence = max_score

        if self._consecutive_unsafe < self.k:
            return StreamDecision(
                kind="continue",
                fired_categories=tuple(fired),
                consecutive_unsafe=self._consecutive_unsafe,
                confidence=max_score,
            )

        # Consecutive threshold met — decide kind from per-category actions.
        # If any fired category has a stop-shaped action, emit "stop". If all
        # fired categories are `flag`, emit "flag". This biases toward
        # caution: a single stop-bound category among the fired set wins.
        any_stop = any(
            self.actions.get(c, "stop") in ("stop", "redact", "regenerate")
            for c in fired
        )
        kind: DecisionKind = "stop" if any_stop else "flag"

        return StreamDecision(
            kind=kind,
            fired_categories=tuple(fired),
            consecutive_unsafe=self._consecutive_unsafe,
            confidence=max_score,
        )

    def reset(self) -> None:
        """Reset stream-local state. Call between requests if reusing
        the instance (most callers construct a fresh state per request
        instead — this is here for tests and special integrations)."""
        self._consecutive_unsafe = 0
        self._last_fired = ()
        self._last_confidence = 0.0


# ── Convenience constructor ──────────────────────────────────────────────────


def make_delay_k_state(
    *,
    threshold: float = 0.5,
    k: int = 3,
    categories: Iterable[str] = (),
    actions: Mapping[str, str] | None = None,
) -> DelayKDecisionState:
    """Build a `DelayKDecisionState` with sensible defaults.

    Hosts that already have a policy in hand pass its `categories[].name`
    + `categories[].action` here so the streaming decision matches the
    policy bindings exactly.
    """
    return DelayKDecisionState(
        threshold=threshold,
        k=k,
        categories=tuple(categories),
        actions=dict(actions or {}),
    )
