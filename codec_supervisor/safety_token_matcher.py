"""Policy-shaped wrapper over `safety_aho_corasick`.

Bridges the policy file's `multi_token_patterns` field (list of dicts)
to the AC automaton (sequences of int sequences) and surfaces a
per-stream matcher that yields policy-meaningful matches (label +
action), not just pattern indices.

Expected policy entry shape:

    {
      "literal":      "<original string for telemetry>",
      "label":        "<optional policy-defined label>",
      "action":       "<stop|redact|regenerate|flag>",
      "tokenizers":   {
        "<tokenizer_id>": [[t_id, t_id, ...], [t_id, ...], ...],
      }
    }

`tokenizers` is the offline-enumerated map: for each tokenizer-id this
policy is bound to, a list of valid token-ID sequences that decode to
the literal. Operators populate this with a tooling step (a script
that loads the HF tokenizer, walks the merge table, and emits every
context-dependent tokenization). We never enumerate at runtime —
that's the whole point of pre-computing.

If `action` is missing, the wrapper defaults to `stop` (cautious). If
`tokenizers` is missing or empty for the engine's loaded tokenizer,
that pattern is silently skipped — a misalignment between policy and
engine is the host's responsibility to fix; the matcher doesn't
silently apply patterns from the wrong vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .safety_aho_corasick import (
    Match as _AcMatch,
    TokenAhoCorasick,
    TokenMatcher as _AcMatcher,
    build_aho_corasick,
)


# ── Public match record ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class PatternMatch:
    """A completed multi-token match annotated with policy context."""

    label: str
    """Policy label (`label` field, falling back to `literal` if unset).
    Used for telemetry and the StreamDecision's fired_categories
    equivalent."""

    action: str
    """`stop` / `redact` / `regenerate` / `flag` — what the caller
    should do. Same semantics as the policy schema's category actions."""

    end_position: int
    """1-based index of the LAST token of the match in the input stream
    (token count at the time `feed()` returned this match)."""

    pattern_length: int
    """How many tokens the match spans. Caller computes
    `start_position = end_position - pattern_length + 1`."""

    literal: str | None = None
    """Original string (`literal` field) if the policy provided one.
    Useful for operator-side telemetry; never sent on the wire."""


# ── Compiled matcher ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _CompiledPattern:
    """Internal record — one entry per (policy-pattern, tokenization)
    pair fed into the AC trie."""

    label: str
    action: str
    literal: str | None


class TokenPatternMatcher:
    """Policy-aware multi-token matcher.

    Construct via `build_token_pattern_matcher(patterns, tokenizer_id)`.
    Per-stream usage:

        m = matcher.stream()
        for tid in generated_token_ids:
            for match in m.feed(tid):
                ...act on match.action...
    """

    __slots__ = ("_ac", "_compiled")

    def __init__(
        self,
        ac: TokenAhoCorasick,
        compiled: tuple[_CompiledPattern, ...],
    ) -> None:
        self._ac = ac
        self._compiled = compiled

    @property
    def pattern_count(self) -> int:
        return self._ac.pattern_count

    def stream(self) -> "TokenPatternStream":
        return TokenPatternStream(self)


class TokenPatternStream:
    """Per-generation stream wrapper. Decorates AC matches with
    policy-level metadata.
    """

    __slots__ = ("_owner", "_inner")

    def __init__(self, owner: TokenPatternMatcher) -> None:
        self._owner = owner
        self._inner = owner._ac.matcher()  # noqa: SLF001

    @property
    def position(self) -> int:
        return self._inner.position

    def reset(self) -> None:
        self._inner.reset()

    def feed(self, token_id: int) -> tuple[PatternMatch, ...]:
        ac_matches = self._inner.feed(token_id)
        if not ac_matches:
            return ()
        return tuple(
            self._decorate(m) for m in ac_matches
        )

    def feed_iter(
        self,
        tokens: Iterable[int],
    ) -> tuple[PatternMatch, ...]:
        out: list[PatternMatch] = []
        for t in tokens:
            out.extend(self.feed(t))
        return tuple(out)

    def _decorate(self, m: _AcMatch) -> PatternMatch:
        c = self._owner._compiled[m.pattern_index]  # noqa: SLF001
        return PatternMatch(
            label=c.label,
            action=c.action,
            end_position=m.end_position,
            pattern_length=m.pattern_length,
            literal=c.literal,
        )


# ── Builder ──────────────────────────────────────────────────────────────────


def build_token_pattern_matcher(
    patterns: Sequence[Mapping],
    tokenizer_id: str,
) -> TokenPatternMatcher:
    """Compile a `TokenPatternMatcher` from a policy's
    `multi_token_patterns` list, restricted to the given
    `tokenizer_id`'s pre-enumerated tokenizations.

    Patterns whose `tokenizers` block doesn't include `tokenizer_id`
    are silently skipped — that's the right behavior when an operator
    has authored cross-tokenizer policies (one per family) and the
    engine has loaded only one of them. The matcher only enforces
    what's actually applicable to the running vocab.

    Returns a matcher with `pattern_count == 0` if no patterns apply.
    The streaming feed against an empty matcher is a constant-time
    no-op, so it's safe to wire unconditionally.
    """
    flat_token_seqs: list[Sequence[int]] = []
    compiled: list[_CompiledPattern] = []

    for entry in patterns:
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"multi_token_patterns entries must be dicts; "
                f"got {type(entry).__name__}",
            )
        tokenizers_block = entry.get("tokenizers") or {}
        if not isinstance(tokenizers_block, Mapping):
            raise TypeError(
                "multi_token_patterns[*].tokenizers must be a dict "
                "of tokenizer-id → list of token-id sequences",
            )
        seqs = tokenizers_block.get(tokenizer_id)
        if not seqs:
            continue
        if not isinstance(seqs, (list, tuple)):
            raise TypeError(
                f"multi_token_patterns[*].tokenizers[{tokenizer_id!r}] "
                "must be a list of token-id sequences",
            )

        action = entry.get("action", "stop")
        if not isinstance(action, str):
            raise TypeError(
                f"multi_token_patterns[*].action must be a string; "
                f"got {type(action).__name__}",
            )
        literal = entry.get("literal")
        if literal is not None and not isinstance(literal, str):
            raise TypeError(
                f"multi_token_patterns[*].literal must be a string; "
                f"got {type(literal).__name__}",
            )
        label_raw = entry.get("label")
        label = (
            label_raw if isinstance(label_raw, str) and label_raw
            else (literal if isinstance(literal, str) else "<unnamed>")
        )

        compiled_entry = _CompiledPattern(
            label=label, action=action, literal=literal,
        )
        for seq in seqs:
            if not isinstance(seq, (list, tuple)):
                raise TypeError(
                    "each tokenization in tokenizers[<id>] must be a "
                    "list/tuple of ints",
                )
            if len(seq) == 0:
                # Skip empty sequences silently — they'd match every
                # position and break everything.
                continue
            flat_token_seqs.append(tuple(int(t) for t in seq))
            compiled.append(compiled_entry)

    ac = build_aho_corasick(flat_token_seqs)
    return TokenPatternMatcher(ac, tuple(compiled))
