"""Aho-Corasick automaton over token-ID sequences.

Standard textbook AC, but the alphabet is integers (token IDs) instead
of characters. Same O(N + matches) streaming complexity. Pure Python,
no external deps — small enough that auditing for adversarial-input
hardening is straightforward.

Why over token IDs and not text:

The banned string `"secret-handshake"` has many valid tokenizations
depending on context (BPE merges differ when the string is at the
start of a sentence vs. mid-word). Matching at the text layer would
require the supervisor to maintain a sliding detok buffer and run
regex over it — exactly the path we're avoiding by staying in token
space. Instead, the operator's policy carries a *list of pre-enumerated
token-ID sequences* per banned pattern (one sequence per valid
tokenization), and this matcher streams over generated token IDs
directly, with no UTF-8 anywhere in the hot path.

Enumeration is offline (operator runs a tokenizer once when authoring
the policy); detection is online and constant-cost-per-token.

Public surface:

  build_aho_corasick(patterns)     — preprocess once
  TokenAhoCorasick.matcher()       — per-stream state holder
  matcher.feed(token_id)           — advance state; yield matches
  matcher.reset()                  — re-arm for a new stream

Each stream gets its own `TokenMatcher` instance so concurrent
generations don't collide. The compiled automaton itself is shared.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence


# ── Internal node ────────────────────────────────────────────────────────────


class _Node:
    """Trie node. Each instance:

    - `goto`: dict[int, _Node]  — direct trie children (deterministic)
    - `fail`: _Node             — longest-proper-suffix link
    - `output`: tuple[int, ...] — pattern indices that complete here OR
                                  at any node reachable via fail-suffix
    """

    __slots__ = ("goto", "fail", "output")

    def __init__(self) -> None:
        self.goto: dict[int, _Node] = {}
        self.fail: _Node | None = None
        self.output: tuple[int, ...] = ()


# ── Match record ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Match:
    """A completed pattern match.

    `pattern_index` is the position of the matching pattern in the
    list passed to `build_aho_corasick`. `end_position` is the
    1-based index of the LAST token of the match in the input stream
    (i.e. the call to `feed()` that completed it returns matches
    whose `end_position` equals the running token count).

    `pattern_length` lets a caller compute `start_position =
    end_position - pattern_length + 1` if it needs span info — useful
    for the policy's `redact` action which must replace the matched
    span with placeholder IDs.
    """

    pattern_index: int
    end_position: int
    pattern_length: int


# ── Build ────────────────────────────────────────────────────────────────────


class TokenAhoCorasick:
    """Compiled multi-pattern matcher over int alphabets.

    Built once per policy load and shared across requests; each
    generation creates its own `TokenMatcher` via `matcher()` so the
    state pointer doesn't race.

    Empty pattern lists produce a degenerate automaton that always
    yields no matches; empty individual patterns (`[]`) are rejected
    at build time because they would match every position.
    """

    __slots__ = ("_root", "_pattern_lengths", "_pattern_count")

    def __init__(self, patterns: Sequence[Sequence[int]]) -> None:
        self._pattern_count = len(patterns)
        self._pattern_lengths: tuple[int, ...] = tuple(len(p) for p in patterns)
        if any(L == 0 for L in self._pattern_lengths):
            raise ValueError(
                "TokenAhoCorasick rejects empty patterns "
                "(would match every position)",
            )
        self._root = _Node()
        if self._pattern_count == 0:
            # Trivial automaton: only the root, no transitions.
            self._root.fail = self._root
            return
        self._build_trie(patterns)
        self._build_fail_links()

    @property
    def pattern_count(self) -> int:
        return self._pattern_count

    def matcher(self) -> "TokenMatcher":
        return TokenMatcher(self)

    # ── Internals ──────────────────────────────────────────────────────────

    def _build_trie(self, patterns: Sequence[Sequence[int]]) -> None:
        """Insert each pattern into the trie, recording the pattern
        index in the leaf node's `output`. Output sets are extended
        later (during fail-link construction) to include matches
        completing at fail-suffix nodes too.
        """
        for idx, pat in enumerate(patterns):
            node = self._root
            for tok in pat:
                tok_int = int(tok)
                child = node.goto.get(tok_int)
                if child is None:
                    child = _Node()
                    node.goto[tok_int] = child
                node = child
            # Multiple patterns may share a leaf (duplicate inputs);
            # accumulate. Tuple keeps it immutable for safety.
            node.output = node.output + (idx,)

    def _build_fail_links(self) -> None:
        """BFS construction of fail links + output extension.

        Root's depth-1 children fail to root. For each non-root node N
        reached via edge `c` from parent P:
          fail(N) = goto*(fail(P), c)
        where `goto*` follows fail links until it finds a child via `c`
        or returns to root. Output set of N then absorbs output of
        fail(N) — that's how a single state captures every pattern
        whose suffix completes here.
        """
        root = self._root
        root.fail = root

        queue: deque[_Node] = deque()
        for child in root.goto.values():
            child.fail = root
            queue.append(child)

        while queue:
            node = queue.popleft()
            for tok, child in node.goto.items():
                # fail link: walk up via fail until we find a state
                # whose goto[tok] exists (or hit root).
                fail_walk = node.fail
                assert fail_walk is not None
                while fail_walk is not root and tok not in fail_walk.goto:
                    fail_walk = fail_walk.fail
                    assert fail_walk is not None
                if tok in fail_walk.goto and fail_walk.goto[tok] is not child:
                    child.fail = fail_walk.goto[tok]
                else:
                    child.fail = root
                # output extension
                child.output = child.output + child.fail.output
                queue.append(child)


# ── Per-stream matcher ───────────────────────────────────────────────────────


class TokenMatcher:
    """Stream-local state pointer.

    Hold one per concurrent generation. `feed(token_id)` advances the
    pointer and returns any matches that completed on this token (most
    feeds return an empty tuple; matches are sparse).
    """

    __slots__ = ("_ac", "_state", "_position")

    def __init__(self, ac: TokenAhoCorasick) -> None:
        self._ac = ac
        self._state = ac._root  # noqa: SLF001 — internal coupling intentional
        self._position = 0

    def reset(self) -> None:
        """Re-arm for a new stream. Idempotent."""
        self._state = self._ac._root  # noqa: SLF001
        self._position = 0

    @property
    def position(self) -> int:
        """Token count fed since construction or last reset."""
        return self._position

    def feed(self, token_id: int) -> tuple[Match, ...]:
        """Advance one token. Return any matches completed on this token.

        Empty automaton (no patterns) returns () unconditionally — the
        per-token cost is still O(1) (one dict lookup against the root)
        so it's cheap to wire in for "maybe later" enforcement paths.
        """
        if self._ac.pattern_count == 0:
            self._position += 1
            return ()

        tok = int(token_id)
        node = self._state
        root = self._ac._root  # noqa: SLF001

        # Walk fail links until we find a state whose goto contains tok
        # — or until we're back at root and there's no transition.
        while node is not root and tok not in node.goto:
            assert node.fail is not None
            node = node.fail
        if tok in node.goto:
            node = node.goto[tok]
        # else: stay at root (no goto for tok from root → no advance)

        self._state = node
        self._position += 1

        if not node.output:
            return ()

        end = self._position
        return tuple(
            Match(
                pattern_index=idx,
                end_position=end,
                pattern_length=self._ac._pattern_lengths[idx],  # noqa: SLF001
            )
            for idx in node.output
        )

    def feed_iter(self, tokens: Iterable[int]) -> tuple[Match, ...]:
        """Convenience: feed many tokens, return concatenated matches.

        Useful for batch tests and for non-streaming consumers that
        want the matches in a single completed sequence.
        """
        out: list[Match] = []
        for t in tokens:
            out.extend(self.feed(t))
        return tuple(out)


# ── Convenience constructor ──────────────────────────────────────────────────


def build_aho_corasick(
    patterns: Sequence[Sequence[int]],
) -> TokenAhoCorasick:
    """Compile a multi-pattern matcher.

    Patterns are sequences of token IDs. Empty individual patterns are
    rejected (would match every position). Duplicate patterns are
    merged at the trie level — each unique pattern appears once in
    output sets per its earliest index.
    """
    return TokenAhoCorasick(patterns)
