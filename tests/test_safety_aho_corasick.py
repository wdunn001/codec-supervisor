"""Tests for the Aho-Corasick automaton over token IDs."""

from __future__ import annotations

import pytest

from codec_supervisor.safety_aho_corasick import (
    Match,
    TokenAhoCorasick,
    build_aho_corasick,
)


# ── Construction ─────────────────────────────────────────────────────────────


def test_empty_pattern_list_builds_a_trivial_automaton():
    ac = build_aho_corasick([])
    assert ac.pattern_count == 0
    m = ac.matcher()
    # Feeding any sequence into an empty automaton yields nothing.
    out = m.feed_iter([1, 2, 3, 4, 5])
    assert out == ()


def test_empty_individual_pattern_is_rejected():
    with pytest.raises(ValueError):
        build_aho_corasick([[1, 2], []])


# ── Single pattern ──────────────────────────────────────────────────────────


def test_single_pattern_matches_when_completed():
    ac = build_aho_corasick([[7, 8, 9]])
    m = ac.matcher()
    assert m.feed(7) == ()
    assert m.feed(8) == ()
    matches = m.feed(9)
    assert len(matches) == 1
    assert matches[0] == Match(pattern_index=0, end_position=3, pattern_length=3)


def test_single_pattern_does_not_match_partial():
    ac = build_aho_corasick([[7, 8, 9]])
    out = ac.matcher().feed_iter([7, 8, 7, 9])  # broken in the middle
    assert out == ()


# ── Streaming + position ─────────────────────────────────────────────────────


def test_position_increments_with_every_feed():
    ac = build_aho_corasick([[1, 2]])
    m = ac.matcher()
    assert m.position == 0
    m.feed(99)
    assert m.position == 1
    m.feed(99)
    assert m.position == 2


def test_match_end_position_is_token_count():
    """A match completing on the 5th fed token has end_position=5."""
    ac = build_aho_corasick([[3, 4]])
    m = ac.matcher()
    out = m.feed_iter([1, 2, 99, 3, 4])
    assert len(out) == 1
    assert out[0].end_position == 5
    assert out[0].pattern_length == 2


# ── Reset ────────────────────────────────────────────────────────────────────


def test_reset_clears_state_and_position():
    ac = build_aho_corasick([[1, 2, 3]])
    m = ac.matcher()
    m.feed(1)
    m.feed(2)
    m.reset()
    assert m.position == 0
    assert m.feed(3) == ()  # no longer mid-pattern


def test_two_streams_dont_share_state():
    """Per-stream isolation: each matcher() returns a fresh pointer."""
    ac = build_aho_corasick([[1, 2]])
    a, b = ac.matcher(), ac.matcher()
    a.feed(1)
    # b is unaffected — feeding 2 into b alone shouldn't match.
    assert b.feed(2) == ()
    # And a, fed 2 next, MUST match.
    out = a.feed(2)
    assert len(out) == 1
    assert out[0].pattern_index == 0


# ── Multiple patterns + overlap ─────────────────────────────────────────────


def test_multiple_patterns_match_independently():
    ac = build_aho_corasick([[1, 2], [3, 4]])
    out = ac.matcher().feed_iter([1, 2, 0, 3, 4])
    indices = sorted(m.pattern_index for m in out)
    assert indices == [0, 1]


def test_overlapping_patterns_both_fire():
    """When pattern A is a suffix of pattern B (A=[2,3], B=[1,2,3]),
    feeding 1,2,3 fires BOTH on token 3."""
    ac = build_aho_corasick([[2, 3], [1, 2, 3]])
    out = ac.matcher().feed_iter([1, 2, 3])
    indices = sorted(m.pattern_index for m in out)
    assert indices == [0, 1]
    # Both end on position 3.
    assert all(m.end_position == 3 for m in out)


def test_pattern_that_is_prefix_of_another_doesnt_force_extra_match():
    """A=[1,2], B=[1,2,3]. Feeding 1,2 fires A only; feeding 3 fires B."""
    ac = build_aho_corasick([[1, 2], [1, 2, 3]])
    m = ac.matcher()
    a_match = m.feed_iter([1, 2])
    assert [x.pattern_index for x in a_match] == [0]
    b_match = m.feed(3)
    assert [x.pattern_index for x in b_match] == [1]


def test_repeated_pattern_at_two_positions():
    """Same pattern AAA in input AAAA → matches at positions 3 and 4."""
    ac = build_aho_corasick([[1, 1, 1]])
    out = ac.matcher().feed_iter([1, 1, 1, 1])
    ends = sorted(m.end_position for m in out)
    assert ends == [3, 4]


# ── Adversarial inputs ──────────────────────────────────────────────────────


def test_unknown_tokens_dont_advance_state():
    """Feeding tokens that don't match any pattern leaves the matcher
    at root — subsequent valid prefix should still match cleanly."""
    ac = build_aho_corasick([[5, 6]])
    m = ac.matcher()
    out = m.feed_iter([99, 88, 77, 5, 6])
    assert len(out) == 1
    assert out[0].pattern_index == 0
    assert out[0].end_position == 5


def test_long_alphabet_doesnt_break_anything():
    """Sanity: million-ID alphabet works because we use dicts not arrays."""
    ac = build_aho_corasick([[999_999, 1_000_000]])
    out = ac.matcher().feed_iter([999_999, 1_000_000])
    assert len(out) == 1
    assert out[0].pattern_index == 0


# ── Duplicate patterns ──────────────────────────────────────────────────────


def test_duplicate_patterns_dont_cause_double_emit_on_same_token():
    """Inserting the same pattern twice — both indices fire once on the
    completing token (the trie de-duplicates the structure but keeps
    both pattern indices in the leaf's output set)."""
    ac = build_aho_corasick([[7, 8], [7, 8]])
    out = ac.matcher().feed_iter([7, 8])
    indices = sorted(m.pattern_index for m in out)
    assert indices == [0, 1]
    # Both end on the same position with the same length.
    assert all(m.end_position == 2 for m in out)
    assert all(m.pattern_length == 2 for m in out)
