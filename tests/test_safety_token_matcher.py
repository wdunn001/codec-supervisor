"""Tests for the policy-shape multi-token matcher wrapper."""

from __future__ import annotations

import pytest

from codec_supervisor.safety import InternalPolicy
from codec_supervisor.safety_enforcement import SafetyEnforcement
from codec_supervisor.safety_token_matcher import (
    PatternMatch,
    build_token_pattern_matcher,
)


TOK_LLAMA = "meta-llama/llama-3"
TOK_QWEN = "qwen/qwen2"


# ── Build behavior ──────────────────────────────────────────────────────────


def test_empty_patterns_yield_zero_pattern_count():
    m = build_token_pattern_matcher([], TOK_LLAMA)
    assert m.pattern_count == 0


def test_pattern_without_matching_tokenizer_is_skipped():
    """A policy entry with no tokenizations under the engine's
    tokenizer-id is silently dropped — that's the right behavior for
    cross-tokenizer policies that publish per-vocab tokenizations."""
    patterns = [
        {
            "literal": "secret",
            "action": "stop",
            "tokenizers": {TOK_QWEN: [[42, 43]]},  # not llama
        },
    ]
    m = build_token_pattern_matcher(patterns, TOK_LLAMA)
    assert m.pattern_count == 0


def test_pattern_with_matching_tokenizer_compiles():
    patterns = [
        {
            "literal": "secret",
            "action": "redact",
            "tokenizers": {TOK_LLAMA: [[42, 43]]},
        },
    ]
    m = build_token_pattern_matcher(patterns, TOK_LLAMA)
    assert m.pattern_count == 1


def test_multiple_tokenizations_for_one_pattern_all_compile():
    """One literal with two valid tokenizations → both routes hit the
    same compiled pattern (label/action/literal) when matched."""
    patterns = [
        {
            "literal": "secret",
            "action": "redact",
            "tokenizers": {
                TOK_LLAMA: [[42, 43], [99, 100, 101]],
            },
        },
    ]
    m = build_token_pattern_matcher(patterns, TOK_LLAMA)
    assert m.pattern_count == 2  # two routes through the trie

    s = m.stream()
    a = s.feed_iter([42, 43])
    s.reset()
    b = s.feed_iter([99, 100, 101])
    assert len(a) == 1
    assert len(b) == 1
    assert a[0].literal == "secret"
    assert b[0].literal == "secret"
    assert a[0].action == b[0].action == "redact"


def test_default_action_is_stop():
    """Missing `action` defaults to stop (cautious default)."""
    patterns = [
        {"tokenizers": {TOK_LLAMA: [[1, 2]]}},  # no action, no label, no literal
    ]
    m = build_token_pattern_matcher(patterns, TOK_LLAMA)
    s = m.stream()
    out = s.feed_iter([1, 2])
    assert len(out) == 1
    assert out[0].action == "stop"
    assert out[0].label == "<unnamed>"
    assert out[0].literal is None


def test_label_falls_back_to_literal_when_unset():
    patterns = [
        {
            "literal": "naming-myself",
            "tokenizers": {TOK_LLAMA: [[1, 2]]},
        },
    ]
    m = build_token_pattern_matcher(patterns, TOK_LLAMA)
    s = m.stream()
    out = s.feed_iter([1, 2])
    assert out[0].label == "naming-myself"


def test_label_overrides_literal_when_both_present():
    patterns = [
        {
            "literal": "secret-string",
            "label": "secrets",
            "tokenizers": {TOK_LLAMA: [[1, 2]]},
        },
    ]
    m = build_token_pattern_matcher(patterns, TOK_LLAMA)
    s = m.stream()
    out = s.feed_iter([1, 2])
    assert out[0].label == "secrets"
    assert out[0].literal == "secret-string"


# ── Validation ──────────────────────────────────────────────────────────────


def test_non_dict_entry_raises():
    with pytest.raises(TypeError):
        build_token_pattern_matcher(["not a dict"], TOK_LLAMA)  # type: ignore[list-item]


def test_non_dict_tokenizers_block_raises():
    with pytest.raises(TypeError):
        build_token_pattern_matcher(
            [{"tokenizers": "not a dict"}],  # type: ignore[dict-item]
            TOK_LLAMA,
        )


def test_non_list_tokenization_raises():
    with pytest.raises(TypeError):
        build_token_pattern_matcher(
            [{"tokenizers": {TOK_LLAMA: "not a list"}}],  # type: ignore[dict-item]
            TOK_LLAMA,
        )


def test_non_string_action_raises():
    with pytest.raises(TypeError):
        build_token_pattern_matcher(
            [{"action": 1, "tokenizers": {TOK_LLAMA: [[1]]}}],  # type: ignore[dict-item]
            TOK_LLAMA,
        )


def test_empty_individual_tokenization_is_silently_skipped():
    """An empty token-id sequence inside the list is dropped (would
    match every position) but doesn't blow up the build."""
    patterns = [
        {"literal": "x", "tokenizers": {TOK_LLAMA: [[], [1, 2]]}},
    ]
    m = build_token_pattern_matcher(patterns, TOK_LLAMA)
    assert m.pattern_count == 1


# ── Streaming behavior ──────────────────────────────────────────────────────


def test_streaming_position_increments():
    patterns = [{"tokenizers": {TOK_LLAMA: [[1, 2]]}}]
    m = build_token_pattern_matcher(patterns, TOK_LLAMA)
    s = m.stream()
    s.feed_iter([99, 88, 77])
    assert s.position == 3


def test_match_carries_end_position_and_length():
    patterns = [{"tokenizers": {TOK_LLAMA: [[10, 20, 30]]}}]
    m = build_token_pattern_matcher(patterns, TOK_LLAMA)
    s = m.stream()
    out = s.feed_iter([99, 10, 20, 30])
    assert len(out) == 1
    assert out[0].end_position == 4
    assert out[0].pattern_length == 3


# ── SafetyEnforcement integration ────────────────────────────────────────────


def make_internal(**overrides):
    payload: dict = {
        "id": "acme/strict-v3",
        "version": "1",
        "tokenizers": ["meta-llama/llama-3"],
        "categories": [{"name": "secrets", "action": "stop"}],
        "classifier": {"family": "llama-guard-3-1b", "host": "server"},
    }
    payload.update(overrides)
    return InternalPolicy.model_validate(payload)


def test_enforcement_make_token_matcher_skips_when_no_patterns():
    policy = make_internal()
    e = SafetyEnforcement.from_policy(policy)
    m = e.make_token_matcher(TOK_LLAMA)
    assert m.pattern_count == 0
    s = m.stream()
    # Empty matcher is a constant-time no-op — safe to feed unconditionally.
    assert s.feed_iter([1, 2, 3]) == ()


def test_enforcement_make_token_matcher_compiles_patterns():
    policy = make_internal(
        multi_token_patterns=[
            {
                "literal": "exfiltrate-this",
                "label": "secrets",
                "action": "redact",
                "tokenizers": {TOK_LLAMA: [[5, 6, 7]]},
            },
        ],
    )
    e = SafetyEnforcement.from_policy(policy)
    m = e.make_token_matcher(TOK_LLAMA)
    assert m.pattern_count == 1
    s = m.stream()
    out = s.feed_iter([99, 5, 6, 7])
    assert len(out) == 1
    assert out[0] == PatternMatch(
        label="secrets",
        action="redact",
        end_position=4,
        pattern_length=3,
        literal="exfiltrate-this",
    )


def test_enforcement_make_token_matcher_per_stream_isolation():
    """Two concurrent streams off the same compiled matcher don't
    share state — slice 9 explicitly designed for per-request reuse."""
    policy = make_internal(
        multi_token_patterns=[
            {"literal": "x", "tokenizers": {TOK_LLAMA: [[1, 2]]}},
        ],
    )
    e = SafetyEnforcement.from_policy(policy)
    m = e.make_token_matcher(TOK_LLAMA)
    s1, s2 = m.stream(), m.stream()
    s1.feed(1)
    # s2 alone fed [2] shouldn't match — its state is independent.
    assert s2.feed(2) == ()
    # s1 next fed 2 MUST match.
    out = s1.feed(2)
    assert len(out) == 1
