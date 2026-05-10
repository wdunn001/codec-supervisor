"""Tests for codec_supervisor.safety_logits + safety_enforcement.

These exercise the math directly (no torch dep — `scores` is a plain
list of floats). The vLLM signature is `(token_ids, scores) → scores`,
which lists satisfy via `__setitem__` + `__len__`.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from codec_supervisor.safety import InternalPolicy, save_policy
from codec_supervisor.safety_enforcement import (
    SafetyEnforcement,
    TokenizerMismatch,
)
from codec_supervisor.safety_logits import (
    BannedTokenLogitsProcessor,
    compose,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def vocab(n: int) -> list[float]:
    """Build a fresh `scores` list of size n, each entry equal to its index.

    Keeps the test predicates obvious: scores[i] = i unless something
    masked it.
    """
    return [float(i) for i in range(n)]


def make_internal_policy(banned: list[int] | None = None, **overrides) -> InternalPolicy:
    payload: dict = {
        "id": "acme/strict-v3",
        "version": "1",
        "tokenizers": ["meta-llama/llama-3"],
        "categories": [{"name": "secrets", "action": "stop"}],
        "classifier": {"family": "llama-guard-3-1b", "host": "server"},
    }
    if banned is not None:
        payload["banned_token_ids"] = banned
    payload.update(overrides)
    return InternalPolicy.model_validate(payload)


# ── BannedTokenLogitsProcessor ───────────────────────────────────────────────


def test_processor_masks_banned_ids_to_negative_infinity():
    proc = BannedTokenLogitsProcessor([3, 7, 11])
    scores = vocab(16)
    out = proc([], scores)
    assert out is scores  # in-place
    assert out[3] == -math.inf
    assert out[7] == -math.inf
    assert out[11] == -math.inf
    # Untouched entries unchanged.
    for i in (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15):
        assert out[i] == float(i)


def test_processor_with_empty_banned_list_is_a_noop():
    proc = BannedTokenLogitsProcessor([])
    scores = vocab(8)
    out = proc([1, 2, 3], scores)
    for i in range(8):
        assert out[i] == float(i)


def test_processor_silently_skips_out_of_range_ids():
    # Vocab of 8; banning 5000 / -1 / 8 is a no-op (8 is one past last
    # valid index, which is 7).
    proc = BannedTokenLogitsProcessor([5000, -1, 8, 3])
    scores = vocab(8)
    proc([], scores)
    assert scores[3] == -math.inf
    for i in (0, 1, 2, 4, 5, 6, 7):
        assert scores[i] == float(i)


def test_processor_dedupes_and_normalizes_inputs():
    # Iterable + duplicates + non-int-but-int-castable.
    proc = BannedTokenLogitsProcessor((3, 3, 5, 3))
    assert proc.banned_ids == (3, 5)
    assert len(proc) == 2


def test_processor_token_ids_are_ignored():
    """Banned set is constant per policy; the prefix doesn't matter."""
    proc = BannedTokenLogitsProcessor([2])
    scores_a = vocab(4)
    scores_b = vocab(4)
    proc([], scores_a)
    proc([0, 1, 2, 3, 4, 5, 6], scores_b)
    assert scores_a == scores_b


def test_processor_handles_large_banned_set():
    n_vocab = 1024
    banned = list(range(0, n_vocab, 2))  # every even ID
    proc = BannedTokenLogitsProcessor(banned)
    scores = vocab(n_vocab)
    proc([], scores)
    for i in range(n_vocab):
        if i % 2 == 0:
            assert scores[i] == -math.inf, f"id {i} should be masked"
        else:
            assert scores[i] == float(i), f"id {i} should be untouched"


# ── compose() ────────────────────────────────────────────────────────────────


def test_compose_single_returns_input_unchanged():
    proc = BannedTokenLogitsProcessor([1])
    composed = compose(proc)
    assert composed is proc


def test_compose_chains_processors_left_to_right():
    p1 = BannedTokenLogitsProcessor([1])
    p2 = BannedTokenLogitsProcessor([3])
    composed = compose(p1, p2)
    scores = vocab(8)
    composed([], scores)  # type: ignore[operator]
    assert scores[1] == -math.inf
    assert scores[3] == -math.inf
    for i in (0, 2, 4, 5, 6, 7):
        assert scores[i] == float(i)


# ── SafetyEnforcement ────────────────────────────────────────────────────────


def test_enforcement_from_policy_exposes_disclosed_and_internal_fields():
    policy = make_internal_policy(
        banned=[5, 9, 11],
        regex_patterns=["foo"],
        grammar_constraints=[{"name": "no_urls"}],
        multi_token_patterns=[{"literal": "abc"}],
    )
    e = SafetyEnforcement.from_policy(policy)
    assert e.policy_id == policy.id
    assert e.tokenizer_ids == ("meta-llama/llama-3",)
    assert e.banned_token_ids == (5, 9, 11)
    assert e.grammar_specs == ({"name": "no_urls"},)
    assert e.multi_token_patterns == ({"literal": "abc"},)
    assert e.categories == {"secrets": "stop"}


def test_enforcement_make_logits_processor_returns_a_working_processor():
    e = SafetyEnforcement.from_policy(make_internal_policy(banned=[2, 5]))
    proc = e.make_logits_processor()
    scores = vocab(8)
    proc([], scores)
    assert scores[2] == -math.inf
    assert scores[5] == -math.inf
    assert scores[3] == 3.0


def test_enforcement_from_policy_id_loads_from_disk(tmp_path: Path):
    policies_dir = tmp_path / "policies"
    policies_dir.mkdir()
    save_policy(policies_dir, make_internal_policy(banned=[42]))
    e = SafetyEnforcement.from_policy_id(policies_dir, "acme/strict-v3")
    assert e.banned_token_ids == (42,)


def test_enforcement_assert_tokenizer_match():
    e = SafetyEnforcement.from_policy(
        make_internal_policy(tokenizers=["meta-llama/llama-3", "qwen/qwen2"]),
    )
    e.assert_tokenizer_match("meta-llama/llama-3")  # no raise
    e.assert_tokenizer_match("qwen/qwen2")  # no raise
    with pytest.raises(TokenizerMismatch):
        e.assert_tokenizer_match("mistralai/mistral-7b")


def test_enforcement_with_no_internal_payloads_yields_empty_tuples():
    """Policy without banned_token_ids etc. → empty tuples, not None."""
    e = SafetyEnforcement.from_policy(make_internal_policy())
    assert e.banned_token_ids == ()
    assert e.grammar_specs == ()
    assert e.multi_token_patterns == ()
    # Empty processor still callable and a no-op.
    proc = e.make_logits_processor()
    scores = vocab(4)
    proc([], scores)
    assert scores == [0.0, 1.0, 2.0, 3.0]
