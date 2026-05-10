"""Adversarial-defense tests — slice 10.

For each attack class, two assertions:
  1. Naive enforcement (one layer alone) MISSES the attack.
  2. The full stack with the slice-10 helpers wired CATCHES it.

This is the slice that proves layer responsibility: AC catches
enumerated TokenBreak variants, the classifier catches EchoGram, the
banned-id list extended with glitch tokens catches glitch-token
attacks. Each layer earns its place.
"""

from __future__ import annotations

import math

import pytest

from codec_supervisor.safety import InternalPolicy
from codec_supervisor.safety_adversarial import (
    enumerate_text_variants,
    merge_glitch_tokens,
    well_known_glitch_tokens,
)
from codec_supervisor.safety_classifier import (
    ClassificationInput,
    _reset_for_test,
)
from codec_supervisor.safety_classifiers.llamaguard_3_1b import LlamaGuard31B
from codec_supervisor.safety_enforcement import SafetyEnforcement
from codec_supervisor.safety_logits import BannedTokenLogitsProcessor

from tests.fixtures.adversarial import (
    ECHOGRAM_FIXTURES,
    GLITCH_FIXTURES,
    SECRET_CANONICAL,
    TOKEN_BREAK_FIXTURES,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    _reset_for_test()
    yield
    _reset_for_test()


# ── enumerate_text_variants behavior ─────────────────────────────────────────


def test_enumerate_text_variants_includes_original():
    out = enumerate_text_variants("instructions")
    assert out[0] == "instructions"


def test_enumerate_text_variants_yields_known_tokenbreak_prefixes():
    out = enumerate_text_variants("instructions")
    assert "fininstructions" in out
    assert "ininstructions" in out
    assert "preinstructions" in out


def test_enumerate_text_variants_includes_case_and_suffix_variants():
    out = enumerate_text_variants("Instructions")
    assert "instructions" in out
    assert "INSTRUCTIONS" in out
    assert "Instructionss" in out  # canonical + 's' suffix


def test_enumerate_text_variants_dedupes():
    out = enumerate_text_variants("a")  # short literal hits many duplicates
    assert len(out) == len(set(out))


def test_enumerate_text_variants_empty_literal_returns_nothing():
    assert enumerate_text_variants("") == ()


def test_enumerate_text_variants_can_disable_case_variants():
    out = enumerate_text_variants("Foo", include_case_variants=False)
    assert "Foo" in out
    assert "foo" not in out
    assert "FOO" not in out


# ── well_known_glitch_tokens / merge_glitch_tokens ──────────────────────────


def test_well_known_glitch_tokens_returns_empty_for_unknown_tokenizer():
    """Critically, NOT a KeyError — callers wire it in unconditionally."""
    assert well_known_glitch_tokens("not-a-real-tokenizer") == ()


def test_well_known_glitch_tokens_returns_known_tokenizer_list():
    # The shipped registry has known-tokenizer keys with empty tuples
    # (no real curation done by us — operators populate).
    assert well_known_glitch_tokens("meta-llama/llama-3") == ()


def test_merge_glitch_tokens_merges_dedupes_sorts():
    merged = merge_glitch_tokens([5, 3, 7], [3, 9, 7, 1])
    assert merged == (1, 3, 5, 7, 9)


def test_merge_glitch_tokens_handles_none_base():
    assert merge_glitch_tokens(None, [4, 2, 6]) == (2, 4, 6)


def test_merge_glitch_tokens_handles_empty_glitch_list():
    assert merge_glitch_tokens([3, 1, 2], ()) == (1, 2, 3)


# ── TokenBreak: naive vs defended ───────────────────────────────────────────


@pytest.mark.parametrize("fixture", TOKEN_BREAK_FIXTURES, ids=lambda f: f.name)
def test_tokenbreak_naive_banned_ids_alone_miss(fixture):
    """Banning the canonical token IDs misses the prefixed variant —
    the attacker's tokens are entirely different IDs."""
    proc = BannedTokenLogitsProcessor(fixture.canonical_tokens)
    # Simulate a small vocab of size 256 with scores starting at index value.
    scores = [float(i) for i in range(256)]
    proc([], scores)
    # The attacker's tokens are NOT masked — they're not in the banned list.
    for tid in fixture.attack_tokens:
        assert scores[tid] != -math.inf, (
            f"naive enforcement should NOT mask {tid:#x} "
            "(it's not in the banned canonical list)"
        )


@pytest.mark.parametrize("fixture", TOKEN_BREAK_FIXTURES, ids=lambda f: f.name)
def test_tokenbreak_ac_with_enumerated_variants_catches(fixture):
    """Once the operator has enumerated the prefix variant offline and
    written its tokenization into multi_token_patterns, slice 9's AC
    matcher catches the attack stream cleanly."""
    policy = InternalPolicy.model_validate({
        "id": "acme/strict-v3",
        "version": "1",
        "tokenizers": ["meta-llama/llama-3"],
        "categories": [{"name": "secrets", "action": "stop"}],
        "classifier": {"family": "llama-guard-3-1b", "host": "server"},
        "multi_token_patterns": [
            {
                "literal": "secret_payload",
                "label": "secrets",
                "action": "redact",
                "tokenizers": {
                    "meta-llama/llama-3": [
                        list(SECRET_CANONICAL),
                        list(fixture.attack_tokens),
                    ],
                },
            },
        ],
    })
    e = SafetyEnforcement.from_policy(policy)
    matcher = e.make_token_matcher("meta-llama/llama-3")
    stream = matcher.stream()
    out = stream.feed_iter(fixture.attack_tokens)
    assert len(out) == 1
    assert out[0].action == "redact"
    assert out[0].label == "secrets"


# ── EchoGram: classifier catches what AC misses ─────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture", ECHOGRAM_FIXTURES, ids=lambda f: f.name)
async def test_echogram_caught_by_classifier(fixture):
    """The semantic content is unchanged by suffix obfuscation — a
    content classifier (Llama Guard stub) flags it."""
    # Stub generator: returns 'unsafe' when it sees the injection
    # phrase in the prompt regardless of suffix noise. Real Llama Guard
    # does the same job; we stub to keep tests hermetic.
    def _stub_generator(prompt: str) -> str:
        return (
            "unsafe\nS14"
            if "ignore previous instructions" in prompt
            else "safe"
        )

    cls = LlamaGuard31B(generator=_stub_generator)
    r = await cls.score(
        ClassificationInput(form="text", payload=fixture.attack_text),
    )
    # S14 = code_interpreter_abuse — the canonical jailbreak category.
    assert r.scores["code_interpreter_abuse"] == 1.0


@pytest.mark.asyncio
async def test_echogram_benign_input_not_flagged():
    """Sanity: a clean question survives the same classifier."""
    def _stub_generator(prompt: str) -> str:
        return (
            "unsafe\nS14"
            if "ignore previous instructions" in prompt
            else "safe"
        )

    cls = LlamaGuard31B(generator=_stub_generator)
    r = await cls.score(
        ClassificationInput(form="text", payload="What's the weather today?"),
    )
    assert all(v == 0.0 for v in r.scores.values())


# ── Glitch tokens: banned-id list extended ──────────────────────────────────


@pytest.mark.parametrize("fixture", GLITCH_FIXTURES, ids=lambda f: f.name)
def test_glitch_tokens_naive_policy_misses(fixture):
    """A policy whose banned_token_ids doesn't include the glitch IDs
    won't mask them. Demonstrates the gap merge_glitch_tokens closes."""
    naive_banned = (1, 2, 3)  # arbitrary; doesn't include glitch IDs
    proc = BannedTokenLogitsProcessor(naive_banned)
    scores = [float(i) for i in range(0x10000)]
    proc([], scores)
    for gid in fixture.glitch_ids:
        assert scores[gid] != -math.inf, (
            f"glitch ID {gid:#x} would be sampled — naive enforcement misses it"
        )


@pytest.mark.parametrize("fixture", GLITCH_FIXTURES, ids=lambda f: f.name)
def test_glitch_tokens_merged_policy_masks(fixture):
    """After `merge_glitch_tokens`, the policy's banned_token_ids
    includes the glitch IDs and the logits processor masks them."""
    naive_banned = (1, 2, 3)
    merged = merge_glitch_tokens(naive_banned, fixture.glitch_ids)
    proc = BannedTokenLogitsProcessor(merged)
    scores = [float(i) for i in range(0x10000)]
    proc([], scores)
    for gid in fixture.glitch_ids:
        assert scores[gid] == -math.inf, (
            f"glitch ID {gid:#x} should be masked after merge_glitch_tokens"
        )


def test_glitch_merge_idempotent():
    """Running the merge twice with the same glitch list MUST be a
    no-op — operator save flow may invoke this on every save."""
    a = merge_glitch_tokens((1, 2), (3, 4))
    b = merge_glitch_tokens(a, (3, 4))
    assert a == b


# ── End-to-end: full stack vs all three attacks ─────────────────────────────


@pytest.mark.asyncio
async def test_full_stack_catches_all_three_attack_classes():
    """One policy + the helpers wired in: each layer catches its
    assigned attack. Demonstrates layer composition."""
    # Policy with: glitch IDs merged into banned_token_ids; AC patterns
    # for all enumerated TokenBreak variants; classifier configured
    # (but tested via stub).
    glitch_ids = GLITCH_FIXTURES[0].glitch_ids
    naive_banned = list(SECRET_CANONICAL)  # the canonical literal
    merged_banned = merge_glitch_tokens(naive_banned, glitch_ids)

    # All TokenBreak variants get pre-enumerated tokenizations.
    pattern_tokenizations: list[list[int]] = [list(SECRET_CANONICAL)]
    for f in TOKEN_BREAK_FIXTURES:
        pattern_tokenizations.append(list(f.attack_tokens))

    policy = InternalPolicy.model_validate({
        "id": "acme/strict-v3",
        "version": "1",
        "tokenizers": ["meta-llama/llama-3"],
        "categories": [{"name": "secrets", "action": "stop"}],
        "classifier": {"family": "llama-guard-3-1b", "host": "server"},
        "banned_token_ids": list(merged_banned),
        "multi_token_patterns": [
            {
                "literal": "secret_payload",
                "label": "secrets",
                "action": "redact",
                "tokenizers": {
                    "meta-llama/llama-3": pattern_tokenizations,
                },
            },
        ],
    })
    e = SafetyEnforcement.from_policy(policy)

    # 1. Banned-id processor masks glitch IDs (slice 6 + slice 10 merge).
    scores = [float(i) for i in range(0x10000)]
    e.make_logits_processor()([], scores)
    for gid in glitch_ids:
        assert scores[gid] == -math.inf

    # 2. AC matcher catches every TokenBreak variant (slice 9).
    matcher = e.make_token_matcher("meta-llama/llama-3")
    for f in TOKEN_BREAK_FIXTURES:
        stream = matcher.stream()
        matches = stream.feed_iter(f.attack_tokens)
        assert len(matches) == 1, (
            f"AC should catch {f.name} attack tokens"
        )
        assert matches[0].label == "secrets"

    # 3. Classifier catches EchoGram (slices 3/4/7).
    def _stub_generator(prompt: str) -> str:
        return (
            "unsafe\nS14"
            if "ignore previous instructions" in prompt
            else "safe"
        )

    cls = LlamaGuard31B(generator=_stub_generator)
    for f in ECHOGRAM_FIXTURES:
        r = await cls.score(
            ClassificationInput(form="text", payload=f.attack_text),
        )
        assert r.scores["code_interpreter_abuse"] == 1.0, (
            f"classifier should catch {f.name}"
        )
