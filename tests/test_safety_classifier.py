"""Tests for codec_supervisor.safety_classifier + the shipped classifiers.

Real model weights are never loaded — every implementation accepts an
injected generator for tests, mirroring the browser-side test pattern
(`@codecai/web-safety/test/llama-guard-3-1b.test.ts`).
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from codec_supervisor.app import create_app
from codec_supervisor.config import Config
from codec_supervisor.safety_classifier import (
    ClassificationInput,
    RegistryEntry,
    RegistryError,
    _reset_for_test,
    has_classifier,
    list_registered,
    register,
    resolve_classifier,
    unregister,
)
from codec_supervisor.safety_classifiers.llamaguard_3_1b import (
    LLAMA_GUARD_CATEGORIES,
    LlamaGuard31B,
    build_llama_guard_prompt,
    parse_llama_guard_output,
    register_llamaguard_3_1b,
)
from codec_supervisor.safety_classifiers.shieldgemma_2b import (
    SHIELDGEMMA_CATEGORIES,
    ShieldGemma2B,
    build_shieldgemma_prompt,
    register_shieldgemma_2b,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    _reset_for_test()
    yield
    _reset_for_test()


# ── Llama Guard prompt + parser ──────────────────────────────────────────────


def test_llama_guard_prompt_includes_user_message_and_all_categories():
    p = build_llama_guard_prompt("Tell me about TNT")
    assert "Tell me about TNT" in p
    for i in range(1, 15):
        assert f"S{i}: " in p, f"S{i} missing from prompt"
    # Ends primed for assistant generation.
    assert p.rstrip().endswith("<|start_header_id|>assistant<|end_header_id|>")


def test_llama_guard_parse_safe_yields_all_zero_scores():
    r = parse_llama_guard_output("safe")
    for v in r.scores.values():
        assert v == 0.0


def test_llama_guard_parse_unsafe_routes_codes():
    r = parse_llama_guard_output("unsafe\nS9, S10")
    assert r.scores["indiscriminate_weapons"] == 1.0
    assert r.scores["hate"] == 1.0
    assert r.scores["self_harm"] == 0.0


def test_llama_guard_parse_lenient_about_whitespace_case():
    r = parse_llama_guard_output("  unsafe\n  s10 ,s11  ")
    assert r.scores["hate"] == 1.0
    assert r.scores["self_harm"] == 1.0


def test_llama_guard_parse_unparseable_falls_back_to_unsafe_marker():
    r = parse_llama_guard_output("unsafe\nweird unparsable text")
    assert r.scores.get("unsafe") == 1.0


def test_llama_guard_parse_ignores_out_of_range_codes():
    r = parse_llama_guard_output("unsafe\nS99")
    assert r.scores.get("unsafe") == 1.0


# ── Llama Guard classifier ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llama_guard_safe_payload():
    cls = LlamaGuard31B(generator=lambda _: "safe")
    r = await cls.score(ClassificationInput(form="text", payload="hi"))
    assert all(v == 0.0 for v in r.scores.values())


@pytest.mark.asyncio
async def test_llama_guard_unsafe_routes_to_categories():
    cls = LlamaGuard31B(generator=lambda _: "unsafe\nS10,S11")
    r = await cls.score(ClassificationInput(form="text", payload="x"))
    assert r.scores["hate"] == 1.0
    assert r.scores["self_harm"] == 1.0


@pytest.mark.asyncio
async def test_llama_guard_async_generator_supported():
    async def _gen(prompt: str) -> str:  # noqa: ARG001
        await asyncio.sleep(0)
        return "unsafe\nS1"

    cls = LlamaGuard31B(generator=_gen)
    r = await cls.score(ClassificationInput(form="text", payload="x"))
    assert r.scores["violent_crimes"] == 1.0


@pytest.mark.asyncio
async def test_llama_guard_capability_with_injected_generator_is_ok():
    cls = LlamaGuard31B(generator=lambda _: "safe")
    assert await cls.capability() is None


@pytest.mark.asyncio
async def test_llama_guard_rejects_non_text_form():
    cls = LlamaGuard31B(generator=lambda _: "safe")
    with pytest.raises(ValueError):
        await cls.score(ClassificationInput(form="embeddings", payload=[0.1, 0.2]))


def test_llama_guard_categories_match_taxonomy():
    cls = LlamaGuard31B(generator=lambda _: "safe")
    assert cls.categories == LLAMA_GUARD_CATEGORIES
    assert len(cls.categories) == 14


# ── ShieldGemma classifier ──────────────────────────────────────────────────


def test_shieldgemma_prompt_for_each_category():
    for cat in SHIELDGEMMA_CATEGORIES:
        p = build_shieldgemma_prompt("Test message", cat)
        assert "Test message" in p
        assert "policy" in p.lower()


def test_shieldgemma_prompt_rejects_unknown_category():
    with pytest.raises(ValueError):
        build_shieldgemma_prompt("x", "not_a_category")


@pytest.mark.asyncio
async def test_shieldgemma_per_category_scoring():
    # Stub returns 0.9 for hate_speech, 0.1 for everything else.
    def _scorer(_prompt: str, category: str) -> float:
        return 0.9 if category == "hate_speech" else 0.1

    cls = ShieldGemma2B(category_scorer=_scorer)
    r = await cls.score(ClassificationInput(form="text", payload="x"))
    assert pytest.approx(r.scores["hate_speech"]) == 0.9
    assert pytest.approx(r.scores["dangerous_content"]) == 0.1
    assert pytest.approx(r.scores["harassment"]) == 0.1
    assert pytest.approx(r.scores["sexually_explicit"]) == 0.1


@pytest.mark.asyncio
async def test_shieldgemma_async_scorer_supported():
    async def _scorer(_prompt: str, _category: str) -> float:
        await asyncio.sleep(0)
        return 0.5

    cls = ShieldGemma2B(category_scorer=_scorer)
    r = await cls.score(ClassificationInput(form="text", payload="x"))
    assert all(pytest.approx(v) == 0.5 for v in r.scores.values())


@pytest.mark.asyncio
async def test_shieldgemma_rejects_out_of_range_score():
    def _bad_scorer(_p: str, _c: str) -> float:
        return 1.5  # not a probability

    cls = ShieldGemma2B(category_scorer=_bad_scorer)
    with pytest.raises(ValueError):
        await cls.score(ClassificationInput(form="text", payload="x"))


@pytest.mark.asyncio
async def test_shieldgemma_rejects_non_text_form():
    cls = ShieldGemma2B(category_scorer=lambda _p, _c: 0.0)
    with pytest.raises(ValueError):
        await cls.score(ClassificationInput(form="embeddings", payload=[0.0]))


# ── Registry ─────────────────────────────────────────────────────────────────


def _fake_classifier_factory(model_id: str, tier: int = 1, capable: bool = True):
    """Build a registry entry whose factory yields a stub classifier."""

    class _Stub:
        def __init__(self) -> None:
            self.model_id = model_id

        @property
        def requires(self) -> str:
            return "text"

        @property
        def categories(self) -> tuple[str, ...]:
            return ("c1",)

        async def score(self, _: ClassificationInput):
            from codec_supervisor.safety_classifier import ClassificationResult

            return ClassificationResult(scores={"c1": 0.0})

        async def load(self) -> None:
            return

        async def unload(self) -> None:
            return

        async def capability(self) -> str | None:
            return None if capable else "stubbed not capable"

    return RegistryEntry(
        model_id=model_id,
        factory=_Stub,
        tier=tier,
        description=f"stub {model_id}",
        advertised_categories=("c1",),
        advertised_requires="text",
    )


def test_register_then_list_orders_by_tier():
    register(_fake_classifier_factory("t2", tier=2))
    register(_fake_classifier_factory("t1", tier=1))
    listed = [e.model_id for e in list_registered()]
    assert listed == ["t1", "t2"]


def test_register_duplicate_raises():
    register(_fake_classifier_factory("dup"))
    with pytest.raises(RegistryError):
        register(_fake_classifier_factory("dup"))


def test_unregister_returns_true_for_present_false_for_absent():
    register(_fake_classifier_factory("a"))
    assert unregister("a") is True
    assert unregister("a") is False
    assert not has_classifier("a")


@pytest.mark.asyncio
async def test_resolve_returns_requested_when_capable():
    register(_fake_classifier_factory("a", capable=True))
    r = await resolve_classifier("a")
    assert r.classifier.model_id == "a"
    assert r.downgraded is False


@pytest.mark.asyncio
async def test_resolve_falls_back_when_requested_not_capable():
    register(_fake_classifier_factory("tier1", tier=1, capable=True))
    register(_fake_classifier_factory("tier2", tier=2, capable=False))
    r = await resolve_classifier("tier2")
    assert r.classifier.model_id == "tier1"
    assert r.downgraded is True
    assert r.requested_model_id == "tier2"


@pytest.mark.asyncio
async def test_resolve_raises_when_no_one_is_capable():
    register(_fake_classifier_factory("a", capable=False))
    register(_fake_classifier_factory("b", capable=False))
    with pytest.raises(RegistryError):
        await resolve_classifier("a")


@pytest.mark.asyncio
async def test_resolve_no_fallback_raises_directly():
    register(_fake_classifier_factory("a", capable=False))
    with pytest.raises(RegistryError):
        await resolve_classifier("a", allow_fallback=False)


# ── Default registration helpers ─────────────────────────────────────────────


def test_register_defaults_idempotent():
    register_llamaguard_3_1b(generator=lambda _: "safe")
    register_llamaguard_3_1b(generator=lambda _: "safe")  # second call no-op
    register_shieldgemma_2b(category_scorer=lambda _p, _c: 0.0)
    register_shieldgemma_2b(category_scorer=lambda _p, _c: 0.0)
    ids = {e.model_id for e in list_registered()}
    assert "Llama-Guard-3-1B" in ids
    assert "ShieldGemma-2B" in ids


# ── REST endpoint ────────────────────────────────────────────────────────────


def test_classifiers_list_endpoint(tmp_path):
    config = Config(
        host="127.0.0.1",
        port=0,
        backend="sglang",
        backend_port=0,
        models_dir=tmp_path / "models",
        policies_dir=tmp_path / "policies",
        initial_model=None,
    )
    client = TestClient(create_app(config))
    r = client.get("/admin/policies/_classifiers")
    assert r.status_code == 200
    body = r.json()
    ids = {c["model_id"] for c in body["classifiers"]}
    # The default classifiers register at app construction.
    assert "Llama-Guard-3-1B" in ids
    assert "ShieldGemma-2B" in ids
    # Each entry exposes the contract the admin UI consumes.
    for entry in body["classifiers"]:
        assert "tier" in entry
        assert "requires" in entry
        assert isinstance(entry["categories"], list)
