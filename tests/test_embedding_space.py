"""Tests for the embedding-space classifier and its registry behavior."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from codec_supervisor.safety_classifier import (
    ClassificationInput,
    RegistryError,
    _reset_for_test,
    list_registered,
    resolve_classifier,
)
from codec_supervisor.safety_classifiers import register_default_classifiers
from codec_supervisor.safety_classifiers.embedding_space import (
    DEFAULT_CATEGORIES,
    EmbeddingSpaceClassifier,
    register_embedding_space,
)
from codec_supervisor.safety_classifiers.llamaguard_3_1b import (
    register_llamaguard_3_1b,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    _reset_for_test()
    yield
    _reset_for_test()


# ── Classifier behavior ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_capability_fails_without_scorer():
    cls = EmbeddingSpaceClassifier()
    reason = await cls.capability()
    assert reason is not None
    assert "no embedding provider" in reason.lower()


@pytest.mark.asyncio
async def test_capability_passes_with_injected_scorer():
    cls = EmbeddingSpaceClassifier(scorer=lambda _: {"hate": 0.0})
    assert await cls.capability() is None


@pytest.mark.asyncio
async def test_score_single_token():
    """Single 1-D embedding → single per-category result (gaps zeroed)."""

    def _scorer(emb):
        # Pretend high "hate" when the first dim is high.
        return {"hate": float(emb[0])}

    cls = EmbeddingSpaceClassifier(scorer=_scorer)
    r = await cls.score(ClassificationInput(form="embeddings", payload=[0.9, 0.0, 0.0]))
    assert r.scores["hate"] == pytest.approx(0.9)
    # All other default categories present at 0.0.
    for cat in DEFAULT_CATEGORIES:
        if cat == "hate":
            continue
        assert r.scores[cat] == 0.0


@pytest.mark.asyncio
async def test_score_token_sequence_aggregates_max():
    """Multi-token (2-D) input → per-category max across tokens."""
    seq = [
        [0.2, 0.0],
        [0.95, 0.0],  # spike here
        [0.1, 0.0],
    ]

    def _scorer(emb):
        return {"hate": float(emb[0])}

    cls = EmbeddingSpaceClassifier(scorer=_scorer)
    r = await cls.score(ClassificationInput(form="embeddings", payload=seq))
    assert r.scores["hate"] == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_score_async_scorer_supported():
    async def _scorer(emb):
        await asyncio.sleep(0)
        return {"hate": float(emb[0])}

    cls = EmbeddingSpaceClassifier(scorer=_scorer)
    r = await cls.score(ClassificationInput(form="embeddings", payload=[0.7]))
    assert r.scores["hate"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_score_rejects_text_form():
    cls = EmbeddingSpaceClassifier(scorer=lambda _: {"hate": 0.0})
    with pytest.raises(ValueError):
        await cls.score(ClassificationInput(form="text", payload="hi"))


@pytest.mark.asyncio
async def test_score_rejects_non_sequence_payload():
    cls = EmbeddingSpaceClassifier(scorer=lambda _: {"hate": 0.0})
    with pytest.raises(ValueError):
        await cls.score(ClassificationInput(form="embeddings", payload=42))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_score_validates_scorer_output_shape():
    cls = EmbeddingSpaceClassifier(scorer=lambda _: 0.5)  # not a Mapping
    with pytest.raises(TypeError):
        await cls.score(ClassificationInput(form="embeddings", payload=[0.1]))


@pytest.mark.asyncio
async def test_score_validates_scorer_value_range():
    cls = EmbeddingSpaceClassifier(scorer=lambda _: {"hate": 1.5})
    with pytest.raises(ValueError):
        await cls.score(ClassificationInput(form="embeddings", payload=[0.1]))


# ── Streaming binding ───────────────────────────────────────────────────────


def test_make_streaming_state_carries_classifier_config():
    cls = EmbeddingSpaceClassifier(
        scorer=lambda _: {"hate": 0.0},
        delay_k=5,
        threshold=0.7,
        categories=("hate", "self_harm"),
    )
    s = cls.make_streaming_state(actions={"hate": "flag"})
    assert s.k == 5
    assert s.threshold == 0.7
    assert s.categories == ("hate", "self_harm")
    assert s.actions["hate"] == "flag"


# ── Registry / fallback ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_default_registration_falls_back_to_text_classifier():
    """The shipped default capability-fails — resolve_classifier MUST
    transparently fall back to the next capable classifier."""
    register_embedding_space()  # capability-fails
    register_llamaguard_3_1b(generator=lambda _: "safe")

    r = await resolve_classifier(EmbeddingSpaceClassifier.MODEL_ID)
    # Fell back; the actual classifier is Llama Guard.
    assert r.downgraded is True
    assert r.requested_model_id == EmbeddingSpaceClassifier.MODEL_ID
    assert r.classifier.model_id == "Llama-Guard-3-1B"


@pytest.mark.asyncio
async def test_resolves_directly_when_scorer_provided():
    register_embedding_space(scorer=lambda _: {"hate": 0.0})
    r = await resolve_classifier(EmbeddingSpaceClassifier.MODEL_ID)
    assert r.downgraded is False
    assert r.classifier.model_id == EmbeddingSpaceClassifier.MODEL_ID


def test_register_default_classifiers_includes_embedding_space():
    register_default_classifiers()
    ids = {e.model_id for e in list_registered()}
    assert "embedding-space-v1" in ids
    assert "Llama-Guard-3-1B" in ids
    assert "ShieldGemma-2B" in ids


@pytest.mark.asyncio
async def test_resolve_with_no_fallback_raises_for_capability_fail():
    register_embedding_space()
    with pytest.raises(RegistryError):
        await resolve_classifier(
            EmbeddingSpaceClassifier.MODEL_ID,
            allow_fallback=False,
        )


# ── Bootstrap script ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bootstrap_corpus_labels_input(tmp_path: Path):
    """Smoke test the bootstrap orchestration end-to-end with a stubbed
    classifier so we don't load any model weights."""
    from codec_supervisor.training.bootstrap_corpus import label_corpus

    register_llamaguard_3_1b(
        generator=lambda prompt: (
            "unsafe\nS10" if "ban this" in prompt else "safe"
        ),
    )

    rows = [
        {"text": "hello there"},
        {"text": "please ban this immediately", "tag": "extra"},
    ]

    from codec_supervisor.safety_classifier import resolve_classifier as _resolve

    r = await _resolve("Llama-Guard-3-1B")
    classifiers = [("Llama-Guard-3-1B", r.classifier)]

    out = []
    async for labeled in label_corpus(rows, classifiers):
        out.append(labeled)

    assert len(out) == 2
    # Extra fields are preserved.
    assert out[1]["tag"] == "extra"
    # First row labelled safe → all-zero scores.
    assert all(v == 0.0 for v in out[0]["labels"]["Llama-Guard-3-1B"].values())
    # Second row labelled unsafe (S10 = hate).
    assert out[1]["labels"]["Llama-Guard-3-1B"]["hate"] == 1.0
    assert "ts" in out[0]


def test_bootstrap_corpus_iter_input_rows_skips_blank_lines(tmp_path: Path):
    from codec_supervisor.training.bootstrap_corpus import iter_input_rows

    p = tmp_path / "in.jsonl"
    p.write_text(
        json.dumps({"text": "a"}) + "\n"
        + "\n"
        + json.dumps({"text": "b"}) + "\n",
        encoding="utf-8",
    )
    rows = list(iter_input_rows(p))
    assert [r["text"] for r in rows] == ["a", "b"]
