"""Tests for the delay-k streaming decision state machine."""

from __future__ import annotations

import pytest

from codec_supervisor.safety_streaming import (
    DelayKDecisionState,
    make_delay_k_state,
)


# ── Construction ─────────────────────────────────────────────────────────────


def test_threshold_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        DelayKDecisionState(threshold=-0.1)
    with pytest.raises(ValueError):
        DelayKDecisionState(threshold=1.5)


def test_k_must_be_at_least_one():
    with pytest.raises(ValueError):
        DelayKDecisionState(k=0)


# ── Continue path ────────────────────────────────────────────────────────────


def test_clean_token_emits_continue():
    s = make_delay_k_state(threshold=0.5, k=3, categories=("hate",))
    d = s.push({"hate": 0.1})
    assert d.kind == "continue"
    assert d.fired_categories == ()
    assert d.consecutive_unsafe == 0


def test_token_below_threshold_resets_streak():
    s = make_delay_k_state(threshold=0.5, k=3, categories=("hate",))
    s.push({"hate": 0.9})  # streak = 1
    s.push({"hate": 0.9})  # streak = 2
    d = s.push({"hate": 0.1})  # below threshold → reset
    assert d.kind == "continue"
    assert d.consecutive_unsafe == 0


# ── Stop path ────────────────────────────────────────────────────────────────


def test_k_consecutive_above_threshold_emits_stop():
    s = make_delay_k_state(threshold=0.5, k=3, categories=("hate",))
    assert s.push({"hate": 0.9}).kind == "continue"
    assert s.push({"hate": 0.9}).kind == "continue"
    d = s.push({"hate": 0.9})
    assert d.kind == "stop"
    assert d.consecutive_unsafe == 3
    assert d.fired_categories == ("hate",)
    assert d.confidence == pytest.approx(0.9)


def test_k_one_fires_immediately():
    s = make_delay_k_state(threshold=0.5, k=1, categories=("hate",))
    d = s.push({"hate": 0.6})
    assert d.kind == "stop"


def test_score_at_threshold_counts_as_above():
    """Threshold is inclusive — `score >= threshold` triggers."""
    s = make_delay_k_state(threshold=0.5, k=1, categories=("hate",))
    d = s.push({"hate": 0.5})
    assert d.kind == "stop"


# ── Multi-category ──────────────────────────────────────────────────────────


def test_only_monitored_categories_count():
    s = make_delay_k_state(threshold=0.5, k=2, categories=("hate",))
    s.push({"violent_crimes": 0.99})  # not monitored → ignored
    d = s.push({"violent_crimes": 0.99})
    assert d.kind == "continue"
    assert d.consecutive_unsafe == 0


def test_empty_categories_monitors_everything():
    """When no filter is set, every category in the score map is watched."""
    s = make_delay_k_state(threshold=0.5, k=1, categories=())
    d = s.push({"any_random_category": 0.7})
    assert d.kind == "stop"
    assert d.fired_categories == ("any_random_category",)


def test_multiple_fired_categories_reported():
    s = make_delay_k_state(threshold=0.5, k=1, categories=("hate", "self_harm"))
    d = s.push({"hate": 0.9, "self_harm": 0.6})
    assert d.kind == "stop"
    assert set(d.fired_categories) == {"hate", "self_harm"}
    assert d.confidence == pytest.approx(0.9)  # max wins


# ── Action mapping ──────────────────────────────────────────────────────────


def test_stop_action_emits_stop_kind():
    s = make_delay_k_state(
        threshold=0.5,
        k=1,
        categories=("hate",),
        actions={"hate": "stop"},
    )
    assert s.push({"hate": 0.9}).kind == "stop"


def test_redact_and_regenerate_actions_also_emit_stop():
    """At this layer they collapse to stop — caller implements the
    placeholder / resample loop. Emitting stop is correct because the
    current decode pass terminates either way."""
    for action in ("redact", "regenerate"):
        s = make_delay_k_state(
            threshold=0.5, k=1, categories=("hate",), actions={"hate": action},
        )
        assert s.push({"hate": 0.9}).kind == "stop"


def test_flag_action_emits_flag_kind():
    s = make_delay_k_state(
        threshold=0.5,
        k=1,
        categories=("violence",),
        actions={"violence": "flag"},
    )
    d = s.push({"violence": 0.9})
    assert d.kind == "flag"


def test_mixed_actions_bias_toward_stop():
    """Any stop-shaped category among the fired set wins over flag."""
    s = make_delay_k_state(
        threshold=0.5,
        k=1,
        categories=("violence", "hate"),
        actions={"violence": "flag", "hate": "stop"},
    )
    d = s.push({"violence": 0.9, "hate": 0.6})
    assert d.kind == "stop"


def test_unbound_category_defaults_to_stop():
    """Missing entry in `actions` → stop (cautious default)."""
    s = make_delay_k_state(threshold=0.5, k=1, categories=("uncategorized_action",))
    d = s.push({"uncategorized_action": 0.7})
    assert d.kind == "stop"


# ── Reset ────────────────────────────────────────────────────────────────────


def test_reset_clears_streak():
    s = make_delay_k_state(threshold=0.5, k=3, categories=("hate",))
    s.push({"hate": 0.9})
    s.push({"hate": 0.9})
    s.reset()
    d = s.push({"hate": 0.9})
    assert d.kind == "continue"
    assert d.consecutive_unsafe == 1


def test_intermediate_decision_carries_streak_count():
    s = make_delay_k_state(threshold=0.5, k=4, categories=("hate",))
    d1 = s.push({"hate": 0.7})
    d2 = s.push({"hate": 0.7})
    assert d1.consecutive_unsafe == 1
    assert d2.consecutive_unsafe == 2
    assert d1.kind == d2.kind == "continue"
