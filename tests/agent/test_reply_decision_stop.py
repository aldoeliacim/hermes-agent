"""Reply-decision forced-nudge guard (agent/reply_decision_stop.py).

Covers the module in isolation (build_reply_decision_nudge, config/env
overrides) and the real regression this exists to prevent — CONFIRMED
2026-07-11: a tool-gated turn produced a complete, correct 1260-char answer
and never called send_message, so the gateway's post-turn suppression
silently dropped it (outcome=undecided). Mirrors
tests/agent/test_verification_stop.py's structure for the sibling
verify-on-stop guard.
"""

from __future__ import annotations

from agent.reply_decision_stop import (
    DEFAULT_MAX_REPLY_DECISION_NUDGES,
    build_reply_decision_nudge,
    max_reply_decision_nudges,
    reply_decision_nudge_enabled,
)


# --------------------------------------------------------------------------- #
# build_reply_decision_nudge
# --------------------------------------------------------------------------- #


def test_no_nudge_when_already_replied_via_tool():
    """tool_sends > 0 means a decision was already made this turn — no nudge."""
    assert build_reply_decision_nudge(
        tool_sends=1, decided_silent=0, attempts=0,
    ) is None


def test_no_nudge_when_explicitly_decided_silent():
    """decided_silent > 0 means the model explicitly chose silence — no nudge."""
    assert build_reply_decision_nudge(
        tool_sends=0, decided_silent=1, attempts=0,
    ) is None


def test_nudges_when_neither_call_happened():
    """The undecided case (CONFIRMED real casualty 2026-07-11) — must nudge."""
    nudge = build_reply_decision_nudge(
        tool_sends=0, decided_silent=0, attempts=0,
    )
    assert nudge is not None
    assert "send_message" in nudge
    assert 'action="silent"' in nudge
    assert 'target="current"' in nudge


def test_nudge_text_mentions_both_valid_actions():
    """The nudge must not bias toward always-reply — silence stays legitimate."""
    nudge = build_reply_decision_nudge(tool_sends=0, decided_silent=0, attempts=0)
    assert nudge is not None
    # Both the reply path and the explicit-silence path must be present so
    # the model isn't pushed to over-reply just to satisfy the gate.
    assert "deliver a reply" in nudge
    assert "choosing not to reply" in nudge


def test_respects_attempt_cap_default():
    """Fails open once the default cap (2) is reached — never loops forever."""
    cap = DEFAULT_MAX_REPLY_DECISION_NUDGES
    assert build_reply_decision_nudge(
        tool_sends=0, decided_silent=0, attempts=cap - 1,
    ) is not None
    assert build_reply_decision_nudge(
        tool_sends=0, decided_silent=0, attempts=cap,
    ) is None
    assert build_reply_decision_nudge(
        tool_sends=0, decided_silent=0, attempts=cap + 5,
    ) is None


def test_respects_explicit_max_attempts_override():
    assert build_reply_decision_nudge(
        tool_sends=0, decided_silent=0, attempts=0, max_attempts=0,
    ) is None
    assert build_reply_decision_nudge(
        tool_sends=0, decided_silent=0, attempts=4, max_attempts=5,
    ) is not None


# --------------------------------------------------------------------------- #
# max_reply_decision_nudges (config resolution)
# --------------------------------------------------------------------------- #


def test_max_nudges_default_when_no_config():
    assert max_reply_decision_nudges(config={}) == DEFAULT_MAX_REPLY_DECISION_NUDGES


def test_max_nudges_reads_agent_config():
    assert max_reply_decision_nudges(
        config={"agent": {"max_reply_decision_nudges": 5}}
    ) == 5


def test_max_nudges_clamps_negative_to_zero():
    assert max_reply_decision_nudges(
        config={"agent": {"max_reply_decision_nudges": -3}}
    ) == 0


def test_max_nudges_falls_back_on_invalid_value():
    assert max_reply_decision_nudges(
        config={"agent": {"max_reply_decision_nudges": "not-a-number"}}
    ) == DEFAULT_MAX_REPLY_DECISION_NUDGES


# --------------------------------------------------------------------------- #
# reply_decision_nudge_enabled (env + config precedence)
# --------------------------------------------------------------------------- #


def test_enabled_by_default():
    assert reply_decision_nudge_enabled(config={}) is True


def test_disabled_via_config_bool():
    assert reply_decision_nudge_enabled(config={"agent": {"reply_decision_nudge": False}}) is False


def test_disabled_via_config_string():
    assert reply_decision_nudge_enabled(
        config={"agent": {"reply_decision_nudge": "off"}}
    ) is False


def test_env_var_overrides_config(monkeypatch):
    monkeypatch.setenv("HERMES_REPLY_DECISION_NUDGE", "0")
    # Even with config saying True, the env var wins.
    assert reply_decision_nudge_enabled(
        config={"agent": {"reply_decision_nudge": True}}
    ) is False
    monkeypatch.setenv("HERMES_REPLY_DECISION_NUDGE", "1")
    assert reply_decision_nudge_enabled(
        config={"agent": {"reply_decision_nudge": False}}
    ) is True


# --------------------------------------------------------------------------- #
# Integration: conversation_loop wiring reads the reply-gate decision counters
# off the identity-stable TurnDeliveryState (reply-gate v2, tools/approval.py)
# that the gateway's post-turn delivery block also reads — a single shared
# ledger, not the per-source counters that diverged across the re-entrant
# follow-up chain. Confirms the nudge module's inputs are wired to the real
# counters, not a parallel state.
# --------------------------------------------------------------------------- #


def test_nudge_inputs_match_approval_module_counter_names():
    """The two counters this module reads must be the exact field names
    tools/approval.py's note_current_message_delivered/silent() write to on
    the TurnDeliveryState — a name drift here would silently defeat the whole
    guard. The legacy source-object mirror is also asserted for back-compat."""
    from tools.approval import TurnDeliveryState
    from gateway.session import SessionSource
    from gateway.config import Platform

    # v2 authoritative ledger.
    state = TurnDeliveryState()
    assert hasattr(state, "tool_sends")
    assert hasattr(state, "decided_silent")
    assert state.tool_sends == 0
    assert state.decided_silent == 0

    # Legacy back-compat mirror still present on the source object.
    src = SessionSource(platform=Platform.WHATSAPP, chat_id="123@g.us", chat_type="group")
    assert hasattr(src, "reply_gate_tool_sends")
    assert hasattr(src, "reply_gate_decided_silent")
    assert src.reply_gate_tool_sends == 0
    assert src.reply_gate_decided_silent == 0
