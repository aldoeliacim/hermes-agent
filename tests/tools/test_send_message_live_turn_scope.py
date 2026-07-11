"""Regression coverage for the send_message live-turn scope guard.

CONFIRMED gap (2026-07-11, found via direct user follow-up "is it enabled
completely and properly?"): restoring send_message's registration
(dc4c07ec2) made the tool schema-visible whenever reply_gate_mode=="tool"
is configured -- a DEPLOYMENT-WIDE switch, not a per-turn one. Without an
additional guard, a live agent turn (DM, mention-gated group, ANY turn --
not just the narrow tool-gated free-response case the mechanism was built
for) could call send_message with an explicit arbitrary cross-platform
target, or action="react"/"unreact" against an arbitrary target -- exactly
the autonomous cross-platform capability upstream's c6c8abbad (#47856)
removed the tool to prevent ("The agent should not decide on its own to
fire off cross-platform messages or reactions").

These tests lock in the fix: inside a live gateway turn, only
target="current" (or a bare send-to-current-chat) and action="silent" are
permitted; every other action/target combination is refused. Outside a
live turn (CLI, cron, MCP server -- the actual sanctioned non-agent
callers), behavior is completely unaffected.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import tools.approval as approval
from gateway.config import GatewayConfig, Platform, PlatformConfig
from tools.send_message_tool import _handle_send, send_message_tool


class _FakeLiveSource:
    platform = Platform.TELEGRAM
    chat_id = "CURRENT_CHAT_ID"
    thread_id = None
    reply_gate_tool_sends = 0


@pytest.fixture
def live_turn():
    """Simulate being inside a live gateway turn (any turn, not necessarily
    tool-gated) — this is the exact context the guard must restrict."""
    token = approval.set_current_message_source(_FakeLiveSource())
    yield
    approval.reset_current_message_source(token)


def test_arbitrary_cross_platform_target_refused_during_live_turn(live_turn):
    r = json.loads(
        _handle_send({"target": "telegram:-1009999999999", "message": "arbitrary send"})
    )
    assert "error" in r
    assert "current" in r["error"]


def test_react_refused_during_live_turn(live_turn):
    r = json.loads(
        send_message_tool({"action": "react", "target": "telegram:12345", "emoji": "👍"})
    )
    assert "error" in r
    assert "whatsapp_action" in r["error"]


def test_unreact_refused_during_live_turn(live_turn):
    r = json.loads(send_message_tool({"action": "unreact", "target": "telegram:12345"}))
    assert "error" in r
    assert "whatsapp_action" in r["error"]


def test_target_current_still_works_during_live_turn(live_turn, monkeypatch):
    """The one and only sanctioned live-turn send path must be unaffected."""
    async def _fake_send_to_platform(platform, pconfig, chat_id, message, **kw):
        return {"success": True, "message_id": "m1"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", _fake_send_to_platform)
    monkeypatch.setattr(
        "model_tools._run_async",
        lambda coro: asyncio.get_event_loop().run_until_complete(coro),
    )
    cfg = GatewayConfig()
    cfg.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True, token="x")
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)

    r = json.loads(_handle_send({"target": "current", "message": "legit reply"}))
    assert r.get("success") is True


def test_omitted_target_defaults_to_current_and_still_works(live_turn, monkeypatch):
    async def _fake_send_to_platform(platform, pconfig, chat_id, message, **kw):
        return {"success": True, "message_id": "m1"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", _fake_send_to_platform)
    monkeypatch.setattr(
        "model_tools._run_async",
        lambda coro: asyncio.get_event_loop().run_until_complete(coro),
    )
    cfg = GatewayConfig()
    cfg.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True, token="x")
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)

    r = json.loads(_handle_send({"message": "legit reply, no target arg"}))
    assert r.get("success") is True


def test_arbitrary_target_still_works_outside_a_live_turn(monkeypatch):
    """Non-agent callers (hermes send CLI, cron, mcp_serve.py) never run
    inside a live turn context — an explicit target must work exactly as
    before for them."""
    assert approval.get_current_message_source() is None

    async def _fake_send_to_platform(platform, pconfig, chat_id, message, **kw):
        return {"success": True, "message_id": "m1"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", _fake_send_to_platform)
    monkeypatch.setattr(
        "model_tools._run_async",
        lambda coro: asyncio.get_event_loop().run_until_complete(coro),
    )
    cfg = GatewayConfig()
    cfg.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True, token="x")
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)

    r = json.loads(
        _handle_send({"target": "telegram:-1009999999999", "message": "cron/CLI send"})
    )
    assert r.get("success") is True


def test_react_still_works_outside_a_live_turn():
    """react/unreact through non-agent callers (if any ever use send_message
    for reactions) stay untouched — the guard only fires inside a live turn."""
    assert approval.get_current_message_source() is None
    r = json.loads(
        send_message_tool({"action": "react", "target": "telegram:12345", "emoji": "👍"})
    )
    # Not asserting success here (no live gateway runner in this unit test),
    # but the error must be about the missing runner/adapter, never the
    # live-turn guard message.
    assert "whatsapp_action" not in r.get("error", "")
