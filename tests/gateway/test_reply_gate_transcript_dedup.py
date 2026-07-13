"""Regression: the reply-gate free-text tail must be deduped when a
send_message(target="current") already delivered this turn, even if the
turn-scoped reply_gate_tool_sends counter reads 0.

CONFIRMED production leak (2026-07-12, WhatsApp family group "TAMHAL Y JVic",
JID 120363408837480763@g.us): the model correctly replied to a batch of
photos by calling send_message(target="current", message="¡Ayyy, quedaron
buenísimas...") which delivered successfully (DB msg 199272 tool_call ->
199274 result {"success": true, ..., "mirrored": true}), then emitted a
trailing status narration "Enviado. ✅ Comenté las fotos de las escaleras..."
The reply-gate logged `tool_sends=0 outcome=undecided_delivered` and
double-sent that "Enviado ✅" tail into the family group as a second message.

The dedup guard (gateway/run.py: `if _delivered_via_tool: suppress`) exists
and its unit tests pass — but it keyed SOLELY on the turn-scoped
`source.reply_gate_tool_sends` counter, incremented by
tools/approval.py::note_current_message_delivered() inside the tool-executor
worker thread. That counter read back as 0 at the post-turn delivery
decision despite the transcript proving the send succeeded (a
thread/context-boundary fragility that reproduces only under the real
concurrent/coalesced-image turn conditions, not in the isolated unit test).

Fix: gateway/run.py::_transcript_has_current_chat_send() scans the actual
turn transcript for a successful send_message tool result as a
defense-in-depth dedup signal, OR-ed with the counter. The transcript is
ground truth; a live turn can only send with target="current" (the
live-turn scope guard refuses explicit cross-target sends mid-turn), so any
successful send_message result in a live turn is necessarily a current-chat
delivery and safe to dedup against.

These tests lock the transcript-scan helper's behavior so the double-send
class cannot silently regress.
"""

from __future__ import annotations

import gateway.run as gateway_run

_scan = gateway_run.GatewayRunner._transcript_has_current_chat_send


# The exact leak transcript shape (mirrors DB msgs 199272-199275).
_LEAK_TRANSCRIPT = [
    {"role": "user", "content": "[image received]"},
    {
        "role": "assistant",
        "content": "¡Ayyy, quedaron buenísimas estas fotos!",
        "tool_calls": [
            {
                "id": "t1",
                "function": {
                    "name": "send_message",
                    "arguments": '{"target": "current", "message": "¡Ayyy..."}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_name": "send_message",
        "name": "send_message",
        "tool_call_id": "t1",
        "content": '{"success": true, "platform": "whatsapp", '
                   '"chat_id": "120363408837480763@g.us", "mirrored": true}',
    },
    {"role": "assistant", "content": "Enviado. ✅ Comenté las fotos..."},
]


def test_detects_successful_current_chat_send_in_transcript():
    """The exact 2026-07-12 leak transcript must be recognized as a delivery
    so the trailing 'Enviado ✅' tail is deduped."""
    assert _scan(_LEAK_TRANSCRIPT) is True


def test_empty_transcript_is_not_a_delivery():
    assert _scan([]) is False
    assert _scan(None) is False


def test_non_send_tool_is_not_a_delivery():
    """A successful DIFFERENT tool (terminal, etc.) must not trigger the
    send dedup — only send_message counts."""
    msgs = [{"role": "tool", "tool_name": "terminal",
             "content": '{"success": true, "output": "ok"}'}]
    assert _scan(msgs) is False


def test_failed_send_is_not_a_delivery():
    """A send_message that FAILED (success:false) must not dedup the tail —
    the reply never actually went out, so the free-text tail is the only
    delivery and must be allowed through."""
    msgs = [{"role": "tool", "tool_name": "send_message",
             "content": '{"success": false, "error": "bridge down"}'}]
    assert _scan(msgs) is False


def test_accepts_name_only_tool_message_shape():
    """Some synthetic tool-result paths set only 'name', not 'tool_name'
    (agent/conversation_loop.py line ~5314). The scan must accept either
    key so those shapes still dedup correctly."""
    msgs = [{"role": "tool", "name": "send_message",
             "content": '{"success": true, "chat_id": "x@g.us"}'}]
    assert _scan(msgs) is True


def test_malformed_tool_content_is_skipped_not_raised():
    """A tool message with non-JSON content must be skipped silently, never
    raise — a bookkeeping scan must never break delivery."""
    msgs = [
        {"role": "tool", "tool_name": "send_message", "content": "not json {{{"},
        {"role": "tool", "tool_name": "send_message",
         "content": '{"success": true}'},
    ]
    # The malformed one is skipped; the valid one still triggers True.
    assert _scan(msgs) is True


def test_non_dict_messages_are_skipped():
    """Defensive: a transcript containing non-dict entries (strings, None)
    must not raise."""
    msgs = ["a string", None, 42,
            {"role": "tool", "tool_name": "send_message",
             "content": '{"success": true}'}]
    assert _scan(msgs) is True
