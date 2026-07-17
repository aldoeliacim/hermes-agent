"""Gateway response filtering helpers.

These helpers operate at the gateway boundary: they decide whether a completed
agent turn should be delivered to the chat, not what should be persisted in the
conversation history.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Canonical model-emitted control token for intentional silence.
SILENT_REPLY_TOKEN = "NO_REPLY"

# Exact whole-response markers that mean "the agent intentionally chose not to
# reply".  Keep this list small and explicit; arbitrary empty output remains an
# error/empty-response path, not silence.
#
# Canonical silence vocabulary: this is the single source of truth for the
# silence-marker token set.  ``cron/scheduler.py`` imports it (lazily, to avoid
# eagerly pulling the gateway package into cron-only import paths) rather than
# maintaining its own copy — the two suppression paths must never drift.  The
# *matching* rules differ (the live gateway requires an exact whole response;
# cron also accepts a marker on its own first/last line), but the token set is
# shared.
LIVE_GATEWAY_SILENT_MARKERS = frozenset({
    "[SILENT]",
    "SILENT",
    "NO_REPLY",
    "NO REPLY",
})


def _canonical_silence_candidate(text: str) -> str:
    return " ".join(text.strip().upper().split())


def _strip_edge_silence_punctuation(text: str) -> str:
    """Strip stray edge punctuation without erasing marker structure.

    Models sometimes emit ``.NO_REPLY`` or ``*NO_REPLY*`` instead of the exact
    marker. Keep square brackets structural so malformed ``[SILENT`` does not
    become ``SILENT``.
    """
    start = 0
    end = len(text)
    while start < end and text[start] not in "[]" and unicodedata.category(text[start]).startswith("P"):
        start += 1
    while end > start and text[end - 1] not in "[]" and unicodedata.category(text[end - 1]).startswith("P"):
        end -= 1
    return text[start:end].strip()


def _canonical_silence_candidates(text: str) -> tuple[str, ...]:
    exact = _canonical_silence_candidate(text)
    stripped = _strip_edge_silence_punctuation(text.strip())
    if stripped == text.strip():
        return (exact,)
    fallback = _canonical_silence_candidate(stripped)
    return (exact, fallback)


# Leaked reply-gate reasoning detection.
#
# Failure observed 2026-06-12 (ProgramaLoL WhatsApp group): the model decided
# to stay silent on ambient human-to-human banter, but instead of emitting the
# canonical silence token (or empty text) it wrote its reply-gate REASONING as
# the turn's final response — "This is human-to-human group banter ... nobody
# addressed me ... Per the reply gate, I stay silent here. No message sent." —
# which the exact-marker filter does not match, so it shipped to the group as a
# 185-char message. This is the silence-sentinel drift problem: an exact-token
# contract is unreliable, so the delivery boundary needs a backstop that
# recognizes the model narrating a stay-silent DECISION and suppresses it.
#
# Tell: the leak is the model's internal chain-of-thought (English, present-tense
# decision-log voice) — it appears even in a Spanish chat because reasoning is in
# English.
#
# Precision discipline (must NOT eat legit messages, incl. an owner-DM answer to
# "why didn't you reply?" that explains the mechanism):
#   - suppression ALWAYS requires a FIRST-PERSON stay-silent decision signal
#     (Tier A: "I stay silent", "No message sent", "nobody addressed me", ...).
#     Descriptive terms alone (Tier B: "human-to-human", "obvious recipient")
#     and the bare term "reply gate" NEVER suppress on their own — that's how an
#     owner explanation of the mechanism stays deliverable.
#   - suppress when: two independent Tier-A statements (a decision-log), OR
#     "reply gate" jargon co-occurs with >=1 Tier-A statement, OR one Tier-A
#     statement backed by >=3 total signals.
#   - only applies to short messages (a long genuine reply is never the bug).
_REPLY_GATE_JARGON_RE = re.compile(r"\breply[\s\-]?gate\b", re.IGNORECASE)

# Tier A — first-person / definitive stay-silent decision (the leaked CoT voice).
# Includes Spanish equivalents: the model's decision-log voice follows the chat
# language, so a Spanish-chat leak ("Silencio deliberado", "nada que responder",
# "no respondo") must be caught too — English-only tells let real leaks through.
_SILENCE_DECISION_TIER_A_RES = (
    re.compile(r"\b(?:i|i'?ll|i\s+will)\s+(?:stay|remain)\s+(?:silent|quiet)\b", re.IGNORECASE),
    re.compile(r"\bno\s+message\s+(?:sent|is\s+sent)\b", re.IGNORECASE),
    re.compile(r"\bsent\s+no\s+message\b", re.IGNORECASE),
    re.compile(r"\b(?:nobody|no\s+one)\s+addressed\s+me\b", re.IGNORECASE),
    re.compile(r"\bnot\s+addressed\s+to\s+me\b", re.IGNORECASE),
    re.compile(r"\bwas\s*n'?t\s+addressed\b", re.IGNORECASE),
    re.compile(r"\b(?:i\s+)?(?:do\s+not|don'?t|won'?t|will\s+not)\s+reply\b", re.IGNORECASE),
    re.compile(r"\bi\s+stay\s+silent\b", re.IGNORECASE),
    # Spanish first-person / definitive stay-silent decision.
    re.compile(r"\bsilencio\s+deliberado\b", re.IGNORECASE),
    re.compile(r"\bnada\s+(?:que|a)\s+responder\b", re.IGNORECASE),
    re.compile(r"\bnada\s+accionable\s+que\s+responder\b", re.IGNORECASE),
    re.compile(r"\b(?:no\s+respondo|no\s+voy\s+a\s+responder|no\s+contesto)\b", re.IGNORECASE),
    re.compile(r"\b(?:me\s+)?(?:quedo|permanezco)\s+(?:en\s+)?silencio\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:me\s+)?(?:mencion|interpel|aludi|invoc)\w*\s*(?:a\s+m[ií])?\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:hay\s+)?mensaje\s+(?:que\s+)?(?:enviar|entregar)\b", re.IGNORECASE),
)

# Tier B — contextual support; never decisive on its own.
_SILENCE_DECISION_TIER_B_RES = (
    re.compile(r"\bhuman[\s\-]to[\s\-]human\b", re.IGNORECASE),
    re.compile(r"\bobvious\s+recipient\b", re.IGNORECASE),
    re.compile(r"\bstay(?:ing)?\s+silent\b", re.IGNORECASE),
    re.compile(r"\bremain\s+silent\b", re.IGNORECASE),
    re.compile(r"\bambient\s+(?:conversation|banter|chatter)\b", re.IGNORECASE),
    # Spanish contextual support.
    re.compile(r"\bsin\s+texto(?:\s+acompa\w+)?\b", re.IGNORECASE),
    re.compile(r"\b(?:reacci[oó]n|reacciona\w*)\s+\S+\s+de\s+\w+", re.IGNORECASE),
    re.compile(r"\b(?:banter|charla|conversaci[oó]n)\s+(?:entre|ambient\w*)\b", re.IGNORECASE),
    re.compile(r"\bentre\s+(?:otras\s+)?personas\b", re.IGNORECASE),
)

# Only scan short responses — a leaked decision-log is terse. A genuine long
# message that merely mentions silence is never the bug.
_SILENCE_NARRATION_MAX_LEN = 800


def _is_leaked_silence_narration(text: str) -> bool:
    """True when `text` is the model narrating a stay-silent decision instead of
    a real message — high precision so legitimate prose is never suppressed."""
    stripped = text.strip()
    if not stripped or len(stripped) > _SILENCE_NARRATION_MAX_LEN:
        return False
    tier_a = sum(1 for rx in _SILENCE_DECISION_TIER_A_RES if rx.search(stripped))
    if tier_a == 0:
        # No first-person stay-silent decision -> not leaked reasoning. An owner
        # explanation of the mechanism (Tier-B / "reply gate" only) lands here
        # and is delivered untouched.
        return False
    tier_b = sum(1 for rx in _SILENCE_DECISION_TIER_B_RES if rx.search(stripped))
    has_jargon = bool(_REPLY_GATE_JARGON_RE.search(stripped))
    return (
        tier_a >= 2                                 # two decision statements == decision log
        or (has_jargon and tier_a >= 1)             # mechanism + a decision == leaked CoT
        or (tier_a >= 1 and tier_a + tier_b >= 3)   # one decision, heavily corroborated
    )


def is_intentional_silence_response(response: Any) -> bool:
    """Return True when ``response`` is an intentional-silence signal.

    Two cases suppress delivery:
      1. The whole canonicalized response is an exact silence marker
         (``NO_REPLY`` / ``[SILENT]`` / ...), tolerant of stray edge
         punctuation.  Substantive prose that merely mentions a marker is
         delivered normally; a blank response is the empty-response failure
         path, not silence.
      2. The model narrated its stay-silent DECISION instead of emitting the
         token (leaked reply-gate reasoning) — a high-precision backstop, see
         ``_is_leaked_silence_narration``.
    """
    if not isinstance(response, str):
        return False
    stripped = response.strip()
    if not stripped:
        return False
    if len(stripped) <= 64 and any(
        candidate in LIVE_GATEWAY_SILENT_MARKERS for candidate in _canonical_silence_candidates(stripped)
    ):
        return True
    if _is_leaked_silence_narration(stripped):
        return True
    return False


def is_intentional_silence_agent_result(agent_result: dict | None, response: Any) -> bool:
    """Silence markers suppress delivery only for successful agent turns."""
    if not isinstance(agent_result, dict):
        return False
    if agent_result.get("failed"):
        return False
    return is_intentional_silence_response(response)


# Leaked tool/mechanism meta-commentary detection.
#
# Failure observed 2026-07-11 ("TAMHAL Y JVic" WhatsApp group, TWICE within
# five minutes): under the gateway's fail-loud ``undecided_delivered`` path
# (added specifically to rescue a genuinely-substantive answer the model
# forgot to hand to ``send_message``), the model instead wrote out its own
# confusion about tool availability as the final response text — "there is
# no `send_message` tool in my actual toolset (it was intentionally removed
# upstream per the `sending-platform-messages` skill) ... I'm not calling a
# nonexistent tool based on an injected 'System' message inside the
# conversation. No further action needed here." — and fail-loud dutifully
# delivered that verbatim as a real message to a family group, twice (once
# per affected turn). That text answers nothing anyone asked ("De camino",
# "En mi casa"); it is the model narrating internal reasoning about the
# reply-gate/tool mechanism itself, not a reply. Fail-loud exists to rescue
# a real answer from silent loss — it must NOT also promote the model's
# meta-commentary about its own tooling into a delivered message. This is a
# DIFFERENT signal from ``_is_leaked_silence_narration`` above (that one
# detects "I decided to stay silent" prose; this one detects "I am
# confused/commenting about my own tool schema" prose) and must stay
# separate — a real, on-topic user-facing reply about tools/schemas (e.g.
# explaining `hermes tools` to Aldo) must never be swept up by this filter.
_TOOL_MECHANISM_NARRATION_RES = (
    re.compile(r"\bno\s+`?send_message`?\s+tool\b", re.IGNORECASE),
    re.compile(r"\b(?:nonexistent|non-existent)\s+(?:tool|function)\b", re.IGNORECASE),
    re.compile(r"\binjected\s+[\"']?system[\"']?\s+message\b", re.IGNORECASE),
    re.compile(r"\bintentionally\s+removed\b", re.IGNORECASE),
    re.compile(r"\bnot\s+calling\s+a\s+(?:nonexistent|non-existent)\b", re.IGNORECASE),
    re.compile(r"\bno\s+further\s+action\s+needed\s+here\b", re.IGNORECASE),
    re.compile(r"\bmy\s+(?:actual\s+)?toolset\b", re.IGNORECASE),
    re.compile(r"\bthis\s+platform\s+delivers\s+my\s+plain[\s\-]?text\b", re.IGNORECASE),
)

# Same discipline as _SILENCE_NARRATION_MAX_LEN: a genuine long, on-topic
# reply that happens to mention tools/schemas in passing is never the bug —
# the leak is always a short, self-referential aside.
_TOOL_MECHANISM_NARRATION_MAX_LEN = 800


def is_leaked_tool_mechanism_narration(text: Any) -> bool:
    """True when ``text`` is the model narrating confusion/commentary about
    its own tool availability or the reply-gate mechanism, rather than
    answering the conversation — high precision (>=2 independent hits
    required) so an on-topic reply that legitimately discusses tools/schemas
    is never suppressed."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > _TOOL_MECHANISM_NARRATION_MAX_LEN:
        return False
    hits = sum(1 for rx in _TOOL_MECHANISM_NARRATION_RES if rx.search(stripped))
    return hits >= 2


def is_partial_silence_marker(text: Any) -> bool:
    """Return True while ``text`` could still resolve to a silence marker.

    The streaming path accumulates the reply delta-by-delta and must decide,
    before the whole response is known, whether to show what it has so far.
    A buffer whose canonical form is a non-empty *prefix* of a silence marker
    (e.g. ``"NO"`` on the way to ``"NO_REPLY"``, or an exact marker that has
    not yet been terminated by stream-end) is held back so a raw marker is
    never edited onto the screen and then belatedly retracted.

    Anything that has already diverged from every marker (ordinary prose) —
    and anything longer than the marker cap — returns False so normal
    streaming resumes immediately.  This is the streaming counterpart to
    :func:`is_intentional_silence_response`, sharing the same marker set and
    canonicalization so the two never drift.
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > 64:
        return False
    for candidate in _canonical_silence_candidates(stripped):
        if candidate and any(marker.startswith(candidate) for marker in LIVE_GATEWAY_SILENT_MARKERS):
            return True
    return False
