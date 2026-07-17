from gateway.response_filters import (
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
)


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_edge_punctuation_silence_tokens_are_intentional_silence():
    for token in (".NO_REPLY", "*NO_REPLY*", " .NO_REPLY ", "*[SILENT]*", "NO_REPLY."):
        assert is_intentional_silence_response(token)


def test_blank_and_prose_mentions_are_not_silence():
    assert not is_intentional_silence_response("")
    assert not is_intentional_silence_response("Use NO_REPLY when no answer is needed.")
    assert not is_intentional_silence_response("The reply was [SILENT], intentionally.")
    assert not is_intentional_silence_response("😄 NO_REPLY")
    assert not is_intentional_silence_response("[SILENT")


def test_failed_agent_result_never_counts_as_intentional_silence():
    assert is_intentional_silence_agent_result({"failed": False}, "NO_REPLY")
    assert not is_intentional_silence_agent_result({"failed": True}, "NO_REPLY")


# --- Spanish leaked-CoT stay-silent narration (2026-07-17) -----------------
# The model's decision-log voice follows the chat language, so Spanish-chat
# turns leak Spanish stay-silent reasoning ("Silencio deliberado", "nada que
# responder", "no me aludió"). English-only tells let those real leaks reach
# the user verbatim. These lock the Spanish Tier-A/B coverage AND guard against
# suppressing legitimate Spanish replies that merely contain "no".

def test_spanish_leaked_silence_narration_is_suppressed():
    # Each is a realistic leaked decision-log the model must NOT deliver.
    leaks = (
        # jargon + one decision
        "Reply gate: silencio deliberado, no me aludió directamente.",
        # two decisions == decision log
        "Silencio deliberado. No respondo porque nada que responder.",
        "No me mencionaron; me quedo en silencio.",
        # one decision + heavy corroboration (tier_a + tier_b >= 3)
        "No respondo: charla entre otras personas, sin texto que enviar.",
    )
    for s in leaks:
        assert is_intentional_silence_response(s), f"should suppress leak: {s!r}"


def test_spanish_aludir_variants_are_detected():
    # Regression for the 'aluli' typo that made these MISS: aludir conjugations
    # ("aludió"/"aludieron") are the natural way the model says "didn't address me".
    for s in (
        "No me aludió y no respondo.",
        "No me aludieron directamente, silencio deliberado.",
    ):
        assert is_intentional_silence_response(s), f"aludir variant missed: {s!r}"


def test_legitimate_spanish_prose_is_not_suppressed():
    # Real replies that contain 'no' / silence-adjacent words but are genuine
    # answers must be delivered untouched (no false positives).
    for s in (
        "No te preocupes, ya lo hago ahora mismo.",
        "No hay problema, te lo mando en un momento.",
        "El servidor no responde todavía, sigo revisando.",
        "Claro que sí, aquí está el resultado.",
    ):
        assert not is_intentional_silence_response(s), f"false positive: {s!r}"
