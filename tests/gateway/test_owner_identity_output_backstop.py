"""Regression tests for the non-owner outbound backstop.

Design note (2026-06-16): the gateway deliberately does NOT cosmetically
rewrite the owner's *name* out of replies on non-owner silos. A blanket
"Aldo" -> "the owner" substitution reads as robotic (arguably more
conspicuous to a contact than a name slipping through once) and a
whole-message regex mangles FILE:/MEDIA: delivery paths, silently dropping
attachments. The PRIMARY owner-identity defense is prompt-side:
skip_user_profile gates USER.md out of a non-owner contact's prompt.

What the send-path backstop DOES do on proven non-owner silos is scrub
genuine sensitive info (credentials/tokens/keys) via
_redact_secrets_for_non_owner — the same secret scrub
_sanitize_gateway_final_response applies to Telegram, extended to all
platforms for non-owner destinations. It must NOT touch names, tone, or
delivery paths.
"""

from gateway.run import _redact_secrets_for_non_owner


def test_empty_input_is_passthrough():
    assert _redact_secrets_for_non_owner("") == ""
    assert _redact_secrets_for_non_owner(None) is None


def test_clean_reply_is_unchanged():
    clean = "Listo, ya quedó. Te puse subtítulos a los 8 episodios."
    assert _redact_secrets_for_non_owner(clean) == clean


def test_owner_name_in_prose_is_NOT_rewritten():
    # The owner's name is intentionally left alone — no robotic "the owner".
    inp = "Hola Aldo, claro que sí."
    out = _redact_secrets_for_non_owner(inp)
    assert out == inp
    assert "the owner" not in out


def test_meta_preamble_is_NOT_stripped():
    # Tone/preamble shaping was removed; conversational openers survive.
    inp = "Everything's wired up and flowing. Listo, ya quedó."
    assert _redact_secrets_for_non_owner(inp) == inp


def test_file_media_delivery_paths_are_untouched():
    # The backstop must never mangle a delivery directive / path.
    for directive in (
        "Here you go FILE:/home/aldo/videos/x.mp4",
        "See MEDIA:/home/aldo/clip.mp4",
        'Attached: FILE:"/home/aldo/My Files/report.pdf"',
    ):
        assert _redact_secrets_for_non_owner(directive) == directive


def test_secret_token_is_redacted():
    # A leaked credential SHOULD be scrubbed. Use a representative key shape;
    # the underlying _redact_gateway_user_facing_secrets owns the patterns, so
    # assert only that a recognizable secret does not pass through verbatim.
    secret = "sk-ant-oat01-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHHIIIIJJJJKKKKLLLL"
    inp = f"Your key is {secret} ok"
    out = _redact_secrets_for_non_owner(inp)
    assert secret not in out or "[REDACTED]" in out
