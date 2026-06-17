"""Tests for the native-vision fast path inside vision_analyze.

When the active main model supports native vision AND the provider supports
image content inside tool-result messages, ``_handle_vision_analyze`` skips
the auxiliary LLM and returns a multimodal envelope so the main model sees
the pixels directly on its next turn.
"""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import patch


from tools.vision_tools import (
    _build_native_vision_tool_result,
    _handle_vision_analyze,
    _supports_media_in_tool_results,
    _vision_analyze_native,
)


# Minimal valid 1x1 PNG bytes.
_TINY_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


# ─── _supports_media_in_tool_results ─────────────────────────────────────────


class TestSupportsMediaInToolResults:
    def test_anthropic_native_yes(self):
        assert _supports_media_in_tool_results("anthropic", "claude-opus-4-6") is True

    def test_openrouter_yes(self):
        assert _supports_media_in_tool_results("openrouter", "anthropic/claude-opus-4.6") is True

    def test_nous_yes(self):
        assert _supports_media_in_tool_results("nous", "anthropic/claude-sonnet-4.6") is True

    def test_openai_chat_yes(self):
        assert _supports_media_in_tool_results("openai", "gpt-5.4") is True

    def test_openai_codex_yes(self):
        assert _supports_media_in_tool_results("openai-codex", "gpt-5-codex") is True

    def test_gemini_3_yes(self):
        assert _supports_media_in_tool_results("google", "gemini-3-flash-preview") is True

    def test_gemini_2_no(self):
        assert _supports_media_in_tool_results("google", "gemini-2.5-pro") is False

    def test_unknown_provider_conservative_no(self):
        assert _supports_media_in_tool_results("brand-new-provider", "any-model") is False

    def test_empty_provider_no(self):
        assert _supports_media_in_tool_results("", "anything") is False
        assert _supports_media_in_tool_results(None, "anything") is False  # type: ignore[arg-type]


# ─── _build_native_vision_tool_result ────────────────────────────────────────


class TestBuildNativeVisionToolResult:
    def test_envelope_shape(self):
        env = _build_native_vision_tool_result(
            image_url="/tmp/foo.png",
            question="what does it say?",
            image_data_url="data:image/png;base64,XYZ",
            image_size_bytes=1024,
        )
        assert env["_multimodal"] is True
        assert isinstance(env["content"], list)
        assert len(env["content"]) == 2
        assert env["content"][0]["type"] == "text"
        assert env["content"][1]["type"] == "image_url"
        assert env["content"][1]["image_url"]["url"] == "data:image/png;base64,XYZ"
        assert "what does it say?" in env["content"][0]["text"]
        assert "Image attached natively" in env["text_summary"]

    def test_no_question_omits_question_section(self):
        env = _build_native_vision_tool_result(
            image_url="/tmp/foo.png",
            question="",
            image_data_url="data:image/png;base64,XYZ",
            image_size_bytes=512,
        )
        text = env["content"][0]["text"]
        assert "Question:" not in text
        assert "Image loaded" in text


# ─── _vision_analyze_native ──────────────────────────────────────────────────


class TestVisionAnalyzeNative:
    def test_local_file_returns_multimodal_envelope(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(_TINY_PNG)
        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(str(img), "what is this?")
        )
        assert isinstance(result, dict)
        assert result.get("_multimodal") is True
        parts = result["content"]
        assert any(p.get("type") == "image_url" for p in parts)
        assert any(p.get("type") == "text" for p in parts)
        url = next(p["image_url"]["url"] for p in parts if p.get("type") == "image_url")
        assert url.startswith("data:image/")

    def test_missing_file_returns_error_string(self, tmp_path):
        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(str(tmp_path / "nope.png"), "?")
        )
        # tool_error returns a JSON string, not the multimodal envelope
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed.get("success") is False
        # Unified resolver: local backend reports a clean not-found.
        err = parsed.get("error", "").lower()
        assert "image file not found" in err or "no active sandbox" in err

    def test_empty_image_url_returns_error(self):
        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native("", "?")
        )
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed.get("success") is False
        assert "image_url is required" in parsed.get("error", "")

    def test_file_url_scheme_resolves(self, tmp_path):
        img = tmp_path / "t.png"
        img.write_bytes(_TINY_PNG)
        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(f"file://{img}", "?")
        )
        assert isinstance(result, dict)
        assert result.get("_multimodal") is True

    def test_oversized_image_resized_under_embed_cap(self, tmp_path):
        """Regression for the wedged-session incident (May 2026).

        A vision tool-result image is baked into conversation history and
        re-sent on every subsequent turn.  Anthropic rejects any single
        base64 image over 5 MB with a 400, and immutable history means the
        bad bytes can't be cleared by retrying — the session is permanently
        wedged.  The native fast path must proactively resize down to the
        embed cap (well under 5 MB) BEFORE embedding, not just at the 20 MB
        hard ceiling.  Skips if Pillow isn't available (resize is a no-op).
        """
        pytest = __import__("pytest")
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed — proactive resize is a no-op")

        from tools.vision_tools import _EMBED_TARGET_BYTES

        # Noisy PNG that base64-encodes to well over 5 MB (won't compress much).
        big = tmp_path / "big.png"
        Image.effect_noise((2600, 2600), 80).convert("RGB").save(big, format="PNG")
        assert big.stat().st_size * 4 // 3 > 5 * 1024 * 1024, "test image not big enough"

        result = asyncio.get_event_loop().run_until_complete(
            _vision_analyze_native(str(big), "describe")
        )
        assert isinstance(result, dict) and result.get("_multimodal") is True
        url = next(
            p["image_url"]["url"]
            for p in result["content"]
            if p.get("type") == "image_url"
        )
        assert len(url) <= _EMBED_TARGET_BYTES, (
            f"embedded image {len(url) / 1024 / 1024:.1f} MB exceeds embed cap "
            f"{_EMBED_TARGET_BYTES / 1024 / 1024:.0f} MB — would wedge sessions on Anthropic"
        )


# ─── _handle_vision_analyze fast-path gating ─────────────────────────────────


class TestHandleVisionAnalyzeFastPath:
    """Verify the dispatcher chooses fast-path vs aux-LLM correctly."""

    def test_vision_capable_main_model_uses_fast_path(self, tmp_path, monkeypatch):
        """Main model supports native vision → fast path returns multimodal."""
        img = tmp_path / "x.png"
        img.write_bytes(_TINY_PNG)

        # Set runtime override so the handler thinks we're on opus@openrouter
        from agent.auxiliary_client import set_runtime_main, clear_runtime_main
        set_runtime_main("openrouter", "anthropic/claude-opus-4.6")
        try:
            # Mock decide_image_input_mode to always return "native" so the
            # fast path fires regardless of model-catalog state in CI.
            with patch(
                "agent.image_routing.decide_image_input_mode",
                return_value="native",
            ):
                coro = _handle_vision_analyze({"image_url": str(img), "question": "?"})
                result = asyncio.get_event_loop().run_until_complete(coro)
        finally:
            clear_runtime_main()

        assert isinstance(result, dict), \
            f"Expected multimodal envelope, got {type(result).__name__}: {str(result)[:200]}"
        assert result.get("_multimodal") is True

    def test_non_vision_main_model_falls_through_to_aux(self, tmp_path, monkeypatch):
        """Non-vision main model → fast path skipped, aux LLM path attempted."""
        img = tmp_path / "x.png"
        img.write_bytes(_TINY_PNG)

        async def _aux_sentinel(*args, **kwargs):
            return '{"sentinel": "aux-path"}'

        from agent.auxiliary_client import set_runtime_main, clear_runtime_main
        set_runtime_main("openrouter", "qwen/qwen3-coder")
        try:
            with patch("tools.vision_tools.vision_analyze_tool", side_effect=_aux_sentinel):
                coro = _handle_vision_analyze({"image_url": str(img), "question": "?"})
                result = asyncio.get_event_loop().run_until_complete(coro)
        finally:
            clear_runtime_main()

        assert not (isinstance(result, dict) and result.get("_multimodal") is True), \
            "Fast path fired for non-vision model; should have fallen through to aux LLM"

    def test_fast_path_disabled_for_unsupported_provider(self, tmp_path, monkeypatch):
        """Even with vision-capable model, unknown provider → fall through."""
        img = tmp_path / "x.png"
        img.write_bytes(_TINY_PNG)

        async def _aux_sentinel(*args, **kwargs):
            return '{"sentinel": "aux-path"}'

        from agent.auxiliary_client import set_runtime_main, clear_runtime_main
        set_runtime_main("brand-new-provider", "anthropic/claude-opus-4.6")
        try:
            with patch("tools.vision_tools.vision_analyze_tool", side_effect=_aux_sentinel):
                coro = _handle_vision_analyze({"image_url": str(img), "question": "?"})
                result = asyncio.get_event_loop().run_until_complete(coro)
        finally:
            clear_runtime_main()

        assert not (isinstance(result, dict) and result.get("_multimodal") is True), \
            "Fast path fired for unknown provider; should have fallen through"

    def test_supports_vision_override_bypasses_provider_allowlist(self, tmp_path):
        """supports_vision=true enables the fast path on an unlisted provider."""
        img = tmp_path / "x.png"
        img.write_bytes(_TINY_PNG)

        async def _aux_sentinel(*args, **kwargs):
            return '{"sentinel": "aux-path"}'

        from agent.auxiliary_client import set_runtime_main, clear_runtime_main
        set_runtime_main("brand-new-provider", "llava-v1.6")
        try:
            with patch(
                "hermes_cli.config.load_config",
                return_value={"model": {"supports_vision": True}},
            ), patch(
                "tools.vision_tools.vision_analyze_tool", side_effect=_aux_sentinel,
            ) as mock_aux:
                coro = _handle_vision_analyze({"image_url": str(img), "question": "?"})
                result = asyncio.get_event_loop().run_until_complete(coro)
        finally:
            clear_runtime_main()

        assert isinstance(result, dict) and result.get("_multimodal") is True
        mock_aux.assert_not_called()

    def test_text_mode_wins_over_supports_vision_override(self, tmp_path):
        """Explicit text routing blocks the fast path even with supports_vision."""
        img = tmp_path / "x.png"
        img.write_bytes(_TINY_PNG)

        async def _aux_sentinel(*args, **kwargs):
            return '{"sentinel": "aux-path"}'

        from agent.auxiliary_client import set_runtime_main, clear_runtime_main
        set_runtime_main("brand-new-provider", "llava-v1.6")
        try:
            with patch(
                "hermes_cli.config.load_config",
                return_value={
                    "agent": {"image_input_mode": "text"},
                    "model": {"supports_vision": True},
                },
            ), patch(
                "tools.vision_tools.vision_analyze_tool", side_effect=_aux_sentinel,
            ) as mock_aux:
                coro = _handle_vision_analyze({"image_url": str(img), "question": "?"})
                result = asyncio.get_event_loop().run_until_complete(coro)
        finally:
            clear_runtime_main()

        assert isinstance(result, str)
        assert json.loads(result) == {"sentinel": "aux-path"}
        mock_aux.assert_called_once()


# ─── stale inbound-cache image guard ─────────────────────────────────────────


class TestStaleInboundCacheGuard:
    """Native fast path must refuse an inbound-cache image that wasn't attached
    on the current turn.

    Regression for 2026-06-17: a stale ``image_cache/img_*.jpg`` path recalled
    from memory was loaded and described as the wrong image. The guard only
    polices files inside the inbound image-cache dir; everything else (browser
    screenshots, generated images, arbitrary local files) and CLI/test contexts
    with no turn set are unaffected.
    """

    def _run(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)

    def test_no_turn_context_allows_any_cache_path(self, tmp_path, monkeypatch):
        """Default (contextvar None) — CLI/tests — never restricts."""
        from tools import vision_tools

        cache_dir = tmp_path / "image_cache"
        cache_dir.mkdir()
        img = cache_dir / "img_deadbeef0001.png"
        img.write_bytes(_TINY_PNG)
        monkeypatch.setattr(
            "gateway.platforms.base.get_image_cache_dir", lambda: cache_dir
        )

        # No set_current_turn_image_paths call -> contextvar is None -> inert.
        result = self._run(vision_tools._vision_analyze_native(str(img), "?"))
        assert isinstance(result, dict) and result.get("_multimodal") is True

    def test_stale_cache_path_rejected_when_not_in_turn(self, tmp_path, monkeypatch):
        """A cache image absent from the current turn's set is refused."""
        from tools import vision_tools

        cache_dir = tmp_path / "image_cache"
        cache_dir.mkdir()
        fresh = cache_dir / "img_fresh00000001.png"
        fresh.write_bytes(_TINY_PNG)
        stale = cache_dir / "img_stale00000002.png"
        stale.write_bytes(_TINY_PNG)
        monkeypatch.setattr(
            "gateway.platforms.base.get_image_cache_dir", lambda: cache_dir
        )

        # This turn attached only the fresh image.
        token = vision_tools.set_current_turn_image_paths([str(fresh)])
        try:
            stale_res = self._run(vision_tools._vision_analyze_native(str(stale), "?"))
            fresh_res = self._run(vision_tools._vision_analyze_native(str(fresh), "?"))
        finally:
            vision_tools.reset_current_turn_image_paths(token)

        # Stale path fails closed with a clear, actionable error.
        assert isinstance(stale_res, str)
        parsed = json.loads(stale_res)
        assert parsed.get("success") is False
        assert "earlier message" in parsed.get("error", "")
        # Fresh path (in the turn set) still works.
        assert isinstance(fresh_res, dict) and fresh_res.get("_multimodal") is True

    def test_empty_turn_set_makes_all_cache_paths_stale(self, tmp_path, monkeypatch):
        """A turn that attached no inbound images rejects any cache path.

        This is the text-routing / recall case: the user's current message had
        no image, but a stale cache path was recalled into context.
        """
        from tools import vision_tools

        cache_dir = tmp_path / "image_cache"
        cache_dir.mkdir()
        stale = cache_dir / "img_stale00000003.png"
        stale.write_bytes(_TINY_PNG)
        monkeypatch.setattr(
            "gateway.platforms.base.get_image_cache_dir", lambda: cache_dir
        )

        token = vision_tools.set_current_turn_image_paths([])  # no images this turn
        try:
            result = self._run(vision_tools._vision_analyze_native(str(stale), "?"))
        finally:
            vision_tools.reset_current_turn_image_paths(token)

        assert isinstance(result, str)
        assert json.loads(result).get("success") is False

    def test_non_cache_path_never_restricted(self, tmp_path, monkeypatch):
        """Files outside the inbound cache dir are out of scope for the guard.

        Browser screenshots, generated images, and arbitrary local paths must
        keep working even when the turn set is empty.
        """
        from tools import vision_tools

        cache_dir = tmp_path / "image_cache"
        cache_dir.mkdir()
        monkeypatch.setattr(
            "gateway.platforms.base.get_image_cache_dir", lambda: cache_dir
        )

        # Image lives OUTSIDE the inbound cache dir (e.g. a generated image).
        other = tmp_path / "generated_image.png"
        other.write_bytes(_TINY_PNG)

        token = vision_tools.set_current_turn_image_paths([])  # empty turn set
        try:
            result = self._run(vision_tools._vision_analyze_native(str(other), "?"))
        finally:
            vision_tools.reset_current_turn_image_paths(token)

        assert isinstance(result, dict) and result.get("_multimodal") is True
