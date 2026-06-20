"""Proof that the existing fallback chain can fail over to the copilot-acp
provider END-TO-END through the REAL resolver — no mock of resolve_provider_client.

This is the load-bearing verification for the Claude ACP subscription fallback:
the win is that Hermes already routes external_process providers through
`resolve_provider_client` + duck-typed client assignment, so a `copilot-acp`
entry in `fallback_providers` activates with the genuine `CopilotACPClient`
WITHOUT any core code change. These tests lock that contract so it can't
silently regress.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


WRAPPER = "/home/aldo/.hermes/acp-fallback/claude-acp-run.sh"


def _make_agent_with_acp_fallback():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.anthropic.com",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent.provider = "anthropic"
        agent.model = "claude-opus-4-8"
        # Inject the ACP entry as the fallback chain directly (mirrors what
        # config `fallback_providers: [{provider: copilot-acp, ...}]` builds).
        agent._fallback_chain = [
            {"provider": "copilot-acp", "model": "claude-opus-4-8"},
        ]
        agent._fallback_index = 0
        return agent


@pytest.mark.skipif(
    not os.path.exists(WRAPPER),
    reason="ACP bridge wrapper not installed on this host",
)
def test_chain_fails_over_to_real_copilot_acp_client(monkeypatch):
    """Force the chain to walk: it must activate and land the REAL
    CopilotACPClient via the live resolver (no resolve_provider_client mock)."""
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", WRAPPER)

    from agent.copilot_acp_client import CopilotACPClient

    agent = _make_agent_with_acp_fallback()
    activated = agent._try_activate_fallback()

    assert activated is True, "fallback chain failed to activate copilot-acp entry"
    assert agent.provider == "copilot-acp"
    assert agent._fallback_index == 1
    # The decisive assertion: the live client the agent will USE is the ACP
    # subprocess client, not a misrouted REST client.
    assert isinstance(agent.client, CopilotACPClient), (
        f"expected CopilotACPClient, got {type(agent.client).__name__}"
    )


@pytest.mark.skipif(
    not os.path.exists(WRAPPER),
    reason="ACP bridge wrapper not installed on this host",
)
def test_acp_resolver_returns_working_client(monkeypatch):
    """Narrower contract: resolve_provider_client builds a real ACP client for
    the copilot-acp provider (the external_process auth branch)."""
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", WRAPPER)
    from agent.auxiliary_client import resolve_provider_client
    from agent.copilot_acp_client import CopilotACPClient

    client, model = resolve_provider_client(
        "copilot-acp", model="claude-opus-4-8", raw_codex=True
    )
    assert client is not None, "resolver returned None -> chain would SKIP acp entry"
    assert isinstance(client, CopilotACPClient)
    assert model == "claude-opus-4-8"
