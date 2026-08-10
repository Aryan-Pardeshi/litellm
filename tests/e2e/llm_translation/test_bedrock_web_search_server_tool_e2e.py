"""Live e2e: the Anthropic web_search server tool over Bedrock Invoke.

Bedrock hosts none of Anthropic's ``web_search_*`` server tools, so a
``/v1/messages`` request carrying one can only be served if the gateway
intercepts it and runs the search itself. When the bedrock messages config
claims it handles web search natively the tool is forwarded to AWS instead and
every such request comes back 400 "The provided request is not valid".

This is deliberately not the same thing as the ``web_search`` cells in the
Claude Code compat matrix. Those drive the CLI's *client-side* ``WebSearch``
tool, an ordinary custom tool the CLI executes and feeds back as a
``tool_result``; the CLI never emits a ``web_search_20250305`` definition, so
those cells stay green while this path is broken.

Prerequisites beyond the usual AWS credentials: the proxy config must enable
web-search interception for bedrock and declare a search backend, e.g.

    litellm_settings:
      websearch_interception_params:
        enabled_providers: ["bedrock"]
    search_tools:
      - search_tool_name: e2e-search
        litellm_params:
          search_provider: searxng
          api_base: http://127.0.0.1:8391

The gateway answers with the search text rather than Anthropic's
``server_tool_use`` / ``web_search_tool_result`` blocks: the interception hooks
rewrite the native tool into LiteLLM's own search tool before the short-circuit
runs, so the short-circuit no longer recognizes the request as native and emits
a plain text block. That is a separate gap from the one under test here.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import unwrap
from endpoints_client import EndpointsClient
from lifecycle import ResourceManager
from models import (
    AnthropicMessagesBody,
    AnthropicWebSearchTool,
    ChatMessage,
    LiteLLMParamsBody,
)

pytestmark = pytest.mark.e2e

BEDROCK_INVOKE_BACKEND = "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0"

WEB_SEARCH_TOOL = AnthropicWebSearchTool(
    type="web_search_20250305",
    name="web_search",
    max_uses=3,
)

SEARCH_PROMPT = "Use web search to tell me one recent news headline about Anthropic."


class TestBedrockWebSearchServerTool:
    @pytest.mark.covers("llm.messages.bedrock_invoke.web_search_server_tool.nonstream.works")
    def test_web_search_server_tool_is_served(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        """A bedrock deployment must answer a web_search server-tool request
        instead of handing the tool to AWS and returning its 400."""
        model = f"e2e-bedrock-websearch-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model=BEDROCK_INVOKE_BACKEND,
                aws_region_name="us-east-1",
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        response = unwrap(
            endpoints_client.proxy.messages(
                key,
                AnthropicMessagesBody(
                    model=model,
                    max_tokens=512,
                    tools=[WEB_SEARCH_TOOL],
                    messages=[ChatMessage(role="user", content=SEARCH_PROMPT)],
                ),
            )
        )

        assert response.content, f"no content blocks in response: {response}"
        answer = "\n".join(block.text or "" for block in response.content)
        assert "URL: http" in answer, (
            "the answer carries no search results, so the gateway did not run the "
            "search; Bedrock rejects the web_search tool itself, so a request that "
            f"reaches it comes back 400 instead. answer={answer[:300]!r}"
        )
