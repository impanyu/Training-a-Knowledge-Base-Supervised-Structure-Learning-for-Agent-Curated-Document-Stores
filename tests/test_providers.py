from ca.agent import LLMPolicy
from ca.providers import make_policy, resolve_endpoint, to_openai_tools


def test_to_openai_tools_wraps_schema():
    tools = [{"name": "pay", "description": "d", "input_schema": {"type": "object", "properties": {}, "required": []}}]
    out = to_openai_tools(tools)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "pay"
    assert out[0]["function"]["parameters"]["type"] == "object"


def test_resolve_endpoint_routing():
    assert resolve_endpoint("deepseek-v4-flash") == ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY")
    assert resolve_endpoint("gpt-5.4-nano") == (None, "OPENAI_API_KEY")


def test_make_policy_routes_claude_to_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert isinstance(make_policy("claude-haiku-4-5"), LLMPolicy)
