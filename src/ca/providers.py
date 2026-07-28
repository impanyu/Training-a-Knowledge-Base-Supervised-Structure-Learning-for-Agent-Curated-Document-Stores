"""Multi-provider LLM policies: Anthropic native + OpenAI-compatible (GPT, DeepSeek)."""
import json
import sys

from ca.agent import Decision, LLMPolicy


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Anthropic tool format -> OpenAI function-calling format."""
    return [{"type": "function",
             "function": {"name": t["name"],
                          "description": t["description"],
                          "parameters": t["input_schema"]}}
            for t in tools]


def resolve_endpoint(model: str) -> tuple[str | None, str]:
    """Return (base_url or None for OpenAI default, env var name for the key)."""
    if model.startswith("deepseek"):
        return "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"
    return None, "OPENAI_API_KEY"


class OpenAICompatPolicy:
    """One tool call per turn against any OpenAI-compatible chat API."""

    def __init__(self, model: str, max_tokens: int = 1024):
        import os

        from openai import OpenAI
        base_url, key_env = resolve_endpoint(model)
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(key_env),
                             max_retries=5)
        self.model = model
        self.max_tokens = max_tokens

    def decide(self, system: str, context: str, tools: list[dict]) -> Decision:
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": context}],
            tools=to_openai_tools(tools),
            tool_choice="required",
        )
        # Newer GPT models (gpt-5*, o*) reject max_tokens in favor of
        # max_completion_tokens; other OpenAI-compatible endpoints (DeepSeek,
        # older GPT) still expect max_tokens.
        if self.model.startswith("gpt-5") or self.model.startswith("o"):
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
        # DeepSeek's OpenAI-compatible endpoint doesn't support
        # parallel_tool_calls; every other endpoint here only ever makes one
        # call per turn anyway, so just omit it for deepseek.
        if not self.model.startswith("deepseek"):
            kwargs["parallel_tool_calls"] = False
        else:
            # DeepSeek V4 defaults to thinking mode, which rejects
            # tool_choice="required"; the forced-single-action turn needs
            # thinking off (symmetric with the no-thinking Anthropic setup).
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            print(f"[llm-error] {e}", file=sys.stderr)
            return Decision("__noop__", {}, 0, 0)
        usage = resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        calls = resp.choices[0].message.tool_calls or []
        if not calls:
            return Decision("__noop__", {}, in_tok, out_tok)
        call = calls[0]
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        return Decision(call.function.name, args, in_tok, out_tok)


def make_policy(model: str, max_tokens: int = 1024):
    if model.startswith("claude"):
        return LLMPolicy(model, max_tokens)
    return OpenAICompatPolicy(model, max_tokens)
