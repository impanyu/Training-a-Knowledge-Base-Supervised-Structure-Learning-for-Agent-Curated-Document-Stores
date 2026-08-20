"""Policies: one tool call per turn. The LLM policy targets any
OpenAI-compatible chat API (spec default gpt-5-mini); ScriptedPolicy is the
deterministic test double. Adapted from ca.providers for kb's static
9-action schema; kept import-independent of ca."""
import json
import sys
from dataclasses import dataclass


@dataclass
class Decision:
    name: str
    inp: dict
    in_tokens: int
    out_tokens: int


class ScriptedPolicy:
    """Replays a fixed (name, input) list; records every (system, context,
    tool-names) triple it was shown, so tests can assert on prompts. An
    exhausted script emits done() — off-mode phases turn that into the
    runtime error result and burn budget, which is exactly what exhaustion
    tests need."""

    def __init__(self, script, in_tokens: int = 0, out_tokens: int = 0):
        self.script = list(script)
        self.in_tokens, self.out_tokens = in_tokens, out_tokens
        self.seen: list[tuple[str, str, tuple[str, ...]]] = []

    def decide(self, system: str, context: str, tools: list[dict]) -> Decision:
        self.seen.append((system, context, tuple(t["name"] for t in tools)))
        if not self.script:
            return Decision("done", {}, self.in_tokens, self.out_tokens)
        name, inp = self.script.pop(0)
        return Decision(name, dict(inp), self.in_tokens, self.out_tokens)


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Anthropic tool format -> OpenAI function-calling format."""
    return [{"type": "function",
             "function": {"name": t["name"],
                          "description": t["description"],
                          "parameters": t["input_schema"]}}
            for t in tools]


class OpenAICompatPolicy:
    """One tool call per turn, tool_choice=required (spec §2/§5)."""

    def __init__(self, model: str, max_tokens: int = 12000,
                 temperature: float = 0.3, reasoning_effort: str = "low"):
        import os

        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=5)
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature

    def decide(self, system: str, context: str, tools: list[dict]) -> Decision:
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": context}],
            tools=to_openai_tools(tools),
            tool_choice="required",
            parallel_tool_calls=False,
        )
        # gpt-5*/o* reasoning models reject non-default temperature and
        # max_tokens (they want max_completion_tokens).
        if self.model.startswith("gpt-5") or self.model.startswith("o"):
            kwargs["max_completion_tokens"] = self.max_tokens
            # Reasoning is the wall clock here: at default effort a curation
            # step took ~16 s and a 300-iteration run projected to 33 hours.
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = self.temperature
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
            # A reasoning model that exhausts its completion budget while
            # thinking returns no tool call at all: the step is lost and its
            # tokens are spent for nothing, so the budget is set well above
            # what a normal turn needs rather than trimmed to it.
            if out_tok >= self.max_tokens - 8:
                print(f"[llm-truncated] no tool call after {out_tok} tokens",
                      file=sys.stderr)
            return Decision("__noop__", {}, in_tok, out_tok)
        call = calls[0]
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        return Decision(call.function.name, args, in_tok, out_tok)


def make_policy(model: str, max_tokens: int = 12000, temperature: float = 0.3,
                reasoning_effort: str = "low"):
    return OpenAICompatPolicy(model, max_tokens, temperature, reasoning_effort)
