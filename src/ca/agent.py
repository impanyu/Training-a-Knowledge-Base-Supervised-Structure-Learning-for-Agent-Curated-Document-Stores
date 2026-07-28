"""Agent = short-term memory + a policy that picks one action per turn."""
import json
from dataclasses import dataclass
from typing import Protocol

import anthropic

from ca.actions import dispatch, is_billable, permission_error, visible_tools
from ca.config import ExperimentConfig
from ca.context import render_turn, system_prompt
from ca.infra import Infra
from ca.memory import FifoMemory, GoalStack


@dataclass
class Decision:
    name: str
    inp: dict
    in_tokens: int
    out_tokens: int


class Policy(Protocol):
    def decide(self, system: str, context: str, tools: list[dict]) -> Decision: ...


class ScriptedPolicy:
    """Deterministic policy for tests: replays a fixed action list."""

    def __init__(self, script: list[tuple[str, dict]], in_tokens: int = 0, out_tokens: int = 0):
        self.script = list(script)
        self.in_tokens, self.out_tokens = in_tokens, out_tokens

    def decide(self, system, context, tools) -> Decision:
        if not self.script:
            return Decision("check_balance", {}, self.in_tokens, self.out_tokens)
        name, inp = self.script.pop(0)
        return Decision(name, inp, self.in_tokens, self.out_tokens)


class LLMPolicy:
    def __init__(self, model: str, max_tokens: int = 1024):
        self.client = anthropic.Anthropic(max_retries=5)
        self.model = model
        self.max_tokens = max_tokens

    def decide(self, system, context, tools) -> Decision:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": context}],
                tools=tools,
                tool_choice={"type": "any", "disable_parallel_tool_use": True},
            )
        except Exception:
            # Deliberately broad: the SDK already retried: whatever still failed
            # must cost this agent one turn, not the whole multi-hour run.
            # Nothing is billed, so a dead API cannot bankrupt anyone either.
            return Decision("__noop__", {}, 0, 0)
        usage = resp.usage
        for block in resp.content:
            if block.type == "tool_use":
                return Decision(block.name, dict(block.input),
                                usage.input_tokens, usage.output_tokens)
        return Decision("__noop__", {}, usage.input_tokens, usage.output_tokens)


class Agent:
    def __init__(self, agent_id: str, cfg: ExperimentConfig, infra: Infra, policy: Policy):
        self.id = agent_id
        self.cfg = cfg
        self.infra = infra
        self.policy = policy
        self.fifo = FifoMemory(cfg.fifo_k, cfg.result_cap_recent,
                               cfg.result_cap_old, cfg.fifo_recent_n)
        self.goals = GoalStack("maximize token balance")
        self._system = system_prompt(cfg.level, agent_id, infra.agent_ids)
        self._tools = visible_tools(cfg.level, agent_id)

    def take_turn(self) -> dict:
        context = render_turn(self.infra, self.id, self.fifo, self.goals)
        d = self.policy.decide(self._system, context, self._tools)
        billable = False
        if d.name == "__noop__":
            result = "ERROR: no valid action produced this turn"
        elif d.name == "push_goal":
            self.goals.push(str(d.inp.get("note", "")))
            result = "goal pushed"
        elif d.name == "pop_goal":
            try:
                result = f"popped goal: {self.goals.pop()}"
            except IndexError as e:
                result = f"ERROR: {e}"
        else:
            err = permission_error(self.infra, self.id, d.name, d.inp)
            if err:
                result = f"ERROR: {err}"
            else:
                result = dispatch(self.infra, self.id, d.name, d.inp)
                billable = is_billable(d.name, d.inp) and not result.startswith("ERROR")
        if billable:
            self.infra.ledger.burn(self.id, d.in_tokens + d.out_tokens)
        self.infra.chat.mark_read(self.id)  # rendered messages are now "seen"
        # store the FULL result; FifoMemory.render() applies the display budget
        self.fifo.add(f"{d.name}({json.dumps(d.inp, ensure_ascii=False)[:120]})", result)
        return {
            "round": self.infra.round, "agent": self.id,
            "action": d.name, "input": d.inp, "result": result,
            "billable": billable, "tokens_in": d.in_tokens, "tokens_out": d.out_tokens,
            "balance_after": self.infra.ledger.balance(self.id),
        }
