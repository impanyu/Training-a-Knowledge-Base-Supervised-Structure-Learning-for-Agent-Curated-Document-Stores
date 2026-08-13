"""Experiment arms (v7: proactive vs passive).

The centralization spectrum is retired. v7 is ONE organizational form -- a
cluster of always-on domain experts over one shared knowledge base -- and the
two arms differ ONLY in what idle time is for:

    P0  proactive: idle turns invent likely questions, answer them, and bank
        the results with record_qa.
    B0  passive baseline: identical world, routing, threads, KB and actions,
        EXCEPT record_qa is absent from the catalog and the system prompt
        carries no proactive protocol.

The `proactive` flag gates exactly those two things and nothing else, so any
measured difference attributes cleanly to the proactive protocol.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LevelConfig:
    level: str
    proactive: bool


CONFIGS: dict[str, LevelConfig] = {
    "P0": LevelConfig("P0", proactive=True),
    "B0": LevelConfig("B0", proactive=False),
}


def agent_ids(n_agents: int) -> list[str]:
    return [f"agent_{i}" for i in range(1, n_agents + 1)]


@dataclass
class ExperimentConfig:
    level: LevelConfig
    seed: int
    n_agents: int = 8            # agents == domain clusters (CLI --agents)
    arrival_rate: float = 0.5    # Poisson lambda: external questions per round
    fifo_k: int = 10
    memory_k: int = 5            # rows per memory_search
    max_rounds: int = 60
    model: str = "gpt-5-mini"
    max_tokens_per_turn: int = 4096  # reasoning models spend thinking tokens from this budget
    temperature: float = 0.3     # low variance for stable agent policies
    checkpoint_every: int = 20   # full-state checkpoint every N rounds (T29)
