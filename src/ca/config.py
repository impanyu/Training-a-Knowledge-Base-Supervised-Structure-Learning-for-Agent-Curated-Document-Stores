"""Centralization configurations C0-C7 (v3: single-factor design).

Configs are NOT cumulative. C0 is the fully decentralized baseline and every
other multi-agent config flips EXACTLY ONE mechanism relative to it, so any
measured effect attributes cleanly to that one mechanism. The flags are
independent booleans, so arbitrary combinations remain mechanically possible
for follow-up experiments -- the C0-C7 table just does not use them.

C7 is the solo baseline (one agent, no collaboration at all).

Info centralization was deleted in v3: `retrieve` reads a shared corpus that is
infrastructure, so there is nothing for an agent-level monopoly to add.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LevelConfig:
    level: str
    n_agents: int
    has_interface: bool
    world_access: str = "all"          # "all" | "interface" (list/claim/deliver to WORLD)
    central_pricing: bool = False      # interface sets ALL contract prices; bargaining off
    central_credit: bool = False       # interface is the sole lender
    star_comms: bool = False           # non-interface agents may only reach the interface
    shared_solution_memory: bool = False  # one solution KV store for everyone (T26)
    collective_goal: bool = False      # root goal = TOTAL system balance, not one's own


CONFIGS: dict[str, LevelConfig] = {
    # demand, pricing, credit and comms centralization each need a hub agent to
    # hold the power; shared memory and the collective goal do not.
    "C0": LevelConfig("C0", 8, False),
    "C1": LevelConfig("C1", 8, True, world_access="interface"),
    "C2": LevelConfig("C2", 8, False, shared_solution_memory=True),
    "C3": LevelConfig("C3", 8, True, central_pricing=True),
    "C4": LevelConfig("C4", 8, True, central_credit=True),
    "C5": LevelConfig("C5", 8, True, star_comms=True),
    "C6": LevelConfig("C6", 8, False, collective_goal=True),
    "C7": LevelConfig("C7", 1, False),
}


def agent_ids(level: LevelConfig) -> list[str]:
    if level.has_interface:
        return ["interface"] + [f"agent_{i}" for i in range(1, level.n_agents)]
    return [f"agent_{i}" for i in range(1, level.n_agents + 1)]


@dataclass
class ExperimentConfig:
    level: LevelConfig
    seed: int
    seed_capital_total: int
    fifo_k: int = 6
    retrieve_k: int = 3          # passages per retrieve call (billable-cost knob)
    list_top_n: int = 20            # open tasks shown per page by list_tasks
    claim_ttl: int = 8              # rounds before an undelivered claim reopens
    max_rounds: int = 60
    model: str = "gpt-5-mini"
    max_tokens_per_turn: int = 4096  # reasoning models spend thinking tokens from this budget
    temperature: float = 0.3     # low variance for stable agent policies
    loan_rate: float = 0.01      # interest charged per round on outstanding loan principal
    interface_turns_per_round: int = 1  # extra interface turns/round at has_interface levels
