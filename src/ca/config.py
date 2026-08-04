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
    has_hub: bool
    world_access: str = "all"          # "all" | "hub" (list/claim/deliver to WORLD)
    central_pricing: bool = False      # hub sets ALL contract prices; bargaining off
    central_credit: bool = False       # hub is the sole lender
    star_comms: bool = False           # non-hub agents may only reach the hub
    shared_memory: bool = False        # one long-term memory for everyone (C2)
    collective_goal: bool = False      # root goal = TOTAL system balance, not one's own


CONFIGS: dict[str, LevelConfig] = {
    # demand, pricing, credit and comms centralization each need a hub agent to
    # hold the power; shared memory and the collective goal do not.
    "C0": LevelConfig("C0", 8, False),
    "C1": LevelConfig("C1", 8, True, world_access="hub"),
    "C2": LevelConfig("C2", 8, False, shared_memory=True),
    "C3": LevelConfig("C3", 8, True, central_pricing=True),
    "C4": LevelConfig("C4", 8, True, central_credit=True),
    "C5": LevelConfig("C5", 8, True, star_comms=True),
    "C6": LevelConfig("C6", 8, False, collective_goal=True),
    "C7": LevelConfig("C7", 1, False),
}


def agent_ids(level: LevelConfig) -> list[str]:
    if level.has_hub:
        return ["hub"] + [f"agent_{i}" for i in range(1, level.n_agents)]
    return [f"agent_{i}" for i in range(1, level.n_agents + 1)]


@dataclass
class ExperimentConfig:
    level: LevelConfig
    seed: int
    seed_capital_total: int
    fifo_k: int = 6
    retrieve_k: int = 3          # passages per retrieve call (billable-cost knob)
    list_top_n: int = 20            # open jobs shown per page by list_jobs
    # rounds before an undelivered job claim reopens. Calibrated, not guessed:
    # at the measured 2.9 turns per answer a solo agent needs ~23 rounds for a
    # job of 8, while a delegating holder needs ~18 (claim + 5 contracts +
    # collection + 3 questions of its own + assembly). 20 sits between the two,
    # so going it alone does not fit and delegating does -- this is the lever
    # that creates the market v4 lost.
    claim_ttl: int = 20
    max_rounds: int = 60
    model: str = "gpt-5-mini"
    max_tokens_per_turn: int = 4096  # reasoning models spend thinking tokens from this budget
    temperature: float = 0.3     # low variance for stable agent policies
    loan_rate: float = 0.01      # interest charged per round on outstanding loan principal
    hub_turns_per_round: int = 1  # extra hub turns/round at has_hub levels
    solo_turns_per_round: int = 1  # solo-agent turns/round at C7 (8 = compute parity with 8-agent configs)
    checkpoint_every: int = 20   # full-state checkpoint every N rounds (T29)
