"""Centralization configurations (v6: single-factor design, no economy).

Configs are NOT cumulative. C0 is the fully decentralized baseline and every
other multi-agent config flips EXACTLY ONE mechanism relative to it, so any
measured effect attributes cleanly to that one mechanism. The flags are
independent booleans, so arbitrary combinations remain mechanically possible
for follow-up experiments -- the table just does not use them.

C7 is the solo baseline (one agent, no collaboration at all).

The gap in the naming (no C3/C4/C6) is deliberate: those three centralized
pricing, credit and the goal function, all of which died with the token
economy in v6. The survivors keep the identities they had in v3-v5 so earlier
results stay referenceable.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LevelConfig:
    level: str
    n_agents: int
    has_hub: bool
    world_access: str = "all"          # "all" | "hub" (list/claim/deliver to WORLD)
    star_comms: bool = False           # non-hub agents may only reach the hub
    shared_memory: bool = False        # one long-term memory for everyone (C2)


CONFIGS: dict[str, LevelConfig] = {
    # task access and comms centralization each need a hub agent to hold the
    # power; shared memory does not.
    "C0": LevelConfig("C0", 8, False),
    "C1": LevelConfig("C1", 8, True, world_access="hub"),
    "C2": LevelConfig("C2", 8, False, shared_memory=True),
    "C5": LevelConfig("C5", 8, True, star_comms=True),
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
    fifo_k: int = 10
    memory_k: int = 5            # rows per memory_search
    list_top_n: int = 20         # open questions shown per page by list_questions
    # None = claims never expire. The TTL exists only to stop an agent squatting
    # on work it will not deliver; v4.1 briefly misused it as a device to force
    # delegation, which produced deadline behaviour (filler answers) rather than
    # cooperation. release_question is the voluntary way to hand work back.
    claim_ttl: int | None = None
    max_rounds: int = 60
    model: str = "gpt-5-mini"
    max_tokens_per_turn: int = 4096  # reasoning models spend thinking tokens from this budget
    temperature: float = 0.3     # low variance for stable agent policies
    hub_turns_per_round: int = 1  # extra hub turns/round at has_hub levels
    solo_turns_per_round: int = 1  # solo-agent turns/round at C7 (8 = compute parity with 8-agent configs)
    checkpoint_every: int = 20   # full-state checkpoint every N rounds (T29)
