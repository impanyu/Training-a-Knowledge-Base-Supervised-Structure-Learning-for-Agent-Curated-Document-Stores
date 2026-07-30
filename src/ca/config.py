"""Centralization levels L0-L5. Levels differ ONLY through these fields."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LevelConfig:
    level: str
    n_agents: int
    has_interface: bool
    world_access: str        # "all" | "interface"  (list/claim/deliver to WORLD)
    retrieve_access: str     # "all" | "interface"
    central_pricing: bool    # interface sets ALL contract prices; bargaining disabled
    star_comms: bool         # non-interface agents may only interact with interface


LEVELS: dict[str, LevelConfig] = {
    "L0": LevelConfig("L0", 8, False, "all", "all", False, False),
    "L1": LevelConfig("L1", 8, True, "interface", "all", False, False),
    "L2": LevelConfig("L2", 8, True, "interface", "interface", False, False),
    "L3": LevelConfig("L3", 8, True, "interface", "interface", True, False),
    "L4": LevelConfig("L4", 8, True, "interface", "interface", True, True),
    "L5": LevelConfig("L5", 1, False, "all", "all", False, False),
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
    list_top_n: int = 20            # open questions shown by list_questions
    claim_ttl: int = 8              # rounds before an undelivered claim reopens
    max_rounds: int = 60
    model: str = "gpt-5-mini"
    max_tokens_per_turn: int = 4096  # reasoning models spend thinking tokens from this budget
    temperature: float = 0.3     # low variance for stable agent policies
    loan_rate: float = 0.01      # interest charged per round on outstanding loan principal
