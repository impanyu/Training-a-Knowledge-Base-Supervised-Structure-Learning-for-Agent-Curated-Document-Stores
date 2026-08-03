"""Full-state checkpoint + resume (T29).

A checkpoint is one JSON file (`checkpoint_XXXX.json`, XXXX = completed round)
holding every piece of MUTABLE run state: ledger, board, contracts, loans,
chat, per-agent short-term memories (fifo / goals / scratchpads), the vector
long-term memory, recorder tallies, and the scheduler's RNG state. Static
structure (question bank, retriever index, policies, action tables) is NOT
serialized -- a resumed run rebuilds it from the same CLI args and `restore`
overwrites only the mutable parts. Restoring the RNG makes the per-round
shuffle -- and therefore the whole continuation -- identical to a run that
never stopped.
"""
import json
import os
import random

from ca.config import ExperimentConfig


def config_state(cfg: ExperimentConfig) -> dict:
    return {"level": cfg.level.level, "seed": cfg.seed,
            "seed_capital_total": cfg.seed_capital_total, "model": cfg.model,
            "fifo_k": cfg.fifo_k, "claim_ttl": cfg.claim_ttl,
            "loan_rate": cfg.loan_rate,
            "hub_turns_per_round": cfg.hub_turns_per_round,
            "solo_turns_per_round": cfg.solo_turns_per_round,
            "checkpoint_every": cfg.checkpoint_every}


def validate(state: dict, cfg: ExperimentConfig) -> None:
    c = state["config"]
    if c["level"] != cfg.level.level:
        raise ValueError(f"checkpoint is for level {c['level']}, "
                         f"but this run is level {cfg.level.level}")
    if c["seed"] != cfg.seed:
        raise ValueError(f"checkpoint is for seed {c['seed']}, "
                         f"but this run uses seed {cfg.seed}")


def rng_state(rng: random.Random) -> list:
    version, internal, gauss_next = rng.getstate()
    return [version, list(internal), gauss_next]


def restore_rng(rng: random.Random, state: list) -> None:
    version, internal, gauss_next = state
    rng.setstate((version, tuple(internal), gauss_next))


def capture(infra, agents, recorder, rng: random.Random, round_no: int) -> dict:
    return {
        "round": round_no,
        "config": config_state(infra.cfg),
        "ledger": infra.ledger.to_state(),
        "board": infra.board.to_state(),
        "contracts": infra.contracts.to_state(),
        "loans": infra.loans.to_state(),
        "chat": infra.chat.to_state(),
        "memory": infra.memory.to_state(),
        "scratchpads": {a: {t: list(notes) for t, notes in pads.items()}
                        for a, pads in infra.scratchpads.items()},
        "agents": {ag.id: {"fifo": ag.fifo.to_state(), "goals": ag.goals.to_state()}
                   for ag in agents},
        "recorder": recorder.to_state(),
        "rng": rng_state(rng),
    }


def save(path, state: dict) -> None:
    """Temp file + atomic rename: a crash mid-write never corrupts the
    previous checkpoint."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, path)


def load(path) -> dict:
    with open(path) as f:
        return json.load(f)


def restore(state: dict, infra, agents, recorder, rng: random.Random) -> None:
    infra.ledger.from_state(state["ledger"])
    infra.board.from_state(state["board"])
    infra.contracts.from_state(state["contracts"])
    infra.loans.from_state(state["loans"])
    infra.chat.from_state(state["chat"])
    infra.memory.from_state(state["memory"])
    infra.scratchpads.clear()
    for a, pads in state["scratchpads"].items():
        for t, notes in pads.items():
            infra.scratchpads[a][t] = list(notes)
    for ag in agents:
        ag.fifo.from_state(state["agents"][ag.id]["fifo"])
        ag.goals.from_state(state["agents"][ag.id]["goals"])
    recorder.from_state(state["recorder"])
    restore_rng(rng, state["rng"])
    infra.round = state["round"]
