"""Full-state checkpoint + resume (T29).

A checkpoint is one JSON file (`checkpoint_XXXX.json`, XXXX = completed round)
holding every piece of MUTABLE run state: the question stream (order position,
its own rng, pending, results), chat threads + unread counters, per-agent
short-term memories (fifo / goals), the shared vector KB (notes / self-QA /
answers only -- the seeded corpus is static), recorder tallies, and the
scheduler's RNG state. Static structure (question bank, corpus + embeddings,
cluster cache, policies, action tables) is NOT serialized -- a resumed run
rebuilds and re-seeds it from the same CLI args and `restore` overwrites only
the mutable parts. Restoring both RNGs makes the arrival schedule and the
per-round shuffle -- and therefore the whole continuation -- identical to a
run that never stopped.
"""
import json
import os
import random

from ca.config import ExperimentConfig


def config_state(cfg: ExperimentConfig) -> dict:
    return {"level": cfg.level.level, "seed": cfg.seed, "model": cfg.model,
            "n_agents": cfg.n_agents, "arrival_rate": cfg.arrival_rate,
            "fifo_k": cfg.fifo_k, "checkpoint_every": cfg.checkpoint_every}


def validate(state: dict, cfg: ExperimentConfig) -> None:
    c = state["config"]
    for key, mine in (("level", cfg.level.level), ("seed", cfg.seed),
                      ("n_agents", cfg.n_agents),
                      ("arrival_rate", cfg.arrival_rate)):
        if c[key] != mine:
            raise ValueError(f"checkpoint is for {key} {c[key]}, "
                             f"but this run uses {key} {mine}")


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
        "stream": infra.stream.to_state(),
        "chat": infra.chat.to_state(),
        "memory": infra.memory.to_state(),
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
    infra.stream.from_state(state["stream"])
    infra.chat.from_state(state["chat"])
    infra.memory.from_state(state["memory"])
    for ag in agents:
        ag.fifo.from_state(state["agents"][ag.id]["fifo"])
        ag.goals.from_state(state["agents"][ag.id]["goals"])
    recorder.from_state(state["recorder"])
    restore_rng(rng, state["rng"])
    infra.round = state["round"]
