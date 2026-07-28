"""Synchronous round-robin scheduler with seeded per-round shuffling."""
import random

from ca.agent import Agent
from ca.config import ExperimentConfig
from ca.infra import Infra
from ca.recorder import Recorder


class Scheduler:
    def __init__(self, infra: Infra, agents: list[Agent], cfg: ExperimentConfig,
                 recorder: Recorder, rng: random.Random):
        self.infra = infra
        self.agents = agents
        self.cfg = cfg
        self.recorder = recorder
        self.rng = rng

    def run(self) -> dict:
        rounds_used = 0
        for r in range(1, self.cfg.max_rounds + 1):
            self.infra.round = r
            rounds_used = r
            self.infra.board.expire_claims(r, self.cfg.claim_ttl)
            order = list(self.agents)
            self.rng.shuffle(order)
            for agent in order:
                event = agent.take_turn()
                self.recorder.log(event)
            assert self.infra.ledger.conservation_ok(), f"conservation violated in round {r}"
            if self.infra.board.all_done():
                break
        summary = self.recorder.write_summary(self.infra, rounds_used)
        self.recorder.close()
        return summary
