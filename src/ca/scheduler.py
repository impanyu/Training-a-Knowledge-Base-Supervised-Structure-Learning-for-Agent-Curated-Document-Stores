"""Synchronous round-robin scheduler with seeded per-round shuffling."""
import random

from ca import checkpoint
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

    def run(self, start_round: int = 1) -> dict:
        rounds_used = start_round - 1
        try:
            for r in range(start_round, self.cfg.max_rounds + 1):
                self.infra.round = r
                rounds_used = r
                if self.cfg.claim_ttl is not None:
                    self.infra.board.expire_claims(r, self.cfg.claim_ttl)
                order = list(self.agents)
                if self.cfg.level.has_hub and self.cfg.hub_turns_per_round > 1:
                    iface = next(a for a in self.agents if a.id == "hub")
                    order += [iface] * (self.cfg.hub_turns_per_round - 1)
                if len(self.agents) == 1 and self.cfg.solo_turns_per_round > 1:
                    order += [self.agents[0]] * (self.cfg.solo_turns_per_round - 1)
                self.rng.shuffle(order)
                for agent in order:
                    event = agent.take_turn()
                    self.recorder.log(event)
                # T28: snapshot each completed round (before the break check,
                # so the final round is captured too); a crash mid-round means
                # no line for that round, but the finally still writes summary.
                self.recorder.log_round(self.infra, r)
                # T29: checkpoint after the round is fully committed (timeseries
                # line included) -- every N rounds, at max_rounds, and whenever
                # the run is about to stop, so the last completed round of ANY
                # finished run is always resumable.
                done = self.infra.board.all_done()
                if (r % self.cfg.checkpoint_every == 0 or r == self.cfg.max_rounds
                        or done):
                    checkpoint.save(
                        self.recorder.dir / f"checkpoint_{r:04d}.json",
                        checkpoint.capture(self.infra, self.agents, self.recorder,
                                           self.rng, r))
                if done:
                    break
        finally:
            # a crash (or a tripped invariant) must not cost us the run's data
            summary = self.recorder.write_summary(self.infra, rounds_used)
            self.recorder.close()
        return summary
