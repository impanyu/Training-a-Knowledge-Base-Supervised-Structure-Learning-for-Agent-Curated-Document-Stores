"""Synchronous round scheduler: arrivals first, then seeded-shuffled turns."""
import random

from ca import checkpoint
from ca.agent import Agent
from ca.chat import EXTERNAL
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
        self.rng = rng   # per-round turn shuffle only; the stream owns its own

    def run(self, start_round: int = 1) -> dict:
        rounds_used = start_round - 1
        try:
            for r in range(start_round, self.cfg.max_rounds + 1):
                self.infra.round = r
                rounds_used = r
                # arrivals land as external-thread messages BEFORE any turn,
                # so the assignee's notification line shows them this round
                for qid, agent in self.infra.stream.tick(r):
                    q = self.infra.bank.questions[qid]
                    self.infra.chat.send(EXTERNAL, agent, f"[{qid}] {q.text}", r)
                order = list(self.agents)
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
                done = self.infra.stream.all_done()
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
