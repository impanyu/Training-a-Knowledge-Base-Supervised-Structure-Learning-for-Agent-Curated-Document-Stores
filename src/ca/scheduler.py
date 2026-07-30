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
        try:
            for r in range(1, self.cfg.max_rounds + 1):
                self.infra.round = r
                rounds_used = r
                self.infra.board.expire_claims(r, self.cfg.claim_ttl)
                for ev in self.infra.loans.interest_tick():
                    verb = "paid" if ev["paid"] else "capitalized"
                    self.recorder.log({
                        "round": r, "agent": ev["borrower"], "action": "__interest__",
                        "input": {"lid": ev["lid"], "lender": ev["lender"]},
                        "result": f"{verb} {ev['interest']}",
                        "category": "admin", "tokens_in": 0, "tokens_out": 0,
                        "balance_after": self.infra.ledger.balance(ev["borrower"]),
                    })
                order = list(self.agents)
                self.rng.shuffle(order)
                for agent in order:
                    event = agent.take_turn()
                    self.recorder.log(event)
                assert self.infra.ledger.conservation_ok(), \
                    f"conservation violated in round {r}"
                if self.infra.board.all_done():
                    break
                if all(self.infra.ledger.is_bankrupt(a) for a in self.infra.agent_ids):
                    break  # terminal: solving actions frozen for everyone, no income possible
        finally:
            # a crash (or a tripped invariant) must not cost us the run's data
            summary = self.recorder.write_summary(self.infra, rounds_used)
            self.recorder.close()
        return summary
