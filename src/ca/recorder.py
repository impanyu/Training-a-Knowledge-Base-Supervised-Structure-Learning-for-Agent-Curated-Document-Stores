"""Crash-safe JSONL trace + end-of-run summary."""
import json
from collections import defaultdict
from pathlib import Path


class Recorder:
    def __init__(self, out_dir: str):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._f = open(self.dir / "trace.jsonl", "w")  # fresh trace per run
        self._tokens = defaultdict(lambda: {"solving": 0, "admin": 0})

    def log(self, event: dict) -> None:
        spent = event["tokens_in"] + event["tokens_out"]
        self._tokens[event["agent"]][event["category"]] += spent
        self._f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._f.flush()

    def write_summary(self, infra, rounds_used: int) -> dict:
        summary = {
            "level": infra.cfg.level.level,
            "seed": infra.cfg.seed,
            "rounds_used": rounds_used,
            # v2: the board posts task TREES. "tasks" is the task-level record;
            # "questions" flattens it to one row per (task, leaf) pair -- the
            # unit that is actually graded and paid, and what the metrics read.
            "tasks": infra.board.results(),
            "questions": infra.board.leaf_results(),
            "balances": {a: infra.ledger.balance(a) for a in infra.agent_ids},
            "tokens": {a: dict(self._tokens[a]) for a in infra.agent_ids},
            "bankrupt": [a for a in infra.agent_ids if infra.ledger.is_bankrupt(a)],
            "n_contracts": len(infra.contracts.contracts),
            "contract_prices": [c.price for c in infra.contracts.contracts.values()
                                if c.status == "delivered"],
            "minted": infra.ledger.minted,
            "burned": infra.ledger.burned,
            "conservation_ok": infra.ledger.conservation_ok(),
        }
        with open(self.dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    def close(self):
        self._f.close()
