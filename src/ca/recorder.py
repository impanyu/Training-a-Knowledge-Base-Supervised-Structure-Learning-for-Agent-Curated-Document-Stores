"""Crash-safe JSONL trace + end-of-run summary."""
import json
from collections import defaultdict
from pathlib import Path


class Recorder:
    def __init__(self, out_dir: str):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._f = open(self.dir / "trace.jsonl", "w")  # fresh trace per run
        self._tokens = defaultdict(lambda: {"billable": 0, "free": 0})

    def log(self, event: dict) -> None:
        spent = event["tokens_in"] + event["tokens_out"]
        key = "billable" if event["billable"] else "free"
        self._tokens[event["agent"]][key] += spent
        self._f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._f.flush()

    def write_summary(self, infra, rounds_used: int) -> dict:
        summary = {
            "level": infra.cfg.level.level,
            "seed": infra.cfg.seed,
            "rounds_used": rounds_used,
            "questions": [{
                "qid": q.qid, "status": q.status, "difficulty": q.difficulty,
                "price": q.price, "claimed_by": q.claimed_by,
                "submitted": q.submitted, "score": q.score, "em": q.em,
                "payout": q.payout,
            } for q in infra.board.results()],
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
