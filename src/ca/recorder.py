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
        task_results = infra.board.results()
        loans = list(infra.loans.loans.values())
        debtors: dict[str, int] = defaultdict(int)
        for loan in loans:
            if loan.status == "active":
                debtors[loan.borrower] += loan.principal
        summary = {
            "level": infra.cfg.level.level,
            "seed": infra.cfg.seed,
            "rounds_used": rounds_used,
            # v2: the board posts task TREES. "tasks" is the task-level record;
            # "questions" flattens it to one row per (task, leaf) pair -- the
            # unit that is actually graded and paid, and what the metrics read.
            "tasks": task_results,
            "questions": infra.board.leaf_results(),
            # per-delivery leaf->agent attribution: the input metrics.specialization
            # needs (which agent delivered which leaves, under which task).
            "deliveries": [
                {"task": t["nid"], "agent": t["claimed_by"], "total_payout": t["payout"],
                 "n_leaves": t["n_leaves"],
                 "per_leaf": [{"qid": l["qid"], "f1": l["score"]} for l in t["leaves"]]}
                for t in task_results if t["status"] == "closed"
            ],
            "balances": {a: infra.ledger.balance(a) for a in infra.agent_ids},
            "tokens": {a: dict(self._tokens[a]) for a in infra.agent_ids},
            "bankrupt": [a for a in infra.agent_ids if infra.ledger.is_bankrupt(a)],
            "n_contracts": len(infra.contracts.contracts),
            "contract_prices": [c.price for c in infra.contracts.contracts.values()
                                if c.status == "delivered"],
            "loans": {
                "n_proposed": len(loans),
                "n_active": sum(1 for l in loans if l.status == "active"),
                "n_repaid": sum(1 for l in loans if l.status == "repaid"),
                "total_principal_outstanding": sum(debtors.values()),
                "total_interest_paid": infra.loans.total_interest_paid,
                "debtors": dict(debtors),
                "bankrupt_with_debt": [a for a in debtors if infra.ledger.is_bankrupt(a)],
            },
            "minted": infra.ledger.minted,
            "burned": infra.ledger.burned,
            "conservation_ok": infra.ledger.conservation_ok(),
        }
        with open(self.dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    def close(self):
        self._f.close()
