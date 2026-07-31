"""Crash-safe JSONL trace + per-round timeseries + end-of-run summary.

T28 timeseries.jsonl schema -- one JSON object per round, appended by
`log_round` after every completed round; every field is CUMULATIVE as of the
end of that round (a crash mid-round loses only that round's line):

    round                          1-based round index
    bankrupt                       [agent, ...]
    balances                       {agent: tokens}
    total_balance / escrow_total / minted / burned
    tokens                         {agent: {solving, admin}} burned
    solving_total / admin_total
    coordination_overhead          admin / (admin + solving), zero-guarded
    coordination_overhead_by_agent {agent: same, per agent}
    answered                       {agent: {n_answered, f1_sum, em_sum}},
                                   attributed to the WORLD deliverer
    n_answered / total_f1 / total_em
    tasks_closed                   {agent: closed task count (deliverer)}
    n_tasks_closed / task_completion_rate (closed / posted)
    board                          {open, claimed, closed} posted-task counts
    n_contracts / contracts_by_status {status: count}
    n_loans / loan_principal_outstanding / interest_paid_total
    solutions                      {agent: {n_recalls, n_recall_hits,
                                   answers_in_memory, decompositions_in_memory}}
    n_recalls / n_recall_hits / answers_in_memory_total

Formulas mirror ca.metrics on the final round (answers_in_memory_total sums
per-agent stats, so at C2 the shared bucket is counted once per agent, exactly
as compute_metrics does)."""
import json
from collections import defaultdict
from pathlib import Path


class Recorder:
    def __init__(self, out_dir: str, append: bool = False):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"  # append = resumed run (T29)
        self._f = open(self.dir / "trace.jsonl", mode)  # fresh trace per run
        self._ts = open(self.dir / "timeseries.jsonl", mode)
        self._tokens = defaultdict(lambda: {"solving": 0, "admin": 0})
        # T27: solution-reuse tallies, counted live off the event stream so
        # write_summary need not re-scan the trace file.
        self._recalls = defaultdict(lambda: {"n_recalls": 0, "n_recall_hits": 0})

    def log(self, event: dict) -> None:
        spent = event["tokens_in"] + event["tokens_out"]
        self._tokens[event["agent"]][event["category"]] += spent
        if event["action"] == "recall_solutions":
            tally = self._recalls[event["agent"]]
            tally["n_recalls"] += 1
            # a "hit" is a recall whose result names at least one known
            # answer: does NOT start with "ERROR" (unresolvable/bankrupt) and
            # does NOT start with "(no stored" (empty store, or structure
            # without answers -- "(no stored answers yet").
            result_str = str(event["result"])
            if not result_str.startswith("ERROR") and not result_str.startswith("(no stored"):
                tally["n_recall_hits"] += 1
        self._f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._f.flush()

    def log_round(self, infra, round_no: int) -> dict:
        """Append one cumulative snapshot (schema in the module docstring) for
        the just-finished round. Cheap enough to recompute from board state
        every tick at this experiment's scale."""
        agents = infra.agent_ids
        tokens = {a: dict(self._tokens[a]) for a in agents}
        solving = sum(t["solving"] for t in tokens.values())
        admin = sum(t["admin"] for t in tokens.values())
        all_tok = solving + admin

        board = {"open": 0, "claimed": 0, "closed": 0}
        tasks_closed = {a: 0 for a in agents}
        answered = {a: {"n_answered": 0, "f1_sum": 0.0, "em_sum": 0.0} for a in agents}
        task_results = infra.board.results()
        for t in task_results:
            board[t["status"]] += 1
            if t["status"] == "closed":
                tasks_closed[t["claimed_by"]] += 1
        for row in infra.board.leaf_results():
            if row["status"] == "closed":
                tally = answered[row["claimed_by"]]
                tally["n_answered"] += 1
                tally["f1_sum"] += row["score"]
                tally["em_sum"] += row["em"]

        contracts_by_status: dict[str, int] = defaultdict(int)
        for c in infra.contracts.contracts.values():
            contracts_by_status[c.status] += 1
        loans = list(infra.loans.loans.values())
        solutions = {
            a: {**self._recalls[a],
                "answers_in_memory": infra.solutions.stats(a)["answers"],
                "decompositions_in_memory": infra.solutions.stats(a)["decompositions"]}
            for a in agents
        }

        snap = {
            "round": round_no,
            "bankrupt": [a for a in agents if infra.ledger.is_bankrupt(a)],
            "balances": {a: infra.ledger.balance(a) for a in agents},
            "total_balance": sum(infra.ledger.balance(a) for a in agents),
            "escrow_total": sum(infra.ledger.escrow.values()),
            "minted": infra.ledger.minted,
            "burned": infra.ledger.burned,
            "tokens": tokens,
            "solving_total": solving,
            "admin_total": admin,
            "coordination_overhead": admin / all_tok if all_tok else 0.0,
            "coordination_overhead_by_agent": {
                a: (t["admin"] / (t["admin"] + t["solving"])
                    if t["admin"] + t["solving"] else 0.0)
                for a, t in tokens.items()
            },
            "answered": answered,
            "n_answered": sum(v["n_answered"] for v in answered.values()),
            "total_f1": sum(v["f1_sum"] for v in answered.values()),
            "total_em": sum(v["em_sum"] for v in answered.values()),
            "tasks_closed": tasks_closed,
            "n_tasks_closed": board["closed"],
            "task_completion_rate": (board["closed"] / len(task_results)
                                     if task_results else 0.0),
            "board": board,
            "n_contracts": len(infra.contracts.contracts),
            "contracts_by_status": dict(contracts_by_status),
            "n_loans": len(loans),
            "loan_principal_outstanding": sum(l.principal for l in loans
                                              if l.status == "active"),
            "interest_paid_total": infra.loans.total_interest_paid,
            "solutions": solutions,
            "n_recalls": sum(v["n_recalls"] for v in solutions.values()),
            "n_recall_hits": sum(v["n_recall_hits"] for v in solutions.values()),
            "answers_in_memory_total": sum(v["answers_in_memory"]
                                           for v in solutions.values()),
        }
        self._ts.write(json.dumps(snap, ensure_ascii=False) + "\n")
        self._ts.flush()
        return snap

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
            # per-agent solution-memory footprint: what's stored (T26) plus
            # how much it got reused this run (T27).
            "solutions": {
                a: {**infra.solutions.stats(a), **self._recalls[a]}
                for a in infra.agent_ids
            },
            "minted": infra.ledger.minted,
            "burned": infra.ledger.burned,
            "conservation_ok": infra.ledger.conservation_ok(),
        }
        with open(self.dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    def to_state(self) -> dict:
        return {"tokens": {a: dict(v) for a, v in self._tokens.items()},
                "recalls": {a: dict(v) for a, v in self._recalls.items()}}

    def from_state(self, state: dict) -> None:
        for a, v in state["tokens"].items():
            self._tokens[a].update(v)
        for a, v in state["recalls"].items():
            self._recalls[a].update(v)

    def close(self):
        self._f.close()
        self._ts.close()
