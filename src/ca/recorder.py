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
    answered                       {agent: {n_answered, f1_sum, em_sum}}
    n_answered / total_f1 / total_em
    board                          {open, active_claims, delivered} question counts
    total_units / remaining_units / demand_absorbed
    n_contracts / contracts_by_status {status: count}
    n_loans / loan_principal_outstanding / interest_paid_total
    memory                         {agent: {answers, notes, n_claims,
                                   n_memory_hits, n_repeat_deliveries, n_improved}}
    n_claims / n_memory_hits / memory_hit_rate / improvement_rate
    answers_in_memory_total

Formulas mirror ca.metrics on the final round (answers_in_memory_total sums
per-agent counts, so at C2 the shared bucket is counted once per agent, exactly
as compute_metrics does)."""
import json
from collections import defaultdict
from pathlib import Path

from ca.actions import IMPROVED_MARKER, MEMORY_HIT_MARKER, STORED_F1_MARKER


def _mem_tally() -> dict:
    return {"n_claims": 0, "n_memory_hits": 0,
            "n_repeat_deliveries": 0, "n_improved": 0}


class Recorder:
    def __init__(self, out_dir: str, append: bool = False):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"  # append = resumed run (T29)
        self._f = open(self.dir / "trace.jsonl", mode)  # fresh trace per run
        self._ts = open(self.dir / "timeseries.jsonl", mode)
        self._tokens = defaultdict(lambda: {"solving": 0, "admin": 0})
        # memory tallies, counted live off the event stream so write_summary
        # need not re-scan the trace file.
        self._memory = defaultdict(_mem_tally)

    def log(self, event: dict) -> None:
        spent = event["tokens_in"] + event["tokens_out"]
        self._tokens[event["agent"]][event["category"]] += spent
        result = str(event["result"])
        tally = self._memory[event["agent"]]
        if event["action"] == "claim_question" and not result.startswith("ERROR"):
            tally["n_claims"] += 1
            # a "hit" is a claim whose result carried a stored answer -- the
            # auto-recall header is the stable marker
            if MEMORY_HIT_MARKER in result:
                tally["n_memory_hits"] += 1
        elif event["action"] == "deliver_work" and STORED_F1_MARKER in result:
            # a repeat delivery of a question this agent already had graded
            tally["n_repeat_deliveries"] += 1
            if IMPROVED_MARKER in result:
                tally["n_improved"] += 1
        self._f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._f.flush()

    def _memory_block(self, infra) -> dict:
        out = {}
        for a in infra.agent_ids:
            answers = infra.memory.n_answers(a)
            out[a] = {"answers": answers,
                      "notes": infra.memory.n_entries(a) - answers,
                      **self._memory[a]}
        return out

    def log_round(self, infra, round_no: int) -> dict:
        """Append one cumulative snapshot (schema in the module docstring) for
        the just-finished round. Cheap enough to recompute from board state
        every tick at this experiment's scale."""
        agents = infra.agent_ids
        tokens = {a: dict(self._tokens[a]) for a in agents}
        solving = sum(t["solving"] for t in tokens.values())
        admin = sum(t["admin"] for t in tokens.values())
        all_tok = solving + admin

        answered = {a: {"n_answered": 0, "f1_sum": 0.0, "em_sum": 0.0} for a in agents}
        for r in infra.board.results:
            tally = answered[r.agent]
            tally["n_answered"] += 1
            tally["f1_sum"] += r.f1
            tally["em_sum"] += r.em
        total_units = infra.bank.total_units()
        board = {"open": sum(1 for n in infra.board.remaining.values() if n > 0),
                 "active_claims": sum(len(c) for c in infra.board.active.values()),
                 "delivered": len(infra.board.results)}

        contracts_by_status: dict[str, int] = defaultdict(int)
        for c in infra.contracts.contracts.values():
            contracts_by_status[c.status] += 1
        loans = list(infra.loans.loans.values())
        memory = self._memory_block(infra)
        n_claims = sum(v["n_claims"] for v in memory.values())
        n_hits = sum(v["n_memory_hits"] for v in memory.values())
        n_repeats = sum(v["n_repeat_deliveries"] for v in memory.values())
        n_improved = sum(v["n_improved"] for v in memory.values())

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
            "board": board,
            "total_units": total_units,
            "remaining_units": sum(infra.board.remaining.values()),
            "demand_absorbed": (len(infra.board.results) / total_units
                                if total_units else 0.0),
            "n_contracts": len(infra.contracts.contracts),
            "contracts_by_status": dict(contracts_by_status),
            "n_loans": len(loans),
            "loan_principal_outstanding": sum(l.principal for l in loans
                                              if l.status == "active"),
            "interest_paid_total": infra.loans.total_interest_paid,
            "memory": memory,
            "n_claims": n_claims,
            "n_memory_hits": n_hits,
            "memory_hit_rate": n_hits / n_claims if n_claims else 0.0,
            "improvement_rate": n_improved / n_repeats if n_repeats else 0.0,
            "answers_in_memory_total": sum(v["answers"] for v in memory.values()),
        }
        self._ts.write(json.dumps(snap, ensure_ascii=False) + "\n")
        self._ts.flush()
        return snap

    def write_summary(self, infra, rounds_used: int) -> dict:
        loans = list(infra.loans.loans.values())
        debtors: dict[str, int] = defaultdict(int)
        for loan in loans:
            if loan.status == "active":
                debtors[loan.borrower] += loan.principal
        summary = {
            "level": infra.cfg.level.level,
            "seed": infra.cfg.seed,
            "rounds_used": rounds_used,
            # v4: one row per DELIVERED UNIT -- the thing that is graded and
            # paid, and what the metrics read (topic rides along for
            # specialization, price/difficulty for post-hoc slicing).
            "deliveries": infra.board.results_json(),
            "total_units": infra.bank.total_units(),
            "remaining_units": sum(infra.board.remaining.values()),
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
            # per-agent memory footprint (what is stored) plus how much it was
            # hit and improved on this run
            "memory": self._memory_block(infra),
            "minted": infra.ledger.minted,
            "burned": infra.ledger.burned,
            "conservation_ok": infra.ledger.conservation_ok(),
        }
        with open(self.dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    def to_state(self) -> dict:
        return {"tokens": {a: dict(v) for a, v in self._tokens.items()},
                "memory": {a: dict(v) for a, v in self._memory.items()}}

    def from_state(self, state: dict) -> None:
        for a, v in state["tokens"].items():
            self._tokens[a].update(v)
        for a, v in state["memory"].items():
            self._memory[a].update(v)

    def close(self):
        self._f.close()
        self._ts.close()
