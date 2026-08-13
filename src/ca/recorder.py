"""Crash-safe JSONL trace + per-round timeseries + end-of-run summary.

timeseries.jsonl schema -- one JSON object per round, appended by `log_round`
after every completed round; every field is CUMULATIVE as of the end of that
round (a crash mid-round loses only that round's line):

    round                          1-based round index
    arrivals_total                 external questions arrived so far
    pending                        arrived, not yet delivered
    answered_total / coverage      delivered, and delivered / arrived
    mean_latency                   mean rounds from arrival to delivery
    total_f1 / total_em
    agents                         {agent: {answered, f1_sum, selfqa, notes}}
    kb_answers / kb_selfqa         KB entries by kind, whole cluster
    n_messages                     peer-to-peer chat messages sent so far
    tokens                         {agent: {solving, admin}} measured, not charged
    solving_total / admin_total
    coordination_overhead          admin / (admin + solving), zero-guarded

Formulas mirror ca.metrics on the final round. Corpus entries never appear in
any tally."""
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
        # solving-turn tallies for proactive_ratio, counted live off the
        # event stream so write_summary need not re-scan the trace file
        self._turns = {"solving": 0, "selfqa": 0}

    def log(self, event: dict) -> None:
        spent = event["tokens_in"] + event["tokens_out"]
        self._tokens[event["agent"]][event["category"]] += spent
        if event["category"] == "solving":
            self._turns["solving"] += 1
            if (event["action"] == "record_qa"
                    and not str(event["result"]).startswith("ERROR")):
                self._turns["selfqa"] += 1
        self._f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._f.flush()

    def _agents_block(self, infra) -> dict:
        out = {a: {"answered": 0, "f1_sum": 0.0, "em_sum": 0.0,
                   "selfqa": infra.memory.count("selfqa", a),
                   "notes": infra.memory.count("note", a)}
               for a in infra.agent_ids}
        for r in infra.stream.results:
            tally = out[r.agent]
            tally["answered"] += 1
            tally["f1_sum"] += r.f1
            tally["em_sum"] += r.em
        return out

    def log_round(self, infra, round_no: int) -> dict:
        """Append one cumulative snapshot (schema in the module docstring) for
        the just-finished round. Cheap enough to recompute from stream state
        every round at this experiment's scale."""
        tokens = {a: dict(self._tokens[a]) for a in infra.agent_ids}
        solving = sum(t["solving"] for t in tokens.values())
        admin = sum(t["admin"] for t in tokens.values())
        all_tok = solving + admin

        results = infra.stream.results
        arrived = infra.stream.pos
        answered = len(results)
        agents = self._agents_block(infra)

        snap = {
            "round": round_no,
            "arrivals_total": arrived,
            "pending": len(infra.stream.pending),
            "answered_total": answered,
            "coverage": answered / arrived if arrived else 0.0,
            "mean_latency": (sum(r.latency for r in results) / answered
                             if answered else 0.0),
            "total_f1": sum(r.f1 for r in results),
            "total_em": sum(r.em for r in results),
            "agents": agents,
            "kb_answers": infra.memory.count("answer"),
            "kb_selfqa": infra.memory.count("selfqa"),
            "n_messages": infra.chat.n_agent_messages,
            "tokens": tokens,
            "solving_total": solving,
            "admin_total": admin,
            "coordination_overhead": admin / all_tok if all_tok else 0.0,
        }
        self._ts.write(json.dumps(snap, ensure_ascii=False) + "\n")
        self._ts.flush()
        return snap

    def write_summary(self, infra, rounds_used: int) -> dict:
        summary = {
            "level": infra.cfg.level.level,
            "seed": infra.cfg.seed,
            "rounds_used": rounds_used,
            # one row per DELIVERED question -- the thing that is graded, and
            # what the metrics read (latency for the headline, topic for
            # specialization, difficulty for post-hoc slicing)
            "deliveries": infra.stream.results_json(),
            "arrived_total": infra.stream.pos,
            "pending": len(infra.stream.pending),
            "tokens": {a: dict(self._tokens[a]) for a in infra.agent_ids},
            "n_messages": infra.chat.n_agent_messages,
            "turns": dict(self._turns),
            "agents": self._agents_block(infra),
            "kb_answers": infra.memory.count("answer"),
            "kb_selfqa": infra.memory.count("selfqa"),
        }
        with open(self.dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    def to_state(self) -> dict:
        return {"tokens": {a: dict(v) for a, v in self._tokens.items()},
                "turns": dict(self._turns)}

    def from_state(self, state: dict) -> None:
        for a, v in state["tokens"].items():
            self._tokens[a].update(v)
        self._turns.update(state["turns"])

    def close(self):
        self._f.close()
        self._ts.close()
