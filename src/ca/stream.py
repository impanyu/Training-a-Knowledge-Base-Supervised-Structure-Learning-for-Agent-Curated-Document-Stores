"""External question stream: seeded arrivals, cluster routing, grading.

Questions ARRIVE over the run instead of sitting in a posted pile. The stream owns
the shuffled arrival order, the Poisson draw per round, the routing table
(cluster i -> agent_{i+1}), the pending set, and the graded results. It has
its OWN rng, independent of the scheduler's, so the arrival schedule depends
only on (bank, seed, N, rate) -- identical for the P0 and B0 arms.

A question is `pending` from arrival until its assignee delivers; there are
no reservations and no expiry. `deliver` is the ONE graded attempt: F1/EM vs the
hidden golds, latency = round_delivered - round_arrived.
"""
import math
import random
from dataclasses import dataclass

from ca.bank import QuestionBank


class StreamError(Exception):
    pass


@dataclass
class Result:
    qid: str
    agent: str
    submitted: str
    f1: float
    em: float
    round_in: int
    round_out: int

    @property
    def latency(self) -> int:
        return self.round_out - self.round_in


class QuestionStream:
    def __init__(self, bank: QuestionBank, n_agents: int, seed: int,
                 arrival_rate: float, assignment: dict[str, int]):
        self.bank = bank
        self.arrival_rate = arrival_rate
        missing = set(bank.questions) - set(assignment)
        if missing:
            raise ValueError(f"assignment misses {len(missing)} bank questions "
                             f"(e.g. {sorted(missing)[0]})")
        bad = {c for qid, c in assignment.items()
               if qid in bank.questions and not 0 <= c < n_agents}
        if bad:
            raise ValueError(f"assignment uses clusters {sorted(bad)} outside "
                             f"0..{n_agents - 1}")
        self.routing = {qid: f"agent_{assignment[qid] + 1}"
                        for qid in bank.questions}
        self.rng = random.Random(seed)
        self.order = sorted(bank.questions)
        self.rng.shuffle(self.order)
        self.pos = 0                                    # arrived = order[:pos]
        self.pending: dict[str, tuple[str, int]] = {}   # qid -> (agent, round_in)
        self.closed: set[str] = set()
        self.results: list[Result] = []
        self._last_tick = 0

    # ---------- arrivals ----------

    def _poisson(self) -> int:
        # Knuth: number of unit-exponential gaps fitting in one rate window
        limit, k, p = math.exp(-self.arrival_rate), 0, 1.0
        while True:
            p *= self.rng.random()
            if p <= limit:
                return k
            k += 1

    def tick(self, round_no: int) -> list[tuple[str, str]]:
        """The round's arrivals as (qid, agent), drawn once per round -- a
        repeat call for the same (or an older) round arrives nothing. The
        caller appends the question message to the assignee's external
        thread. Exhausts gracefully: when the order runs out, no more
        arrivals ever."""
        if round_no <= self._last_tick:
            return []
        self._last_tick = round_no
        if self.pos >= len(self.order):
            return []
        arrivals = []
        for _ in range(self._poisson()):
            if self.pos >= len(self.order):
                break
            qid = self.order[self.pos]
            self.pos += 1
            agent = self.routing[qid]
            self.pending[qid] = (agent, round_no)
            arrivals.append((qid, agent))
        return arrivals

    # ---------- delivery ----------

    def deliver(self, agent: str, qid: str, answer: str, round_no: int) -> Result:
        from ca.grader import exact_match, f1
        q = self.bank.get(qid)                  # unknown ids error with near ids
        if q.qid in self.closed:
            raise StreamError(f"{q.qid} has already been answered and closed")
        held = self.pending.get(q.qid)
        if held is None:
            raise StreamError(f"{q.qid} is not an open external question - "
                              "answer the questions in your external thread")
        assignee, round_in = held
        if assignee != agent:
            raise StreamError(f"{q.qid} is assigned to {assignee}, not you")
        submitted = str(answer)
        r = Result(q.qid, agent, submitted, f1(submitted, q.answers),
                   exact_match(submitted, q.answers), round_in, round_no)
        del self.pending[q.qid]
        self.closed.add(q.qid)
        self.results.append(r)
        return r

    def all_done(self) -> bool:
        return self.pos >= len(self.order) and not self.pending

    # ---------- reporting ----------

    def results_json(self) -> list[dict]:
        """One row per delivered question. `topic`/`difficulty` are hidden
        bank metadata, carried for post-hoc slicing."""
        rows = []
        for r in self.results:
            q = self.bank.questions[r.qid]
            rows.append({"qid": r.qid, "agent": r.agent, "submitted": r.submitted,
                         "f1": r.f1, "em": r.em, "round_in": r.round_in,
                         "round_out": r.round_out, "latency": r.latency,
                         "topic": q.topic, "difficulty": q.difficulty})
        return rows

    # ---------- checkpoint ----------

    def to_state(self) -> dict:
        version, internal, gauss_next = self.rng.getstate()
        return {
            "pos": self.pos,
            "rng": [version, list(internal), gauss_next],
            "pending": {qid: [a, r] for qid, (a, r) in self.pending.items()},
            "results": [[r.qid, r.agent, r.submitted, r.f1, r.em,
                         r.round_in, r.round_out] for r in self.results],
            "last_tick": self._last_tick,
        }

    def from_state(self, state: dict) -> None:
        self.pos = state["pos"]
        version, internal, gauss_next = state["rng"]
        self.rng.setstate((version, tuple(internal), gauss_next))
        self.pending = {qid: (a, r) for qid, (a, r) in state["pending"].items()}
        self.results = [Result(*row) for row in state["results"]]
        self.closed = {r.qid for r in self.results}
        self._last_tick = state["last_tick"]
