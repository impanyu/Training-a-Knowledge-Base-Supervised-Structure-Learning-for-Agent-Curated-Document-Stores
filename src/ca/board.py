"""WORLD's question board (v4): quota accounting over the flat bank.

Each question is posted with `quota` claimable units. Claiming takes a unit out
of the pool, an expired claim puts it back (a failed attempt must not destroy
demand), and a delivery consumes it for good. Strikes are per (question, agent)
and are never returned, so an agent gets at most two attempts at any question
while other agents remain free to claim it.
"""
import zlib
from dataclasses import dataclass

from ca.bank import BankError, Question, QuestionBank
from ca.economy import Ledger
from ca.grader import exact_match, f1

MAX_STRIKES = 2


class BoardError(Exception):
    pass


@dataclass
class Claim:
    agent: str
    round: int


@dataclass
class Result:
    qid: str
    agent: str
    submitted: str
    f1: float
    em: float
    payout: int
    round: int


class QuestionBoard:
    def __init__(self, bank: QuestionBank, ledger: Ledger):
        self.bank = bank
        self.ledger = ledger
        self.remaining: dict[str, int] = {qid: q.quota
                                          for qid, q in bank.questions.items()}
        self.active: dict[str, list[Claim]] = {qid: [] for qid in bank.questions}
        self.strikes: dict[tuple[str, str], int] = {}
        self.results: list[Result] = []
        self.round = 0

    # ---------- listing ----------

    def open_questions(self, viewer: str | None = None) -> list[Question]:
        """Questions with units left. Without a viewer: qid order (stable, for
        tests and summaries). With a viewer: a per-viewer stable shuffle --
        everyone sees the same SET, each in its own fixed order, so the whole
        economy does not stampede the same top-priced question."""
        qids = [qid for qid, n in self.remaining.items() if n > 0]
        if viewer is None:
            qids.sort()
        else:
            qids.sort(key=lambda qid: zlib.crc32(f"{viewer}:{qid}".encode()))
        return [self.bank.questions[qid] for qid in qids]

    # ---------- lifecycle ----------

    def claim(self, agent: str, qid: str, round_no: int = 0) -> Question:
        q = self._get(qid)
        self.round = max(self.round, round_no)
        if self.strikes.get((q.qid, agent), 0) >= MAX_STRIKES:
            raise BoardError(f"you have already claimed {q.qid} twice; it is "
                             "closed to you - work on a different question")
        if any(c.agent == agent for c in self.active[q.qid]):
            raise BoardError(f"you already hold an active claim on {q.qid}; "
                             "deliver it before claiming it again")
        if self.remaining[q.qid] == 0:
            raise BoardError(f"{q.qid} has no units left - its quota is taken; "
                             "call list_questions to see what is open NOW")
        self.remaining[q.qid] -= 1
        self.active[q.qid].append(Claim(agent, round_no))
        self.strikes[(q.qid, agent)] = self.strikes.get((q.qid, agent), 0) + 1
        return q

    def expire_claims(self, round_no: int, ttl: int) -> list[str]:
        """Drop claims older than `ttl` rounds and return their unit to the
        pool, so one agent sitting on claims cannot stall the board. The strike
        stays: the attempt was spent."""
        self.round = max(self.round, round_no)
        expired: list[str] = []
        for qid, claims in self.active.items():
            keep = []
            for c in claims:
                if round_no - c.round > ttl:
                    self.remaining[qid] += 1
                    expired.append(qid)
                else:
                    keep.append(c)
            self.active[qid] = keep
        return expired

    def deliver(self, agent: str, qid: str, answer: str,
                round_no: int | None = None) -> Result:
        """Grade a short answer against the gold answers, mint price x F1, and
        consume the unit (remaining is NOT restored)."""
        q = self._get(qid)
        claim = next((c for c in self.active[q.qid] if c.agent == agent), None)
        if claim is None:
            raise BoardError(f"you hold no active claim on {q.qid} - "
                             "claim_question it first")
        rnd = self.round if round_no is None else round_no
        self.round = max(self.round, rnd)
        submitted = str(answer)
        score = f1(submitted, q.answers)
        payout = round(q.price * score)
        self.active[q.qid].remove(claim)
        r = Result(q.qid, agent, submitted, score,
                   exact_match(submitted, q.answers), payout, rnd)
        self.results.append(r)
        if payout > 0:
            self.ledger.mint(agent, payout)
        return r

    def all_done(self) -> bool:
        return (all(n == 0 for n in self.remaining.values())
                and not any(self.active.values()))

    def _get(self, qid: str) -> Question:
        try:
            return self.bank.get(qid)
        except BankError as e:
            raise BoardError(str(e)) from e

    # ---------- reporting ----------

    def results_json(self) -> list[dict]:
        """One row per delivered unit -- the thing that is graded and paid."""
        rows = []
        for r in self.results:
            q = self.bank.questions[r.qid]
            rows.append({"qid": r.qid, "agent": r.agent, "submitted": r.submitted,
                         "f1": r.f1, "em": r.em, "payout": r.payout,
                         "round": r.round, "price": q.price,
                         "difficulty": q.difficulty, "topic": q.topic})
        return rows

    # ---------- checkpoint (T29) ----------

    def to_state(self) -> dict:
        return {
            "round": self.round,
            "remaining": dict(self.remaining),
            "active": {qid: [[c.agent, c.round] for c in claims]
                       for qid, claims in self.active.items() if claims},
            "strikes": [[qid, agent, n] for (qid, agent), n in self.strikes.items()],
            "results": [[r.qid, r.agent, r.submitted, r.f1, r.em, r.payout, r.round]
                        for r in self.results],
        }

    def from_state(self, state: dict) -> None:
        self.round = state["round"]
        self.remaining = dict(state["remaining"])
        self.active = {qid: [] for qid in self.bank.questions}
        for qid, claims in state["active"].items():
            self.active[qid] = [Claim(a, r) for a, r in claims]
        self.strikes = {(qid, agent): n for qid, agent, n in state["strikes"]}
        self.results = [Result(*row) for row in state["results"]]
