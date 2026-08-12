"""WORLD's question board (v6): claim, release and deliver per QUESTION.

A question holds ONE claimant at a time. Delivery is the short answer itself
(a bare string), graded by F1/EM and then closed for good. Releasing a claim
puts the question back on the open board for anyone. Strikes are per
(question, agent) and are never returned -- releasing included -- so an agent
gets at most two attempts at any question.
"""
import zlib
from dataclasses import dataclass

from ca.bank import BankError, Question, QuestionBank
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
    round: int


class QuestionBoard:
    def __init__(self, bank: QuestionBank):
        self.bank = bank
        self.active: dict[str, Claim] = {}          # qid -> its ONE claimant
        self.closed: set[str] = set()               # qid -> delivered, off the board
        self.strikes: dict[tuple[str, str], int] = {}
        self.results: list[Result] = []
        self.round = 0

    # ---------- listing ----------

    def open_questions(self, viewer: str | None = None) -> list[Question]:
        """Questions nobody holds and nobody finished. Without a viewer: qid
        order (stable, for tests and summaries). With a viewer: a per-viewer
        stable shuffle -- everyone sees the same SET, each in its own fixed
        order, so the whole system does not stampede the same question."""
        qids = [qid for qid in self.bank.questions
                if qid not in self.active and qid not in self.closed]
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
                             "closed to you - call list_questions and take another one")
        if q.qid in self.closed:
            raise BoardError(f"{q.qid} has already been answered and closed; "
                             "call list_questions to see what is open NOW")
        held = self.active.get(q.qid)
        if held is not None:
            if held.agent == agent:
                raise BoardError(f"you already hold {q.qid}; deliver it before "
                                 "claiming it again")
            raise BoardError(f"{q.qid} is held by another agent right now - "
                             "call list_questions to see what is open NOW")
        self.active[q.qid] = Claim(agent, round_no)
        self.strikes[(q.qid, agent)] = self.strikes.get((q.qid, agent), 0) + 1
        return q

    def release(self, agent: str, qid: str) -> Question:
        """Hand a claim back to the open board so anyone may take it."""
        q = self._get(qid)
        claim = self.active.get(q.qid)
        if claim is None or claim.agent != agent:
            raise BoardError(f"you hold no active claim on {q.qid} - "
                             "there is nothing to release")
        del self.active[q.qid]
        return q

    def expire_claims(self, round_no: int, ttl: int) -> list[str]:
        """Drop claims older than `ttl` rounds and put the question back on the
        board, so one agent sitting on a claim cannot stall everyone else.
        The strike stays: the attempt was spent."""
        self.round = max(self.round, round_no)
        expired = [qid for qid, c in self.active.items() if round_no - c.round > ttl]
        for qid in expired:
            del self.active[qid]
        return expired

    def deliver(self, agent: str, qid: str, answer: str,
                round_no: int | None = None) -> Result:
        """Grade the answer and close the question. One graded attempt per claim."""
        q = self._get(qid)
        claim = self.active.get(q.qid)
        if claim is None or claim.agent != agent:
            raise BoardError(f"you hold no active claim on {q.qid} - "
                             "claim_question it first")
        rnd = self.round if round_no is None else round_no
        self.round = max(self.round, rnd)
        submitted = str(answer)
        r = Result(q.qid, agent, submitted, f1(submitted, q.answers),
                   exact_match(submitted, q.answers), rnd)
        del self.active[q.qid]
        self.closed.add(q.qid)
        self.results.append(r)
        return r

    def all_done(self) -> bool:
        return len(self.closed) == len(self.bank.questions)

    def _get(self, qid: str) -> Question:
        try:
            return self.bank.get(qid)
        except BankError as e:
            raise BoardError(str(e)) from e

    # ---------- reporting ----------

    def results_json(self) -> list[dict]:
        """One row per delivered question -- the thing that is graded. `price`
        and `difficulty` are inert bank metadata, carried for post-hoc slicing."""
        rows = []
        for r in self.results:
            q = self.bank.questions[r.qid]
            rows.append({"qid": r.qid, "agent": r.agent, "submitted": r.submitted,
                         "f1": r.f1, "em": r.em, "round": r.round,
                         "price": q.price, "difficulty": q.difficulty,
                         "topic": q.topic})
        return rows

    # ---------- checkpoint (T29) ----------

    def to_state(self) -> dict:
        return {
            "round": self.round,
            "active": {qid: [c.agent, c.round] for qid, c in self.active.items()},
            "closed": sorted(self.closed),
            "strikes": [[qid, agent, n] for (qid, agent), n in self.strikes.items()],
            "results": [[r.qid, r.agent, r.submitted, r.f1, r.em, r.round]
                        for r in self.results],
        }

    def from_state(self, state: dict) -> None:
        self.round = state["round"]
        self.active = {qid: Claim(a, r) for qid, (a, r) in state["active"].items()}
        self.closed = set(state["closed"])
        self.strikes = {(qid, agent): n for qid, agent, n in state["strikes"]}
        self.results = [Result(*row) for row in state["results"]]
