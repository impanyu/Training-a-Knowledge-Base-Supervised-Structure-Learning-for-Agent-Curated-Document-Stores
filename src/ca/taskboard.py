"""WORLD's task board: open questions are external contracts at posted prices."""
from dataclasses import dataclass, field

from ca.economy import Ledger
from ca.grader import exact_match, f1


class BoardError(Exception):
    pass


@dataclass
class Question:
    qid: str
    text: str
    answers: list[str]
    difficulty: str
    price: int
    status: str = "open"            # open / claimed / closed
    claimed_by: str | None = None
    claimed_round: int = 0
    submitted: str | None = None
    score: float = 0.0              # F1
    em: float = 0.0
    payout: int = 0


class TaskBoard:
    def __init__(self, questions: list[Question], ledger: Ledger):
        self.questions: dict[str, Question] = {q.qid: q for q in questions}
        self.ledger = ledger

    def get(self, qid: str) -> Question:
        if qid not in self.questions:
            raise BoardError(f"unknown question {qid}")
        return self.questions[qid]

    def list_open(self) -> list[Question]:
        return [q for q in self.questions.values() if q.status == "open"]

    def claim(self, agent: str, qid: str, round_no: int = 0) -> Question:
        q = self.get(qid)
        if q.status != "open":
            raise BoardError(f"{qid} is {q.status}")
        q.status = "claimed"
        q.claimed_by = agent
        q.claimed_round = round_no
        return q

    def expire_claims(self, current_round: int, ttl: int) -> list[str]:
        """Reopen questions claimed more than `ttl` rounds ago and never
        delivered, so one agent hoarding claims cannot stall the whole pool."""
        reopened = []
        for q in self.questions.values():
            if q.status == "claimed" and current_round - q.claimed_round > ttl:
                q.status = "open"
                q.claimed_by = None
                q.claimed_round = 0
                reopened.append(q.qid)
        return reopened

    def deliver(self, agent: str, qid: str, answer: str) -> tuple[float, int]:
        q = self.get(qid)
        if q.status != "claimed":
            raise BoardError(f"{qid} is {q.status}, claim it first")
        if q.claimed_by != agent:
            raise BoardError(f"{qid} was claimed by {q.claimed_by}")
        q.submitted = answer
        q.score = f1(answer, q.answers)
        q.em = exact_match(answer, q.answers)
        q.payout = round(q.price * q.score)
        q.status = "closed"
        if q.payout > 0:
            self.ledger.mint(agent, q.payout)
        return q.score, q.payout

    def all_done(self) -> bool:
        return all(q.status == "closed" for q in self.questions.values())

    def results(self) -> list[Question]:
        return list(self.questions.values())
