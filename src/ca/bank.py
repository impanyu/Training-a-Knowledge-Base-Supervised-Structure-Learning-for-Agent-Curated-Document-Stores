"""v4 question bank: a FLAT set of questions, each posted with a quota.

A task is one question. The quota says how many times that question may be
claimed and paid for, so total system demand is simply the sum of the quotas --
repeats are explicit demand rather than an artefact of a task tree.

Addressing is by qid only: no sentences, no fuzzy resolution.
"""
import json
from dataclasses import dataclass, fields


class BankError(Exception):
    pass


@dataclass
class Question:
    qid: str
    text: str
    answers: list[str]
    difficulty: str
    price: int
    quota: int = 1
    topic: str = ""


class QuestionBank:
    def __init__(self, questions: list[Question]):
        self.questions: dict[str, Question] = {}
        for q in questions:
            if q.qid in self.questions:
                raise BankError(f"duplicate question id {q.qid}")
            if q.quota < 1:
                raise BankError(f"{q.qid} has quota {q.quota}; a posted question "
                                "must be claimable at least once")
            self.questions[q.qid] = q

    @classmethod
    def from_json(cls, path: str) -> "QuestionBank":
        with open(path) as f:
            raw = json.load(f)
        known = {f.name for f in fields(Question)}
        return cls([Question(**{k: v for k, v in row.items() if k in known})
                    for row in raw["questions"]])

    def get(self, qid: str) -> Question:
        key = str(qid).strip()
        q = self.questions.get(key)
        if q is None:
            raise BankError(f"unknown question id '{key}'; questions are addressed "
                            f"by id only (e.g. {', '.join(self._near(key))}) - "
                            "call list_questions to see what is open")
        return q

    def total_units(self) -> int:
        return sum(q.quota for q in self.questions.values())

    def _near(self, key: str, n: int = 3) -> list[str]:
        """A few real ids to anchor the agent, closest first when `key` looks
        like a qid at all."""
        ids = sorted(self.questions)
        digits = "".join(ch for ch in key if ch.isdigit())
        if digits:
            target = int(digits)
            ids = sorted(ids, key=lambda i: abs(int("".join(
                ch for ch in i if ch.isdigit()) or 0) - target))
        return ids[:n]
