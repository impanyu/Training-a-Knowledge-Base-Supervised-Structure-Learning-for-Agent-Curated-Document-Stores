"""v5 question bank: a task IS one question.

The job layer is gone -- the WORLD posts individual questions, each claimed and
answered on its own. Addressing is by id only (qid). `topic` is bank metadata
for the specialization metric; the agents never see it.
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
    topic: str = ""


class QuestionBank:
    def __init__(self, questions: list[Question]):
        self.questions: dict[str, Question] = {}
        for q in questions:
            if q.qid in self.questions:
                raise BankError(f"duplicate question id {q.qid}")
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
                            f"by id only (e.g. {', '.join(_near(self.questions, key))})"
                            " - call list_questions to see what is open")
        return q

    def total_units(self) -> int:
        """One unit per question: the WORLD's whole posted demand."""
        return len(self.questions)


def _near(ids, key: str, n: int = 3) -> list[str]:
    """A few real ids to anchor the agent, closest first when `key` looks like
    an id at all."""
    out = sorted(ids)
    digits = "".join(ch for ch in key if ch.isdigit())
    if digits:
        target = int(digits)
        out = sorted(out, key=lambda i: abs(int("".join(
            ch for ch in i if ch.isdigit()) or 0) - target))
    return out[:n]
