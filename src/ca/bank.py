"""v4.1 question bank: a flat set of questions, posted in JOBS.

A question is still one stand-alone question -- there is no tree, no
decomposition, no sentence addressing. What the WORLD *posts* is a JOB: 6-10
questions drawn from one topic cluster, claimed and delivered as a unit. Job
membership is the repeat mechanism (a question sits in ~3 jobs), so the
per-question quota of v4 is retired.

Total system demand = sum of job sizes = the (job, question) units.
Addressing is by id only: qid for a question, jid for a job.
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


@dataclass
class Job:
    jid: str
    qids: list[str]
    price: int


class QuestionBank:
    def __init__(self, questions: list[Question], jobs: list[Job] | None = None):
        self.questions: dict[str, Question] = {}
        for q in questions:
            if q.qid in self.questions:
                raise BankError(f"duplicate question id {q.qid}")
            self.questions[q.qid] = q
        self.jobs: dict[str, Job] = {}
        for j in jobs or []:
            if j.jid in self.jobs:
                raise BankError(f"duplicate job id {j.jid}")
            if not j.qids:
                raise BankError(f"{j.jid} posts no questions")
            if len(set(j.qids)) != len(j.qids):
                raise BankError(f"{j.jid} lists the same question twice")
            for qid in j.qids:
                if qid not in self.questions:
                    raise BankError(f"{j.jid} references unknown question {qid}")
            self.jobs[j.jid] = j

    @classmethod
    def from_json(cls, path: str) -> "QuestionBank":
        with open(path) as f:
            raw = json.load(f)
        known = {f.name for f in fields(Question)}
        questions = [Question(**{k: v for k, v in row.items() if k in known})
                     for row in raw["questions"]]
        jobs = [Job(row["jid"], list(row["qids"]), row["price"])
                for row in raw.get("jobs", [])]
        return cls(questions, jobs)

    def get(self, qid: str) -> Question:
        key = str(qid).strip()
        q = self.questions.get(key)
        if q is None:
            raise BankError(f"unknown question id '{key}'; questions are addressed "
                            f"by id only (e.g. {', '.join(_near(self.questions, key))})"
                            " - claim_job reveals the questions of a job")
        return q

    def get_job(self, jid: str) -> Job:
        key = str(jid).strip()
        j = self.jobs.get(key)
        if j is None:
            raise BankError(f"unknown job id '{key}'; jobs are addressed by id only "
                            f"(e.g. {', '.join(_near(self.jobs, key))}) - "
                            "call list_jobs to see what is open")
        return j

    def job_price(self, jid: str) -> int:
        return self.get_job(jid).price

    def total_units(self) -> int:
        """(job, question) units: the WORLD's whole posted demand."""
        return sum(len(j.qids) for j in self.jobs.values())


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
