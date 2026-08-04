"""WORLD's job board (v4.1): claim, expire and deliver at JOB level.

A job holds ONE claimant at a time. Delivery is all-or-nothing: a single map
{qid: answer} covering exactly the job's questions, graded per question and
settled in one payment. A malformed or incomplete map is rejected WITHOUT
consuming the attempt, so an agent is punished for missing the deadline, never
for mistyping. Expiry returns the job to the pool; strikes are per (job, agent)
and are never returned, so an agent gets at most two attempts at any job.
"""
import zlib
from dataclasses import dataclass

from ca.bank import BankError, Job, QuestionBank
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
    jid: str
    agent: str
    submitted: str
    f1: float
    em: float
    payout: int
    round: int


class JobBoard:
    def __init__(self, bank: QuestionBank, ledger: Ledger):
        self.bank = bank
        self.ledger = ledger
        self.active: dict[str, Claim] = {}          # jid -> its ONE claimant
        self.closed: set[str] = set()               # jid -> delivered, off the board
        self.strikes: dict[tuple[str, str], int] = {}
        self.results: list[Result] = []
        self.round = 0

    # ---------- listing ----------

    def open_jobs(self, viewer: str | None = None) -> list[Job]:
        """Jobs nobody holds and nobody finished. Without a viewer: jid order
        (stable, for tests and summaries). With a viewer: a per-viewer stable
        shuffle -- everyone sees the same SET, each in its own fixed order, so
        the whole economy does not stampede the same top-priced job."""
        jids = [jid for jid in self.bank.jobs
                if jid not in self.active and jid not in self.closed]
        if viewer is None:
            jids.sort()
        else:
            jids.sort(key=lambda jid: zlib.crc32(f"{viewer}:{jid}".encode()))
        return [self.bank.jobs[jid] for jid in jids]

    # ---------- lifecycle ----------

    def claim(self, agent: str, jid: str, round_no: int = 0) -> Job:
        job = self._get(jid)
        self.round = max(self.round, round_no)
        if self.strikes.get((job.jid, agent), 0) >= MAX_STRIKES:
            raise BoardError(f"you have already claimed {job.jid} twice; it is "
                             "closed to you - call list_jobs and take another one")
        if job.jid in self.closed:
            raise BoardError(f"{job.jid} has already been delivered and closed; "
                             "call list_jobs to see what is open NOW")
        held = self.active.get(job.jid)
        if held is not None:
            if held.agent == agent:
                raise BoardError(f"you already hold {job.jid}; deliver it before "
                                 "claiming it again")
            raise BoardError(f"{job.jid} is held by another agent right now - "
                             "call list_jobs to see what is open NOW")
        self.active[job.jid] = Claim(agent, round_no)
        self.strikes[(job.jid, agent)] = self.strikes.get((job.jid, agent), 0) + 1
        return job

    def expire_claims(self, round_no: int, ttl: int) -> list[str]:
        """Drop claims older than `ttl` rounds and put the job back on the
        board, so one agent sitting on a job cannot stall the whole economy.
        The strike stays: the attempt was spent."""
        self.round = max(self.round, round_no)
        expired = [jid for jid, c in self.active.items() if round_no - c.round > ttl]
        for jid in expired:
            del self.active[jid]
        return expired

    def deliver(self, agent: str, jid: str, answers: dict[str, str],
                round_no: int | None = None) -> list[Result]:
        """Grade EVERY member question of the job and settle once. The map must
        cover exactly the job's qids; anything else raises without touching the
        claim, so the agent may simply try again."""
        job = self._get(jid)
        claim = self.active.get(job.jid)
        if claim is None or claim.agent != agent:
            raise BoardError(f"you hold no active claim on {job.jid} - "
                             "claim_job it first")
        given = {str(k).strip(): v for k, v in dict(answers).items()}
        missing = [qid for qid in job.qids if qid not in given]
        extra = sorted(set(given) - set(job.qids))
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing an answer for {', '.join(missing)}")
            if extra:
                parts.append(f"{', '.join(extra)} is not part of {job.jid}")
            raise BoardError(
                f"{job.jid} is delivered all-or-nothing: " + "; ".join(parts) +
                f". Your claim is STILL YOURS ({len(job.qids)} questions) - "
                "send one map covering exactly "
                f"{', '.join(job.qids)} and nothing else.")
        rnd = self.round if round_no is None else round_no
        self.round = max(self.round, rnd)
        out = []
        for qid in job.qids:
            q = self.bank.questions[qid]
            submitted = str(given[qid])
            score = f1(submitted, q.answers)
            out.append(Result(qid, job.jid, agent, submitted, score,
                              exact_match(submitted, q.answers),
                              round(q.price * score), rnd))
        del self.active[job.jid]
        self.closed.add(job.jid)
        self.results += out
        total = sum(r.payout for r in out)
        if total > 0:
            self.ledger.mint(agent, total)          # ONE settlement per job
        return out

    def all_done(self) -> bool:
        return len(self.closed) == len(self.bank.jobs)

    def _get(self, jid: str) -> Job:
        try:
            return self.bank.get_job(jid)
        except BankError as e:
            raise BoardError(str(e)) from e

    # ---------- reporting ----------

    def results_json(self) -> list[dict]:
        """One row per delivered (job, question) unit -- the thing that is
        graded and paid."""
        rows = []
        for r in self.results:
            q = self.bank.questions[r.qid]
            rows.append({"qid": r.qid, "jid": r.jid, "agent": r.agent,
                         "submitted": r.submitted, "f1": r.f1, "em": r.em,
                         "payout": r.payout, "round": r.round, "price": q.price,
                         "difficulty": q.difficulty, "topic": q.topic})
        return rows

    # ---------- checkpoint (T29) ----------

    def to_state(self) -> dict:
        return {
            "round": self.round,
            "active": {jid: [c.agent, c.round] for jid, c in self.active.items()},
            "closed": sorted(self.closed),
            "strikes": [[jid, agent, n] for (jid, agent), n in self.strikes.items()],
            "results": [[r.qid, r.jid, r.agent, r.submitted, r.f1, r.em,
                         r.payout, r.round] for r in self.results],
        }

    def from_state(self, state: dict) -> None:
        self.round = state["round"]
        self.active = {jid: Claim(a, r) for jid, (a, r) in state["active"].items()}
        self.closed = set(state["closed"])
        self.strikes = {(jid, agent): n for jid, agent, n in state["strikes"]}
        self.results = [Result(*row) for row in state["results"]]
