"""WORLD's task board (v2): posted *tasks* are nodes of the shared task tree.

A task is claimed as a whole and delivered as a whole: one JSON map
{qid: answer} covering every leaf of its subtree, graded leaf by leaf and paid
Sum(price(leaf) x F1(leaf)) in a single settlement. Pay is per (task, leaf), so
the same question under two posted tasks earns twice -- that repeat is the
specialization dividend the experiment is meant to measure.
"""
from dataclasses import dataclass, field

from ca.economy import Ledger
from ca.grader import exact_match, f1


class BoardError(Exception):
    pass


@dataclass
class Question:
    """A leaf of the task tree. Still owns its gold answers and its price;
    grading and pricing are per question even though delivery is per task."""
    qid: str
    text: str
    answers: list[str]
    difficulty: str
    price: int


@dataclass
class LeafResult:
    qid: str
    submitted: str
    score: float           # F1
    em: float
    price: int
    payout: int


@dataclass
class PostedTask:
    nid: str
    status: str = "open"                # open / claimed / closed
    claimed_by: str | None = None
    claimed_round: int = 0
    claim_counts: dict = field(default_factory=dict)   # agent -> times claimed
    leaves: list[LeafResult] = field(default_factory=list)
    payout: int = 0


class TaskBoard:
    def __init__(self, library, posted: list[str], ledger: Ledger):
        self.library = library
        self.ledger = ledger
        self.tasks: dict[str, PostedTask] = {}
        for nid in posted:
            library.get(nid)            # posting an unknown node is a build error
            self.tasks[nid] = PostedTask(nid)

    # ---------- addressing ----------

    def get(self, ref: str) -> PostedTask:
        """Resolve a short id or sentence to a POSTED task. Nodes that exist in
        the library but were never posted are deliberately not claimable."""
        r = str(ref).strip()
        if r in self.tasks:
            return self.tasks[r]
        try:
            node = self.library.resolve(r)
        except Exception as e:
            raise BoardError(str(e)) from e
        if node.nid not in self.tasks:
            raise BoardError(f"{node.nid} is a subtask, not a task posted on the board")
        return self.tasks[node.nid]

    def price(self, nid: str) -> int:
        return self.library.price(nid)

    def sentence(self, nid: str) -> str:
        return self.library.sentence(nid)

    def leaf_ids(self, nid: str) -> list[str]:
        return self.library.leaves(nid)

    # ---------- lifecycle ----------

    def list_open(self) -> list[PostedTask]:
        return sorted((t for t in self.tasks.values() if t.status == "open"),
                      key=lambda t: (-self.price(t.nid), t.nid))

    def claim(self, agent: str, ref: str, round_no: int = 0) -> PostedTask:
        t = self.get(ref)
        if t.status != "open":
            raise BoardError(f"{t.nid} is {t.status}")
        if t.claim_counts.get(agent, 0) >= 2:
            raise BoardError(f"you have already claimed and abandoned {t.nid} twice; "
                             "it is closed to you - work on something else")
        t.status = "claimed"
        t.claimed_by = agent
        t.claim_counts[agent] = t.claim_counts.get(agent, 0) + 1
        t.claimed_round = round_no
        return t

    def expire_claims(self, current_round: int, ttl: int) -> list[str]:
        """Reopen tasks claimed more than `ttl` rounds ago and never delivered,
        so one agent hoarding claims cannot stall the whole board."""
        reopened = []
        for t in self.tasks.values():
            if t.status == "claimed" and current_round - t.claimed_round > ttl:
                t.status = "open"
                t.claimed_by = None
                t.claimed_round = 0
                reopened.append(t.nid)
        return reopened

    def deliver(self, agent: str, ref: str,
                answers: dict[str, str]) -> tuple[list[tuple[str, float, int]], int]:
        """Packaged delivery. Coverage is validated BEFORE anything mutates: an
        incomplete map costs the agent nothing and leaves the claim intact, so
        only a well-formed complete map burns the task's single attempt."""
        t = self.get(ref)
        if t.status != "claimed":
            raise BoardError(f"{t.nid} is {t.status}, claim it first")
        if t.claimed_by != agent:
            raise BoardError(f"{t.nid} was claimed by {t.claimed_by}")
        wanted = self.leaf_ids(t.nid)
        got = set(answers)
        missing = [q for q in wanted if q not in got]
        extra = sorted(got - set(wanted))
        if missing or extra:
            parts = []
            if missing:
                parts.append("missing " + ", ".join(missing))
            if extra:
                parts.append("unknown " + ", ".join(extra))
            raise BoardError(
                f"{t.nid} needs exactly one answer per leaf ({', '.join(wanted)}): "
                f"{'; '.join(parts)}. Nothing was submitted - your delivery "
                "attempt is still available.")
        per_leaf: list[tuple[str, float, int]] = []
        total = 0
        for qid in wanted:
            q = self.library.questions[qid]
            answer = str(answers[qid])
            score = f1(answer, q.answers)
            payout = round(q.price * score)
            total += payout
            t.leaves.append(LeafResult(qid, answer, score,
                                       exact_match(answer, q.answers), q.price, payout))
            per_leaf.append((qid, score, payout))
        t.payout = total
        t.status = "closed"
        if total > 0:
            self.ledger.mint(agent, total)
        return per_leaf, total

    # ---------- reporting ----------

    def all_done(self) -> bool:
        return all(t.status == "closed" for t in self.tasks.values())

    def results(self) -> list[dict]:
        out = []
        for t in self.tasks.values():
            out.append({
                "nid": t.nid,
                "sentence": self.sentence(t.nid),
                "status": t.status,
                "claimed_by": t.claimed_by,
                "price": self.price(t.nid),
                "payout": t.payout,
                "n_leaves": len(self.leaf_ids(t.nid)),
                "depth": self.library.depth(t.nid),
                "leaves": [{"qid": l.qid, "submitted": l.submitted, "score": l.score,
                            "em": l.em, "price": l.price, "payout": l.payout}
                           for l in t.leaves],
            })
        return out

    def leaf_results(self) -> list[dict]:
        """One row per (task, leaf) pair -- the unit that is graded and paid."""
        rows = []
        for t in self.tasks.values():
            graded = {l.qid: l for l in t.leaves}
            for qid in self.leaf_ids(t.nid):
                q = self.library.questions[qid]
                l = graded.get(qid)
                rows.append({
                    "nid": t.nid, "qid": qid, "difficulty": q.difficulty,
                    "price": q.price, "claimed_by": t.claimed_by,
                    "status": "closed" if l else t.status,
                    "submitted": l.submitted if l else None,
                    "score": l.score if l else 0.0,
                    "em": l.em if l else 0.0,
                    "payout": l.payout if l else 0,
                })
        return rows
