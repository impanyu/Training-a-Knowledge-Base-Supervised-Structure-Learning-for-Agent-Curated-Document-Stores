"""Solution memory (v3): a per-agent KEY-VALUE store of what has been SOLVED.

Fully parallel to, and independent of, the free-text `LongTermMemory`
(memory_write / memory_search). Two mapping kinds live here:

    subtask nid  -> [child ids]      (written by `decompose`)
    question qid -> {answer, f1?}    (written when answers are delivered)

Nothing is written by an explicit agent action: the store fills itself at the
three moments an agent demonstrably learns something (decomposing a node,
delivering answers, receiving a node-bound deliverable). Reading it back is
the one deliberate act -- the `recall_solutions` action -- which walks the
stored decompositions recursively so a subtask solved once can be re-served
whole.

Keys are ids only (t#### / q####), never sentences: this module is
library-agnostic on purpose, and the ACTIONS layer resolves whatever the agent
typed (id or sentence) to an id before touching the store.

C2 (`shared_solution_memory`) flips one thing: every agent id maps to a single
bucket, so the whole system reads and writes one knowledge base.
"""
import re
from collections import defaultdict

_SHARED_BUCKET = "__shared__"
_QID_RE = re.compile(r"q\d+")


def is_question(key: str) -> bool:
    """Leaf ids are the q#### namespace; anything else is a subtask node."""
    return bool(_QID_RE.fullmatch(str(key).strip()))


class SolutionMemory:
    def __init__(self, shared: bool = False):
        self.shared = shared
        self._decomp: dict[str, dict[str, list[str]]] = defaultdict(dict)
        self._answers: dict[str, dict[str, dict]] = defaultdict(dict)

    def _bucket(self, agent: str) -> str:
        return _SHARED_BUCKET if self.shared else agent

    # ---------- writing (automatic) ----------

    def record_decomposition(self, agent: str, name: str, children: list[str]) -> None:
        self._decomp[self._bucket(agent)][str(name)] = [str(c) for c in children]

    def record_answer(self, agent: str, qid: str, answer: str,
                      f1: float | None = None) -> None:
        """Later writes win, EXCEPT that an ungraded answer never displaces a
        graded one: contract deliverables arrive ungraded, and a known F1 is
        strictly more informative than no F1 for the same question."""
        store = self._answers[self._bucket(agent)]
        qid = str(qid)
        prev = store.get(qid)
        if f1 is None and prev is not None and "f1" in prev:
            return
        rec: dict = {"answer": str(answer)}
        if f1 is not None:
            rec["f1"] = float(f1)
        store[qid] = rec

    # ---------- reading ----------

    def mapping(self, agent: str, name: str) -> list[str] | None:
        found = self._decomp[self._bucket(agent)].get(str(name))
        return list(found) if found is not None else None

    def has_decomposition(self, agent: str, name: str) -> bool:
        return str(name) in self._decomp[self._bucket(agent)]

    def decomposed_ids(self, agent: str) -> list[str]:
        """Keys of every stored decomposition, in insertion order."""
        return list(self._decomp[self._bucket(agent)])

    def decomposition(self, agent: str, name: str) -> list[str] | None:
        found = self._decomp[self._bucket(agent)].get(str(name))
        return list(found) if found is not None else None

    def answer(self, agent: str, qid: str) -> dict | None:
        found = self._answers[self._bucket(agent)].get(str(qid))
        return dict(found) if found is not None else None

    def recall(self, agent: str, name: str) -> dict:
        """Expand `name` through the stored decompositions and report what is
        underneath it: answers we hold, leaves we know about but cannot
        answer, and branches we never decomposed (so their leaves are still
        unknown unknowns). Cycle-safe: a corrupt or self-referential mapping
        must not hang a turn."""
        bucket = self._bucket(agent)
        decomp, answers = self._decomp[bucket], self._answers[bucket]
        known: dict[str, dict] = {}
        missing: list[str] = []
        unexpanded: list[str] = []
        visited: set[str] = set()

        def walk(key: str) -> None:
            if key in visited:
                return
            visited.add(key)
            if is_question(key):
                if key in answers:
                    known[key] = dict(answers[key])
                else:
                    missing.append(key)
            elif key in decomp:
                for child in decomp[key]:
                    walk(child)
            else:
                unexpanded.append(key)

        walk(str(name).strip())
        return {"known": known, "missing": missing, "unexpanded": unexpanded}

    def to_state(self) -> dict:
        return {"decomp": {b: {k: list(v) for k, v in d.items()}
                           for b, d in self._decomp.items()},
                "answers": {b: {q: dict(rec) for q, rec in d.items()}
                            for b, d in self._answers.items()}}

    def from_state(self, state: dict) -> None:
        self._decomp.clear()
        for b, d in state["decomp"].items():
            self._decomp[b] = {k: list(v) for k, v in d.items()}
        self._answers.clear()
        for b, d in state["answers"].items():
            self._answers[b] = {q: dict(rec) for q, rec in d.items()}

    def stats(self, agent: str) -> dict:
        bucket = self._bucket(agent)
        return {"answers": len(self._answers[bucket]),
                "decompositions": len(self._decomp[bucket])}
