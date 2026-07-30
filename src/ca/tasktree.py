"""Hierarchical task library (v2).

A *node* is an internal subtask: a short id (t0001) plus a globally unique
one-sentence summary. Its children are either other node ids or question ids
(q0001) -- questions are always the leaves. A *task* posted on the board is
simply one node of this library together with its whole subtree, so tasks
nest and share subtrees by construction.

Addressing convention (spec §5): every node-taking action accepts either the
short id or the sentence. Sentences are normalized (lowercase, punctuation and
whitespace stripped) and then matched exactly, or approximately with difflib.
"""
import json
import string
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from ca.taskboard import Question

FUZZY_MIN = 0.85     # below this a sentence simply does not name a node
FUZZY_GAP = 0.03     # two candidates this close together are indistinguishable


class TreeError(Exception):
    pass


class UnknownNodeError(TreeError):
    pass


class AmbiguousError(TreeError):
    pass


@dataclass
class TaskNode:
    nid: str
    sentence: str
    children: list[str] = field(default_factory=list)   # node ids and/or qids
    parent: str | None = None


def normalize(s: str) -> str:
    """Sentence identity: case, punctuation and whitespace carry no meaning."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


class TaskLibrary:
    def __init__(self, nodes, questions, posted: list[str] | None = None):
        self.nodes: dict[str, TaskNode] = {
            n.nid: n for n in (nodes.values() if isinstance(nodes, dict) else nodes)}
        self.questions: dict[str, Question] = {
            q.qid: q for q in (questions.values() if isinstance(questions, dict) else questions)}
        self.posted: list[str] = list(posted or [])
        self._validate()
        self._link_parents()
        self._norm_index = {normalize(n.sentence): n.nid for n in self.nodes.values()}

    # ---------- construction ----------

    def _validate(self) -> None:
        """A malformed library must fail loudly at load time, not crash mid
        -run and destroy whatever summary was in progress. Checks: node ids
        and question ids are disjoint namespaces; every child reference
        resolves to a known node or question (no dangling refs); the
        children graph has no cycles."""
        overlap = set(self.nodes) & set(self.questions)
        if overlap:
            raise ValueError(
                f"node ids and question ids must be disjoint: {sorted(overlap)}")
        for node in self.nodes.values():
            for child in node.children:
                if child not in self.nodes and child not in self.questions:
                    raise ValueError(
                        f"{node.nid} references unknown child {child!r}")
        state: dict[str, int] = {}   # 0=unvisited (absent), 1=in progress, 2=done

        def visit(nid: str) -> None:
            s = state.get(nid, 0)
            if s == 1:
                raise ValueError(f"cycle in task tree at {nid}")
            if s == 2:
                return
            state[nid] = 1
            for child in self.nodes[nid].children:
                if child in self.nodes:
                    visit(child)
            state[nid] = 2

        for nid in self.nodes:
            if state.get(nid, 0) == 0:
                visit(nid)

    def _link_parents(self) -> None:
        for node in self.nodes.values():
            for child in node.children:
                if child in self.nodes:
                    self.nodes[child].parent = node.nid

    @classmethod
    def from_json(cls, path: str) -> "TaskLibrary":
        with open(path) as f:
            raw = json.load(f)
        nodes = [TaskNode(n["nid"], n["sentence"], list(n.get("children", [])))
                 for n in raw["nodes"]]
        questions = [Question(q["qid"], q["text"], q["answers"], q["difficulty"], q["price"])
                     for q in raw["questions"]]
        return cls(nodes, questions, raw.get("posted"))

    def to_json(self, path: str, posted: list[str] | None = None) -> None:
        payload = {
            "nodes": [{"nid": n.nid, "sentence": n.sentence, "children": n.children}
                      for n in self.nodes.values()],
            "questions": [{"qid": q.qid, "text": q.text, "answers": q.answers,
                           "difficulty": q.difficulty, "price": q.price}
                          for q in self.questions.values()],
            "posted": list(posted if posted is not None else self.posted),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ---------- addressing ----------

    def get(self, nid: str) -> TaskNode:
        if nid not in self.nodes:
            raise UnknownNodeError(f"unknown task node {nid}")
        return self.nodes[nid]

    def resolve(self, text: str) -> TaskNode:
        """Short id or (approximate) sentence -> node. Never guesses silently:
        an unclear reference is an error listing concrete candidates."""
        ref = str(text).strip()
        if ref in self.nodes:
            return self.nodes[ref]
        key = normalize(ref)
        if key in self._norm_index:
            return self.nodes[self._norm_index[key]]
        if not self.nodes:
            raise UnknownNodeError(f"no such task: {ref!r}")
        scored = sorted(
            ((SequenceMatcher(None, key, normalize(n.sentence)).ratio(), n)
             for n in self.nodes.values()),
            key=lambda p: (-p[0], p[1].nid))
        best_r, best = scored[0]
        if best_r < FUZZY_MIN:
            near = ", ".join(f"[{n.nid}] {n.sentence}" for _, n in scored[:3])
            raise UnknownNodeError(
                f"no task matches {ref!r}; did you mean one of: {near}?")
        if len(scored) > 1 and best_r - scored[1][0] < FUZZY_GAP:
            both = ", ".join(f"[{n.nid}] {n.sentence}" for _, n in scored[:2])
            raise AmbiguousError(
                f"{ref!r} is ambiguous between: {both}; use the short id instead")
        return best

    def resolve_exact(self, text: str) -> TaskNode | None:
        """Short id or EXACT normalized sentence -> node, or None. No fuzzy
        matching: used where opportunistic/approximate binding would be
        unsafe (e.g. contract proposals, where `task` is free text chosen by
        the proposer and merely resembling a node sentence is not the same
        as naming it)."""
        ref = str(text).strip()
        if ref in self.nodes:
            return self.nodes[ref]
        key = normalize(ref)
        if key in self._norm_index:
            return self.nodes[self._norm_index[key]]
        return None

    # ---------- structure ----------

    def leaves(self, nid: str) -> list[str]:
        """All question ids under a node, depth-first, de-duplicated."""
        self.get(nid)
        out: list[str] = []
        seen: set[str] = set()

        def walk(cur: str, stack: tuple[str, ...]) -> None:
            if cur in stack:
                raise TreeError(f"cycle in task tree at {cur}")
            for child in self.get(cur).children:
                if child in self.nodes:
                    walk(child, stack + (cur,))
                elif child not in seen:
                    seen.add(child)
                    out.append(child)

        walk(nid, ())
        return out

    def price(self, nid: str) -> int:
        return sum(self.questions[q].price for q in self.leaves(nid)
                   if q in self.questions)

    def depth(self, nid: str, _stack: tuple[str, ...] = ()) -> int:
        """Levels in the subtree, counting this node and the leaf level.
        Cycle-guarded regardless of construction-time validation (defense in
        depth, not just a load-time check) so a corrupt tree fails fast here
        too rather than recursing forever."""
        if nid in _stack:
            raise TreeError(f"cycle in task tree at {nid}")
        node = self.get(nid)
        if not node.children:
            return 1
        return 1 + max(
            self.depth(c, _stack + (nid,)) if c in self.nodes else 1
            for c in node.children)

    def children_view(self, nid: str) -> list[dict]:
        """What `decompose` reveals: one row per child, subtasks summarized by
        their sentence, leaves by their full question text."""
        rows = []
        for child in self.get(nid).children:
            if child in self.nodes:
                sub = self.nodes[child]
                rows.append({"kind": "subtask", "nid": sub.nid, "sentence": sub.sentence,
                             "leaf_count": len(self.leaves(sub.nid)),
                             "price": self.price(sub.nid)})
            else:
                q = self.questions.get(child)
                rows.append({"kind": "leaf", "qid": child,
                             "text": q.text if q else "(missing question)",
                             "price": q.price if q else 0})
        return rows

    def sentence(self, nid: str) -> str:
        return self.get(nid).sentence

    def base_subtask(self, task_nid: str, qid: str) -> str:
        """The level-1 grouping key for a leaf delivered under `task_nid`
        (used by metrics.specialization): the immediate child of task_nid
        that contains qid in its subtree, or qid itself when the leaf hangs
        directly off task_nid with no wrapping subtask node. Relative to the
        task it was delivered under, so a leaf shared by two posted tasks
        (e.g. a bare leaf under both) buckets independently per delivery."""
        node = self.get(task_nid)
        if qid in node.children:
            return qid
        for child in node.children:
            if child in self.nodes and qid in self.leaves(child):
                return child
        raise UnknownNodeError(f"{qid} is not a leaf of {task_nid}")
