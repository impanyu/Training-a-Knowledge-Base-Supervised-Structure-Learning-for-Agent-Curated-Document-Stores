"""Per-agent short-term (FIFO, goal stack) and long-term (vector) memory."""
import re
import uuid
from collections import deque

import chromadb

SHARED_BUCKET = "shared"


class FifoMemory:
    """Recent-action buffer. K is the pair budget — when maxlen is reached,
    the oldest (action, result) pair is evicted. All pairs are rendered in full."""

    def __init__(self, k: int = 10):
        self.items: deque[tuple[str, str]] = deque(maxlen=k)

    def add(self, action: str, result: str) -> None:
        self.items.append((action, result))

    def render(self) -> str:
        if not self.items:
            return "(no recent actions)"
        lines = []
        for a, r in self.items:
            lines.append(f"- {a} -> {r}")
        return "\n".join(lines)

    def to_state(self) -> list:
        return [[a, r] for a, r in self.items]

    def from_state(self, state: list) -> None:
        self.items.clear()
        self.items.extend((a, r) for a, r in state)


class GoalStack:
    def __init__(self, root: str):
        self._root = root
        self._stack: list[str] = []

    def push(self, note: str) -> None:
        self._stack.append(note)

    def pop(self) -> str:
        if not self._stack:
            raise IndexError("cannot pop the root goal")
        return self._stack.pop()

    def render(self) -> str:
        lines = [f"[0] {self._root} (root, permanent)"]
        lines += [f"[{i+1}] {n}" for i, n in enumerate(self._stack)]
        lines[-1] += "   <- current focus"
        return "\n".join(lines)

    def to_state(self) -> list:
        return list(self._stack)

    def from_state(self, state: list) -> None:
        self._stack = list(state)


class AgentMemory:
    """One local Chroma collection per bucket (bucket = agent, or a single
    shared one at C2). Append-only: an answer is an ordinary entry that merely
    carries extra metadata, so it is found both by meaning (`search`) and by id
    (`answer`), and answering a question twice leaves both attempts on record.

    `embedding_function` defaults to chroma's local ONNX model; unit tests pass
    a cheap deterministic stub instead (as KeywordBackend does for retrieval).
    """

    def __init__(self, shared: bool = False, persist_dir: str | None = None,
                 embedding_function=None):
        self.shared = shared
        self._client = (chromadb.PersistentClient(path=persist_dir)
                        if persist_dir else chromadb.EphemeralClient())
        self._ef = embedding_function
        # chroma hands every in-process EphemeralClient the SAME in-memory db,
        # so an instance-unique prefix is what actually isolates two memories.
        self._prefix = "mem" if persist_dir else f"mem{uuid.uuid4().hex[:8]}"
        self._cols: dict[str, object] = {}
        self._seq = 0

    # ---------- buckets ----------

    def _bucket(self, agent: str) -> str:
        return SHARED_BUCKET if self.shared else str(agent)

    def _col(self, agent: str):
        bucket = self._bucket(agent)
        if bucket not in self._cols:
            name = f"{self._prefix}-" + re.sub(r"[^a-zA-Z0-9_-]", "-", bucket) + "-0"
            kw = {"embedding_function": self._ef} if self._ef is not None else {}
            self._cols[bucket] = self._client.get_or_create_collection(name, **kw)
        return self._cols[bucket]

    # ---------- writing (append-only) ----------

    def write(self, agent: str, text: str, *, kind: str = "note",
              qid: str | None = None, f1: float | None = None) -> None:
        self._seq += 1
        meta: dict = {"kind": str(kind), "seq": self._seq}
        if qid is not None:
            meta["qid"] = str(qid)
        if f1 is not None:
            meta["f1"] = float(f1)
        self._col(agent).add(ids=[f"m{self._seq}"], documents=[str(text)],
                             metadatas=[meta])

    # ---------- reading ----------

    def search(self, agent: str, query: str, k: int = 5) -> list[dict]:
        col = self._col(agent)
        n = min(k, col.count())
        if n <= 0:
            return []
        res = col.query(query_texts=[str(query)], n_results=n)
        return [_row(doc, meta)
                for doc, meta in zip(res["documents"][0], res["metadatas"][0])]

    def answer(self, agent: str, qid: str) -> dict | None:
        """The best answer stored for `qid`: highest F1, ties broken by
        recency. An ungraded answer (F1 unknown) never displaces a graded one."""
        got = self._col(agent).get(where={"$and": [{"kind": "answer"},
                                                   {"qid": str(qid)}]})
        if not got["ids"]:
            return None
        best = max(zip(got["documents"], got["metadatas"]),
                   key=lambda dm: (dm[1].get("f1", -1.0), dm[1]["seq"]))
        return _row(*best)

    def n_answers(self, agent: str) -> int:
        return len(self._col(agent).get(where={"kind": "answer"})["ids"])

    def n_entries(self, agent: str) -> int:
        return self._col(agent).count()

    # ---------- checkpoint (T29) ----------

    def to_state(self) -> dict:
        """(id, text, metadata) triples per bucket; embeddings are recomputed
        on restore rather than stored."""
        state = {}
        for bucket in sorted(self._cols):
            got = self._cols[bucket].get()
            rows = sorted(zip(got["ids"], got["documents"], got["metadatas"]),
                          key=lambda r: r[2]["seq"])
            state[bucket] = [[i, doc, dict(meta)] for i, doc, meta in rows]
        return state

    def from_state(self, state: dict) -> None:
        for col in self._cols.values():
            self._client.delete_collection(col.name)
        self._cols = {}
        self._seq = 0
        for bucket, rows in state.items():
            if not rows:
                continue
            self._bucket_col(bucket).add(
                ids=[r[0] for r in rows], documents=[r[1] for r in rows],
                metadatas=[dict(r[2]) for r in rows])
            self._seq = max(self._seq, max(r[2]["seq"] for r in rows))

    def _bucket_col(self, bucket: str):
        """_col() by bucket name (from_state restores whatever was dumped,
        including the shared bucket)."""
        saved, self.shared = self.shared, False
        try:
            return self._col(bucket)
        finally:
            self.shared = saved


def _row(doc: str, meta: dict) -> dict:
    return {"text": doc, "kind": meta.get("kind"),
            "qid": meta.get("qid"), "f1": meta.get("f1")}
