"""Per-agent short-term (FIFO, goal stack) memory and the cluster's ONE
shared long-term memory (the knowledge base, KB).

v7: the KB is a single physical vector store for ALL agents, seeded at birth
with the full paragraph corpus (precomputed embeddings, so seeding never
re-embeds). `search` is the single knowledge query, over corpus + notes +
recorded Q&As + delivered answers alike; every write carries its author in
metadata so per-agent tallies stay possible. Corpus entries are static
furniture: excluded from checkpoints and re-seeded deterministically on
resume.
"""
import json
import re
import uuid
from collections import deque

import chromadb


def load_corpus(corpus_path, emb_path):
    """corpus.jsonl ({title, text} per line) + corpus_emb.npy (float32, row i =
    paragraph i, chroma default-ONNX space)."""
    import numpy as np
    with open(corpus_path) as f:
        paras = [json.loads(line) for line in f]
    emb = np.load(emb_path)
    if len(emb) != len(paras):
        raise ValueError(f"{emb_path} has {len(emb)} rows for {len(paras)} paragraphs")
    return paras, emb


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
    """The cluster's shared KB: one Chroma collection for everyone.
    Append-only; every non-corpus entry carries {kind, agent, seq} metadata
    (kind = note | selfqa | answer), so it is found by meaning (`search`) and
    counted per author (`count`).

    `embedding_function` defaults to chroma's local ONNX model; unit tests
    pass a cheap deterministic stub instead.
    """

    def __init__(self, persist_dir: str | None = None, embedding_function=None):
        self._client = (chromadb.PersistentClient(path=persist_dir)
                        if persist_dir else chromadb.EphemeralClient())
        self._ef = embedding_function
        # chroma hands every in-process EphemeralClient the SAME in-memory db,
        # so an instance-unique prefix is what actually isolates two memories.
        self._prefix = "kb" if persist_dir else f"kb{uuid.uuid4().hex[:8]}"
        self._collection = None
        self._seq = 0
        self._n_corpus = 0

    def _col(self):
        if self._collection is None:
            kw = {"embedding_function": self._ef} if self._ef is not None else {}
            self._collection = self._client.get_or_create_collection(
                f"{self._prefix}-shared-0", **kw)
        return self._collection

    # ---------- seeding (the corpus is memory) ----------

    def seed_corpus(self, paras: list[dict], embeddings) -> None:
        """Add every paragraph as a `{kind: corpus, title}` entry, using the
        PRECOMPUTED embeddings so nothing is ever re-embedded. Corpus entries
        consume seq 1..N first -- identically on every (re-)seed -- so
        note/answer ids stay deterministic across save/restore."""
        col = self._col()
        n = len(paras)
        for i in range(0, n, 1000):     # stay under chroma batch limits
            batch = paras[i:i + 1000]
            col.add(
                ids=[f"m{i + j + 1}" for j in range(len(batch))],
                documents=[p["text"] for p in batch],
                embeddings=embeddings[i:i + len(batch)],
                metadatas=[{"kind": "corpus", "title": p["title"], "seq": i + j + 1}
                           for j, p in enumerate(batch)])
        self._n_corpus = n
        self._seq = max(self._seq, n)

    # ---------- writing (append-only) ----------

    def write(self, agent: str, text: str, *, kind: str = "note",
              qid: str | None = None, f1: float | None = None) -> None:
        self._seq += 1
        meta: dict = {"kind": str(kind), "agent": str(agent), "seq": self._seq}
        if qid is not None:
            meta["qid"] = str(qid)
        if f1 is not None:
            meta["f1"] = float(f1)
        self._col().add(ids=[f"m{self._seq}"], documents=[str(text)],
                        metadatas=[meta])

    # ---------- reading ----------

    def search(self, query: str, k: int = 5) -> list[dict]:
        col = self._col()
        n = min(k, col.count())
        if n <= 0:
            return []
        res = col.query(query_texts=[str(query)], n_results=n)
        return [_row(doc, meta)
                for doc, meta in zip(res["documents"][0], res["metadatas"][0])]

    def count(self, kind: str, agent: str | None = None) -> int:
        where = ({"kind": kind} if agent is None
                 else {"$and": [{"kind": kind}, {"agent": str(agent)}]})
        return len(self._col().get(where=where)["ids"])

    def n_entries(self) -> int:
        return self._col().count()

    # ---------- checkpoint (T29) ----------

    def to_state(self) -> list:
        """(id, text, metadata) triples, notes/selfqa/answers ONLY -- the
        corpus is static furniture, re-seeded on restore rather than
        serialized 12k-fold."""
        got = self._col().get(where={"kind": {"$ne": "corpus"}})
        rows = sorted(zip(got["ids"], got["documents"], got["metadatas"]),
                      key=lambda r: r[2]["seq"])
        return [[i, doc, dict(meta)] for i, doc, meta in rows]

    def from_state(self, state: list) -> None:
        """Assumes a freshly re-seeded store: wipes notes/answers (never the
        corpus) and re-adds the dumped rows on top. Embeddings are recomputed;
        the corpus keeps its precomputed ones."""
        col = self._col()
        got = col.get(where={"kind": {"$ne": "corpus"}})
        if got["ids"]:
            col.delete(ids=got["ids"])
        self._seq = self._n_corpus
        if state:
            col.add(ids=[r[0] for r in state], documents=[r[1] for r in state],
                    metadatas=[dict(r[2]) for r in state])
            self._seq = max(self._seq, max(r[2]["seq"] for r in state))


def _row(doc: str, meta: dict) -> dict:
    return {"text": doc, "kind": meta.get("kind"), "agent": meta.get("agent"),
            "qid": meta.get("qid"), "f1": meta.get("f1"),
            "title": meta.get("title")}
