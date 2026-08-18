"""Offline fixtures for kb tests: a deterministic bag-of-words embedding stub
(no ONNX model ever loads), a mini universe, and a tool-aware driving policy
for scripted end-to-end runs."""
import math
import re
import zlib
from collections import defaultdict

from chromadb.api.types import EmbeddingFunction

from kb.build import Universe, build_universe
from kb.policy import Decision
from kb.store import Store

DIM = 64


class HashEmbedding(EmbeddingFunction):
    """Deterministic bag-of-words embedding: cosine order == token overlap."""

    def __init__(self):
        pass

    @staticmethod
    def name() -> str:
        return "hash-stub"

    def get_config(self) -> dict:
        return {}

    def __call__(self, input):
        out = []
        for doc in input:
            v = [0.0] * DIM
            for w in str(doc).lower().split():
                v[zlib.crc32(w.encode()) % DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def mini_universe(seed: int = 0, n_people: int = 12,
                  sizes=(8, 6, 5), eval_sizes=(3, 2), **kw) -> Universe:
    return build_universe(seed, n_people, sizes, eval_sizes, **kw)


def mini_store(universe: Universe | None = None) -> Store:
    u = universe or mini_universe()
    return Store.from_nodes(u.nodes, embedding_function=HashEmbedding())


class ListLog:
    """In-memory RunLog stand-in: same interface, rows kept in lists."""

    def __init__(self):
        self.rows = {"trace": [], "train": [], "test": [], "stats": []}
        self.tokens = defaultdict(lambda: {"in": 0, "out": 0})

    def trace(self, row):
        t = self.tokens[row["kind"]]
        t["in"] += row.get("tokens_in", 0)
        t["out"] += row.get("tokens_out", 0)
        self.rows["trace"].append(row)

    def train(self, row):
        self.rows["train"].append(row)

    def test(self, row):
        self.rows["test"].append(row)

    def stats(self, row):
        self.rows["stats"].append(row)

    def close(self):
        pass


class DrivePolicy:
    """Tool-aware deterministic driver for e2e runs. Answer phases (train
    Phase 1 and test): search once, then answer (from `answers` by qid, else
    "unknown"). Backprop: one add, then done. Detects the phase from the
    schema subset + task block, so one instance can drive a whole run."""

    def __init__(self, answers: dict[str, str] | None = None,
                 in_tokens: int = 7, out_tokens: int = 3):
        self.answers = answers or {}
        self.in_tokens, self.out_tokens = in_tokens, out_tokens
        self._key, self._n = None, 0

    def decide(self, system, context, tools) -> Decision:
        names = {t["name"] for t in tools}
        m = re.search(r"q\d{4}", context)
        qid = m.group(0) if m else "?"
        phase = ("p2" if "[phase 2" in context
                 else "p1" if "TRAIN QUESTION" in context else "test")
        if (qid, phase) != self._key:
            self._key, self._n = (qid, phase), 0
        self._n += 1
        tk = (self.in_tokens, self.out_tokens)
        if "answer" in names:
            if self._n == 1:
                return Decision("search", {"query": context.splitlines()[1]}, *tk)
            return Decision("answer", {"text": self.answers.get(qid, "unknown")}, *tk)
        if self._n == 1:
            return Decision("add", {"text": f"Navigation note for {qid}."}, *tk)
        return Decision("done", {}, *tk)
