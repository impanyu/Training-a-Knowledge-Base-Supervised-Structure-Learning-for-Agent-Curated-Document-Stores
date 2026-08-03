"""Shared in-test fixtures: a tiny flat question bank plus an Infra factory.

    q0001  capital of France?        -> Paris        2hop  100  quota 2  k01
    q0002  longest river in France?  -> Loire        2hop  200  quota 1  k01
    q0003  2+2?                      -> 4 / four     2hop  300  quota 2  k07
    q0004  3+3?                      -> 6 / six      3hop  400  quota 1  k07
    q0005  which rock type is chalk? -> sedimentary  2hop   50  quota 1  k02

7 claimable units in total; two topics carry two questions each, so per-agent
specialization over topics is measurable on the demo bank alone.

Infra's vector memory is built with a deterministic bag-of-words embedding
stub, so no test ever loads the ONNX model (same trick as KeywordBackend
standing in for ChromaBackend).
"""
import math
import zlib

from chromadb.api.types import EmbeddingFunction

from ca.bank import Question, QuestionBank
from ca.config import CONFIGS, ExperimentConfig

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


def demo_questions() -> list[Question]:
    return [
        Question("q0001", "capital of France?", ["Paris"], "2hop", 100, 2, "k01"),
        Question("q0002", "longest river in France?", ["Loire"], "2hop", 200, 1, "k01"),
        Question("q0003", "2+2?", ["4", "four"], "2hop", 300, 2, "k07"),
        Question("q0004", "3+3?", ["6", "six"], "3hop", 400, 1, "k07"),
        Question("q0005", "which rock type is chalk?", ["sedimentary"], "2hop", 50, 1, "k02"),
    ]


def demo_bank() -> QuestionBank:
    return QuestionBank(demo_questions())


def demo_infra(level: str = "C0", capital: int = 1000, retriever=None,
               bank: QuestionBank | None = None, **cfg_kw):
    from ca.infra import Infra
    cfg = ExperimentConfig(level=CONFIGS[level], seed=0, seed_capital_total=capital,
                           **cfg_kw)
    return Infra(cfg, bank or demo_bank(), retriever=retriever,
                 embedding_function=HashEmbedding())
