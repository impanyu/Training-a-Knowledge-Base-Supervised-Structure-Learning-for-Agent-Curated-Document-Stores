"""Shared in-test fixtures: a tiny question bank, a tiny corpus, an Infra
factory.

    q0001  capital of France?        -> Paris        2hop  100  k01
    q0002  longest river in France?  -> Loire        2hop  200  k01
    q0003  2+2?                      -> 4 / four     2hop  300  k07
    q0004  3+3?                      -> 6 / six      3hop  400  k07
    q0005  which rock type is chalk? -> sedimentary  2hop   50  k02

5 questions = 5 units of demand; two topics carry two questions each so
per-agent specialization over topics is measurable on the demo bank alone.

The demo corpus is 4 paragraphs whose token overlap makes the deterministic
bag-of-words embedding stub (`HashEmbedding`) rank them sensibly for the demo
questions -- every Infra memory is seeded with it through the SAME
`seed_corpus` path production uses (precomputed stub embeddings, no
re-embedding), so no test ever loads the ONNX model.
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
        Question("q0001", "capital of France?", ["Paris"], "2hop", 100, "k01"),
        Question("q0002", "longest river in France?", ["Loire"], "2hop", 200, "k01"),
        Question("q0003", "2+2?", ["4", "four"], "2hop", 300, "k07"),
        Question("q0004", "3+3?", ["6", "six"], "3hop", 400, "k07"),
        Question("q0005", "which rock type is chalk?", ["sedimentary"], "2hop", 50, "k02"),
    ]


DEMO_CORPUS = [
    {"title": "Paris", "text": "Paris is the capital of France."},
    {"title": "Loire", "text": "The Loire is the longest river of France."},
    {"title": "Chalk", "text": "Chalk is a sedimentary rock type."},
    {"title": "Arithmetic", "text": "Adding numbers: 2+2 makes 4 and 3+3 makes 6."},
]


def demo_corpus_embeddings(corpus=None):
    corpus = corpus or DEMO_CORPUS
    return HashEmbedding()([p["text"] for p in corpus])


def demo_bank() -> QuestionBank:
    return QuestionBank(demo_questions())


def demo_infra(level: str = "C0", capital: int = 1000,
               bank: QuestionBank | None = None, corpus=None, seed: int = 0,
               **cfg_kw):
    from ca.infra import Infra
    cfg = ExperimentConfig(level=CONFIGS[level], seed=seed, seed_capital_total=capital,
                           **cfg_kw)
    corpus = corpus or DEMO_CORPUS
    return Infra(cfg, bank or demo_bank(), corpus=corpus,
                 corpus_embeddings=demo_corpus_embeddings(corpus),
                 embedding_function=HashEmbedding())
