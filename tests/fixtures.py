"""Shared in-test fixtures: a tiny question bank in two obvious clusters, a
tiny corpus, an Infra factory, and a deterministic-arrival helper.

    q0001  capital of France?          -> Paris        2hop  100  k01
    q0002  longest river in France?    -> Loire        2hop  200  k01
    q0003  largest city in France?     -> Paris        2hop  300  k01
    q0004  highest mountain in France? -> Mont Blanc   3hop  400  k01
    q0005  sum of 2 and 2?             -> 4 / four     2hop  100  k07
    q0006  sum of 3 and 3?             -> 6 / six      2hop  200  k07
    q0007  sum of 4 and 4?             -> 8 / eight    3hop  300  k07
    q0008  sum of 5 and 5?             -> 10 / ten     2hop  400  k07

8 questions in two lexical families ("... France?" vs "sum of ... and ...?")
that the deterministic bag-of-words embedding stub (`HashEmbedding`) separates
cleanly, so k-means with k=2 recovers exactly the France/arithmetic split and
routing is testable. Two topics carry four questions each, so per-agent
specialization stays measurable on the demo bank alone.

The demo corpus is 4 paragraphs whose token overlap makes the stub rank them
sensibly for the demo questions -- every Infra memory is seeded with it
through the SAME `seed_corpus` path production uses (precomputed stub
embeddings, no re-embedding), so no test ever loads the ONNX model.
"""
import math
import zlib

from chromadb.api.types import EmbeddingFunction

from ca.bank import Question, QuestionBank
from ca.chat import EXTERNAL
from ca.cluster import exemplar_qids, kmeans
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
        Question("q0003", "largest city in France?", ["Paris"], "2hop", 300, "k01"),
        Question("q0004", "highest mountain in France?", ["Mont Blanc"], "3hop", 400, "k01"),
        Question("q0005", "sum of 2 and 2?", ["4", "four"], "2hop", 100, "k07"),
        Question("q0006", "sum of 3 and 3?", ["6", "six"], "2hop", 200, "k07"),
        Question("q0007", "sum of 4 and 4?", ["8", "eight"], "3hop", 300, "k07"),
        Question("q0008", "sum of 5 and 5?", ["10", "ten"], "2hop", 400, "k07"),
    ]


DEMO_CORPUS = [
    {"title": "Paris", "text": "Paris is the capital of France and its largest city."},
    {"title": "Loire", "text": "The Loire is the longest river of France."},
    {"title": "Mont Blanc", "text": "Mont Blanc is the highest mountain in France."},
    {"title": "Arithmetic", "text": "Sums: the sum of 2 and 2 is 4, the sum of "
                                    "3 and 3 is 6, the sum of 4 and 4 is 8, and "
                                    "the sum of 5 and 5 is 10."},
]


def demo_corpus_embeddings(corpus=None):
    corpus = corpus or DEMO_CORPUS
    return HashEmbedding()([p["text"] for p in corpus])


def demo_bank() -> QuestionBank:
    return QuestionBank(demo_questions())


def demo_domains(bank: QuestionBank, n_agents: int):
    """(assignment, exemplars) for the demo bank, through the same kmeans the
    production cache builder uses -- stub embeddings, seed 0."""
    qids = sorted(bank.questions)
    emb = HashEmbedding()([bank.questions[q].text for q in qids])
    centroids, assign = kmeans(emb, n_agents, seed=0)
    assignment = {qid: int(a) for qid, a in zip(qids, assign)}
    ex = exemplar_qids(qids, emb, centroids, assign)
    exemplars = {f"agent_{j + 1}": [bank.questions[q].text for q in ex[j]]
                 for j in ex}
    return assignment, exemplars


def demo_infra(level: str = "P0", bank: QuestionBank | None = None,
               corpus=None, seed: int = 0, n_agents: int = 2, **cfg_kw):
    from ca.infra import Infra
    cfg = ExperimentConfig(level=CONFIGS[level], seed=seed, n_agents=n_agents,
                           **cfg_kw)
    bank = bank or demo_bank()
    corpus = corpus or DEMO_CORPUS
    assignment, exemplars = demo_domains(bank, n_agents)
    return Infra(cfg, bank, assignment=assignment, corpus=corpus,
                 corpus_embeddings=demo_corpus_embeddings(corpus),
                 embedding_function=HashEmbedding(), exemplars=exemplars)


def arrive(infra, qid: str, round_no: int | None = None) -> str:
    """Force one bank question to arrive NOW (deterministic tests): advances
    the stream exactly as a tick arrival would -- pending entry, external
    message, unread counter -- and returns the assignee."""
    s = infra.stream
    r = round_no if round_no is not None else max(infra.round, 1)
    i = s.order.index(qid)
    assert i >= s.pos, f"{qid} already arrived"
    s.order.pop(i)
    s.order.insert(s.pos, qid)
    s.pos += 1
    agent = s.routing[qid]
    s.pending[qid] = (agent, r)
    infra.chat.send(EXTERNAL, agent, f"[{qid}] {infra.bank.questions[qid].text}", r)
    return agent
