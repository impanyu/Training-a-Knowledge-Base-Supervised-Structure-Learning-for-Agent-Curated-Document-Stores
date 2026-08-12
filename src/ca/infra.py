"""Aggregate of all authoritative world state."""

from ca.bank import QuestionBank
from ca.board import QuestionBoard
from ca.chat import ChatSystem
from ca.config import ExperimentConfig, agent_ids
from ca.memory import AgentMemory


class Infra:
    def __init__(self, cfg: ExperimentConfig, bank: QuestionBank,
                 corpus: list[dict] | None = None, corpus_embeddings=None,
                 embedding_function=None):
        self.cfg = cfg
        self.bank = bank
        self.agent_ids = agent_ids(cfg.level)
        self.chat = ChatSystem()
        self.board = QuestionBoard(bank)
        # notes AND graded answers; one shared bucket at C2. Born knowing the
        # corpus: every bucket is seeded with the full paragraph set.
        self.memory = AgentMemory(shared=cfg.level.shared_memory,
                                  embedding_function=embedding_function)
        if corpus:
            self.memory.seed_corpus(corpus, corpus_embeddings, self.agent_ids)
        self.round = 0
