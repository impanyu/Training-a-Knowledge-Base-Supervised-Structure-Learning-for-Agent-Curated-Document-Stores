"""Aggregate of all authoritative world state."""

from ca.bank import QuestionBank
from ca.chat import ChatSystem
from ca.config import ExperimentConfig, agent_ids
from ca.memory import AgentMemory
from ca.stream import QuestionStream


class Infra:
    def __init__(self, cfg: ExperimentConfig, bank: QuestionBank,
                 assignment: dict[str, int], corpus: list[dict] | None = None,
                 corpus_embeddings=None, embedding_function=None,
                 exemplars: dict[str, list[str]] | None = None):
        self.cfg = cfg
        self.bank = bank
        self.agent_ids = agent_ids(cfg.n_agents)
        self.chat = ChatSystem()
        self.stream = QuestionStream(bank, cfg.n_agents, cfg.seed,
                                     cfg.arrival_rate, assignment)
        # the cluster's ONE shared knowledge base, born knowing the corpus
        self.memory = AgentMemory(embedding_function=embedding_function)
        if corpus:
            self.memory.seed_corpus(corpus, corpus_embeddings)
        # agent -> its domain's exemplar question texts (system prompt)
        self.exemplars = exemplars or {a: [] for a in self.agent_ids}
        self.round = 0
