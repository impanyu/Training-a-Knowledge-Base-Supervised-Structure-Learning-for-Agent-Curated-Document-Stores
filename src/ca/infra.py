"""Aggregate of all authoritative world state."""
from collections import defaultdict

from ca.bank import QuestionBank
from ca.board import QuestionBoard
from ca.chat import ChatSystem
from ca.config import ExperimentConfig, agent_ids
from ca.contracts import ContractSystem
from ca.economy import Ledger
from ca.loans import LoanSystem
from ca.memory import AgentMemory


class Infra:
    def __init__(self, cfg: ExperimentConfig, bank: QuestionBank, retriever,
                 embedding_function=None):
        self.cfg = cfg
        self.bank = bank
        self.agent_ids = agent_ids(cfg.level)
        n = len(self.agent_ids)
        base, rem = divmod(cfg.seed_capital_total, n)
        seed_capital = {a: base for a in self.agent_ids}
        seed_capital[self.agent_ids[0]] += rem
        self.ledger = Ledger(seed_capital)
        self.chat = ChatSystem()
        self.contracts = ContractSystem(self.ledger)
        self.loans = LoanSystem(self.ledger, cfg.loan_rate)
        self.board = QuestionBoard(bank, self.ledger)
        # notes AND graded answers; one shared bucket at C2
        self.memory = AgentMemory(shared=cfg.level.shared_memory,
                                  embedding_function=embedding_function)
        self.retriever = retriever
        self.scratchpads: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.round = 0
