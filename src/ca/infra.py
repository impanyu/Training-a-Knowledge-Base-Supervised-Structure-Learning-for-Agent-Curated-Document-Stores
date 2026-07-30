"""Aggregate of all authoritative world state."""
from collections import defaultdict

from ca.chat import ChatSystem
from ca.config import ExperimentConfig, agent_ids
from ca.contracts import ContractSystem
from ca.economy import Ledger
from ca.loans import LoanSystem
from ca.memory import LongTermMemory
from ca.taskboard import Question, TaskBoard


class Infra:
    def __init__(self, cfg: ExperimentConfig, questions: list[Question], retriever):
        self.cfg = cfg
        self.agent_ids = agent_ids(cfg.level)
        n = len(self.agent_ids)
        base, rem = divmod(cfg.seed_capital_total, n)
        seed_capital = {a: base for a in self.agent_ids}
        seed_capital[self.agent_ids[0]] += rem
        self.ledger = Ledger(seed_capital)
        self.chat = ChatSystem()
        self.contracts = ContractSystem(self.ledger)
        self.loans = LoanSystem(self.ledger, cfg.loan_rate)
        self.board = TaskBoard(questions, self.ledger)
        self.ltm = LongTermMemory()
        self.retriever = retriever
        self.scratchpads: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.round = 0
