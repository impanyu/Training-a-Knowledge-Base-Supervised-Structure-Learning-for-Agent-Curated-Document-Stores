"""Shared in-test fixtures: a tiny hierarchical task library.

    t0001 <<answer the french geography questions>>        [posted, 600]
      +-- t0002 <<name the capital and the river>>         [300]
      |     +-- q0001  capital of France?      (100)
      |     +-- q0002  longest river in France? (200)
      +-- q0003  2+2?                          (300)

    t0004 <<resolve the two arithmetic warmup questions>>   [posted, 700]
      +-- q0003  2+2?                          (300)   <- SHARED with t0001
      +-- q0004  3+3?                          (400)

    t0005 <<answer the french geology questions>>          [not posted]
      +-- q0005  which rock type is chalk?      (50)

t0005 exists mainly so that fuzzy resolution has a near-twin of t0001 to be
ambiguous against.
"""
from ca.config import LEVELS, ExperimentConfig
from ca.taskboard import Question
from ca.tasktree import TaskLibrary, TaskNode


def demo_questions() -> list[Question]:
    return [
        Question("q0001", "capital of France?", ["Paris"], "easy", 100),
        Question("q0002", "longest river in France?", ["Loire"], "easy", 200),
        Question("q0003", "2+2?", ["4", "four"], "easy", 300),
        Question("q0004", "3+3?", ["6", "six"], "easy", 400),
        Question("q0005", "which rock type is chalk?", ["sedimentary"], "easy", 50),
    ]


def demo_library() -> TaskLibrary:
    nodes = [
        TaskNode("t0001", "answer the french geography questions", ["t0002", "q0003"]),
        TaskNode("t0002", "name the capital and the river", ["q0001", "q0002"]),
        TaskNode("t0004", "resolve the two arithmetic warmup questions", ["q0003", "q0004"]),
        TaskNode("t0005", "answer the french geology questions", ["q0005"]),
    ]
    return TaskLibrary(nodes, demo_questions())


def demo_posted() -> list[str]:
    return ["t0001", "t0004"]


def demo_infra(level: str = "L0", capital: int = 1000, retriever=None,
               library: TaskLibrary | None = None, posted: list[str] | None = None,
               **cfg_kw):
    from ca.infra import Infra
    cfg = ExperimentConfig(level=LEVELS[level], seed=0, seed_capital_total=capital,
                           **cfg_kw)
    return Infra(cfg, library or demo_library(),
                 demo_posted() if posted is None else posted, retriever=retriever)
