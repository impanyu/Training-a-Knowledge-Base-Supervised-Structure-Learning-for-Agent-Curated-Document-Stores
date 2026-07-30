"""CLI entry point: run one (level, seed) experiment."""
import argparse
import json
import random

from ca.agent import Agent
from ca.config import CONFIGS, ExperimentConfig
from ca.infra import Infra
from ca.metrics import compute_metrics
from ca.providers import make_policy
from ca.recorder import Recorder
from ca.retrieval import ChromaBackend
from ca.scheduler import Scheduler
from ca.taskboard import Question
from ca.tasktree import TaskLibrary, TaskNode


def load_questions(path: str) -> list[Question]:
    """Raw question pool (pool.jsonl), the input the task library is built from."""
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out.append(Question(r["qid"], r["text"], r["answers"],
                                r["difficulty"], r["price"]))
    return out


def load_library(path: str) -> TaskLibrary:
    """`library.json` (nodes + questions [+ posted]) is the real format.

    A bare `pool.jsonl` is also accepted and wrapped into a degenerate library
    of one single-leaf task per question -- an interim shim so live runs work
    before the clustering builder (T21) exists. It produces depth-2 trees, so
    decompose/packaging still exercise the real code path."""
    if path.endswith(".jsonl"):
        questions = load_questions(path)
        nodes = [TaskNode(f"t{i:04d}", q.text, [q.qid])
                 for i, q in enumerate(questions, start=1)]
        return TaskLibrary(nodes, questions)
    return TaskLibrary.from_json(path)


def load_posted(library: TaskLibrary, path: str | None = None) -> list[str]:
    """Which library nodes the WORLD puts on the board. An explicit posted.json
    (a JSON list of node ids) wins; otherwise the library's own `posted` field;
    otherwise every root, so a library alone is always runnable."""
    if path:
        with open(path) as f:
            return list(json.load(f))
    if library.posted:
        return list(library.posted)
    return [n.nid for n in library.nodes.values() if n.parent is None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=list(CONFIGS))
    ap.add_argument("--library", required=True,
                    help="library.json (task tree + question pool); a bare pool.jsonl is also accepted and wrapped into one single-leaf task per question")
    ap.add_argument("--posted", default=None,
                    help="posted.json: JSON list of node ids to put on the board; "
                         "defaults to the library's own posted list, else its roots")
    ap.add_argument("--index", required=True, help="chroma persist dir from prepare_data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capital", type=int, default=400_000)
    ap.add_argument("--max-rounds", type=int, default=60)
    ap.add_argument("--turns", type=int, default=None,
                    help="total agent-turn budget; overrides --max-rounds with turns//n_agents")
    ap.add_argument("--solo-turns", type=int, default=1,
                    help="solo-agent turns per round at C7 (8 = compute parity with 8-agent configs)")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from ca.config import agent_ids
    level = CONFIGS[args.level]
    cfg = ExperimentConfig(level=level, seed=args.seed,
                           seed_capital_total=args.capital,
                           max_rounds=args.max_rounds, model=args.model,
                           solo_turns_per_round=args.solo_turns)
    if args.turns is not None:
        slots_per_round = len(agent_ids(level)) + (
            cfg.hub_turns_per_round - 1 if level.has_hub else 0) + (
            cfg.solo_turns_per_round - 1 if level.n_agents == 1 else 0)
        cfg.max_rounds = max(1, args.turns // slots_per_round)
    library = load_library(args.library)
    infra = Infra(cfg, library, load_posted(library, args.posted),
                  retriever=ChromaBackend.load(args.index))
    agents = [Agent(a, cfg, infra, make_policy(cfg.model, cfg.max_tokens_per_turn, cfg.temperature))
              for a in infra.agent_ids]
    sched = Scheduler(infra, agents, cfg, Recorder(args.out), random.Random(args.seed))
    summary = sched.run()
    metrics = compute_metrics(summary, library=library)
    print(json.dumps(metrics, indent=2))
    with open(f"{args.out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
