"""CLI entry point: run one (level, seed) experiment."""
import argparse
import json
import random

from ca.agent import Agent
from ca.config import LEVELS, ExperimentConfig
from ca.infra import Infra
from ca.metrics import compute_metrics
from ca.providers import make_policy
from ca.recorder import Recorder
from ca.retrieval import ChromaBackend
from ca.scheduler import Scheduler
from ca.taskboard import Question


def load_questions(path: str) -> list[Question]:
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out.append(Question(r["qid"], r["text"], r["answers"],
                                r["difficulty"], r["price"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=list(LEVELS))
    ap.add_argument("--questions", required=True)
    ap.add_argument("--index", required=True, help="chroma persist dir from prepare_data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capital", type=int, default=400_000)
    ap.add_argument("--max-rounds", type=int, default=60)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = ExperimentConfig(level=LEVELS[args.level], seed=args.seed,
                           seed_capital_total=args.capital,
                           max_rounds=args.max_rounds, model=args.model)
    infra = Infra(cfg, load_questions(args.questions),
                  retriever=ChromaBackend.load(args.index))
    agents = [Agent(a, cfg, infra, make_policy(cfg.model, cfg.max_tokens_per_turn))
              for a in infra.agent_ids]
    sched = Scheduler(infra, agents, cfg, Recorder(args.out), random.Random(args.seed))
    summary = sched.run()
    metrics = compute_metrics(summary)
    print(json.dumps(metrics, indent=2))
    with open(f"{args.out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
