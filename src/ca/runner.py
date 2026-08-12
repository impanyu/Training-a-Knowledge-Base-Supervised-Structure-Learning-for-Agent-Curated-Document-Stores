"""CLI entry point: run one (level, seed) experiment."""
import argparse
import json
import random
from pathlib import Path

from ca import checkpoint
from ca.agent import Agent
from ca.bank import QuestionBank
from ca.config import CONFIGS, ExperimentConfig, agent_ids
from ca.infra import Infra
from ca.memory import load_corpus
from ca.metrics import compute_metrics
from ca.providers import make_policy
from ca.recorder import Recorder
from ca.scheduler import Scheduler


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=list(CONFIGS))
    ap.add_argument("--bank", required=True,
                    help="bank.json from build_bank.py: the question bank (text, "
                         "gold answers, topic). corpus.jsonl and "
                         "corpus_emb.npy are read from the same directory to "
                         "seed every agent's memory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rounds", type=int, default=60)
    ap.add_argument("--turns", type=int, default=None,
                    help="total agent-turn budget; overrides --max-rounds with turns//n_agents")
    ap.add_argument("--solo-turns", type=int, default=1,
                    help="solo-agent turns per round at C7 (8 = compute parity with 8-agent configs)")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-every", type=int, default=20,
                    help="save a full-state checkpoint every N rounds (T29)")
    ap.add_argument("--resume", default=None,
                    help="checkpoint_XXXX.json from a previous run: restore its "
                         "state and continue from the next round to --max-rounds "
                         "(level and seed must match; trace/timeseries in --out "
                         "are appended to, not rewritten)")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    level = CONFIGS[args.level]
    cfg = ExperimentConfig(level=level, seed=args.seed,
                           max_rounds=args.max_rounds, model=args.model,
                           solo_turns_per_round=args.solo_turns,
                           checkpoint_every=args.checkpoint_every)
    if args.turns is not None:
        slots_per_round = len(agent_ids(level)) + (
            cfg.hub_turns_per_round - 1 if level.has_hub else 0) + (
            cfg.solo_turns_per_round - 1 if level.n_agents == 1 else 0)
        cfg.max_rounds = max(1, args.turns // slots_per_round)
    bank = QuestionBank.from_json(args.bank)
    data_dir = Path(args.bank).parent
    corpus, corpus_emb = load_corpus(data_dir / "corpus.jsonl",
                                     data_dir / "corpus_emb.npy")
    state = None
    if args.resume:
        state = checkpoint.load(args.resume)
        checkpoint.validate(state, cfg)  # level + seed must match
    infra = Infra(cfg, bank, corpus=corpus, corpus_embeddings=corpus_emb)
    agents = [Agent(a, cfg, infra, make_policy(cfg.model, cfg.max_tokens_per_turn, cfg.temperature))
              for a in infra.agent_ids]
    recorder = Recorder(args.out, append=state is not None)
    rng = random.Random(args.seed)
    if state is not None:
        checkpoint.restore(state, infra, agents, recorder, rng)
    sched = Scheduler(infra, agents, cfg, recorder, rng)
    summary = sched.run(start_round=state["round"] + 1 if state else 1)
    metrics = compute_metrics(summary)
    print(json.dumps(metrics, indent=2))
    with open(f"{args.out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
