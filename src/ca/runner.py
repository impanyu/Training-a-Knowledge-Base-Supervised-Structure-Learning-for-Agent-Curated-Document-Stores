"""CLI entry point: run one (level, seed) experiment."""
import argparse
import json
import random
from pathlib import Path

from ca import checkpoint
from ca.agent import Agent
from ca.bank import QuestionBank
from ca.cluster import build_cache, exemplar_texts, load_cache
from ca.config import CONFIGS, ExperimentConfig
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
                         "gold answers, topic). corpus.jsonl, corpus_emb.npy and "
                         "the qclusters_{N}.json cache are read from (or written "
                         "to) the same directory")
    ap.add_argument("--agents", type=int, default=8,
                    help="cluster size: one agent per question-topic cluster")
    ap.add_argument("--arrival-rate", type=float, default=0.5,
                    help="Poisson lambda: external questions arriving per round")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rounds", type=int, default=60)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-every", type=int, default=20,
                    help="save a full-state checkpoint every N rounds (T29)")
    ap.add_argument("--resume", default=None,
                    help="checkpoint_XXXX.json from a previous run: restore its "
                         "state and continue from the next round to --max-rounds "
                         "(level/seed/agents/arrival-rate must match; trace/"
                         "timeseries in --out are appended to, not rewritten)")
    return ap


def _onnx_ef():
    """Chroma's default ONNX embedder -- the same model the corpus was
    embedded with, so question clusters live in the corpus's own space."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    return DefaultEmbeddingFunction()


def question_clusters(bank: QuestionBank, data_dir: Path, n_agents: int) -> dict:
    """The cluster cache for this bank at this N: load it, or embed the
    question texts and build it once."""
    path = data_dir / f"qclusters_{n_agents}.json"
    if path.exists():
        return load_cache(path)
    return build_cache(path, bank, _onnx_ef(), n_agents)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    cfg = ExperimentConfig(level=CONFIGS[args.level], seed=args.seed,
                           n_agents=args.agents, arrival_rate=args.arrival_rate,
                           max_rounds=args.max_rounds, model=args.model,
                           checkpoint_every=args.checkpoint_every)
    bank = QuestionBank.from_json(args.bank)
    data_dir = Path(args.bank).parent
    corpus, corpus_emb = load_corpus(data_dir / "corpus.jsonl",
                                     data_dir / "corpus_emb.npy")
    cache = question_clusters(bank, data_dir, cfg.n_agents)
    state = None
    if args.resume:
        state = checkpoint.load(args.resume)
        checkpoint.validate(state, cfg)
    infra = Infra(cfg, bank, assignment=cache["assignment"], corpus=corpus,
                  corpus_embeddings=corpus_emb,
                  exemplars=exemplar_texts(cache, bank))
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
