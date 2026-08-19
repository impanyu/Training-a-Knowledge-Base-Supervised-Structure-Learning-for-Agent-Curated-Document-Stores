"""CLI: train the KB on the train split for N epochs (spec §9).

Training only trains and snapshots; kb.test evaluates any snapshot. The
--eval-each-epoch convenience flag runs the small EVAL split (both flavors)
after the epoch-0 baseline snapshot and every epoch; test_in/test_out are
never touched during training (T39.1)."""
import argparse
import json
from pathlib import Path

from kb.build import Universe
from kb.loops import K, M, N1, N2, run_training
from kb.policy import make_policy
from kb.recorder import RunLog
from kb.store import DEDUP_THRESHOLD, LLMDuplicateJudge, Store


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-size", type=int, default=None)
    ap.add_argument("--n1", type=int, default=N1)
    ap.add_argument("--n2", type=int, default=N2)
    ap.add_argument("--m", type=int, default=M)
    ap.add_argument("--eval-each-epoch", action="store_true",
                    help="run the eval split after every snapshot")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="also snapshot every N train iterations as "
                         "kb_iter_XXXX.json (0 = per-epoch only)")
    ap.add_argument("--no-dedup", action="store_true",
                    help="disable infrastructure dedup (no judge wired)")
    ap.add_argument("--dedup-threshold", type=float, default=DEDUP_THRESHOLD,
                    help="cosine similarity at which the duplicate judge "
                         "is consulted on add/edit")
    ap.add_argument("--judge-model", default="gpt-5-mini",
                    help="model for the duplicate judge")
    args = ap.parse_args(argv)

    universe = Universe.load(args.universe)
    judge = None if args.no_dedup else LLMDuplicateJudge(args.judge_model)
    store = Store.from_nodes(universe.nodes, judge=judge,
                             dedup_threshold=args.dedup_threshold)
    policy = make_policy(args.model)
    log = RunLog(args.out)
    run_training(store, policy, universe, log, args.out, epochs=args.epochs,
                 seed=args.seed, n1=args.n1, n2=args.n2, m=args.m, k=K,
                 train_size=args.train_size,
                 eval_each_epoch=args.eval_each_epoch,
                 universe_path=args.universe,
                 snapshot_every=args.snapshot_every)
    meta = {"universe": args.universe, "epochs": args.epochs,
            "seed": args.seed, "model": args.model,
            "n1": args.n1, "n2": args.n2, "m": args.m,
            "train_size": args.train_size,
            "snapshot_every": args.snapshot_every,
            "dedup": {"enabled": judge is not None,
                      "threshold": args.dedup_threshold,
                      "judge_model": None if judge is None else args.judge_model,
                      "merges": store.merges,
                      "judge_tokens": None if judge is None else
                      {"in": judge.tokens_in, "out": judge.tokens_out}},
            "tokens": {kind: dict(t) for kind, t in log.tokens.items()}}
    with open(Path(args.out) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.close()
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
