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
from kb.store import LLMSummarizer, Store


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
    ap.add_argument("--summarizer-model", default="gpt-5-mini")
    args = ap.parse_args(argv)

    universe = Universe.load(args.universe)
    store = Store.from_docs(universe.docs,
                            summarizer=LLMSummarizer(args.summarizer_model))
    policy = make_policy(args.model)
    log = RunLog(args.out)
    run_training(store, policy, universe, log, args.out, epochs=args.epochs,
                 seed=args.seed, n1=args.n1, n2=args.n2, m=args.m, k=K,
                 train_size=args.train_size,
                 eval_each_epoch=args.eval_each_epoch,
                 universe_path=args.universe)
    meta = {"universe": args.universe, "epochs": args.epochs,
            "seed": args.seed, "model": args.model,
            "n1": args.n1, "n2": args.n2, "m": args.m,
            "train_size": args.train_size,
            "build_tokens": universe.meta.get("build_tokens"),
            "tokens": {kind: dict(t) for kind, t in log.tokens.items()},
            "summarizer_tokens": {"in": store.summarizer.tokens_in,
                                  "out": store.summarizer.tokens_out}}
    with open(Path(args.out) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.close()
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
