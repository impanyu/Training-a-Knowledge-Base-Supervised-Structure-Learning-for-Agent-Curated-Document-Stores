"""CLI: examine a frozen KB with the independent reader (spec §9).

--kb accepts either a per-epoch snapshot (kb_epoch_N.json, which remembers
its universe path) or a universe.json directly (= the untrained store, the
pure RAG baseline). Logs are opened in append mode, so pointing --out at a
training run's directory adds the one final test_in/test_out pass next to
that run's eval learning curve without clobbering it (T39.1 protocol: full
test exactly once, after training, plus the manual epoch-0 baseline)."""
import argparse
import json
from pathlib import Path

from kb.build import Universe
from kb.loops import K, M, run_test
from kb.policy import make_policy
from kb.recorder import RunLog
from kb.store import Store, load_snapshot


def load_kb(kb_path: str, universe_path: str | None = None,
            summarizer=None, embedding_function=None):
    """(store, universe) from a snapshot or a universe file."""
    with open(kb_path) as f:
        data = json.load(f)
    if "questions" in data:                     # a universe: untrained store
        u = Universe.from_json(data)
        return Store.from_docs(u.docs, summarizer, embedding_function), u
    store, upath = load_snapshot(kb_path, summarizer, embedding_function)
    upath = universe_path or upath
    if not upath:
        raise SystemExit("snapshot carries no universe path; pass --universe")
    return store, Universe.load(upath)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True,
                    help="kb_epoch_N.json snapshot or universe.json")
    ap.add_argument("--split", required=True,
                    help='test_in | test_out | eval | both, or a comma list '
                         'like "test_in,test_out" (budget sweeps / headroom '
                         'probes); eval rows carry their in/out flavor')
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", type=int, default=M)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--epoch", type=int, default=0,
                    help="epoch label recorded on every test_log row")
    ap.add_argument("--universe", default=None,
                    help="override the universe path stored in the snapshot")
    args = ap.parse_args(argv)

    store, universe = load_kb(args.kb, args.universe)
    policy = make_policy(args.model)
    log = RunLog(args.out, append=True)
    splits = (["test_in", "test_out"] if args.split == "both"
              else [s.strip() for s in args.split.split(",") if s.strip()])
    for split in splits:
        if split not in universe.splits:
            raise SystemExit(f"unknown split {split}; available: "
                             f"{', '.join(universe.splits)}")
    rows = []
    for split in splits:
        rows += run_test(store, policy, universe, split, log,
                         epoch=args.epoch, m=args.m, k=K, limit=args.limit)
    meta = {"kb": args.kb, "split": args.split, "epoch": args.epoch,
            "model": args.model, "m": args.m, "limit": args.limit,
            "n_questions": len(rows),
            "tokens": {kind: dict(t) for kind, t in log.tokens.items()}}
    # test_meta.json, NOT meta.json: appending into a train run's directory
    # must not clobber the training meta the report reads
    with open(Path(args.out) / "test_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.close()
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
