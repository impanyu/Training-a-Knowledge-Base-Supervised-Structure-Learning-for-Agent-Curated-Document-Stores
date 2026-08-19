"""Trace-replay KB reconstruction (T39.8): rebuild the store exactly as it
was at the end of any train iteration, from universe + trace.jsonl alone.

Mechanism: load the universe, then re-apply every SUCCESSFUL phase-2 edit
action (add / edit / delete / link / unlink; rows whose result starts with
"ERROR" are skipped) from the trace in order. An iteration = contiguous
train rows of one (epoch, qid); --at N stops after iteration N (0 = the
initial store). Ids minted by `add` are not re-minted: the assigned id is
parsed from the recorded result string ("added s0421") and the store's id
counter is set so the same id is reproduced — every replayed action's
result string is then compared against the recorded one, so any divergence
fails loudly instead of silently drifting.

Embeddings: replay is for analysis, which reads state, not similarity —
by default nodes are "embedded" with a cheap deterministic stub (offline,
no model); pass --embed to rebuild real embeddings with the chroma default
EF, the same way snapshot loading does.

Modes:
    --at N --out state.json          one reconstructed state (snapshot format,
                                     loadable by kb.test's load_kb)
    --series stats --out s.jsonl     one stats row per iteration (plus the
                                     iteration-0 baseline) — the per-timestep
                                     trajectory, no embeddings needed
    --verify runs/X/kb_epoch_N.json  replay to that epoch boundary and diff
                                     node/link/flag/origin state against the
                                     stored snapshot
    --all-epochs runs/X              verify every kb_epoch_*.json in the run

The trace format is consumed as-is (kind/epoch/qid/phase/step/action/input/
result rows); a half-written final line — a live run mid-write — is
tolerated and dropped."""
import argparse
import json
import re
import zlib
from pathlib import Path

from chromadb.api.types import EmbeddingFunction

from kb.actions import EDIT_ACTIONS, dispatch
from kb.build import Universe
from kb.store import Store, save_snapshot

_EDITS = set(EDIT_ACTIONS)
_ADD_RESULT = re.compile(r"added (s\d+)$")
_ADD_MERGE_RESULT = re.compile(
    r"duplicate of (s\d+) - merged, no new note created$")
_EDIT_MERGE_RESULT = re.compile(r"merged into (s\d+)$")


class ReplayError(Exception):
    pass


class StubEmbedding(EmbeddingFunction):
    """Deterministic offline stand-in (crc bag-of-words, dim 8): replayed
    stores get placeholder vectors so no ONNX model ever loads. Do not
    search a stub-embedded store and expect meaningful ranking."""

    def __init__(self):
        pass

    @staticmethod
    def name() -> str:
        return "replay-stub"

    def get_config(self) -> dict:
        return {}

    def __call__(self, input):
        out = []
        for doc in input:
            v = [0.0] * 8
            for w in str(doc).split():
                v[zlib.crc32(w.encode()) % 8] += 1.0
            out.append(v)
        return out


def _train_rows(trace_path) -> list[dict]:
    with open(trace_path) as f:
        lines = f.read().splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break               # live run mid-write: drop the partial tail
            raise
        if row.get("kind") == "train":
            rows.append(row)
    return rows


def _apply(store: Store, row: dict) -> None:
    """Re-apply one trace row if it is a successful phase-2 edit. Recorded
    dedup merges (T42) are replayed from their result strings — the judge's
    verdict is in the trace, so no judge (and no API) is ever needed."""
    action, result = row["action"], str(row["result"])
    if row.get("phase") != 2 or action not in _EDITS:
        return
    if result.startswith("ERROR"):
        return
    if action == "add" and result.startswith("duplicate of "):
        if not _ADD_MERGE_RESULT.fullmatch(result):
            raise ReplayError(f"unparseable add-merge result {result!r} "
                              f"(epoch {row['epoch']} {row['qid']} "
                              f"step {row['step']})")
        store.merges += 1               # nothing was created
        return
    if action == "edit" and (m := _EDIT_MERGE_RESULT.fullmatch(result)):
        store.merge(str(row["input"]["id"]).strip(), m.group(1))
        return
    if action == "add":
        m = _ADD_RESULT.fullmatch(result)
        if not m:
            raise ReplayError(f"unparseable add result {result!r} "
                              f"(epoch {row['epoch']} {row['qid']} "
                              f"step {row['step']})")
        # reproduce the live-minted id instead of re-minting
        store._next_id = int(m.group(1)[1:])
    got = dispatch(store, action, dict(row["input"]))
    if got != result:
        raise ReplayError(
            f"replay diverged at epoch {row['epoch']} {row['qid']} "
            f"step {row['step']}: {action} -> {got!r}, trace has {result!r}")


def replay(universe: Universe, trace_path, at: int | None = None,
           upto_epoch: int | None = None, embedding_function=None,
           on_iteration=None) -> tuple[Store, int]:
    """(store, iterations_applied). `at` counts train iterations across
    epochs in trace order, 1-based; 0 (or an empty trace) = initial store.
    `upto_epoch` instead applies every iteration with epoch <= N (the
    kb_epoch_N.json boundary). on_iteration(n, (epoch, qid), store) fires
    at the end of each applied iteration."""
    store = Store.from_nodes(universe.nodes, embedding_function)
    cur, n = None, 0
    for row in _train_rows(trace_path):
        if upto_epoch is not None and row["epoch"] > upto_epoch:
            break
        key = (row["epoch"], row["qid"])
        if key != cur:
            if cur is not None and on_iteration:
                on_iteration(n, cur, store)
            if at is not None and n >= at:
                return store, n
            n += 1
            cur = key
        _apply(store, row)
    if cur is not None and on_iteration:
        on_iteration(n, cur, store)
    return store, n


# ---------------- verification against epoch snapshots ----------------

def _canon(state: dict) -> dict:
    """id -> (text, origin, flag, link set, absorbed set) over a store-JSON
    dict (absorbed: T42 merge provenance; absent in pre-v11 snapshots)."""
    return {n["id"]: (n["text"], n["origin"], n["flag"],
                      tuple(sorted(n["links"])),
                      tuple(sorted(n.get("absorbed", []))))
            for n in state["nodes"]}


def _diff(replayed: dict, snapshot: dict) -> list[str]:
    lines = []
    for nid in sorted(set(replayed) - set(snapshot)):
        lines.append(f"replay-only node {nid}: {replayed[nid]}")
    for nid in sorted(set(snapshot) - set(replayed)):
        lines.append(f"snapshot-only node {nid}: {snapshot[nid]}")
    for nid in sorted(set(replayed) & set(snapshot)):
        if replayed[nid] != snapshot[nid]:
            lines.append(f"node {nid} differs: replay={replayed[nid]} "
                         f"snapshot={snapshot[nid]}")
    return lines


def verify_epoch(universe: Universe, trace_path, snap_path) -> list[str]:
    """Replay to the snapshot's epoch boundary; [] iff state matches."""
    with open(snap_path) as f:
        snap = json.load(f)
    m = re.search(r"kb_epoch_(\d+)", Path(snap_path).stem)
    if not m:
        raise ReplayError(f"{snap_path} is not a kb_epoch_N.json snapshot")
    store, _ = replay(universe, trace_path, upto_epoch=int(m.group(1)),
                      embedding_function=StubEmbedding())
    return _diff(_canon(store.to_json()), _canon(snap["store"]))


def _load_universe(universe_arg, snap_path=None) -> Universe:
    if universe_arg:
        return Universe.load(universe_arg)
    if snap_path:
        with open(snap_path) as f:
            upath = json.load(f).get("universe")
        if upath and Path(upath).exists():
            return Universe.load(upath)
    raise SystemExit("cannot resolve the universe; pass --universe")


# ---------------- CLI ----------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=None)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--at", type=int, default=None,
                    help="stop after this train iteration (0 = initial; "
                         "default: the whole trace)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--series", choices=["stats"], default=None,
                    help="emit one stats row per iteration instead of one "
                         "state")
    ap.add_argument("--verify", default=None,
                    help="kb_epoch_N.json to check the replay against")
    ap.add_argument("--all-epochs", default=None,
                    help="run directory: verify every kb_epoch_*.json in it")
    ap.add_argument("--embed", action="store_true",
                    help="rebuild real embeddings (chroma default EF); "
                         "default is a cheap offline stub")
    args = ap.parse_args(argv)
    ef = None if args.embed else StubEmbedding()

    if args.all_epochs:
        run = Path(args.all_epochs)
        trace = args.trace or run / "trace.jsonl"
        snaps = sorted(run.glob("kb_epoch_*.json"),
                       key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
        if not snaps:
            raise SystemExit(f"no kb_epoch_*.json in {run}")
        universe = _load_universe(args.universe, snaps[0])
        bad = 0
        for snap in snaps:
            diff = verify_epoch(universe, trace, snap)
            if diff:
                bad += 1
                print(f"{snap.name}: MISMATCH ({len(diff)} nodes)")
                for line in diff[:20]:
                    print(f"  {line}")
            else:
                print(f"{snap.name}: ok")
        if bad:
            raise SystemExit(f"{bad} snapshot(s) mismatched")
        return

    if args.verify:
        trace = args.trace or Path(args.verify).parent / "trace.jsonl"
        universe = _load_universe(args.universe, args.verify)
        diff = verify_epoch(universe, trace, args.verify)
        if diff:
            print(f"{Path(args.verify).name}: MISMATCH ({len(diff)} nodes)")
            for line in diff[:20]:
                print(f"  {line}")
            raise SystemExit(1)
        print(f"{Path(args.verify).name}: ok")
        return

    if not (args.trace and args.universe and args.out):
        raise SystemExit("state/series replay needs --universe, --trace "
                         "and --out")
    universe = Universe.load(args.universe)

    if args.series == "stats":
        base = Store.from_nodes(universe.nodes, StubEmbedding())
        rows = [{"iteration": 0, "epoch": 0, "qid": None, **base.stats()}]

        def emit(n, key, store):
            rows.append({"iteration": n, "epoch": key[0], "qid": key[1],
                         **store.stats()})

        _, n = replay(universe, args.trace, at=args.at,
                      embedding_function=StubEmbedding(), on_iteration=emit)
        with open(args.out, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"iterations": n, "rows": len(rows),
                          "out": args.out}))
        return

    store, n = replay(universe, args.trace, at=args.at,
                      embedding_function=ef)
    save_snapshot(store, args.out, args.universe)
    print(json.dumps({"iterations_applied": n, "out": args.out,
                      **store.stats()}))


if __name__ == "__main__":
    main()
