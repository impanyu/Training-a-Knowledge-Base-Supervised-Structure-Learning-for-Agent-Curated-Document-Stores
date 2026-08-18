"""T39.8: trace-replay reconstruction. A scripted mini-run (with an edit mix
that includes ERROR results from iteration 2 on) is replayed and checked
against the live store, the per-iteration kb_iter snapshots, and the
per-epoch snapshots."""
import json
import re

import pytest

from kb.loops import run_training
from kb.policy import Decision
from kb.recorder import RunLog
from kb.replay import main as replay_main
from kb.replay import replay, verify_epoch
from kb.store import Store

from .fixtures import HashEmbedding, mini_store, mini_universe


class EditPolicy:
    """Backprop per iteration: add a note, edit s0001, link s0001 -> s0002
    (succeeds once, ERROR every later iteration), delete s0003 (same), done.
    Forward: search once, answer "unknown"."""

    def __init__(self):
        self._key, self._n = None, 0

    def decide(self, system, context, tools) -> Decision:
        names = {t["name"] for t in tools}
        m = re.search(r"q\d{4}", context)
        qid = m.group(0) if m else "?"
        phase = "p2" if "[phase 2" in context else "p1"
        if (qid, phase) != self._key:
            self._key, self._n = (qid, phase), 0
        self._n += 1
        if "answer" in names:
            if self._n == 1:
                return Decision("search", {"query": "x"}, 3, 1)
            return Decision("answer", {"text": "unknown"}, 3, 1)
        step = {1: ("add", {"text": f"Replay note for {qid}."}),
                2: ("edit", {"id": "s0001", "text": f"Edited during {qid}."}),
                3: ("link", {"a": "s0001", "b": "s0002"}),
                4: ("delete", {"id": "s0003"})}.get(self._n, ("done", {}))
        return Decision(step[0], dict(step[1]), 3, 1)


@pytest.fixture()
def run(tmp_path):
    u = mini_universe()
    upath = tmp_path / "universe.json"
    u.save(upath)
    store = mini_store(u)
    run_dir = tmp_path / "run"
    log = RunLog(run_dir)
    run_training(store, EditPolicy(), u, log, run_dir, epochs=2, seed=0,
                 train_size=3, universe_path=upath, snapshot_every=1)
    log.close()
    return u, upath, store, run_dir


def test_full_replay_matches_the_live_store(run):
    u, upath, live, run_dir = run
    store, n = replay(u, run_dir / "trace.jsonl",
                      embedding_function=HashEmbedding())
    assert n == 6                                  # 2 epochs x 3 iterations
    assert store.to_json() == live.to_json()       # ids, flags, links, counter
    assert store.stats() == live.stats()


def test_replay_at_matches_the_per_iteration_snapshots(run):
    u, upath, live, run_dir = run
    for k in (1, 3, 6):
        with open(run_dir / f"kb_iter_{k:04d}.json") as f:
            snap = json.load(f)
        store, n = replay(u, run_dir / "trace.jsonl", at=k,
                          embedding_function=HashEmbedding())
        assert n == k
        assert store.to_json() == snap["store"]


def test_replay_at_zero_is_the_initial_store(run):
    u, upath, live, run_dir = run
    store, n = replay(u, run_dir / "trace.jsonl", at=0,
                      embedding_function=HashEmbedding())
    assert n == 0
    with open(run_dir / "kb_epoch_0.json") as f:
        assert store.to_json() == json.load(f)["store"]


def test_error_result_edits_are_skipped(run):
    u, upath, live, run_dir = run
    rows = [json.loads(l) for l in
            (run_dir / "trace.jsonl").read_text().splitlines()]
    errors = [r for r in rows if r["phase"] == 2
              and str(r["result"]).startswith("ERROR")]
    assert errors                                  # iter >= 2 link/delete fail
    store, _ = replay(u, run_dir / "trace.jsonl",
                      embedding_function=HashEmbedding())
    assert store.nodes["s0001"].links == ["s0002"]  # linked exactly once
    assert "s0003" not in store.nodes               # deleted exactly once


def test_add_ids_are_reproduced_from_the_recorded_results(run):
    u, upath, live, run_dir = run
    rows = [json.loads(l) for l in
            (run_dir / "trace.jsonl").read_text().splitlines()]
    added = [(re.fullmatch(r"added (s\d+)", r["result"]).group(1),
              r["input"]["text"])
             for r in rows if r["action"] == "add"
             and not str(r["result"]).startswith("ERROR")]
    assert len(added) == 6
    store, _ = replay(u, run_dir / "trace.jsonl",
                      embedding_function=HashEmbedding())
    for nid, text in added:
        assert store.nodes[nid].text == text
        assert store.nodes[nid].flag == "authored"
    assert store.to_json()["next_id"] == live.to_json()["next_id"]


def test_epoch_boundaries_verify_against_snapshots(run):
    u, upath, live, run_dir = run
    for epoch in (0, 1, 2):
        assert verify_epoch(u, run_dir / "trace.jsonl",
                            run_dir / f"kb_epoch_{epoch}.json") == []


def test_all_epochs_cli_and_mismatch_diff(run, capsys):
    u, upath, live, run_dir = run
    replay_main(["--all-epochs", str(run_dir), "--universe", str(upath)])
    out = capsys.readouterr().out
    for epoch in (0, 1, 2):
        assert f"kb_epoch_{epoch}.json: ok" in out
    # tamper a snapshot: verification must fail loudly with a node diff
    with open(run_dir / "kb_epoch_2.json") as f:
        snap = json.load(f)
    snap["store"]["nodes"][0]["text"] = "Tampered."
    with open(run_dir / "kb_epoch_2.json", "w") as f:
        json.dump(snap, f)
    with pytest.raises(SystemExit):
        replay_main(["--verify", str(run_dir / "kb_epoch_2.json"),
                     "--trace", str(run_dir / "trace.jsonl"),
                     "--universe", str(upath)])
    out = capsys.readouterr().out
    assert "MISMATCH" in out and "differs" in out


def test_series_stats_trajectory(run, tmp_path):
    u, upath, live, run_dir = run
    out = tmp_path / "series.jsonl"
    replay_main(["--universe", str(upath), "--trace",
                 str(run_dir / "trace.jsonl"), "--series", "stats",
                 "--out", str(out)])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 7                          # baseline + 6 iterations
    assert [r["iteration"] for r in rows] == list(range(7))
    base = rows[0]
    assert base["authored_statements"] == 0 and base["n_links"] == 0
    for k in range(1, 7):                          # one authored note per iter
        assert rows[k]["authored_statements"] == k
        assert rows[k]["edited_statements"] == 1   # s0001, edited in place
        assert rows[k]["n_links"] == 1             # link succeeded once
    # iter 1: +1 add -1 delete = net 0; every later iter adds one node
    assert rows[1]["n_nodes"] == base["n_nodes"]
    assert rows[6]["n_nodes"] == base["n_nodes"] + 5
    assert rows[6] == {"iteration": 6, "epoch": 2,
                       "qid": rows[6]["qid"], **live.stats()}


def test_state_cli_writes_a_loadable_snapshot(run, tmp_path):
    u, upath, live, run_dir = run
    out = tmp_path / "state.json"
    replay_main(["--universe", str(upath), "--trace",
                 str(run_dir / "trace.jsonl"), "--at", "2",
                 "--out", str(out)])
    with open(out) as f:
        state = json.load(f)
    assert state["universe"] == str(upath)         # load_kb-compatible wrapper
    with open(run_dir / "kb_iter_0002.json") as f:
        assert state["store"] == json.load(f)["store"]
    s = Store.from_json(state["store"], HashEmbedding())
    assert s.stats()["authored_statements"] == 2


def test_half_written_trace_tail_is_tolerated(run, tmp_path):
    u, upath, live, run_dir = run
    trace = tmp_path / "live_trace.jsonl"
    trace.write_text((run_dir / "trace.jsonl").read_text()
                     + '{"kind": "train", "epo')     # a run mid-write
    store, n = replay(u, trace, embedding_function=HashEmbedding())
    assert n == 6
    assert store.to_json() == live.to_json()


def test_snapshot_every_writes_iteration_snapshots(run):
    u, upath, live, run_dir = run
    files = sorted(p.name for p in run_dir.glob("kb_iter_*.json"))
    assert files == [f"kb_iter_{k:04d}.json" for k in range(1, 7)]
    with open(run_dir / "kb_iter_0006.json") as f:
        assert json.load(f)["store"] == live.to_json()
