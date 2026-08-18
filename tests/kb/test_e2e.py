"""Scripted end-to-end: mini universe -> one train epoch + test passes ->
report over the run artifacts. Everything offline (DrivePolicy, counting
summarizer, hash embeddings)."""
import json

import kb.report as report_mod
from kb.loops import run_training
from kb.recorder import RunLog
from kb.report import build_report
from kb.store import Store
from kb.test import load_kb

from .fixtures import DrivePolicy, HashEmbedding, mini_store, mini_universe


def _run(tmp_path, epochs=1):
    u = mini_universe()
    upath = tmp_path / "universe.json"
    u.save(upath)
    store = mini_store(u)
    run_dir = tmp_path / "run"
    log = RunLog(run_dir)
    answers = {qid: u.questions[qid].golds[0]
               for split in ("train", "test_in", "test_out")
               for qid in u.splits[split][:4]}
    run_training(store, DrivePolicy(answers), u, log, run_dir, epochs=epochs,
                 seed=0, eval_each_epoch=True, universe_path=upath)
    with open(run_dir / "meta.json", "w") as f:
        json.dump({"universe": str(upath),
                   "build_tokens": u.meta["build_tokens"],
                   "summarizer_tokens": {"in": store.summarizer.tokens_in,
                                         "out": store.summarizer.tokens_out}}, f)
    log.close()
    return u, store, run_dir


def test_full_flow_files_and_report(tmp_path):
    u, store, run_dir = _run(tmp_path)
    for name in ("trace", "train_log", "test_log", "kb_stats"):
        assert (run_dir / f"{name}.jsonl").exists()
    with open(run_dir / "train_log.jsonl") as f:
        train_rows = [json.loads(l) for l in f]
    assert len(train_rows) == len(u.splits["train"])
    assert all(r["edits"]["create_doc"] == 1 for r in train_rows)

    rep = build_report(run_dir)
    curve = rep["learning_curve"]
    assert set(curve) == {0, 1}
    assert set(curve[0]) == {"test_in", "test_out"}
    assert curve[0]["test_in"]["n"] == len(u.splits["test_in"])
    assert curve[1]["test_in"]["f1"] > 0          # some gold answers scripted
    assert rep["train_forward"]["by_epoch"][1]["n"] == len(u.splits["train"])
    assert rep["edit_mix"][1]["create_doc"] == len(u.splits["train"])
    assert [r["epoch"] for r in rep["kb_stats"]] == [0, 1]
    assert rep["kb_stats"][1]["created_docs"] == len(u.splits["train"])
    # E2: token tallies separated, all nonzero, build from the universe meta
    assert rep["tokens"]["train"]["in"] > 0
    assert rep["tokens"]["test"]["in"] > 0
    assert rep["tokens"]["build"] == {"in": 0, "out": 0}
    assert rep["link_alignment"] is not None      # universe resolvable


def test_snapshots_are_loadable_and_frozen_kb_examinable(tmp_path):
    u, store, run_dir = _run(tmp_path)
    with open(run_dir / "kb_epoch_1.json") as f:
        snap = json.load(f)
    s2 = Store.from_json(snap["store"], embedding_function=HashEmbedding())
    assert s2.stats() == store.stats()
    # kb.test's loader resolves the universe path recorded in the snapshot
    s3, u3 = load_kb(run_dir / "kb_epoch_1.json",
                     embedding_function=HashEmbedding())
    assert u3.splits == u.splits
    assert s3.search(u.questions[u.splits["test_in"][0]].text, 1)
    # a universe file loads as the untrained store (epoch-0 RAG baseline)
    s4, _ = load_kb(tmp_path / "universe.json",
                    embedding_function=HashEmbedding())
    assert s4.stats()["created_docs"] == 0


def test_report_cli_writes_report_json(tmp_path, capsys):
    _, _, run_dir = _run(tmp_path)
    report_mod.main([str(run_dir)])
    out = capsys.readouterr().out
    assert "E1 learning curve" in out
    with open(run_dir / "report.json") as f:
        rep = json.load(f)
    assert "learning_curve" in rep and "edit_mix" in rep
