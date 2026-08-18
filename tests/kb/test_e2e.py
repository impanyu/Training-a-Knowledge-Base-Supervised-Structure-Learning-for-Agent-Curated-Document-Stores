"""Scripted end-to-end: mini universe -> one train epoch (eval split after
every snapshot) -> the single final test pass -> report over the run
artifacts. Everything offline (DrivePolicy, hash embeddings)."""
import json

import kb.report as report_mod
import kb.test as kbtest_mod
from kb.loops import run_test, run_training
from kb.policy import ScriptedPolicy
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
               for split in ("train", "test_in", "test_out", "eval")
               for qid in u.splits[split][:4]}
    pol = DrivePolicy(answers)
    run_training(store, pol, u, log, run_dir, epochs=epochs,
                 seed=0, eval_each_epoch=True, universe_path=upath)
    # T39.1 protocol: the FULL test runs exactly once, after training
    for split in ("test_in", "test_out"):
        run_test(store, pol, u, split, log, epoch=epochs)
    with open(run_dir / "meta.json", "w") as f:
        json.dump({"universe": str(upath)}, f)
    log.close()
    return u, store, run_dir


def test_full_flow_files_and_report(tmp_path):
    u, store, run_dir = _run(tmp_path)
    for name in ("trace", "train_log", "test_log", "kb_stats"):
        assert (run_dir / f"{name}.jsonl").exists()
    with open(run_dir / "train_log.jsonl") as f:
        train_rows = [json.loads(l) for l in f]
    assert len(train_rows) == len(u.splits["train"])
    assert all(r["edits"]["add"] == 1 for r in train_rows)

    rep = build_report(run_dir)
    curve = rep["learning_curve"]
    assert set(curve) == {0, 1}
    assert set(curve[0]) == {"eval_in", "eval_out"}        # eval split only
    assert curve[1]["eval_in"]["f1"] > 0                   # gold answers scripted
    final = rep["final_test"]
    assert set(final) == {"test_in", "test_out"}           # single final point
    assert list(final["test_in"]) == [1]
    assert final["test_in"][1]["n"] == len(u.splits["test_in"])
    assert rep["train_forward"]["by_epoch"][1]["n"] == len(u.splits["train"])
    assert rep["edit_mix"][1]["add"] == len(u.splits["train"])
    assert [r["epoch"] for r in rep["kb_stats"]] == [0, 1]
    assert rep["kb_stats"][1]["authored_statements"] == len(u.splits["train"])
    assert rep["kb_stats"][1]["n_nodes"] == \
        rep["kb_stats"][0]["n_nodes"] + len(u.splits["train"])
    # E2: train/test token tallies separated, nonzero (no build column: the
    # node-graph build makes no API calls)
    assert rep["tokens"]["train"]["in"] > 0
    assert rep["tokens"]["test"]["in"] > 0
    assert set(rep["tokens"]) == {"train", "test"}
    assert rep["link_alignment"] is not None               # universe resolvable


def test_report_headline_costs_and_kb_size(tmp_path):
    u, store, run_dir = _run(tmp_path)
    rep = build_report(run_dir)
    tc = rep["costs"]["train"]
    assert tc["iterations"] == len(u.splits["train"])
    assert tc["tokens_in"] > 0 and tc["seconds"] > 0
    assert tc["mean_tokens_per_iteration"] > 0
    assert tc["mean_seconds_per_iteration"] > 0
    for split in ("eval", "test_in", "test_out"):
        c = rep["costs"][split]
        assert c["questions"] > 0
        assert c["mean_tokens"] > 0 and c["mean_seconds"] > 0
        assert c["mean_steps"] == 2.0                      # DrivePolicy: search+answer
    ks = rep["kb_size"]
    assert set(ks) == {"n_nodes", "n_links", "statement_tokens"}
    assert ks["statement_tokens"] > 0                      # ~chars // 4


def test_appended_baseline_pass_lands_next_to_the_curve(tmp_path):
    u, store, run_dir = _run(tmp_path)
    # simulate the manual epoch-0 baseline: kb.test appends into the run dir
    log = RunLog(run_dir, append=True)
    base_store, base_u = load_kb(run_dir / "kb_epoch_0.json",
                                 embedding_function=HashEmbedding())
    run_test(base_store, DrivePolicy(), base_u, "test_in", log, epoch=0)
    log.close()
    rep = build_report(run_dir)
    assert list(rep["final_test"]["test_in"]) == [0, 1]    # baseline + final
    assert set(rep["learning_curve"]) == {0, 1}            # curve untouched


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
    assert s4.stats()["authored_statements"] == 0
    assert s4.stats()["n_links"] == 0


def test_kb_test_cli_records_budget_m(tmp_path, monkeypatch):
    u = mini_universe()
    store = mini_store(u)
    monkeypatch.setattr(kbtest_mod, "load_kb",
                        lambda kb, universe=None: (store, u))
    monkeypatch.setattr(kbtest_mod, "make_policy",
                        lambda model: ScriptedPolicy([]))   # never answers
    out = tmp_path / "t"
    kbtest_mod.main(["--kb", "snap.json", "--split", "test_out",
                     "--m", "4", "--out", str(out)])
    meta = json.loads((out / "test_meta.json").read_text())
    assert meta["m"] == 4 and meta["split"] == "test_out"   # budget recorded
    rows = [json.loads(l)
            for l in (out / "test_log.jsonl").read_text().splitlines()]
    assert len(rows) == len(u.splits["test_out"])
    assert all(r["steps"] == 4 and r["status"] == "unanswered" for r in rows)


def test_kb_test_cli_takes_comma_lists_and_eval(tmp_path, monkeypatch):
    u = mini_universe()
    store = mini_store(u)
    monkeypatch.setattr(kbtest_mod, "load_kb",
                        lambda kb, universe=None: (store, u))
    monkeypatch.setattr(kbtest_mod, "make_policy",
                        lambda model: DrivePolicy())
    out = tmp_path / "t"
    kbtest_mod.main(["--kb", "snap.json", "--split", "test_out,eval",
                     "--out", str(out)])
    rows = [json.loads(l)
            for l in (out / "test_log.jsonl").read_text().splitlines()]
    assert {r["split"] for r in rows} == {"test_out", "eval"}
    ev = [r for r in rows if r["split"] == "eval"]
    assert len(ev) == len(u.splits["eval"])
    assert {r["flavor"] for r in ev} == {"in", "out"}      # flavors recorded
    meta = json.loads((out / "test_meta.json").read_text())
    assert meta["split"] == "test_out,eval"


def test_kb_test_cli_rejects_unknown_split(tmp_path, monkeypatch):
    u = mini_universe()
    store = mini_store(u)
    monkeypatch.setattr(kbtest_mod, "load_kb",
                        lambda kb, universe=None: (store, u))
    monkeypatch.setattr(kbtest_mod, "make_policy",
                        lambda model: DrivePolicy())
    import pytest
    with pytest.raises(SystemExit, match="unknown split"):
        kbtest_mod.main(["--kb", "snap.json", "--split", "test_nope",
                         "--out", str(tmp_path / "x")])


def test_report_cli_writes_report_json(tmp_path, capsys):
    _, _, run_dir = _run(tmp_path)
    report_mod.main([str(run_dir)])
    out = capsys.readouterr().out
    assert "E1 learning curve" in out
    assert "headline costs" in out
    with open(run_dir / "report.json") as f:
        rep = json.load(f)
    assert "learning_curve" in rep and "edit_mix" in rep
    assert "final_test" in rep and "costs" in rep and "kb_size" in rep
