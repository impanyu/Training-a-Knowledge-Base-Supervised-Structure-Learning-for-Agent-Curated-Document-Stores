from kb.loops import READER_SYSTEM, TRAIN_SYSTEM, run_test, run_training, train_iteration
from kb.loops import test_iteration as exam_iteration  # alias: not a pytest test
from kb.policy import ScriptedPolicy

from .fixtures import DrivePolicy, ListLog, mini_store, mini_universe

U = mini_universe()


def _q(split="train", i=0):
    return U.questions[U.splits[split][i]]


def _answerable(split="train"):
    return next(U.questions[q] for q in U.splits[split]
                if not U.questions[q].unanswerable)


# ---------------- train phase 1 ----------------

def test_answer_result_reveals_gold_and_f1():
    store, log = mini_store(U), ListLog()
    q = _answerable()
    pol = ScriptedPolicy([("search", {"query": q.text}),
                          ("answer", {"text": q.golds[0]})])
    row = train_iteration(store, pol, q, log, epoch=1)
    assert row["f1"] == 1.0 and row["em"] == 1.0
    assert row["answered_at"] == 2
    ans = log.rows["trace"][1]
    assert ans["phase"] == 1 and ans["action"] == "answer"
    assert f'gold answer: "{q.golds[0]}"' in ans["result"]
    assert "your F1: 1.00" in ans["result"]


def test_answer_ends_phase1_immediately_leftover_forfeited():
    store, log = mini_store(U), ListLog()
    q = _answerable()
    pol = ScriptedPolicy([("answer", {"text": "wrong guess"}),
                          ("done", {})])
    row = train_iteration(store, pol, q, log, epoch=1, n1=15)
    assert row["p1_steps"] == 1                     # 14 leftover steps forfeited
    assert len([r for r in log.rows["trace"] if r["phase"] == 1]) == 1


def test_phase1_exhaustion_scores_zero_but_still_reveals_gold():
    store, log = mini_store(U), ListLog()
    q = _answerable()
    pol = ScriptedPolicy([("search", {"query": "x"})] * 99)
    row = train_iteration(store, pol, q, log, epoch=1, n1=4, n2=2)
    assert row["answered_at"] is None and row["f1"] == 0.0
    assert row["p1_steps"] == 4
    p2_ctx = next(ctx for _, ctx, _ in pol.seen if "[phase 2" in ctx)
    assert f'gold answer: "{q.golds[0]}"' in p2_ctx  # revealed at phase 2 start
    assert "budget exhausted" in p2_ctx


def test_phase1_only_sees_forward_tools_and_rejects_offmode():
    store, log = mini_store(U), ListLog()
    q = _answerable()
    pol = ScriptedPolicy([("add", {"text": "Sneaky note."}),
                          ("answer", {"text": "x"})])
    train_iteration(store, pol, q, log, epoch=1)
    assert pol.seen[0][2] == ("search", "read", "answer")
    first = log.rows["trace"][0]
    assert first["result"] == "ERROR: add is not available in this phase"
    assert store.stats()["authored_statements"] == 0


# ---------------- train phase 2 ----------------

def test_phase2_edits_dispatch_and_done_ends():
    store, log = mini_store(U), ListLog()
    q = _answerable()
    a, b = U.nodes[0]["id"], U.nodes[1]["id"]
    pol = ScriptedPolicy([("answer", {"text": q.golds[0]}),
                          ("add", {"text": "A navigation note."}),
                          ("link", {"a": a, "b": b}),
                          ("link", {"a": a, "b": "s9999"}),   # ERROR result
                          ("done", {})])
    row = train_iteration(store, pol, q, log, epoch=1)
    assert row["p2_steps"] == 4
    assert row["edits"]["add"] == 1
    assert row["edits"]["link"] == 1            # ERROR results are not edits
    p2 = [r for r in log.rows["trace"] if r["phase"] == 2]
    assert p2[0]["result"].startswith("added ")
    assert p2[1]["result"] == f"linked {a} -> {b}"
    assert p2[2]["result"].startswith("ERROR:")
    assert p2[3]["action"] == "done" and p2[3]["result"] == "phase complete"
    assert pol.seen[1][2] == ("search", "read", "add",
                              "edit", "delete", "link", "link_many",
                              "unlink", "done")


def test_phase2_budget_exhaustion_ends_without_done():
    store, log = mini_store(U), ListLog()
    q = _answerable()
    pol = ScriptedPolicy([("answer", {"text": "x"})]
                         + [("search", {"query": "y"})] * 99)
    row = train_iteration(store, pol, q, log, epoch=1, n2=3)
    assert row["p2_steps"] == 3
    assert len([r for r in log.rows["trace"] if r["phase"] == 2]) == 3


def test_phase2_authoring_edits_count_and_reembed():
    store, log = mini_store(U), ListLog()
    q = _answerable()
    nid = U.nodes[0]["id"]
    pol = ScriptedPolicy([("answer", {"text": q.golds[0]}),
                          ("add", {"text": "A new authored fact."}),
                          ("edit", {"id": nid, "text": "A rewritten fact."}),
                          ("edit", {"id": "s9999", "text": "x"}),   # ERROR
                          ("done", {})])
    row = train_iteration(store, pol, q, log, epoch=1)
    assert row["edits"]["add"] == 1
    assert row["edits"]["edit"] == 1            # the ERROR one doesn't count
    assert row["regens"] == 2                   # both dirty nodes; batch at end
    assert store.dirty == set()
    stats = store.stats()
    assert stats["authored_statements"] == 1
    assert stats["edited_statements"] == 1


def test_dedup_merges_are_counted_per_iteration_and_in_stats():
    store, log = mini_store(U, judge=lambda a, b: a == b), ListLog()
    q = _answerable()
    dup = U.nodes[0]
    pol = ScriptedPolicy([("answer", {"text": q.golds[0]}),
                          ("add", {"text": dup["text"]}),      # exact dup
                          ("add", {"text": "A very novel quokka note."}),
                          ("done", {})])
    row = train_iteration(store, pol, q, log, epoch=1)
    assert row["merges"] == 1
    assert row["edits"]["add"] == 2             # both adds were successful
    p2 = [r for r in log.rows["trace"] if r["phase"] == 2]
    assert p2[0]["result"] == (f"duplicate of {dup['id']} - merged, "
                               "no new note created")
    assert p2[1]["result"].startswith("added ")
    stats = store.stats()
    assert stats["merges"] == 1                 # cumulative, into kb_stats
    assert stats["authored_statements"] == 1    # only the novel note exists


def test_answer_is_not_available_in_phase2():
    store, log = mini_store(U), ListLog()
    q = _answerable()
    pol = ScriptedPolicy([("answer", {"text": "first"}),
                          ("answer", {"text": "second"}),
                          ("done", {})])
    row = train_iteration(store, pol, q, log, epoch=1)
    p2 = [r for r in log.rows["trace"] if r["phase"] == 2]
    assert p2[0]["result"] == "ERROR: answer is not available in this phase"
    assert row["answer"] == "first"             # the single graded attempt


def test_memory_resets_per_iteration_and_survives_phase_transition():
    store, log = mini_store(U), ListLog()
    q = _answerable()
    pol = ScriptedPolicy([("search", {"query": "marker-alpha"}),
                          ("answer", {"text": "x"}), ("done", {}),
                          ("answer", {"text": "y"}), ("done", {})])
    train_iteration(store, pol, q, log, epoch=1)
    train_iteration(store, pol, q, log, epoch=1)
    p2_ctx = next(ctx for _, ctx, _ in pol.seen if "[phase 2" in ctx)
    assert "marker-alpha" in p2_ctx             # FIFO retained across phases
    second_start = pol.seen[3][1]
    assert "marker-alpha" not in second_start   # but reset per iteration
    assert "(no actions yet)" in second_start


# ---------------- prompts ----------------

def test_two_skills_carry_goal_means_and_example():
    """Retrieval and curation are separate skills; Phase 1 and the exam share
    the retrieval one, so the store is the only difference between them."""
    from kb.loops import RETRIEVAL_SKILL, CURATION_SKILL
    for skill in (RETRIEVAL_SKILL, CURATION_SKILL):
        assert "GOAL." in skill and "MEANS." in skill and "EXAMPLE" in skill
    for cue in ("search(query, k)", "read(id)", "answer(text)",
                "the full text of every note linked to it",
                "A second case."):
        assert cue in RETRIEVAL_SKILL
    # curation: build an index for every key the question named
    for cue in ("GOAL. Build the INDEX NOTES this question needed",
                "each named thing is a KEY",
                "You do not have to abstract the question",
                "Examples of the mapping",
                "AN INDEX POINTS; IT DOES NOT LIST",
                "COMPLETENESS."):
        assert cue in CURATION_SKILL
    for cue in ("search(query, k) returns the k most similar notes",
                "link_many costs ONE action",
                "ONE note holds ONE statement"):
        assert cue in CURATION_SKILL
    assert "doc" not in CURATION_SKILL.lower()




# ---------------- test iterations ----------------

def test_answer_result_is_empty_in_test_and_gold_never_appears():
    store, log = mini_store(U), ListLog()
    q = _answerable("test_in")
    pol = ScriptedPolicy([("search", {"query": q.text}),
                          ("answer", {"text": q.golds[0]})])
    row = exam_iteration(store, pol, q, log, "test_in", epoch=0)
    assert row["status"] == "answered" and row["f1"] == 1.0
    assert log.rows["trace"][-1]["result"] == "submitted"
    for _, ctx, _ in pol.seen:
        assert "gold" not in ctx
    assert all(system == pol.seen[0][0] for system, _, _ in pol.seen)


def test_wrong_answer_is_distinct_from_unanswered():
    store, log = mini_store(U), ListLog()
    q = _answerable("test_in")
    wrong = exam_iteration(store, ScriptedPolicy([("answer", {"text": "zzz"})]),
                           q, log, "test_in", epoch=0)
    assert wrong["status"] == "wrong" and wrong["steps"] == 1
    none = exam_iteration(store, ScriptedPolicy([("search", {"query": "x"})] * 99),
                          q, log, "test_in", epoch=0, m=5)
    assert none["status"] == "unanswered" and none["steps"] == 5
    assert none["f1"] == 0.0 and none["answer"] is None


def test_test_mode_has_no_edit_tools_kb_stays_frozen():
    store, log = mini_store(U), ListLog()
    q = _answerable("test_out")
    pol = ScriptedPolicy([("add", {"text": "Sneaky note."}),
                          ("answer", {"text": "x"})])
    exam_iteration(store, pol, q, log, "test_out", epoch=0)
    assert pol.seen[0][2] == ("search", "read", "answer")
    assert store.stats()["authored_statements"] == 0
    assert log.rows["trace"][0]["result"].startswith("ERROR")


def test_run_test_respects_limit_and_split():
    store, log = mini_store(U), ListLog()
    pol = DrivePolicy()
    rows = run_test(store, pol, U, "test_out", log, epoch=0, limit=2)
    assert len(rows) == 2
    assert all(r["split"] == "test_out" for r in rows)


# ---------------- epoch driver ----------------

def test_eval_each_epoch_touches_only_the_eval_split(tmp_path):
    store, log = mini_store(U), ListLog()
    run_training(store, DrivePolicy(), U, log, tmp_path, epochs=1, seed=0,
                 train_size=2, eval_each_epoch=True)
    rows = log.rows["test"]
    assert rows
    assert {r["split"] for r in rows} == {"eval"}          # never test_in/out
    assert {r["epoch"] for r in rows} == {0, 1}            # baseline + epoch 1
    assert {r["flavor"] for r in rows} == {"in", "out"}    # both flavors
    per_epoch = [r for r in rows if r["epoch"] == 1]
    assert len(per_epoch) == len(U.splits["eval"])         # whole split, no limit


def test_cost_fields_on_rows_and_kb_stats(tmp_path):
    store, log = mini_store(U), ListLog()
    pol = DrivePolicy(in_tokens=7, out_tokens=3)
    run_training(store, pol, U, log, tmp_path, epochs=1, seed=0,
                 train_size=2, eval_each_epoch=True)
    for r in log.rows["train"]:
        # DrivePolicy: 2 forward turns + 2 backward turns = 4 decisions
        assert r["tokens_in"] == 28 and r["tokens_out"] == 12
        assert r["seconds"] > 0
    for r in log.rows["test"]:
        assert r["tokens_in"] == 14 and r["tokens_out"] == 6   # 2 turns
        assert r["seconds"] > 0
    e0, e1 = log.rows["stats"]
    assert e0["train_iterations"] == 0 and e0["train_seconds"] == 0
    assert e1["train_iterations"] == 2
    assert e1["train_tokens_in"] == 56 and e1["train_tokens_out"] == 24
    assert e1["train_seconds"] > 0
    assert e1["statement_tokens"] > 0          # approximate KB size in tokens
    assert e1["n_nodes"] == e0["n_nodes"] + 2  # one authored note per iteration
    assert e1["authored_statements"] == 2


def test_epoch_shuffle_is_seeded_and_snapshots_written(tmp_path):
    store, log = mini_store(U), ListLog()
    run_training(store, DrivePolicy(), U, log, tmp_path, epochs=2, seed=0,
                 train_size=3)
    assert (tmp_path / "kb_epoch_0.json").exists()
    assert (tmp_path / "kb_epoch_2.json").exists()
    assert [r["epoch"] for r in log.rows["stats"]] == [0, 1, 2]
    e1 = [r["qid"] for r in log.rows["train"] if r["epoch"] == 1]
    e2 = [r["qid"] for r in log.rows["train"] if r["epoch"] == 2]
    assert sorted(e1) == sorted(e2) == sorted(U.splits["train"][:3])
    store2, log2 = mini_store(U), ListLog()
    run_training(store2, DrivePolicy(), U, log2, tmp_path / "b", epochs=2,
                 seed=0, train_size=3)
    assert [r["qid"] for r in log2.rows["train"]] == e1 + e2   # same seed
