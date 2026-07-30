"""T27: Recorder solution-reuse tallies -- per-agent `solutions` summary block
(answers/decompositions stored, from infra.solutions.stats) plus n_recalls /
n_recall_hits counted live off the event stream in `log()`.

T28: per-round cumulative snapshots (`log_round`) -> timeseries.jsonl."""
import json

from fixtures import demo_infra

from ca.recorder import Recorder


def _event(agent, action, result, round_no=1):
    return {"round": round_no, "agent": agent, "action": action, "input": {},
            "result": result, "category": "solving", "tokens_in": 1, "tokens_out": 1,
            "balance_after": 0}


def test_solutions_block_reflects_stats_and_recall_tallies(tmp_path):
    infra = demo_infra("C0")
    infra.solutions.record_decomposition("agent_1", "t0001", ["t0002", "q0003"])
    infra.solutions.record_answer("agent_1", "q0003", "4", f1=1.0)
    rec = Recorder(str(tmp_path))
    rec.log(_event("agent_1", "recall_solutions",
                   'known 1/1 answers under t0001: {"q0003": "4" (F1 1.00)}'))
    rec.log(_event("agent_1", "recall_solutions", "(no stored solutions under t0002)"))
    rec.log(_event("agent_2", "recall_solutions", "(no stored solutions under t0001)"))
    summary = rec.write_summary(infra, rounds_used=1)
    rec.close()

    assert summary["solutions"]["agent_1"] == {
        "answers": 1, "decompositions": 1, "n_recalls": 2, "n_recall_hits": 1,
    }
    assert summary["solutions"]["agent_2"] == {
        "answers": 0, "decompositions": 0, "n_recalls": 1, "n_recall_hits": 0,
    }
    # every agent gets an entry, even one that never called recall_solutions
    assert summary["solutions"]["agent_3"] == {
        "answers": 0, "decompositions": 0, "n_recalls": 0, "n_recall_hits": 0,
    }


def test_non_recall_events_do_not_affect_the_tally(tmp_path):
    infra = demo_infra("C0")
    rec = Recorder(str(tmp_path))
    rec.log(_event("agent_1", "check_balance", "balance: 100"))
    rec.log(_event("agent_1", "decompose", "[t0002] ..."))
    summary = rec.write_summary(infra, rounds_used=1)
    rec.close()
    assert summary["solutions"]["agent_1"]["n_recalls"] == 0
    assert summary["solutions"]["agent_1"]["n_recall_hits"] == 0


def test_shared_c2_bucket_is_reflected_per_agent_via_stats(tmp_path):
    infra = demo_infra("C2")
    infra.solutions.record_answer("agent_1", "q0001", "Paris", f1=1.0)
    rec = Recorder(str(tmp_path))
    summary = rec.write_summary(infra, rounds_used=0)
    rec.close()
    # the shared bucket is the same store no matter which agent's id you ask
    # through, so both agents see the one answer that lives in it
    assert summary["solutions"]["agent_1"]["answers"] == 1
    assert summary["solutions"]["agent_2"]["answers"] == 1


def test_error_recalls_are_not_hits(tmp_path):
    """ERROR results from recall_solutions should NOT be counted as hits,
    even though they are not "(no stored solutions" responses."""
    infra = demo_infra("C0")
    rec = Recorder(str(tmp_path))
    # Log an ERROR result (e.g. bankrupt agent, unresolvable name)
    rec.log(_event("agent_1", "recall_solutions",
                   'ERROR: bankrupt: agent_2 has no active balance'))
    # Log an empty-store result (should not be a hit)
    rec.log(_event("agent_1", "recall_solutions",
                   "(no stored solutions under t0001)"))
    # Log a valid result (should be a hit)
    rec.log(_event("agent_1", "recall_solutions",
                   'known 2/3 answers: {"q0001": "42", "q0002": "yes"}'))
    summary = rec.write_summary(infra, rounds_used=1)
    rec.close()

    # All three should increment n_recalls
    assert summary["solutions"]["agent_1"]["n_recalls"] == 3
    # Only the valid result should increment n_recall_hits
    # (ERROR and "(no stored solutions" should not)
    assert summary["solutions"]["agent_1"]["n_recall_hits"] == 1


def test_log_round_snapshot_shape_and_content(tmp_path):
    infra = demo_infra("C0")
    rec = Recorder(str(tmp_path))
    rec.log(_event("agent_1", "retrieve", "docs ..."))            # 2 solving tokens
    admin_ev = _event("agent_2", "list_tasks", "t0001, t0004")
    admin_ev["category"] = "admin"
    rec.log(admin_ev)                                             # 2 admin tokens
    snap = rec.log_round(infra, 1)
    rec.close()

    lines = [json.loads(l) for l in open(tmp_path / "timeseries.jsonl")]
    assert lines == [snap]
    roster = set(infra.agent_ids)
    for key in ("balances", "tokens", "coordination_overhead_by_agent",
                "answered", "tasks_closed", "solutions"):
        assert set(snap[key]) == roster, key
    assert snap["round"] == 1
    assert snap["bankrupt"] == []
    assert snap["total_balance"] == 1000 and snap["escrow_total"] == 0
    assert snap["minted"] == 0 and snap["burned"] == 0
    assert snap["tokens"]["agent_1"] == {"solving": 2, "admin": 0}
    assert snap["tokens"]["agent_2"] == {"solving": 0, "admin": 2}
    assert snap["solving_total"] == 2 and snap["admin_total"] == 2
    assert snap["coordination_overhead"] == 0.5
    assert snap["coordination_overhead_by_agent"]["agent_2"] == 1.0
    assert snap["coordination_overhead_by_agent"]["agent_3"] == 0.0  # zero-guarded
    assert snap["board"] == {"open": 2, "claimed": 0, "closed": 0}
    assert snap["n_tasks_closed"] == 0 and snap["task_completion_rate"] == 0.0
    assert snap["n_answered"] == 0 and snap["total_f1"] == 0.0 and snap["total_em"] == 0.0
    assert snap["n_contracts"] == 0 and snap["contracts_by_status"] == {}
    assert snap["n_loans"] == 0 and snap["loan_principal_outstanding"] == 0
    assert snap["interest_paid_total"] == 0
    assert snap["n_recalls"] == 0 and snap["n_recall_hits"] == 0
    assert snap["answers_in_memory_total"] == 0


def test_log_round_attributes_answers_and_tasks_to_deliverer(tmp_path):
    infra = demo_infra("C0")
    infra.board.claim("agent_3", "t0004", round_no=1)
    infra.board.deliver("agent_3", "t0004", {"q0003": "4", "q0004": "6"})
    # the actions layer auto-records delivered answers; done by hand here
    infra.solutions.record_answer("agent_3", "q0003", "4", f1=1.0)
    infra.solutions.record_answer("agent_3", "q0004", "6", f1=1.0)
    rec = Recorder(str(tmp_path))
    rec.log(_event("agent_3", "recall_solutions", "(no stored solutions under t0004)"))
    snap1 = rec.log_round(infra, 1)
    snap2 = rec.log_round(infra, 2)
    rec.close()

    assert snap1["answered"]["agent_3"] == {"n_answered": 2, "f1_sum": 2.0, "em_sum": 2.0}
    assert snap1["answered"]["agent_1"] == {"n_answered": 0, "f1_sum": 0.0, "em_sum": 0.0}
    assert snap1["n_answered"] == 2 and snap1["total_f1"] == 2.0
    assert snap1["tasks_closed"]["agent_3"] == 1 and snap1["n_tasks_closed"] == 1
    assert snap1["task_completion_rate"] == 0.5    # 1 of 2 posted tasks closed
    assert snap1["board"] == {"open": 1, "claimed": 0, "closed": 1}
    assert snap1["minted"] == 700 and snap1["total_balance"] == 1700
    # delivering auto-records answers into agent_3's solution memory
    assert snap1["solutions"]["agent_3"] == {
        "n_recalls": 1, "n_recall_hits": 0, "answers_in_memory": 2,
        "decompositions_in_memory": 0,
    }
    assert snap1["answers_in_memory_total"] == 2
    # one line per log_round call, cumulative so round 2 repeats the state
    lines = [json.loads(l) for l in open(tmp_path / "timeseries.jsonl")]
    assert [s["round"] for s in lines] == [1, 2]
    assert lines[1] == {**snap1, "round": 2} == snap2
