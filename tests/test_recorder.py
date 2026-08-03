"""Recorder: the per-agent `memory` summary block (answers/notes stored, plus
claim/hit and repeat/improvement tallies counted live off the event stream in
`log()`), and the T28 per-round cumulative snapshots -> timeseries.jsonl."""
import json

from fixtures import demo_infra

from ca.actions import dispatch
from ca.recorder import Recorder


def _event(agent, action, result, round_no=1, category="solving"):
    return {"round": round_no, "agent": agent, "action": action, "input": {},
            "result": result, "category": category, "tokens_in": 1, "tokens_out": 1,
            "balance_after": 0}


CLAIM_HIT = ('claimed [q0001] capital of France? (2hop, reward 100, 1 left)\n'
             'memory: stored answer [q0001] capital of France? -> "Paris" '
             '(F1 1.00) - GOOD (F1 1.00 of 1.00): you can deliver it as-is')
CLAIM_MISS = "claimed [q0004] 3+3? (3hop, reward 400, 0 left)"


def test_memory_block_reflects_the_store_and_the_live_tallies(tmp_path):
    infra = demo_infra("C0")
    infra.memory.write("agent_1", "a note")
    infra.memory.write("agent_1", "an answer", kind="answer", qid="q0001", f1=1.0)
    rec = Recorder(str(tmp_path))
    rec.log(_event("agent_1", "claim_question", CLAIM_HIT, category="admin"))
    rec.log(_event("agent_1", "claim_question", CLAIM_MISS, category="admin"))
    rec.log(_event("agent_2", "claim_question", CLAIM_MISS, category="admin"))
    summary = rec.write_summary(infra, rounds_used=1)
    rec.close()

    assert summary["memory"]["agent_1"] == {
        "answers": 1, "notes": 1, "n_claims": 2, "n_memory_hits": 1,
        "n_repeat_deliveries": 0, "n_improved": 0,
    }
    assert summary["memory"]["agent_2"]["n_claims"] == 1
    # every agent gets an entry, even one that never claimed anything
    assert summary["memory"]["agent_3"] == {
        "answers": 0, "notes": 0, "n_claims": 0, "n_memory_hits": 0,
        "n_repeat_deliveries": 0, "n_improved": 0,
    }


def test_failed_claims_and_other_actions_do_not_move_the_tally(tmp_path):
    infra = demo_infra("C0")
    rec = Recorder(str(tmp_path))
    rec.log(_event("agent_1", "claim_question", "ERROR: q0002 has no units left",
                   category="admin"))
    # the marker only counts on claim_question events, wherever else it echoes
    rec.log(_event("agent_1", "retrieve", "docs say memory: stored answer ..."))
    rec.log(_event("agent_1", "check_balance", "balance: 100", category="admin"))
    summary = rec.write_summary(infra, rounds_used=1)
    rec.close()
    assert summary["memory"]["agent_1"]["n_claims"] == 0
    assert summary["memory"]["agent_1"]["n_memory_hits"] == 0


def test_repeat_delivery_tallies_split_improved_from_not_improved(tmp_path):
    infra = demo_infra("C0")
    rec = Recorder(str(tmp_path))
    rec.log(_event("agent_1", "deliver_work",
                   "delivered q0001: F1=1.00 -> 100 tokens "
                   "(IMPROVED on your stored F1 0.30)"))
    rec.log(_event("agent_1", "deliver_work",
                   "delivered q0003: F1=0.00 -> 0 tokens "
                   "(no better than your stored F1 1.00)"))
    rec.log(_event("agent_1", "deliver_work", "delivered q0004: F1=1.00 -> 400 tokens"))
    summary = rec.write_summary(infra, rounds_used=1)
    rec.close()
    tally = summary["memory"]["agent_1"]
    assert tally["n_repeat_deliveries"] == 2 and tally["n_improved"] == 1


def test_shared_c2_bucket_is_reflected_per_agent(tmp_path):
    infra = demo_infra("C2")
    infra.memory.write("agent_1", "an answer", kind="answer", qid="q0001", f1=1.0)
    rec = Recorder(str(tmp_path))
    summary = rec.write_summary(infra, rounds_used=0)
    rec.close()
    # the shared bucket is the same store no matter which agent's id you ask
    # through, so both agents see the one answer that lives in it
    assert summary["memory"]["agent_1"]["answers"] == 1
    assert summary["memory"]["agent_2"]["answers"] == 1


def test_summary_carries_delivery_rows_and_the_demand_denominator(tmp_path):
    infra = demo_infra("C0")
    dispatch(infra, "agent_3", "claim_question", {"qid": "q0004"})
    dispatch(infra, "agent_3", "deliver_work", {"target_id": "q0004", "content": "6"})
    rec = Recorder(str(tmp_path))
    summary = rec.write_summary(infra, rounds_used=1)
    rec.close()
    assert summary["total_units"] == 7 and summary["remaining_units"] == 6
    assert summary["deliveries"] == [{
        "qid": "q0004", "agent": "agent_3", "submitted": "6", "f1": 1.0, "em": 1.0,
        "payout": 400, "round": 0, "price": 400, "difficulty": "3hop", "topic": "k07",
    }]
    # v3's task/question split is gone
    assert "tasks" not in summary and "questions" not in summary


def test_log_round_snapshot_shape_and_content(tmp_path):
    infra = demo_infra("C0")
    rec = Recorder(str(tmp_path))
    rec.log(_event("agent_1", "retrieve", "docs ..."))            # 2 solving tokens
    rec.log(_event("agent_2", "list_questions", "q0001 ...", category="admin"))
    snap = rec.log_round(infra, 1)
    rec.close()

    lines = [json.loads(l) for l in open(tmp_path / "timeseries.jsonl")]
    assert lines == [snap]
    roster = set(infra.agent_ids)
    for key in ("balances", "tokens", "coordination_overhead_by_agent",
                "answered", "memory"):
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
    assert snap["board"] == {"open": 5, "active_claims": 0, "delivered": 0}
    assert snap["total_units"] == 7 and snap["remaining_units"] == 7
    assert snap["demand_absorbed"] == 0.0
    assert snap["n_answered"] == 0 and snap["total_f1"] == 0.0 and snap["total_em"] == 0.0
    assert snap["n_contracts"] == 0 and snap["contracts_by_status"] == {}
    assert snap["n_loans"] == 0 and snap["loan_principal_outstanding"] == 0
    assert snap["interest_paid_total"] == 0
    assert snap["n_claims"] == 0 and snap["n_memory_hits"] == 0
    assert snap["memory_hit_rate"] == 0.0 and snap["improvement_rate"] == 0.0
    assert snap["answers_in_memory_total"] == 0


def test_log_round_attributes_answers_and_memory_to_the_deliverer(tmp_path):
    infra = demo_infra("C0")
    for qid, ans in (("q0003", "4"), ("q0004", "6")):
        dispatch(infra, "agent_3", "claim_question", {"qid": qid})
        dispatch(infra, "agent_3", "deliver_work", {"target_id": qid, "content": ans})
    rec = Recorder(str(tmp_path))
    rec.log(_event("agent_3", "claim_question", CLAIM_MISS, category="admin"))
    snap1 = rec.log_round(infra, 1)
    snap2 = rec.log_round(infra, 2)
    rec.close()

    assert snap1["answered"]["agent_3"] == {"n_answered": 2, "f1_sum": 2.0, "em_sum": 2.0}
    assert snap1["answered"]["agent_1"] == {"n_answered": 0, "f1_sum": 0.0, "em_sum": 0.0}
    assert snap1["n_answered"] == 2 and snap1["total_f1"] == 2.0
    # q0003 had quota 2, so it is still open with one unit left; q0004 is gone
    assert snap1["board"] == {"open": 4, "active_claims": 0, "delivered": 2}
    assert snap1["remaining_units"] == 5
    assert snap1["demand_absorbed"] == 2 / 7
    assert snap1["minted"] == 700 and snap1["total_balance"] == 1700
    # delivering auto-records the graded answers into agent_3's memory
    assert snap1["memory"]["agent_3"] == {
        "answers": 2, "notes": 0, "n_claims": 1, "n_memory_hits": 0,
        "n_repeat_deliveries": 0, "n_improved": 0,
    }
    assert snap1["answers_in_memory_total"] == 2
    # one line per log_round call, cumulative so round 2 repeats the state
    lines = [json.loads(l) for l in open(tmp_path / "timeseries.jsonl")]
    assert [s["round"] for s in lines] == [1, 2]
    assert lines[1] == {**snap1, "round": 2} == snap2


def test_log_round_counts_outstanding_claims(tmp_path):
    infra = demo_infra("C0")
    dispatch(infra, "agent_1", "claim_question", {"qid": "q0001"})
    dispatch(infra, "agent_2", "claim_question", {"qid": "q0001"})
    rec = Recorder(str(tmp_path))
    snap = rec.log_round(infra, 1)
    rec.close()
    assert snap["board"] == {"open": 4, "active_claims": 2, "delivered": 0}
    assert snap["remaining_units"] == 5
