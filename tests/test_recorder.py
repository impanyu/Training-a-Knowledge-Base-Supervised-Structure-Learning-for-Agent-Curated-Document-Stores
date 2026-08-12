"""Recorder: the per-agent `memory` summary block (answers/notes stored --
corpus excluded -- plus claim/hit and repeat/improvement tallies counted live
off the event stream in `log()`), and the per-round cumulative snapshots
-> timeseries.jsonl."""
import json

from fixtures import demo_infra

from ca.actions import dispatch
from ca.recorder import Recorder


def _event(agent, action, result, round_no=1, category="solving"):
    return {"round": round_no, "agent": agent, "action": action, "input": {},
            "result": result, "category": category, "tokens_in": 1, "tokens_out": 1}


CLAIM_HIT = ('claimed [q0001] capital of France? (2hop)\n'
             'deliver ONE short answer: ...\n'
             'memory: stored answer: [q0001] capital of France? -> "Paris" '
             '(F1 1.00) GOOD')
CLAIM_MISS = "claimed [q0002] longest river in France? (2hop)"


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
    # every agent gets an entry, even one that never claimed anything --
    # and the seeded corpus counts as neither answers nor notes
    assert summary["memory"]["agent_3"] == {
        "answers": 0, "notes": 0, "n_claims": 0, "n_memory_hits": 0,
        "n_repeat_deliveries": 0, "n_improved": 0,
    }


def test_failed_claims_and_other_actions_do_not_move_the_tally(tmp_path):
    infra = demo_infra("C0")
    rec = Recorder(str(tmp_path))
    rec.log(_event("agent_1", "claim_question", "ERROR: q0002 is held by another agent",
                   category="admin"))
    # the marker only counts on claim_question events, wherever else it echoes
    rec.log(_event("agent_1", "memory_search", "docs say memory: stored answer ..."))
    rec.log(_event("agent_1", "list_agents", "agent_1, agent_2", category="admin"))
    summary = rec.write_summary(infra, rounds_used=1)
    rec.close()
    assert summary["memory"]["agent_1"]["n_claims"] == 0
    assert summary["memory"]["agent_1"]["n_memory_hits"] == 0


def test_repeat_delivery_tallies(tmp_path):
    infra = demo_infra("C0")
    rec = Recorder(str(tmp_path))
    rec.log(_event("agent_1", "deliver_work",
                   "delivered q0001: F1 1.00 (IMPROVED on your stored F1 0.30)"))
    rec.log(_event("agent_1", "deliver_work",
                   "delivered q0002: F1 0.00 (no better than your stored F1 1.00)"))
    rec.log(_event("agent_1", "deliver_work", "delivered q0005: F1 1.00"))
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
    dispatch(infra, "agent_3", "claim_question", {"qid": "q0005"})
    dispatch(infra, "agent_3", "deliver_work",
             {"target_id": "q0005", "content": "sedimentary"})
    rec = Recorder(str(tmp_path))
    summary = rec.write_summary(infra, rounds_used=1)
    rec.close()
    assert summary["total_units"] == 5 and summary["remaining_units"] == 4
    assert summary["deliveries"] == [{
        "qid": "q0005", "agent": "agent_3", "submitted": "sedimentary",
        "f1": 1.0, "em": 1.0, "round": 0, "price": 50,
        "difficulty": "2hop", "topic": "k02",
    }]
    # the job layer and the economy are gone
    assert "jobs_posted" not in summary and "jobs_closed" not in summary
    for dead in ("balances", "bankrupt", "contracts", "contract_prices",
                 "n_contracts", "loans", "minted", "burned", "conservation_ok"):
        assert dead not in summary, dead


def test_summary_counts_chat_messages(tmp_path):
    infra = demo_infra("C0")
    dispatch(infra, "agent_1", "send_message", {"to": "agent_2", "text": "what is chalk?"})
    dispatch(infra, "agent_2", "send_message", {"to": "agent_1", "text": "sedimentary"})
    rec = Recorder(str(tmp_path))
    summary = rec.write_summary(infra, rounds_used=1)
    rec.close()
    assert summary["n_messages"] == 2


def test_log_round_snapshot_shape_and_content(tmp_path):
    infra = demo_infra("C0")
    rec = Recorder(str(tmp_path))
    rec.log(_event("agent_1", "memory_search", "docs ..."))       # 2 solving tokens
    rec.log(_event("agent_2", "list_questions", "q0001 ...", category="admin"))
    snap = rec.log_round(infra, 1)
    rec.close()

    lines = [json.loads(l) for l in open(tmp_path / "timeseries.jsonl")]
    assert lines == [snap]
    roster = set(infra.agent_ids)
    for key in ("tokens", "coordination_overhead_by_agent",
                "answered", "memory"):
        assert set(snap[key]) == roster, key
    assert snap["round"] == 1
    assert snap["tokens"]["agent_1"] == {"solving": 2, "admin": 0}
    assert snap["tokens"]["agent_2"] == {"solving": 0, "admin": 2}
    assert snap["solving_total"] == 2 and snap["admin_total"] == 2
    assert snap["coordination_overhead"] == 0.5
    assert snap["coordination_overhead_by_agent"]["agent_2"] == 1.0
    assert snap["coordination_overhead_by_agent"]["agent_3"] == 0.0  # zero-guarded
    assert snap["board"] == {"open": 5, "active_claims": 0, "closed": 0}
    assert snap["total_units"] == 5 and snap["remaining_units"] == 5
    assert snap["demand_absorbed"] == 0.0
    assert "jobs_posted" not in snap and "job_completion_rate" not in snap
    assert "delegation_rate" not in snap
    for dead in ("balances", "total_balance", "escrow_total", "minted", "burned",
                 "bankrupt", "n_contracts", "contracts_by_status", "n_loans",
                 "loan_principal_outstanding", "interest_paid_total"):
        assert dead not in snap, dead
    assert snap["n_answered"] == 0 and snap["total_f1"] == 0.0 and snap["total_em"] == 0.0
    assert snap["n_messages"] == 0
    assert snap["n_claims"] == 0 and snap["n_memory_hits"] == 0
    assert snap["memory_hit_rate"] == 0.0 and snap["improvement_rate"] == 0.0
    assert snap["answers_in_memory_total"] == 0


def test_log_round_attributes_answers_and_memory_to_the_deliverer(tmp_path):
    infra = demo_infra("C0")
    dispatch(infra, "agent_3", "claim_question", {"qid": "q0003"})
    dispatch(infra, "agent_3", "deliver_work", {"target_id": "q0003", "content": "4"})
    rec = Recorder(str(tmp_path))
    rec.log(_event("agent_3", "claim_question", CLAIM_MISS, category="admin"))
    snap1 = rec.log_round(infra, 1)
    snap2 = rec.log_round(infra, 2)
    rec.close()

    assert snap1["answered"]["agent_3"] == {"n_answered": 1, "f1_sum": 1.0, "em_sum": 1.0}
    assert snap1["answered"]["agent_1"] == {"n_answered": 0, "f1_sum": 0.0, "em_sum": 0.0}
    assert snap1["n_answered"] == 1 and snap1["total_f1"] == 1.0
    assert snap1["board"] == {"open": 4, "active_claims": 0, "closed": 1}
    assert snap1["remaining_units"] == 4
    assert snap1["demand_absorbed"] == 1 / 5
    # delivering auto-records the graded answer into agent_3's memory
    assert snap1["memory"]["agent_3"] == {
        "answers": 1, "notes": 0, "n_claims": 1, "n_memory_hits": 0,
        "n_repeat_deliveries": 0, "n_improved": 0,
    }
    assert snap1["answers_in_memory_total"] == 1
    # one line per log_round call, cumulative so round 2 repeats the state
    lines = [json.loads(l) for l in open(tmp_path / "timeseries.jsonl")]
    assert [s["round"] for s in lines] == [1, 2]
    assert lines[1] == {**snap1, "round": 2} == snap2


def test_log_round_counts_chat_traffic(tmp_path):
    infra = demo_infra("C0")
    rec = Recorder(str(tmp_path))
    assert rec.log_round(infra, 1)["n_messages"] == 0
    dispatch(infra, "agent_1", "send_message", {"to": "agent_2", "text": "chalk?"})
    dispatch(infra, "agent_2", "send_message", {"to": "agent_1", "text": "sedimentary"})
    assert rec.log_round(infra, 2)["n_messages"] == 2
    rec.close()


def test_log_round_counts_outstanding_claims(tmp_path):
    infra = demo_infra("C0")
    dispatch(infra, "agent_1", "claim_question", {"qid": "q0001"})
    dispatch(infra, "agent_2", "claim_question", {"qid": "q0002"})
    rec = Recorder(str(tmp_path))
    snap = rec.log_round(infra, 1)
    rec.close()
    assert snap["board"] == {"open": 3, "active_claims": 2, "closed": 0}
    assert snap["remaining_units"] == 5     # claimed, not yet absorbed
