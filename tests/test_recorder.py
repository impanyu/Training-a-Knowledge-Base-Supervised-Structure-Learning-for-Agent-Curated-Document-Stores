import json

import pytest
from fixtures import arrive, demo_infra

from ca.actions import dispatch
from ca.recorder import Recorder


def event(agent, action, result="ok", category="admin", tokens=(10, 5)):
    return {"round": 1, "agent": agent, "action": action, "input": {},
            "result": result, "category": category,
            "tokens_in": tokens[0], "tokens_out": tokens[1]}


def test_token_tallies_split_by_category(tmp_path):
    rec = Recorder(str(tmp_path))
    rec.log(event("agent_1", "memory_search", category="solving"))
    rec.log(event("agent_1", "read_chat", category="admin"))
    rec.log(event("agent_2", "deliver_work", category="solving", tokens=(100, 0)))
    infra = demo_infra()
    summary = rec.write_summary(infra, 1)
    rec.close()
    assert summary["tokens"]["agent_1"] == {"solving": 15, "admin": 15}
    assert summary["tokens"]["agent_2"] == {"solving": 100, "admin": 0}


def test_turn_tallies_count_selfqa_inside_solving(tmp_path):
    rec = Recorder(str(tmp_path))
    rec.log(event("agent_1", "memory_search", category="solving"))
    rec.log(event("agent_1", "record_qa",
                  result="recorded to the shared knowledge base", category="solving"))
    rec.log(event("agent_1", "record_qa", result="ERROR: 'question'",
                  category="solving"))       # a failed record is not a banked QA
    rec.log(event("agent_1", "list_agents", category="admin"))
    infra = demo_infra()
    summary = rec.write_summary(infra, 1)
    rec.close()
    assert summary["turns"] == {"solving": 3, "selfqa": 1}


def test_summary_carries_deliveries_arrivals_and_kb_counts(tmp_path):
    infra = demo_infra()
    infra.round = 3
    arrive(infra, "q0005", 1)
    arrive(infra, "q0001", 2)
    dispatch(infra, "agent_1", "deliver_work", {"target_id": "q0005", "content": "4"})
    dispatch(infra, "agent_2", "record_qa", {"question": "q?", "answer": "a"})
    dispatch(infra, "agent_2", "memory_write", {"content": "note"})
    dispatch(infra, "agent_1", "send_message", {"to": "agent_2", "text": "hi"})
    rec = Recorder(str(tmp_path))
    summary = rec.write_summary(infra, 3)
    rec.close()
    assert summary["level"] == "P0" and summary["seed"] == 0
    (d,) = summary["deliveries"]
    assert d["qid"] == "q0005" and d["latency"] == 2 and d["topic"] == "k07"
    assert summary["arrived_total"] == 2 and summary["pending"] == 1
    assert summary["kb_answers"] == 1 and summary["kb_selfqa"] == 1
    assert summary["n_messages"] == 1          # external traffic not counted
    assert summary["agents"]["agent_1"] == {
        "answered": 1, "f1_sum": 1.0, "em_sum": 1.0, "selfqa": 0, "notes": 0}
    assert summary["agents"]["agent_2"] == {
        "answered": 0, "f1_sum": 0.0, "em_sum": 0.0, "selfqa": 1, "notes": 1}
    on_disk = json.loads((tmp_path / "summary.json").read_text())
    assert on_disk == summary


def test_log_round_snapshot_shape_and_content(tmp_path):
    infra = demo_infra()
    infra.round = 4
    arrive(infra, "q0005", 1)
    arrive(infra, "q0006", 2)
    arrive(infra, "q0001", 3)
    dispatch(infra, "agent_1", "deliver_work", {"target_id": "q0005", "content": "4"})
    dispatch(infra, "agent_1", "deliver_work", {"target_id": "q0006", "content": "7"})
    rec = Recorder(str(tmp_path))
    rec.log(event("agent_1", "deliver_work", category="solving"))
    snap = rec.log_round(infra, 4)
    rec.close()
    assert snap["round"] == 4
    assert snap["arrivals_total"] == 3 and snap["pending"] == 1
    assert snap["answered_total"] == 2
    assert snap["coverage"] == pytest.approx(2 / 3)
    assert snap["mean_latency"] == pytest.approx((3 + 2) / 2)
    assert snap["total_f1"] == pytest.approx(1.0) and snap["total_em"] == 1.0
    assert snap["agents"]["agent_1"]["answered"] == 2
    assert snap["kb_answers"] == 2 and snap["kb_selfqa"] == 0
    assert snap["solving_total"] == 15 and snap["admin_total"] == 0
    assert snap["coordination_overhead"] == 0.0
    line = json.loads((tmp_path / "timeseries.jsonl").read_text())
    assert line == snap


def test_log_round_zero_guards_before_anything_happened(tmp_path):
    infra = demo_infra()
    rec = Recorder(str(tmp_path))
    snap = rec.log_round(infra, 1)
    rec.close()
    assert snap["coverage"] == 0.0 and snap["mean_latency"] == 0.0
    assert snap["coordination_overhead"] == 0.0


def test_the_dead_snapshot_fields_are_gone(tmp_path):
    infra = demo_infra()
    rec = Recorder(str(tmp_path))
    snap = rec.log_round(infra, 1)
    summary = rec.write_summary(infra, 1)
    rec.close()
    for dead in ("board", "demand_absorbed", "n_claims", "memory_hit_rate",
                 "total_units", "remaining_units", "memory",
                 "answers_in_memory_total"):
        assert dead not in snap, dead
        assert dead not in summary, dead


def test_state_roundtrip_and_append_mode(tmp_path):
    rec = Recorder(str(tmp_path / "a"))
    rec.log(event("agent_1", "memory_search", category="solving"))
    rec.log(event("agent_1", "record_qa",
                  result="recorded to the shared knowledge base", category="solving"))
    state = json.loads(json.dumps(rec.to_state()))
    rec.close()

    rec2 = Recorder(str(tmp_path / "a"), append=True)
    rec2.from_state(state)
    rec2.log(event("agent_1", "read_chat", category="admin"))
    infra = demo_infra()
    summary = rec2.write_summary(infra, 2)
    rec2.close()
    assert summary["tokens"]["agent_1"] == {"solving": 30, "admin": 15}
    assert summary["turns"] == {"solving": 2, "selfqa": 1}
    # append mode: the trace file kept the first run's lines
    lines = (tmp_path / "a" / "trace.jsonl").read_text().splitlines()
    assert len(lines) == 3
