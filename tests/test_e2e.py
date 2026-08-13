"""v7 end-to-end scripted flows over the real scheduler: the arrival ->
notification -> read -> search -> deliver pipeline with graded latency, the
proactive idle cycle banking a Q&A that a peer later finds, the peer-assist
round trip, P0/B0 arm-invariance of the arrival schedule, the per-round
timeseries contract, and a source-wide straggler sweep. Deterministic, no
LLM.

The demo world (fixtures, seed 0, rate 0.5, N=2) has a PINNED arrival
schedule: r2 q0005->agent_1, r5 q0002->agent_2, r7 q0006->agent_1,
r9 q0003+q0001+q0004->agent_2 and q0008->agent_1, r10 q0007->agent_1.
"""
import json
import pathlib
import random
import re

import pytest
from fixtures import (DEMO_CORPUS, HashEmbedding, demo_bank,
                      demo_corpus_embeddings, demo_domains)

from ca.agent import Agent, ScriptedPolicy
from ca.config import CONFIGS, ExperimentConfig
from ca.infra import Infra
from ca.metrics import compute_metrics
from ca.recorder import Recorder
from ca.scheduler import Scheduler

IDLE = ("list_agents", {})     # a turn that touches no state

REPO = pathlib.Path(__file__).resolve().parents[1]


def build(level, scripts, tmp_path, in_tokens=10, out_tokens=5,
          max_rounds=10, seed=0, n_agents=2, **cfg_kw):
    cfg = ExperimentConfig(level=CONFIGS[level], seed=seed, n_agents=n_agents,
                           max_rounds=max_rounds, **cfg_kw)
    bank = demo_bank()
    assignment, exemplars = demo_domains(bank, n_agents)
    infra = Infra(cfg, bank, assignment=assignment, corpus=DEMO_CORPUS,
                  corpus_embeddings=demo_corpus_embeddings(),
                  embedding_function=HashEmbedding(), exemplars=exemplars)
    agents = [Agent(a, cfg, infra, ScriptedPolicy(scripts.get(a, []),
                                                  in_tokens=in_tokens,
                                                  out_tokens=out_tokens))
              for a in infra.agent_ids]
    return infra, Scheduler(infra, agents, cfg, Recorder(str(tmp_path)),
                            random.Random(cfg.seed))


def _trace(tmp_path):
    return [json.loads(l) for l in open(tmp_path / "trace.jsonl")]


def _results(trace, agent, action):
    return [e["result"] for e in trace if e["agent"] == agent and e["action"] == action]


# ---------------- the headline pipeline ----------------

def test_arrival_to_graded_delivery_with_latency(tmp_path):
    """q0005 arrives at r2; agent_1 reads the thread, searches the KB and
    delivers at r4 -- graded, latency recorded, echoed back to external."""
    scripts = {"agent_1": [
        IDLE,                                                     # r1
        ("read_chat", {"with_agent": "external"}),                # r2: arrival round
        ("memory_search", {"query": "sum of 2 and 2"}),           # r3
        ("deliver_work", {"target_id": "q0005", "content": "4"}),  # r4
    ]}
    infra, sched = build("P0", scripts, tmp_path, max_rounds=4)
    summary = sched.run()
    trace = _trace(tmp_path)
    # the arrival reached the external thread and was READ, verbatim
    assert _results(trace, "agent_1", "read_chat") == \
        ["[r2] external: [q0005] sum of 2 and 2?"]
    # the corpus answered the search
    assert "[Arithmetic]" in _results(trace, "agent_1", "memory_search")[0]
    # graded delivery with latency
    assert _results(trace, "agent_1", "deliver_work") == ["delivered q0005: F1 1.00"]
    (d,) = summary["deliveries"]
    assert d["qid"] == "q0005" and d["f1"] == 1.0 and d["em"] == 1.0
    assert (d["round_in"], d["round_out"], d["latency"]) == (2, 4, 2)
    # the answer went back out on the external thread
    msgs, _ = infra.chat.read("agent_1", "external")
    assert [m.text for m in msgs] == ["[q0005] sum of 2 and 2?", "[q0005] 4"]
    m = compute_metrics(summary)
    assert m["n_answered"] == 1 and m["mean_latency"] == 2.0
    assert m["coverage"] == 1.0                    # only q0005 had arrived by r4


def test_idle_agent_banks_a_selfqa_that_a_peer_later_finds(tmp_path):
    """The proactive loop end-to-end: agent_1 records a Q&A while idle;
    agent_2's later memory_search surfaces it."""
    scripts = {
        "agent_1": [("record_qa", {"question": "sum of 9 and 9?",
                                   "answer": "18"})],
        "agent_2": [IDLE, ("memory_search", {"query": "sum of 9 and 9?"})],
    }
    infra, sched = build("P0", scripts, tmp_path, max_rounds=2)
    summary = sched.run()
    trace = _trace(tmp_path)
    assert "Q: sum of 9 and 9?\nA: 18" in _results(trace, "agent_2", "memory_search")[0]
    assert summary["kb_selfqa"] == 1
    assert summary["agents"]["agent_1"]["selfqa"] == 1
    m = compute_metrics(summary)
    assert m["selfqa_total"] == 1
    assert m["proactive_ratio"] == pytest.approx(1 / 2)   # 1 record_qa, 2 solving turns


def test_peer_assist_round_trip(tmp_path):
    """agent_2 holds a France question but asks agent_1 for the fact; agent_1
    answers out of the shared KB; agent_2 delivers correctly."""
    scripts = {
        "agent_2": [
            IDLE, IDLE, IDLE, IDLE,
            ("read_chat", {"with_agent": "external"}),            # r5: q0002 arrives
            ("send_message", {"to": "agent_1",
                              "text": "what is the longest river of France?"}),
            IDLE, IDLE, IDLE,
            ("deliver_work", {"target_id": "q0002", "content": "Loire"}),  # r10
        ],
        "agent_1": [
            IDLE, IDLE, IDLE, IDLE, IDLE, IDLE,
            ("read_chat", {"with_agent": "agent_2"}),             # r7
            ("memory_search", {"query": "longest river of France"}),  # r8
            ("send_message", {"to": "agent_2",
                              "text": "the longest river of France is the Loire"}),
        ],
    }
    infra, sched = build("P0", scripts, tmp_path, max_rounds=10)
    summary = sched.run()
    trace = _trace(tmp_path)
    assert "[Loire]" in _results(trace, "agent_1", "memory_search")[0]
    assert "longest river of France?" in _results(trace, "agent_1", "read_chat")[0]
    d = next(x for x in summary["deliveries"] if x["qid"] == "q0002")
    assert d["agent"] == "agent_2" and d["f1"] == 1.0 and d["latency"] == 5
    assert summary["n_messages"] == 2
    assert compute_metrics(summary)["messages_per_answer"] == pytest.approx(2.0)


def test_the_arrival_schedule_is_identical_across_arms(tmp_path):
    """Same bank+seed+N: every question arrives at the same round to the same
    agent whether the arm is P0 or B0."""
    runs = {}
    for arm in ("P0", "B0"):
        infra, sched = build(arm, {}, tmp_path / arm, max_rounds=12)
        sched.run()
        runs[arm] = {
            "order": list(infra.stream.order),
            "pending": dict(infra.stream.pending),
            "arrivals": [json.loads(l)["arrivals_total"]
                         for l in open(tmp_path / arm / "timeseries.jsonl")],
            "threads": {k: [(m.sender, m.text, m.round_no) for m in v]
                        for k, v in infra.chat.threads.items()},
        }
    assert runs["P0"] == runs["B0"]


def test_a_different_seed_moves_the_arrivals(tmp_path):
    _, sched_a = build("P0", {}, tmp_path / "a", max_rounds=12, seed=0)
    _, sched_b = build("P0", {}, tmp_path / "b", max_rounds=12, seed=1)
    sched_a.run()
    sched_b.run()
    a = [json.loads(l)["arrivals_total"] for l in open(tmp_path / "a" / "timeseries.jsonl")]
    b = [json.loads(l)["arrivals_total"] for l in open(tmp_path / "b" / "timeseries.jsonl")]
    assert a != b


# ---------------- run mechanics ----------------

def test_max_rounds_stops_a_run_that_answers_nothing(tmp_path):
    infra, sched = build("B0", {}, tmp_path, max_rounds=6)
    summary = sched.run()
    assert summary["rounds_used"] == 6
    assert summary["deliveries"] == []
    assert summary["arrived_total"] == 2           # r2 and r5 arrivals came in
    assert summary["pending"] == 2


def test_the_run_stops_when_the_stream_is_drained(tmp_path):
    scripts = {
        "agent_1": [("deliver_work", {"target_id": q, "content": "x"})
                    for q in ("q0005", "q0006", "q0007", "q0008")],
        "agent_2": [("deliver_work", {"target_id": q, "content": "x"})
                    for q in ("q0001", "q0002", "q0003", "q0004")],
    }
    infra, sched = build("P0", scripts, tmp_path, max_rounds=30,
                         arrival_rate=10.0)
    summary = sched.run()
    assert summary["rounds_used"] == 4
    assert infra.stream.all_done()
    assert compute_metrics(summary)["coverage"] == 1.0


def test_timeseries_one_cumulative_snapshot_per_round(tmp_path):
    scripts = {"agent_1": [
        IDLE,
        ("read_chat", {"with_agent": "external"}),
        ("record_qa", {"question": "q?", "answer": "a"}),
        ("deliver_work", {"target_id": "q0005", "content": "4"}),
    ]}
    _, sched = build("P0", scripts, tmp_path, max_rounds=6)
    sched.run()
    lines = [json.loads(l) for l in open(tmp_path / "timeseries.jsonl")]
    assert [s["round"] for s in lines] == [1, 2, 3, 4, 5, 6]
    assert [s["arrivals_total"] for s in lines] == [0, 1, 1, 1, 2, 2]
    assert [s["answered_total"] for s in lines] == [0, 0, 0, 1, 1, 1]
    assert [s["kb_selfqa"] for s in lines] == [0, 0, 1, 1, 1, 1]
    assert lines[3]["mean_latency"] == 2.0
    assert lines[3]["coverage"] == 1.0 and lines[4]["coverage"] == 0.5
    for s in lines:                                # cumulative, never regressing
        assert s["pending"] == s["arrivals_total"] - s["answered_total"]


def test_summary_is_written_even_when_a_turn_crashes(tmp_path):
    class ExplodingPolicy:
        def decide(self, system, context, tools):
            raise RuntimeError("boom")

    infra, sched = build("P0", {}, tmp_path, max_rounds=3)
    sched.agents[0].policy = ExplodingPolicy()
    with pytest.raises(RuntimeError):
        sched.run()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["level"] == "P0" and summary["rounds_used"] == 1


def test_specialization_is_computable_from_a_real_run(tmp_path):
    scripts = {
        "agent_1": [("deliver_work", {"target_id": q, "content": "x"})
                    for q in ("q0005", "q0006", "q0007", "q0008")],
        "agent_2": [("deliver_work", {"target_id": q, "content": "x"})
                    for q in ("q0001", "q0002", "q0003", "q0004")],
    }
    _, sched = build("P0", scripts, tmp_path, max_rounds=10, arrival_rate=10.0)
    m = compute_metrics(sched.run())
    # routing by domain makes every agent fully specialized over topics
    assert m["specialization"] == {"agent_1": 1.0, "agent_2": 1.0}
    assert m["mean_specialization"] == 1.0


# ---------------- straggler sweep ----------------

def test_no_source_module_references_the_dead_machinery():
    """§10: nothing in src/ or scripts/ may still speak the v6 language of
    boards, claims, hubs or C-levels."""
    dead = re.compile(r"\b(board|claim(?:s|ed|ing)?|hub|star_comms|"
                      r"world_access|shared_memory|solo_turns|has_hub|C[0-7])\b")
    for path in sorted((REPO / "src").rglob("*.py")) + \
            sorted((REPO / "scripts").glob("*")):
        if path.is_dir() or "__pycache__" in str(path):
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            assert not dead.search(line), (path.name, i, line)
