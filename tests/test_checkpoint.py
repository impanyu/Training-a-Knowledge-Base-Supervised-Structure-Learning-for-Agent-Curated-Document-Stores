"""T29 under v7: full-state checkpoint every N rounds + resume. The stream's
own rng, the chat threads with their unread counters, and the shared KB
(restored on top of a freshly re-seeded corpus) all cross the boundary."""
import json
import random

import pytest
from fixtures import (DEMO_CORPUS, HashEmbedding, demo_bank,
                      demo_corpus_embeddings, demo_domains)

from ca import checkpoint
from ca.agent import Agent, ScriptedPolicy
from ca.config import CONFIGS, ExperimentConfig
from ca.infra import Infra
from ca.recorder import Recorder
from ca.scheduler import Scheduler


def _infra(level, seed, max_rounds, **cfg_kw):
    cfg = ExperimentConfig(level=CONFIGS[level], seed=seed, n_agents=2,
                           max_rounds=max_rounds, **cfg_kw)
    bank = demo_bank()
    assignment, exemplars = demo_domains(bank, 2)
    return Infra(cfg, bank, assignment=assignment, corpus=DEMO_CORPUS,
                 corpus_embeddings=demo_corpus_embeddings(),
                 embedding_function=HashEmbedding(), exemplars=exemplars)


def build(level, scripts, out_dir, seed=0, max_rounds=6, **cfg_kw):
    infra = _infra(level, seed, max_rounds, **cfg_kw)
    agents = [Agent(a, infra.cfg, infra,
                    ScriptedPolicy(scripts.get(a, []), in_tokens=10, out_tokens=5))
              for a in infra.agent_ids]
    sched = Scheduler(infra, agents, infra.cfg, Recorder(str(out_dir)),
                      random.Random(seed))
    return infra, agents, sched


def resume(level, scripts, out_dir, ck_path, seed=0, max_rounds=6, **cfg_kw):
    """Rebuild everything fresh from 'CLI args' -- including the corpus
    seeding and the stream's shuffled order -- then restore the checkpoint:
    the exact shape of the runner's --resume path."""
    infra = _infra(level, seed, max_rounds, **cfg_kw)
    agents = [Agent(a, infra.cfg, infra,
                    ScriptedPolicy(scripts.get(a, []), in_tokens=10, out_tokens=5))
              for a in infra.agent_ids]
    recorder = Recorder(str(out_dir), append=True)
    rng = random.Random(seed)
    with open(ck_path) as f:
        state = json.load(f)
    checkpoint.validate(state, infra.cfg)
    checkpoint.restore(state, infra, agents, recorder, rng)
    sched = Scheduler(infra, agents, infra.cfg, recorder, rng)
    return infra, sched.run(start_round=state["round"] + 1)


# 6-round P0 workload over the seed-0 arrival schedule (q0005 -> agent_1 at
# r2, q0002 -> agent_2 at r5), with activity on both sides of a round-3
# checkpoint boundary: chat, goals, self-QA and the stream itself all cross
# it -- q0002 ARRIVES after the boundary, so the resumed stream rng must
# reproduce the arrival exactly.
SCRIPTS = {
    "agent_1": [
        ("memory_write", {"content": "sums are my domain"}),
        ("read_chat", {"with_agent": "external"}),
        ("deliver_work", {"target_id": "q0005", "content": "4"}),
        ("record_qa", {"question": "sum of 6 and 6?", "answer": "12"}),
        ("send_message", {"to": "agent_2", "text": "France is all yours"}),
        ("list_agents", {}),
    ],
    "agent_2": [
        ("push_goal", {"note": "wait for France questions"}),
        ("send_message", {"to": "agent_1", "text": "ping"}),
        ("memory_search", {"query": "France"}),
        ("memory_write", {"content": "the Loire is the longest river"}),
        ("read_chat", {"with_agent": "external"}),
        ("deliver_work", {"target_id": "q0002", "content": "Loire"}),
    ],
}


def _lines(path):
    with open(path) as f:
        return f.read().splitlines()


# ---------- capture shape ----------


def test_capture_holds_stream_chat_memory_agents_recorder_rng(tmp_path):
    _, agents, sched = build("P0", SCRIPTS, tmp_path, max_rounds=2)
    sched.run()
    state = json.load(open(tmp_path / "checkpoint_0002.json"))
    assert set(state) == {"round", "config", "stream", "chat", "memory",
                          "agents", "recorder", "rng"}
    assert state["round"] == 2
    assert state["config"]["level"] == "P0" and state["config"]["n_agents"] == 2
    assert state["config"]["arrival_rate"] == 0.5
    assert state["stream"]["pos"] == 1              # q0005 arrived at r2
    assert set(state["agents"]) == {"agent_1", "agent_2"}


def test_capture_holds_no_board_or_economy_state(tmp_path):
    infra, agents, sched = build("P0", {}, tmp_path, max_rounds=1)
    sched.run()
    state = json.load(open(tmp_path / "checkpoint_0001.json"))
    for dead in ("board", "economy", "contracts", "loans"):
        assert dead not in state, dead
    for dead in ("claim_ttl", "hub_turns_per_round", "solo_turns_per_round"):
        assert dead not in state["config"], dead


def test_checkpoint_never_contains_the_corpus(tmp_path):
    _, _, sched = build("P0", SCRIPTS, tmp_path, max_rounds=4,
                        checkpoint_every=4)
    sched.run()
    state = json.load(open(tmp_path / "checkpoint_0004.json"))
    # the KB dump holds the 4 written rows, never the seeded corpus (the
    # corpus text may still echo through FIFO'd search RESULTS -- that is
    # transcript, not store)
    assert all(row[2]["kind"] != "corpus" for row in state["memory"])
    assert len(state["memory"]) == 4    # 2 notes + 1 answer + 1 selfqa
    corpus_texts = {p["text"] for p in DEMO_CORPUS}
    assert all(row[1] not in corpus_texts for row in state["memory"])


def test_rng_state_survives_json():
    rng = random.Random(3)
    rng.random()
    state = json.loads(json.dumps(checkpoint.rng_state(rng)))
    clone = random.Random(0)
    checkpoint.restore_rng(clone, state)
    assert clone.random() == rng.random()


def test_validate_rejects_any_identity_mismatch(tmp_path):
    infra, agents, sched = build("P0", {}, tmp_path, max_rounds=1)
    sched.run()
    state = json.load(open(tmp_path / "checkpoint_0001.json"))

    def cfg(**kw):
        base = dict(level=CONFIGS["P0"], seed=0, n_agents=2)
        base.update(kw)
        return ExperimentConfig(**base)

    checkpoint.validate(state, cfg())               # the matching one passes
    with pytest.raises(ValueError, match="level"):
        checkpoint.validate(state, cfg(level=CONFIGS["B0"]))
    with pytest.raises(ValueError, match="seed"):
        checkpoint.validate(state, cfg(seed=8))
    with pytest.raises(ValueError, match="n_agents"):
        checkpoint.validate(state, cfg(n_agents=4))
    with pytest.raises(ValueError, match="arrival_rate"):
        checkpoint.validate(state, cfg(arrival_rate=1.5))


# ---------- file lifecycle ----------


def test_checkpoint_files_every_n_and_at_final_round(tmp_path):
    _, _, sched = build("P0", {}, tmp_path, max_rounds=5, checkpoint_every=2)
    sched.run()
    names = sorted(p.name for p in tmp_path.glob("checkpoint_*.json"))
    assert names == ["checkpoint_0002.json", "checkpoint_0004.json",
                     "checkpoint_0005.json"]


def test_checkpoint_written_when_the_stream_finishes_early(tmp_path):
    # rate 10: everything arrives at r1; two agents drain their four each
    scripts = {
        "agent_1": [("deliver_work", {"target_id": q, "content": "x"})
                    for q in ("q0005", "q0006", "q0007", "q0008")],
        "agent_2": [("deliver_work", {"target_id": q, "content": "x"})
                    for q in ("q0001", "q0002", "q0003", "q0004")],
    }
    infra, _, sched = build("P0", scripts, tmp_path, max_rounds=30,
                            arrival_rate=10.0, checkpoint_every=10)
    summary = sched.run()
    assert summary["rounds_used"] == 4              # done long before 30
    assert (tmp_path / "checkpoint_0004.json").exists()
    assert not (tmp_path / "checkpoint_0003.json").exists()


# ---------- the fidelity test ----------


def test_resume_reproduces_a_straight_run_exactly(tmp_path):
    """Run A: 6 rounds straight. Run B: 3 rounds, checkpoint, resume 4-6 on a
    freshly rebuilt world. Summary, timeseries and trace must be
    indistinguishable -- including the post-boundary ARRIVAL of q0002."""
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"

    _, _, sched_a = build("P0", SCRIPTS, dir_a, max_rounds=6)
    summary_a = sched_a.run()
    assert summary_a["rounds_used"] == 6
    assert {d["qid"]: d["latency"] for d in summary_a["deliveries"]} == \
        {"q0005": 1, "q0002": 1}
    assert summary_a["n_messages"] == 2
    assert summary_a["kb_selfqa"] == 1

    _, _, sched_b = build("P0", SCRIPTS, dir_b, max_rounds=3, checkpoint_every=3)
    sched_b.run()
    assert (dir_b / "checkpoint_0003.json").exists()

    # each agent takes exactly one turn per round: entries 3+ remain
    rest = {a: s[3:] for a, s in SCRIPTS.items()}
    _, summary_b = resume("P0", rest, dir_b, dir_b / "checkpoint_0003.json",
                          max_rounds=6, checkpoint_every=3)

    assert summary_b == summary_a
    assert _lines(dir_b / "timeseries.jsonl") == _lines(dir_a / "timeseries.jsonl")
    trace_a = _lines(dir_a / "trace.jsonl")
    trace_b = _lines(dir_b / "trace.jsonl")
    assert trace_b == trace_a                       # RNG restore => same shuffle
    order_a = [json.loads(l)["agent"] for l in trace_a]
    order_b = [json.loads(l)["agent"] for l in trace_b]
    assert order_a == order_b


def test_resume_restores_threads_unread_and_the_kb(tmp_path):
    """agent_2's r2 ping is still unread at the boundary (agent_1 read only
    the external thread), the KB holds the pre-boundary answer on top of a
    fresh corpus re-seed, and both survive the round trip."""
    dir_b = tmp_path / "b"
    _, _, sched = build("P0", SCRIPTS, dir_b, max_rounds=3, checkpoint_every=3)
    sched.run()
    infra_b, _ = resume("P0", {}, dir_b, dir_b / "checkpoint_0003.json",
                        max_rounds=4, checkpoint_every=3)
    assert infra_b.chat.unread_partners("agent_1") == [("agent_2", 1)]
    msgs, _ = infra_b.chat.read("agent_1", "external")
    assert [m.text for m in msgs] == ["[q0005] sum of 2 and 2?", "[q0005] 4"]
    hits = infra_b.memory.search("sum of 2 and 2", k=8)
    assert any(h["kind"] == "answer" and h["qid"] == "q0005" for h in hits)
    assert any(h["kind"] == "corpus" for h in
               infra_b.memory.search("capital of France", k=1))


def test_resume_continues_to_a_larger_max_rounds(tmp_path):
    _, _, sched = build("P0", {}, tmp_path, max_rounds=2, checkpoint_every=10)
    sched.run()
    _, summary = resume("P0", {}, tmp_path, tmp_path / "checkpoint_0002.json",
                        max_rounds=4, checkpoint_every=10)
    assert summary["rounds_used"] == 4
    lines = [json.loads(l) for l in _lines(tmp_path / "timeseries.jsonl")]
    assert [s["round"] for s in lines] == [1, 2, 3, 4]
    assert (tmp_path / "checkpoint_0004.json").exists()
