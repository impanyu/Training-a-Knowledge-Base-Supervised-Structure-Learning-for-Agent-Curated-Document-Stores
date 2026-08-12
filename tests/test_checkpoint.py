"""T29: full-state checkpoint every N rounds + resume, with the seeded memory:
the corpus never enters a checkpoint, and a resumed run restores notes/answers
on top of a freshly re-seeded store."""
import json
import random

import pytest
from fixtures import (DEMO_CORPUS, HashEmbedding, demo_bank,
                      demo_corpus_embeddings, demo_infra)

from ca import checkpoint
from ca.agent import Agent, ScriptedPolicy
from ca.config import CONFIGS, ExperimentConfig
from ca.infra import Infra
from ca.memory import FifoMemory, GoalStack
from ca.recorder import Recorder
from ca.scheduler import Scheduler


def _infra(level, seed, max_rounds, **cfg_kw):
    cfg = ExperimentConfig(level=CONFIGS[level], seed=seed,
                           max_rounds=max_rounds, **cfg_kw)
    return Infra(cfg, demo_bank(), corpus=DEMO_CORPUS,
                 corpus_embeddings=demo_corpus_embeddings(),
                 embedding_function=HashEmbedding())


def build(level, scripts, out_dir, seed=7, max_rounds=6, **cfg_kw):
    infra = _infra(level, seed, max_rounds, **cfg_kw)
    agents = [Agent(a, infra.cfg, infra,
                    ScriptedPolicy(scripts.get(a, []), in_tokens=10, out_tokens=5))
              for a in infra.agent_ids]
    sched = Scheduler(infra, agents, infra.cfg, Recorder(str(out_dir)),
                      random.Random(seed))
    return infra, agents, sched


def resume(level, scripts, out_dir, ck_path, seed=7, max_rounds=6, **cfg_kw):
    """Rebuild everything fresh from 'CLI args' -- including the corpus
    seeding -- then restore the checkpoint: the exact shape of the runner's
    --resume path."""
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


# 6-round C0 workload that touches every stateful subsystem, with activity on
# both sides of a round-3 checkpoint boundary: chat, goals and seeded memories
# all cross it, and the q0004 hand-off (agent_4 releases in r4, agent_5 claims
# it in r5 and delivers in r6) happens entirely after the boundary.
SCRIPTS = {
    "agent_1": [
        ("list_questions", {}),
        ("claim_question", {"qid": "q0001"}),
        ("memory_search", {"query": "capital of France"}),
        ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
        ("claim_question", {"qid": "q0002"}),
        ("deliver_work", {"target_id": "q0002", "content": "Loire"}),
    ],
    "agent_2": [
        ("claim_question", {"qid": "q0005"}),
        ("memory_write", {"content": "chalk is sedimentary"}),
        ("send_message", {"to": "agent_3", "text": "chalk is a sedimentary rock"}),
        ("list_questions", {}),
        ("deliver_work", {"target_id": "q0005", "content": "sedimentary"}),
    ],
    "agent_3": [
        ("read_chat", {"with_agent": "agent_2"}),
        ("memory_write", {"content": "agent_2 knows the geology questions"}),
        ("memory_search", {"query": "chalk sedimentary rock"}),
        ("send_message", {"to": "agent_2", "text": "thanks, noted"}),
        ("list_agents", {}),
    ],
    "agent_4": [
        ("claim_question", {"qid": "q0004"}),
        ("push_goal", {"note": "answer q0004 or hand it on"}),
        ("list_questions", {}),
        ("release_question", {"qid": "q0004"}),     # r4: hand-off, after the boundary
        ("read_chat", {"with_agent": "agent_5"}),
        ("pop_goal", {}),
    ],
    "agent_5": [
        ("memory_search", {"query": "3+3 arithmetic"}),
        ("memory_write", {"content": "3+3 makes 6"}),
        ("read_chat", {"with_agent": "agent_4"}),
        ("list_questions", {}),
        ("claim_question", {"qid": "q0004"}),       # r5: picks up what r4 released
        ("deliver_work", {"target_id": "q0004", "content": "6"}),
    ],
    "agent_6": [
        ("memory_write", {"content": "q0003 looks like easy arithmetic"}),
        ("list_questions", {}),
        ("list_agents", {}),
        ("send_message", {"to": "agent_7", "text": "anything left on the board?"}),
    ],
    "agent_7": [
        ("push_goal", {"note": "find a niche"}),
        ("list_questions", {}),
        ("pop_goal", {}),
        ("read_chat", {"with_agent": "agent_6"}),
        ("list_agents", {}),
    ],
    "agent_8": [
        ("list_questions", {}),
        ("list_agents", {}),
        ("memory_write", {"content": "nothing worth claiming yet"}),
    ],
}


def _lines(path):
    with open(path) as f:
        return f.read().splitlines()


# ---------- subsystem round-trips ----------


def test_board_state_roundtrip():
    infra = demo_infra("C0")
    infra.board.claim("agent_1", "q0001", round_no=2)
    infra.board.claim("agent_2", "q0005", round_no=3)
    infra.board.deliver("agent_2", "q0005", "sedimentary")
    infra.board.claim("agent_3", "q0004", round_no=3)
    infra.board.release("agent_3", "q0004")
    state = json.loads(json.dumps(infra.board.to_state()))
    fresh = demo_infra("C0")
    fresh.board.from_state(state)
    assert fresh.board.open_questions() == infra.board.open_questions()
    assert fresh.board.active["q0001"].agent == "agent_1"
    assert fresh.board.active["q0001"].round == 2
    assert "q0005" not in fresh.board.active
    assert "q0004" not in fresh.board.active          # released, still open
    assert fresh.board.strikes == infra.board.strikes
    assert fresh.board.results_json() == infra.board.results_json()


def test_chat_state_roundtrip_preserves_unread_cursors():
    infra = demo_infra("C0")
    infra.chat.send("agent_1", "agent_2", "hello", 1)
    infra.chat.mark_read("agent_2")
    infra.chat.send("agent_3", "agent_2", "still unread", 2)
    state = json.loads(json.dumps(infra.chat.to_state()))
    fresh = demo_infra("C0")
    fresh.chat.from_state(state)
    assert [m.text for m in fresh.chat.unread("agent_2")] == ["still unread"]
    assert [m.text for m in fresh.chat.history("agent_1", "agent_2")] == ["hello"]


def test_short_term_memories_state_roundtrip():
    fifo = FifoMemory(k=2)
    fifo.add("a1", "r1")
    fifo.add("a2", "r2")
    f2 = FifoMemory(k=2)
    f2.from_state(json.loads(json.dumps(fifo.to_state())))
    assert f2.render() == fifo.render()
    f2.add("a3", "r3")                       # maxlen survives the round-trip
    assert len(f2.items) == 2

    goals = GoalStack("root")
    goals.push("sub-goal")
    g2 = GoalStack("root")
    g2.from_state(json.loads(json.dumps(goals.to_state())))
    assert g2.render() == goals.render()
    assert g2.pop() == "sub-goal"


def test_agent_memory_state_roundtrip_including_the_shared_bucket():
    infra = demo_infra("C2")                 # shared bucket, corpus-seeded
    infra.memory.write("agent_1", "paris facts")
    infra.memory.write("agent_2", '[q0003] 2+2? -> "4" (F1 1.00)',
                       kind="answer", qid="q0003", f1=1.0)
    state = json.loads(json.dumps(infra.memory.to_state()))
    fresh = demo_infra("C2")                 # freshly re-seeded shared store
    fresh.memory.from_state(state)
    assert fresh.memory.answer("agent_5", "q0003")["f1"] == 1.0
    assert fresh.memory.search("agent_5", "paris facts")[0]["text"] == "paris facts"
    assert fresh.memory.n_answers("agent_1") == infra.memory.n_answers("agent_1")
    assert fresh.memory.n_entries("agent_1") == len(DEMO_CORPUS) + 2


def test_recorder_tallies_roundtrip_and_append_mode(tmp_path):
    rec = Recorder(str(tmp_path))
    rec.log({"round": 1, "agent": "agent_1", "action": "memory_search", "input": {},
             "result": "ok", "category": "solving", "tokens_in": 10, "tokens_out": 5})
    rec.log({"round": 1, "agent": "agent_1", "action": "claim_question", "input": {},
             "result": 'claimed [q0001] capital of France? (2hop)\n'
                       'memory: stored answer: ... "Paris" (F1 1.00)',
             "category": "admin", "tokens_in": 1, "tokens_out": 1})
    state = json.loads(json.dumps(rec.to_state()))
    rec.close()

    rec2 = Recorder(str(tmp_path), append=True)
    rec2.from_state(state)
    assert rec2._tokens["agent_1"] == {"solving": 15, "admin": 2}
    assert rec2._memory["agent_1"] == {"n_claims": 1, "n_memory_hits": 1,
                                       "n_repeat_deliveries": 0, "n_improved": 0}
    rec2.log({"round": 2, "agent": "agent_1", "action": "list_agents", "input": {},
              "result": "agent_1", "category": "admin", "tokens_in": 1, "tokens_out": 1})
    rec2.close()
    assert len(_lines(tmp_path / "trace.jsonl")) == 3    # append, no rewrite


def test_rng_state_survives_json():
    rng = random.Random(7)
    rng.random()
    state = checkpoint.rng_state(rng)
    rng2 = random.Random(0)
    checkpoint.restore_rng(rng2, json.loads(json.dumps(state)))
    assert [rng2.random() for _ in range(5)] == [rng.random() for _ in range(5)]


def test_capture_holds_no_economy_state(tmp_path):
    infra, agents, sched = build("C7", {}, tmp_path, max_rounds=1)
    sched.run()
    ck = json.loads((tmp_path / "checkpoint_0001.json").read_text())
    assert set(ck) == {"round", "config", "board", "chat", "memory", "agents",
                       "recorder", "rng"}
    assert "seed_capital_total" not in ck["config"]
    assert "loan_rate" not in ck["config"]


# ---------- scheduler save discipline ----------


def test_checkpoint_files_every_n_and_at_final_round(tmp_path):
    _, _, sched = build("C7", {}, tmp_path, max_rounds=5, checkpoint_every=2)
    sched.run()
    names = sorted(p.name for p in tmp_path.glob("checkpoint_*.json"))
    assert names == ["checkpoint_0002.json", "checkpoint_0004.json",
                     "checkpoint_0005.json"]
    ck = json.loads((tmp_path / "checkpoint_0004.json").read_text())
    assert ck["round"] == 4
    assert ck["config"]["level"] == "C7" and ck["config"]["seed"] == 7


def test_checkpoint_written_when_run_stops_early(tmp_path):
    """C7 solo agent absorbs the demo bank's whole 5-question demand, so the
    run breaks on all_done() before max_rounds."""
    script = []
    for qid, ans in (("q0001", "Paris"), ("q0002", "Loire"), ("q0003", "4"),
                     ("q0004", "6"), ("q0005", "sedimentary")):
        script += [("claim_question", {"qid": qid}),
                   ("deliver_work", {"target_id": qid, "content": ans})]
    _, _, sched = build("C7", {"agent_1": script}, tmp_path,
                        max_rounds=40, checkpoint_every=50)
    summary = sched.run()
    assert summary["rounds_used"] == 10        # 5 questions x (claim + deliver)
    assert summary["remaining_units"] == 0
    assert (tmp_path / "checkpoint_0010.json").exists()


def test_checkpoint_never_contains_the_corpus(tmp_path):
    """The seeded corpus is static furniture: 12k paragraphs must not be
    serialized into every checkpoint file."""
    _, _, sched = build("C0", SCRIPTS, tmp_path, max_rounds=3, checkpoint_every=3)
    sched.run()
    ck = json.loads((tmp_path / "checkpoint_0003.json").read_text())
    rows = [row for rows in ck["memory"].values() for row in rows]
    assert rows                                          # notes/answers ARE saved
    assert all(row[2]["kind"] != "corpus" for row in rows)


def test_validate_rejects_level_or_seed_mismatch(tmp_path):
    _, _, sched = build("C7", {}, tmp_path, max_rounds=2)
    sched.run()
    state = json.loads((tmp_path / "checkpoint_0002.json").read_text())
    checkpoint.validate(state, ExperimentConfig(level=CONFIGS["C7"], seed=7))
    with pytest.raises(ValueError, match="level"):
        checkpoint.validate(state, ExperimentConfig(level=CONFIGS["C0"], seed=7))
    with pytest.raises(ValueError, match="seed"):
        checkpoint.validate(state, ExperimentConfig(level=CONFIGS["C7"], seed=8))


# ---------- the fidelity test ----------


def test_resume_reproduces_a_straight_run_exactly(tmp_path):
    """Run A: 6 rounds straight. Run B: 3 rounds, checkpoint, resume 4-6 on a
    freshly re-seeded store. Summary, timeseries and trace must be
    indistinguishable."""
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"

    _, _, sched_a = build("C0", SCRIPTS, dir_a, max_rounds=6)
    summary_a = sched_a.run()
    assert summary_a["rounds_used"] == 6
    # the workload really did exercise the board across the boundary: the
    # question agent_4 released in r4 was answered by agent_5 in r6
    assert len(summary_a["deliveries"]) == 4
    assert [d["agent"] for d in summary_a["deliveries"] if d["qid"] == "q0004"] \
        == ["agent_5"]
    assert summary_a["n_messages"] == 3

    _, _, sched_b = build("C0", SCRIPTS, dir_b, max_rounds=3, checkpoint_every=3)
    sched_b.run()
    assert (dir_b / "checkpoint_0003.json").exists()

    # each C0 agent takes exactly one turn per round: entries 3+ remain
    rest = {a: s[3:] for a, s in SCRIPTS.items()}
    _, summary_b = resume("C0", rest, dir_b, dir_b / "checkpoint_0003.json",
                          max_rounds=6, checkpoint_every=3)

    assert summary_b == summary_a
    assert _lines(dir_b / "timeseries.jsonl") == _lines(dir_a / "timeseries.jsonl")
    trace_a = _lines(dir_a / "trace.jsonl")
    trace_b = _lines(dir_b / "trace.jsonl")
    assert trace_b == trace_a                       # RNG restore => same shuffle
    order_a = [json.loads(l)["agent"] for l in trace_a]
    order_b = [json.loads(l)["agent"] for l in trace_b]
    assert order_a == order_b


def test_resume_restores_the_vector_memory_contents(tmp_path):
    """The store is rebuilt from the checkpoint on top of a fresh re-seed: an
    answer written before the boundary is still recallable after it, the
    corpus is still searchable, and privacy still holds."""
    dir_b = tmp_path / "b"
    _, _, sched = build("C0", SCRIPTS, dir_b, max_rounds=3, checkpoint_every=3)
    sched.run()
    infra_b, _ = resume("C0", {a: s[3:] for a, s in SCRIPTS.items()}, dir_b,
                        dir_b / "checkpoint_0003.json", max_rounds=4,
                        checkpoint_every=3)
    rec = infra_b.memory.answer("agent_1", "q0001")
    assert rec is not None and "Paris" in rec["text"] and rec["f1"] == 1.0
    assert infra_b.memory.answer("agent_2", "q0001") is None      # private at C0
    hit = infra_b.memory.search("agent_8", "capital of France.", k=1)[0]
    assert hit["kind"] == "corpus" and hit["title"] == "Paris"    # re-seeded


def test_resume_continues_to_a_larger_max_rounds(tmp_path):
    _, _, sched = build("C7", {}, tmp_path, max_rounds=2, checkpoint_every=10)
    sched.run()
    _, summary = resume("C7", {}, tmp_path, tmp_path / "checkpoint_0002.json",
                        max_rounds=4, checkpoint_every=10)
    assert summary["rounds_used"] == 4
    lines = [json.loads(l) for l in _lines(tmp_path / "timeseries.jsonl")]
    assert [s["round"] for s in lines] == [1, 2, 3, 4]
    assert (tmp_path / "checkpoint_0004.json").exists()
