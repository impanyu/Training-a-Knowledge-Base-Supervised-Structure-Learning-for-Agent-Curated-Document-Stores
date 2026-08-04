"""T29: full-state checkpoint every N rounds + resume."""
import json
import random

import pytest
from fixtures import HashEmbedding, demo_bank, demo_infra

from ca import checkpoint
from ca.agent import Agent, ScriptedPolicy
from ca.config import CONFIGS, ExperimentConfig
from ca.infra import Infra
from ca.memory import FifoMemory, GoalStack
from ca.recorder import Recorder
from ca.retrieval import KeywordBackend
from ca.scheduler import Scheduler

DOCS = [{"title": "Paris", "text": "Paris is the capital of France."}]


def build(level, scripts, out_dir, seed=7, capital=4000, max_rounds=6, **cfg_kw):
    cfg = ExperimentConfig(level=CONFIGS[level], seed=seed, seed_capital_total=capital,
                           max_rounds=max_rounds, **cfg_kw)
    infra = Infra(cfg, demo_bank(), retriever=KeywordBackend(DOCS),
                  embedding_function=HashEmbedding())
    agents = [Agent(a, cfg, infra,
                    ScriptedPolicy(scripts.get(a, []), in_tokens=10, out_tokens=5))
              for a in infra.agent_ids]
    sched = Scheduler(infra, agents, cfg, Recorder(str(out_dir)), random.Random(seed))
    return infra, agents, sched


def resume(level, scripts, out_dir, ck_path, seed=7, capital=4000, max_rounds=6, **cfg_kw):
    """Rebuild everything fresh from 'CLI args', then restore the checkpoint --
    the exact shape of the runner's --resume path."""
    cfg = ExperimentConfig(level=CONFIGS[level], seed=seed, seed_capital_total=capital,
                           max_rounds=max_rounds, **cfg_kw)
    infra = Infra(cfg, demo_bank(), retriever=KeywordBackend(DOCS),
                  embedding_function=HashEmbedding())
    agents = [Agent(a, cfg, infra,
                    ScriptedPolicy(scripts.get(a, []), in_tokens=10, out_tokens=5))
              for a in infra.agent_ids]
    recorder = Recorder(str(out_dir), append=True)
    rng = random.Random(seed)
    with open(ck_path) as f:
        state = json.load(f)
    checkpoint.validate(state, cfg)
    checkpoint.restore(state, infra, agents, recorder, rng)
    sched = Scheduler(infra, agents, cfg, recorder, rng)
    return infra, sched.run(start_round=state["round"] + 1)


# 6-round C0 workload that touches every stateful subsystem, with activity on
# both sides of a round-3 checkpoint boundary (loan interest, an open escrowed
# contract, chat, scratchpads, memories all cross it).
SCRIPTS = {
    "agent_1": [
        ("list_jobs", {}),
        ("claim_job", {"jid": "j0001"}),
        ("retrieve", {"query": "capital of France"}),
        ("deliver_work", {"target_id": "j0001",
                          "content": '{"q0001": "Paris", "q0002": "Loire", "q0003": "4"}'}),
        ("claim_job", {"jid": "j0002"}),      # q0003 comes back with its stored answer
        ("deliver_work", {"target_id": "j0002", "content": '{"q0003": "4", "q0004": "6"}'}),
    ],
    "agent_2": [
        ("claim_job", {"jid": "j0003"}),
        ("propose_loan", {"to": "agent_3", "amount": 100}),
        ("work_on", {"question_id": "q0005", "thought": "chalk is sedimentary"}),
        ("check_balance", {}),
        ("deliver_work", {"target_id": "j0003", "content": '{"q0005": "sedimentary"}'}),
    ],
    "agent_3": [
        ("check_balance", {}),
        ("check_balance", {}),
        ("accept_loan", {"loan_id": "n0001"}),
        ("memory_write", {"content": "agent_2 owes me 100 tokens"}),
        ("memory_search", {"query": "tokens owed"}),
    ],
    "agent_4": [
        ("propose_contract", {"to": "agent_5", "task": "q0002", "price": 40}),
        ("push_goal", {"note": "await c0001 delivery"}),
        ("check_balance", {}),
        ("check_balance", {}),
        ("read_chat", {"with_agent": "agent_5"}),
    ],
    "agent_5": [
        ("check_balance", {}),
        ("accept_contract", {"contract_id": "c0001"}),
        ("retrieve", {"query": "capital of France"}),
        ("work_on", {"question_id": "q0002", "thought": "the Loire is longest"}),
        ("deliver_work", {"target_id": "c0001", "content": "Loire"}),
    ],
    "agent_6": [
        ("memory_write", {"content": "q0003 looks like easy arithmetic"}),
        ("check_balance", {}),
        ("check_balance", {}),
        ("send_message", {"to": "agent_7", "text": "anything left on the board?"}),
    ],
    "agent_7": [
        ("push_goal", {"note": "find a niche"}),
        ("list_jobs", {}),
        ("pop_goal", {}),
        ("check_balance", {}),
        ("read_chat", {"with_agent": "agent_6"}),
    ],
    "agent_8": [
        ("list_jobs", {}),
        ("check_balance", {}),
        ("check_balance", {}),
    ],
}


def _lines(path):
    with open(path) as f:
        return f.read().splitlines()


# ---------- subsystem round-trips ----------


def test_ledger_state_roundtrip():
    infra = demo_infra("C0", capital=800)
    infra.ledger.mint("agent_1", 50)
    infra.ledger.burn("agent_2", 30)
    infra.ledger.lock_escrow("c0001", "agent_3", 25)
    state = json.loads(json.dumps(infra.ledger.to_state()))
    fresh = demo_infra("C0", capital=800)
    fresh.ledger.from_state(state)
    assert fresh.ledger.balances == infra.ledger.balances
    assert fresh.ledger.escrow == {"c0001": 25}
    assert fresh.ledger.minted == 50 and fresh.ledger.burned == 30
    assert fresh.ledger.conservation_ok()


def test_board_state_roundtrip():
    infra = demo_infra("C0")
    infra.board.claim("agent_1", "j0001", round_no=2)
    infra.board.claim("agent_2", "j0003", round_no=3)
    infra.board.deliver("agent_2", "j0003", {"q0005": "sedimentary"})
    state = json.loads(json.dumps(infra.board.to_state()))
    fresh = demo_infra("C0")
    fresh.board.from_state(state)
    assert fresh.board.open_jobs() == infra.board.open_jobs()
    assert fresh.board.active["j0001"].agent == "agent_1"
    assert fresh.board.active["j0001"].round == 2
    assert "j0003" not in fresh.board.active
    assert fresh.board.strikes == infra.board.strikes
    assert fresh.board.results_json() == infra.board.results_json()


def test_contracts_state_roundtrip_id_counter_continues():
    infra = demo_infra("C0")
    infra.contracts.propose("agent_1", "agent_2", "some work", 40)
    infra.contracts.accept("agent_2", "c0001")
    c = infra.contracts.propose("agent_3", "agent_4", "q0002", 10)
    c.qid = "q0002"
    state = json.loads(json.dumps(infra.contracts.to_state()))
    fresh = demo_infra("C0")
    fresh.contracts.from_state(state)
    got = fresh.contracts.get("c0001")
    assert got.status == "accepted" and got.price == 40 and got.contractor == "agent_2"
    assert fresh.contracts.get("c0002").qid == "q0002"
    assert fresh.contracts.propose("agent_1", "agent_5", "new", 5).cid == "c0003"


def test_loans_state_roundtrip_id_counter_continues():
    infra = demo_infra("C0")
    infra.loans.propose("agent_1", "agent_2", 100)
    infra.loans.accept("agent_2", "n0001")
    infra.loans.interest_tick()
    state = json.loads(json.dumps(infra.loans.to_state()))
    fresh = demo_infra("C0")
    fresh.loans.from_state(state)
    loan = fresh.loans.get("n0001")
    assert loan.status == "active" and loan.principal == 100
    assert fresh.loans.total_interest_paid == 1
    assert fresh.loans.propose("agent_3", "agent_4", 5).lid == "n0002"


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
    infra = demo_infra("C2")                 # shared bucket
    infra.memory.write("agent_1", "paris facts")
    infra.memory.write("agent_2", '[q0003] 2+2? -> "4" (F1 1.00)',
                       kind="answer", qid="q0003", f1=1.0)
    state = json.loads(json.dumps(infra.memory.to_state()))
    fresh = demo_infra("C2")
    fresh.memory.from_state(state)
    assert fresh.memory.answer("agent_5", "q0003")["f1"] == 1.0
    assert fresh.memory.search("agent_5", "paris facts")[0]["text"] == "paris facts"
    assert fresh.memory.n_answers("agent_1") == infra.memory.n_answers("agent_1")


def test_recorder_tallies_roundtrip_and_append_mode(tmp_path):
    rec = Recorder(str(tmp_path))
    rec.log({"round": 1, "agent": "agent_1", "action": "retrieve", "input": {},
             "result": "ok", "category": "solving", "tokens_in": 10, "tokens_out": 5,
             "balance_after": 0})
    rec.log({"round": 1, "agent": "agent_1", "action": "claim_job", "input": {},
             "result": 'claimed [j0001] 3 questions, reward 600\n'
                       '  q0001 capital of France? -- memory: stored answer "Paris" (F1 1.00)',
             "category": "admin", "tokens_in": 1, "tokens_out": 1, "balance_after": 0})
    state = json.loads(json.dumps(rec.to_state()))
    rec.close()

    rec2 = Recorder(str(tmp_path), append=True)
    rec2.from_state(state)
    assert rec2._tokens["agent_1"] == {"solving": 15, "admin": 2}
    assert rec2._memory["agent_1"] == {"n_claims": 1, "n_memory_hits": 1,
                                       "n_repeat_deliveries": 0, "n_improved": 0}
    rec2.log({"round": 2, "agent": "agent_1", "action": "check_balance", "input": {},
              "result": "0", "category": "admin", "tokens_in": 1, "tokens_out": 1,
              "balance_after": 0})
    rec2.close()
    assert len(_lines(tmp_path / "trace.jsonl")) == 3    # append, no rewrite


def test_rng_state_survives_json():
    rng = random.Random(7)
    rng.random()
    state = checkpoint.rng_state(rng)
    rng2 = random.Random(0)
    checkpoint.restore_rng(rng2, json.loads(json.dumps(state)))
    assert [rng2.random() for _ in range(5)] == [rng.random() for _ in range(5)]


# ---------- scheduler save discipline ----------


def test_checkpoint_files_every_n_and_at_final_round(tmp_path):
    _, _, sched = build("C7", {}, tmp_path, capital=1000, max_rounds=5,
                        checkpoint_every=2)
    sched.run()
    names = sorted(p.name for p in tmp_path.glob("checkpoint_*.json"))
    assert names == ["checkpoint_0002.json", "checkpoint_0004.json",
                     "checkpoint_0005.json"]
    ck = json.loads((tmp_path / "checkpoint_0004.json").read_text())
    assert ck["round"] == 4
    assert ck["config"]["level"] == "C7" and ck["config"]["seed"] == 7


def test_checkpoint_written_when_run_stops_early(tmp_path):
    """C7 solo agent absorbs the demo bank's whole 6-unit demand, so the run
    breaks on all_done() before max_rounds."""
    script = []
    for jid, answers in (("j0001", {"q0001": "Paris", "q0002": "Loire", "q0003": "4"}),
                         ("j0002", {"q0003": "4", "q0004": "6"}),
                         ("j0003", {"q0005": "sedimentary"})):
        script += [("claim_job", {"jid": jid}),
                   ("deliver_work", {"target_id": jid, "content": json.dumps(answers)})]
    _, _, sched = build("C7", {"agent_1": script}, tmp_path, capital=1000,
                        max_rounds=40, checkpoint_every=50)
    summary = sched.run()
    assert summary["rounds_used"] == 6         # 3 jobs x (claim + deliver)
    assert summary["remaining_units"] == 0
    assert (tmp_path / "checkpoint_0006.json").exists()


def test_validate_rejects_level_or_seed_mismatch(tmp_path):
    _, _, sched = build("C7", {}, tmp_path, capital=1000, max_rounds=2)
    sched.run()
    state = json.loads((tmp_path / "checkpoint_0002.json").read_text())
    cfg_ok = ExperimentConfig(level=CONFIGS["C7"], seed=7, seed_capital_total=1000)
    checkpoint.validate(state, cfg_ok)
    with pytest.raises(ValueError, match="level"):
        checkpoint.validate(state, ExperimentConfig(
            level=CONFIGS["C0"], seed=7, seed_capital_total=1000))
    with pytest.raises(ValueError, match="seed"):
        checkpoint.validate(state, ExperimentConfig(
            level=CONFIGS["C7"], seed=8, seed_capital_total=1000))


# ---------- the fidelity test ----------


def test_resume_reproduces_a_straight_run_exactly(tmp_path):
    """Run A: 6 rounds straight. Run B: 3 rounds, checkpoint, resume 4-6.
    Summary, timeseries and trace must be indistinguishable."""
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"

    _, _, sched_a = build("C0", SCRIPTS, dir_a, max_rounds=6)
    summary_a = sched_a.run()
    assert summary_a["rounds_used"] == 6
    assert summary_a["conservation_ok"] is True
    assert summary_a["n_contracts"] == 1 and summary_a["loans"]["n_active"] == 1
    # the workload really did exercise memory across the boundary
    assert summary_a["memory"]["agent_1"]["n_memory_hits"] == 1
    assert len(summary_a["deliveries"]) >= 3

    _, _, sched_b = build("C0", SCRIPTS, dir_b, max_rounds=3, checkpoint_every=3)
    sched_b.run()
    assert (dir_b / "checkpoint_0003.json").exists()

    # each C0 agent takes exactly one turn per round: entries 3+ remain
    rest = {a: s[3:] for a, s in SCRIPTS.items()}
    infra_b, summary_b = resume("C0", rest, dir_b, dir_b / "checkpoint_0003.json",
                                max_rounds=6, checkpoint_every=3)

    assert infra_b.ledger.conservation_ok()
    assert summary_b == summary_a
    assert _lines(dir_b / "timeseries.jsonl") == _lines(dir_a / "timeseries.jsonl")
    trace_a = _lines(dir_a / "trace.jsonl")
    trace_b = _lines(dir_b / "trace.jsonl")
    assert trace_b == trace_a                       # RNG restore => same shuffle
    order_a = [json.loads(l)["agent"] for l in trace_a]
    order_b = [json.loads(l)["agent"] for l in trace_b]
    assert order_a == order_b


def test_resume_restores_the_vector_memory_contents(tmp_path):
    """The store is rebuilt from the checkpoint, not merely re-created empty:
    an answer written before the boundary is still recallable after it."""
    dir_b = tmp_path / "b"
    _, _, sched = build("C0", SCRIPTS, dir_b, max_rounds=3, checkpoint_every=3)
    sched.run()
    infra_b, _ = resume("C0", {a: s[3:] for a, s in SCRIPTS.items()}, dir_b,
                        dir_b / "checkpoint_0003.json", max_rounds=4,
                        checkpoint_every=3)
    rec = infra_b.memory.answer("agent_1", "q0001")
    assert rec is not None and "Paris" in rec["text"] and rec["f1"] == 1.0
    assert infra_b.memory.answer("agent_2", "q0001") is None      # private at C0


def test_resume_continues_to_a_larger_max_rounds(tmp_path):
    _, _, sched = build("C7", {}, tmp_path, capital=1000, max_rounds=2,
                        checkpoint_every=10)
    sched.run()
    _, summary = resume("C7", {}, tmp_path, tmp_path / "checkpoint_0002.json",
                        capital=1000, max_rounds=4, checkpoint_every=10)
    assert summary["rounds_used"] == 4
    lines = [json.loads(l) for l in _lines(tmp_path / "timeseries.jsonl")]
    assert [s["round"] for s in lines] == [1, 2, 3, 4]
    assert (tmp_path / "checkpoint_0004.json").exists()
