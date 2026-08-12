"""v6 end-to-end scripted flows: the claim -> search -> deliver pipeline over
single questions, the peer-assist round trip that replaces subcontracting,
release_question as the hand-off mechanism, C1's board monopoly, C5's star
topology, C2's shared memory against its C0 control (corpus identical
everywhere, notes/answers private), and the per-round timeseries contract.
Deterministic, no LLM.
"""
import json
import random

import pytest
from fixtures import (DEMO_CORPUS, HashEmbedding, demo_bank,
                      demo_corpus_embeddings)

from ca.actions import permission_error
from ca.agent import Agent, ScriptedPolicy
from ca.bank import Question, QuestionBank
from ca.config import CONFIGS, ExperimentConfig
from ca.infra import Infra
from ca.metrics import compute_metrics
from ca.recorder import Recorder
from ca.scheduler import Scheduler

IDLE = ("list_agents", {})     # a turn that touches no state


def paris_bank(n=1) -> QuestionBank:
    """n easy questions with the same gold answer."""
    return QuestionBank([
        Question(f"q{i:04d}", "capital of France?", ["Paris"], "2hop", 100, "k00")
        for i in range(1, n + 1)])


def build(level, scripts, tmp_path, bank=None, in_tokens=10, out_tokens=5,
          max_rounds=10, **cfg_kw):
    cfg = ExperimentConfig(level=CONFIGS[level], seed=7,
                           max_rounds=max_rounds, **cfg_kw)
    infra = Infra(cfg, bank or paris_bank(1), corpus=DEMO_CORPUS,
                  corpus_embeddings=demo_corpus_embeddings(),
                  embedding_function=HashEmbedding())
    agents = [Agent(a, cfg, infra, ScriptedPolicy(scripts.get(a, []), in_tokens=in_tokens,
                                                  out_tokens=out_tokens))
              for a in infra.agent_ids]
    return infra, Scheduler(infra, agents, cfg, Recorder(str(tmp_path)),
                            random.Random(cfg.seed))


def _trace(tmp_path):
    return [json.loads(l) for l in open(tmp_path / "trace.jsonl")]


def _results(trace, agent, action):
    return [e["result"] for e in trace if e["agent"] == agent and e["action"] == action]


# ---------------- the solo pipeline (C7) ----------------

def test_solo_answer_flow_C7(tmp_path):
    scripts = {"agent_1": [
        ("list_questions", {}),
        ("claim_question", {"qid": "q0001"}),
        ("memory_search", {"query": "capital of France"}),
        ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
    ]}
    infra, sched = build("C7", scripts, tmp_path)
    summary = sched.run()
    assert summary["deliveries"][0]["f1"] == 1.0
    assert summary["remaining_units"] == 0
    # solving turns: memory_search + deliver_work = 2 * 15 tokens
    assert summary["tokens"]["agent_1"]["solving"] == 30
    # admin turns: list_questions + claim_question = 2 * 15 tokens
    assert summary["tokens"]["agent_1"]["admin"] == 30
    trace = _trace(tmp_path)
    assert len(trace) >= 4
    # the corpus answered the search
    assert "[Paris] Paris is the capital of France." in \
        _results(trace, "agent_1", "memory_search")[0]
    m = compute_metrics(summary)
    assert m["total_f1"] == 1.0 and m["n_answered"] == 1
    assert m["demand_absorbed"] == 1.0
    assert m["coordination_overhead"] == pytest.approx(0.5)
    assert m["n_messages"] == 0


def test_two_agents_absorb_two_questions(tmp_path):
    scripts = {
        "agent_1": [("claim_question", {"qid": "q0001"}),
                    ("deliver_work", {"target_id": "q0001", "content": "Paris"})],
        "agent_2": [("claim_question", {"qid": "q0002"}),
                    ("deliver_work", {"target_id": "q0002", "content": "Paris"})],
    }
    infra, sched = build("C0", scripts, tmp_path, bank=paris_bank(2), max_rounds=3)
    summary = sched.run()
    assert {d["agent"] for d in summary["deliveries"]} == {"agent_1", "agent_2"}
    assert infra.board.open_questions() == []
    assert compute_metrics(summary)["demand_absorbed"] == 1.0


def test_max_rounds_stops_a_run_that_answers_nothing(tmp_path):
    infra, sched = build("C7", {}, tmp_path)
    summary = sched.run()
    assert summary["rounds_used"] == 10
    assert summary["deliveries"] == [] and summary["remaining_units"] == 1


# ---------------- cooperation over chat ----------------

def test_peer_assist_round_trip_C0(tmp_path):
    """The behaviour that replaces subcontracting: agent_1 holds a question it
    does not search for itself, asks agent_2, agent_2 answers out of its own
    memory, and agent_1 delivers a correct answer."""
    scripts = {
        "agent_1": [
            ("claim_question", {"qid": "q0001"}),
            ("send_message", {"to": "agent_2", "text": "what is the capital of France?"}),
            IDLE,
            IDLE,
            ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
        ],
        "agent_2": [
            IDLE,
            IDLE,
            ("memory_search", {"query": "capital of France"}),
            ("send_message", {"to": "agent_1", "text": "the capital of France is Paris"}),
        ],
    }
    infra, sched = build("C0", scripts, tmp_path, max_rounds=5)
    summary = sched.run()
    trace = _trace(tmp_path)

    # agent_2 answered out of its own (corpus-seeded) memory
    assert "[Paris] Paris is the capital of France." in \
        _results(trace, "agent_2", "memory_search")[0]
    assert any("capital of France is Paris" in m.text
               for m in infra.chat.history("agent_1", "agent_2"))
    # agent_1 never searched: its only solving turn was the delivery
    assert summary["tokens"]["agent_1"]["solving"] == 15
    assert summary["deliveries"] == [{
        "qid": "q0001", "agent": "agent_1", "submitted": "Paris", "f1": 1.0,
        "em": 1.0, "round": 5, "price": 100, "difficulty": "2hop", "topic": "k00"}]
    m = compute_metrics(summary)
    assert m["n_messages"] == 2 and m["messages_per_answer"] == 2.0


def test_release_hands_a_question_to_another_agent(tmp_path):
    """release_question is the only hand-off left: agent_1 gives up, agent_2
    picks the question up and answers it."""
    scripts = {
        "agent_1": [
            ("claim_question", {"qid": "q0001"}),
            ("release_question", {"qid": "q0001"}),
            IDLE,
        ],
        "agent_2": [
            IDLE,
            IDLE,
            ("claim_question", {"qid": "q0001"}),
            ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
        ],
    }
    infra, sched = build("C0", scripts, tmp_path, max_rounds=4)
    summary = sched.run()
    trace = _trace(tmp_path)
    assert _results(trace, "agent_1", "release_question") == \
        ["released q0001; it is open on the board again for any agent"]
    assert not _results(trace, "agent_2", "claim_question")[0].startswith("ERROR")
    assert [d["agent"] for d in summary["deliveries"]] == ["agent_2"]
    # the releaser keeps its strike, the taker spends one of its own
    assert infra.board.strikes[("q0001", "agent_1")] == 1
    assert infra.board.strikes[("q0001", "agent_2")] == 1


# ---------------- C1: the board monopoly ----------------

def test_board_monopoly_C1(tmp_path):
    """Only the hub may touch the board; a worker's board actions all error.
    The worker contributes the only way it can -- by answering the hub's
    question over chat -- and the hub delivers."""
    scripts = {
        "hub": [
            ("claim_question", {"qid": "q0001"}),
            ("send_message", {"to": "agent_1", "text": "what is the capital of France?"}),
            IDLE,
            IDLE,
            ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
        ],
        "agent_1": [
            ("list_questions", {}),                                    # r1: refused
            ("claim_question", {"qid": "q0001"}),                      # r2: refused
            ("memory_search", {"query": "capital of France"}),         # r3: allowed
            ("send_message", {"to": "hub", "text": "the capital of France is Paris"}),
            ("deliver_work", {"target_id": "q0001", "content": "Paris"}),  # r5: refused
        ],
    }
    infra, sched = build("C1", scripts, tmp_path, max_rounds=5)
    summary = sched.run()
    trace = _trace(tmp_path)

    for action in ("list_questions", "claim_question", "deliver_work"):
        out = _results(trace, "agent_1", action)[0]
        assert out.startswith("ERROR") and "only the hub agent" in out, action
    assert not _results(trace, "agent_1", "memory_search")[0].startswith("ERROR")
    assert not _results(trace, "hub", "claim_question")[0].startswith("ERROR")
    assert [d["agent"] for d in summary["deliveries"]] == ["hub"]
    assert summary["deliveries"][0]["f1"] == 1.0
    assert permission_error(infra, "agent_1", "release_question",
                            {"qid": "q0001"}) is not None


# ---------------- C5: the star topology ----------------

def test_star_comms_C5_workers_reach_only_the_hub(tmp_path):
    """C5 flips comms topology ONLY: a worker may message and read only the
    hub, while its board access is untouched."""
    scripts = {
        "agent_1": [
            ("send_message", {"to": "agent_2", "text": "hello"}),      # r1: refused
            ("read_chat", {"with_agent": "agent_2"}),                  # r2: refused
            ("send_message", {"to": "hub", "text": "hello"}),          # r3: allowed
            ("claim_question", {"qid": "q0001"}),                      # r4: allowed
            ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
        ],
        "hub": [IDLE, IDLE, IDLE,
                ("send_message", {"to": "agent_2", "text": "relayed"})],  # hub may reach all
    }
    infra, sched = build("C5", scripts, tmp_path, max_rounds=5)
    summary = sched.run()
    trace = _trace(tmp_path)

    for action in ("send_message", "read_chat"):
        out = _results(trace, "agent_1", action)[0]
        assert out.startswith("ERROR") and "only interact with the hub" in out, action
    assert _results(trace, "agent_1", "send_message")[1] == "sent to hub"
    assert not _results(trace, "hub", "send_message")[0].startswith("ERROR")
    # the board is untouched at C5
    assert [d["agent"] for d in summary["deliveries"]] == ["agent_1"]


# ---------------- claim expiry and the two-strike rule ----------------

def test_expired_claim_returns_the_question_but_keeps_the_strike(tmp_path):
    """A claim nobody delivers must not destroy demand: the question goes back
    into the pool for anyone else, while the hoarder has spent one of its two
    attempts. After the second, the question is closed to it for good. The TTL
    (and with it the strike rule) is opt-in: cfg.claim_ttl defaults to None."""
    scripts = {
        "agent_1": [
            ("claim_question", {"qid": "q0001"}),   # r1: claimed, then idles
            IDLE, IDLE, IDLE,
            ("claim_question", {"qid": "q0001"}),   # r5: strike 2 (question is back)
            IDLE, IDLE, IDLE,
            ("claim_question", {"qid": "q0001"}),   # r9: refused, two strikes
        ],
    }
    infra, sched = build("C0", scripts, tmp_path, claim_ttl=2, max_rounds=9)
    sched.run()
    claims = _results(_trace(tmp_path), "agent_1", "claim_question")
    assert not claims[0].startswith("ERROR")
    assert not claims[1].startswith("ERROR")        # the expired question was reusable
    assert claims[2].startswith("ERROR") and "twice" in claims[2]
    assert infra.board.strikes[("q0001", "agent_1")] == 2


def test_claims_never_expire_without_a_ttl(tmp_path):
    """cfg.claim_ttl=None (the default): the scheduler never expires claims, so
    a question claimed in round 1 is still held at the end of the run."""
    scripts = {"agent_1": [("claim_question", {"qid": "q0001"})]}
    infra, sched = build("C0", scripts, tmp_path, max_rounds=5)
    sched.run()
    assert infra.board.active["q0001"].agent == "agent_1"
    assert infra.board.strikes[("q0001", "agent_1")] == 1


def test_adversarial_scripted(tmp_path):
    """Every known exploit attempt is refused, and the run still terminates."""
    scripts = {
        "agent_1": [
            ("send_message", {"to": "agent_99", "text": "hi"}),        # unknown recipient
            ("release_question", {"qid": "q0001"}),                    # holds no claim
        ],
        "agent_3": [
            ("claim_question", {"qid": "q0001"}),                      # claims, then idles
        ],
        "agent_4": [
            ("claim_question", {"qid": "q0002"}),
            ("deliver_work", {"target_id": "q0002", "content": "Paris"}),
            ("deliver_work", {"target_id": "q0002", "content": "Paris"}),  # no claim left
        ],
        "agent_5": [
            IDLE,
            ("claim_question", {"qid": "q0002"}),                      # already closed
            ("deliver_work", {"target_id": "q0001", "content": "Paris"}),  # never claimed it
        ],
    }
    infra, sched = build("C0", scripts, tmp_path, bank=paris_bank(2), claim_ttl=2)
    summary = sched.run()
    trace = _trace(tmp_path)

    assert _results(trace, "agent_1", "send_message")[0].startswith("ERROR")
    assert _results(trace, "agent_1", "release_question")[0].startswith("ERROR")
    delivers = _results(trace, "agent_4", "deliver_work")
    assert not delivers[0].startswith("ERROR") and delivers[1].startswith("ERROR")
    assert _results(trace, "agent_5", "claim_question")[0].startswith("ERROR")
    assert _results(trace, "agent_5", "deliver_work")[0].startswith("ERROR")

    # the hoarded claim was returned to the pool; the answered question stays closed
    assert [q.qid for q in infra.board.open_questions()] == ["q0001"]
    assert "q0001" not in infra.board.active
    assert [r.agent for r in infra.board.results] == ["agent_4"]
    assert summary["rounds_used"] == 10


# ---------------- C2 shared memory vs its C0 control ----------------

def _cross_agent_scripts():
    """agent_1 answers q0001; agent_2 then searches its memory for that very
    answer. Under shared memory the graded answer is there; under private
    memory only the (identical) corpus is."""
    return {
        "agent_1": [
            ("claim_question", {"qid": "q0001"}),
            ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
            IDLE,
        ],
        "agent_2": [
            IDLE,
            IDLE,
            ("memory_search", {"query": 'capital of France? -> "Paris"'}),   # r3
        ],
    }


def test_shared_memory_reuse_across_agents_C2(tmp_path):
    """The load-bearing C2 control: agent_1 answers q0001 and the graded
    answer lands in the ONE shared store, where agent_2's search finds it
    without agent_2 ever doing any work of its own."""
    infra, sched = build("C2", _cross_agent_scripts(), tmp_path,
                         bank=paris_bank(2), max_rounds=3)
    sched.run()
    out = _results(_trace(tmp_path), "agent_2", "memory_search")[0]
    assert '[q0001] capital of France? -> "Paris" (F1 1.00)' in out
    assert infra.memory.answer("agent_2", "q0001")["f1"] == 1.0


def test_private_memory_does_not_leak_across_agents_C0(tmp_path):
    """The C0 control for the test above: agent_2's private store holds the
    IDENTICAL corpus (the Paris paragraph shows up), yet agent_1's graded
    answer is invisible to it."""
    infra, sched = build("C0", _cross_agent_scripts(), tmp_path,
                         bank=paris_bank(2), max_rounds=3)
    sched.run()
    out = _results(_trace(tmp_path), "agent_2", "memory_search")[0]
    assert "[Paris] Paris is the capital of France." in out      # corpus: identical at birth
    assert "(F1 1.00)" not in out                                # the answer: private
    assert infra.memory.answer("agent_2", "q0001") is None
    assert infra.memory.answer("agent_1", "q0001")["f1"] == 1.0


def test_reuse_and_improvement_show_up_in_the_metrics(tmp_path):
    """A stored answer makes the later claim a memory hit, and the graded
    delivery reports beating the stored attempt: memory_hit_rate and
    improvement_rate both move."""
    scripts = {"agent_1": [
        ("claim_question", {"qid": "q0002"}),      # r1: bare claim, no hit
        ("claim_question", {"qid": "q0001"}),      # r2: hit (pre-stored answer)
        ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
    ]}
    infra, sched = build("C7", scripts, tmp_path, bank=paris_bank(2), max_rounds=5)
    infra.memory.write("agent_1", '[q0001] capital of France? -> "Lyon" (F1 0.00)',
                       kind="answer", qid="q0001", f1=0.0)
    summary = sched.run()
    trace = _trace(tmp_path)
    assert "LOW QUALITY" in _results(trace, "agent_1", "claim_question")[1]
    assert "IMPROVED on your stored F1 0.00" in _results(trace, "agent_1", "deliver_work")[0]
    m = compute_metrics(summary)
    assert m["memory_hit_rate"] == pytest.approx(1 / 2)
    assert m["improvement_rate"] == pytest.approx(1.0)
    assert m["answers_in_memory_total"] == 2       # append-only: both attempts


# ---------------- timeseries / scheduler contracts ----------------

def test_timeseries_one_cumulative_snapshot_per_round(tmp_path):
    """Every round appends one cumulative system snapshot to timeseries.jsonl;
    the last line agrees with summary.json / metrics."""
    scripts = {"agent_1": [
        ("list_questions", {}),
        ("claim_question", {"qid": "q0001"}),
        ("memory_search", {"query": "capital of France"}),
        ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
        ("claim_question", {"qid": "q0002"}),
        ("deliver_work", {"target_id": "q0002", "content": "Paris"}),
    ],
        "agent_2": [("send_message", {"to": "agent_1", "text": "good luck"})],
    }
    infra, sched = build("C0", scripts, tmp_path, bank=paris_bank(2), max_rounds=6)
    summary = sched.run()
    lines = [json.loads(l) for l in open(tmp_path / "timeseries.jsonl")]

    assert len(lines) == summary["rounds_used"] == 6
    assert [s["round"] for s in lines] == list(range(1, 7))
    roster = set(infra.agent_ids)
    for s in lines:
        for key in ("tokens", "answered", "memory"):
            assert set(s[key]) == roster, key
    # cumulative counters never go down tick over tick
    for key in ("solving_total", "admin_total", "n_answered", "total_f1",
                "n_messages", "n_claims", "n_memory_hits", "demand_absorbed"):
        vals = [s[key] for s in lines]
        assert vals == sorted(vals), key
    assert [s["board"]["closed"] for s in lines] == [0, 0, 0, 1, 1, 2]
    # a unit leaves the pool when its question closes, not when it is claimed
    assert [s["remaining_units"] for s in lines] == [2, 2, 2, 1, 1, 0]

    last = lines[-1]
    m = compute_metrics(summary)
    assert last["tokens"] == summary["tokens"]
    assert last["memory"] == summary["memory"]
    assert last["n_messages"] == summary["n_messages"] == 1
    assert last["total_f1"] == pytest.approx(m["total_f1"])
    assert last["total_em"] == pytest.approx(m["total_em"])
    assert last["n_answered"] == m["n_answered"]
    assert last["coordination_overhead"] == pytest.approx(m["coordination_overhead"])
    assert last["demand_absorbed"] == pytest.approx(m["demand_absorbed"])
    assert last["n_claims"] == m["n_claims"]
    assert last["memory_hit_rate"] == pytest.approx(m["memory_hit_rate"])
    assert last["improvement_rate"] == pytest.approx(m["improvement_rate"])
    assert last["answers_in_memory_total"] == m["answers_in_memory_total"]


def test_timeseries_line_count_matches_max_rounds_run(tmp_path):
    infra, sched = build("C7", {}, tmp_path)  # nobody answers -> full 10 rounds
    summary = sched.run()
    lines = [json.loads(l) for l in open(tmp_path / "timeseries.jsonl")]
    assert len(lines) == summary["rounds_used"] == 10
    assert lines[-1]["board"] == {"open": 1, "active_claims": 0, "closed": 0}


def test_summary_is_written_even_when_a_turn_crashes(tmp_path):
    """A crash mid-run must not cost us the whole run's data."""
    class BoomPolicy:
        def decide(self, system, context, tools):
            raise RuntimeError("policy exploded")

    infra, sched = build("C7", {}, tmp_path)
    sched.agents[0].policy = BoomPolicy()
    with pytest.raises(RuntimeError):
        sched.run()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["level"] == "C7"
    assert summary["deliveries"] == []


def test_hub_turns_per_round_knob(tmp_path):
    infra, sched = build("C1", {}, tmp_path, hub_turns_per_round=3)
    sched.cfg.max_rounds = 1
    sched.run()
    round1 = [e for e in _trace(tmp_path) if e["round"] == 1]
    assert sum(1 for e in round1 if e["agent"] == "hub") == 3
    assert sum(1 for e in round1 if e["agent"] == "agent_1") == 1


def test_solo_turns_per_round_knob(tmp_path):
    infra, sched = build("C7", {}, tmp_path, solo_turns_per_round=8)
    sched.cfg.max_rounds = 2
    summary = sched.run()
    round1 = [e for e in _trace(tmp_path) if e["round"] == 1]
    assert sum(1 for e in round1 if e["agent"] == "agent_1") == 8
    assert summary["rounds_used"] == 2


def test_specialization_is_computable_from_a_real_run(tmp_path):
    """Topics never reach the agents, but every delivery row carries one, so
    the metric works straight off the summary."""
    scripts = {
        "agent_1": [("claim_question", {"qid": "q0001"}),        # k01
                    ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
                    ("claim_question", {"qid": "q0003"}),        # k07
                    ("deliver_work", {"target_id": "q0003", "content": "4"})],
        "agent_2": [("claim_question", {"qid": "q0005"}),        # k02 only
                    ("deliver_work", {"target_id": "q0005", "content": "sedimentary"})],
    }
    infra, sched = build("C0", scripts, tmp_path, bank=demo_bank(), max_rounds=4)
    summary = sched.run()
    m = compute_metrics(summary)
    assert m["specialization"]["agent_1"] == pytest.approx(0.5)
    assert m["specialization"]["agent_2"] == pytest.approx(1.0)   # k02 alone
    assert m["mean_specialization"] == pytest.approx(0.75)
    assert m["demand_absorbed"] == pytest.approx(3 / 5)
