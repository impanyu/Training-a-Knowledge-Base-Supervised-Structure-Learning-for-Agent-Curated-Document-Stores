import json

import pytest
from fixtures import demo_bank, demo_domains

from ca.stream import QuestionStream, StreamError


def make(seed=0, n_agents=2, rate=0.5, bank=None):
    bank = bank or demo_bank()
    assignment, _ = demo_domains(bank, n_agents)
    return QuestionStream(bank, n_agents, seed, rate, assignment)


def drain(stream, max_rounds=200):
    """{round: [(qid, agent)]} until the order is exhausted."""
    out = {}
    for r in range(1, max_rounds + 1):
        arrivals = stream.tick(r)
        if arrivals:
            out[r] = arrivals
        if stream.pos >= len(stream.order):
            break
    return out


# ---------------- routing ----------------

def test_routing_sends_each_cluster_to_its_owner():
    s = make()
    # fixtures: cluster 0 = arithmetic -> agent_1, cluster 1 = France -> agent_2
    assert {s.routing[q] for q in ("q0005", "q0006", "q0007", "q0008")} == {"agent_1"}
    assert {s.routing[q] for q in ("q0001", "q0002", "q0003", "q0004")} == {"agent_2"}


def test_assignment_must_cover_the_bank_and_stay_in_range():
    bank = demo_bank()
    assignment, _ = demo_domains(bank, 2)
    with pytest.raises(ValueError, match="misses"):
        QuestionStream(bank, 2, 0, 0.5, {k: v for k, v in assignment.items()
                                         if k != "q0003"})
    bad = dict(assignment, q0001=5)
    with pytest.raises(ValueError, match="outside"):
        QuestionStream(bank, 2, 0, 0.5, bad)


# ---------------- determinism ----------------

def test_same_bank_seed_and_n_give_an_identical_schedule():
    a, b = make(seed=3), make(seed=3)
    assert a.order == b.order and a.routing == b.routing
    assert drain(a) == drain(b)


def test_different_seeds_give_different_orders_but_the_same_routing():
    a, b = make(seed=0), make(seed=1)
    assert a.routing == b.routing              # routing is the cluster cache
    assert a.order != b.order


def test_the_schedule_is_arm_invariant_because_the_stream_owns_its_rng():
    """Interleaving OTHER consumers of randomness (the scheduler's shuffles,
    which differ between arms) must not move a single arrival."""
    import random
    a, b = make(), make()
    plain = drain(a)
    perturbed = {}
    scheduler_rng = random.Random(0)
    for r in range(1, 200):
        scheduler_rng.random()                 # a different arm's turn shuffle
        arrivals = b.tick(r)
        if arrivals:
            perturbed[r] = arrivals
        if b.pos >= len(b.order):
            break
    assert plain == perturbed


# ---------------- arrivals ----------------

def test_arrival_counts_are_poisson_draws_per_round():
    got = drain(make())
    # deterministic for (demo bank, seed 0, N=2, rate 0.5); pinned on purpose
    assert got == {
        2: [("q0005", "agent_1")],
        5: [("q0002", "agent_2")],
        7: [("q0006", "agent_1")],
        9: [("q0003", "agent_2"), ("q0001", "agent_2"),
            ("q0004", "agent_2"), ("q0008", "agent_1")],
        10: [("q0007", "agent_1")],
    }


def test_arrivals_set_pending_with_the_arrival_round():
    s = make()
    got = drain(s)
    assert s.pending["q0005"] == ("agent_1", 2)
    assert s.pending["q0007"] == ("agent_1", 10)
    assert len(s.pending) == 8 and s.pos == 8
    assert got[9][0][0] == "q0003"


def test_tick_is_idempotent_per_round():
    s = make()
    assert s.tick(1) == []
    assert s.tick(2) == [("q0005", "agent_1")]
    assert s.tick(2) == []                     # a repeat draws nothing
    assert s.tick(1) == []                     # and neither does the past
    assert s.pos == 1


def test_exhaustion_stops_arrivals_for_good():
    s = make(rate=10.0)
    first = s.tick(1)
    assert [q for q, _ in first] == s.order    # rate 10: everything at once
    for r in range(2, 6):
        assert s.tick(r) == []


def test_zero_rate_never_arrives_anything():
    s = make(rate=0.0)
    assert drain(s, max_rounds=50) == {}
    assert s.pos == 0


# ---------------- delivery ----------------

def test_deliver_grades_and_records_latency():
    s = make()
    drain(s)
    r = s.deliver("agent_1", "q0005", "4", 12)
    assert r.f1 == 1.0 and r.em == 1.0
    assert (r.round_in, r.round_out, r.latency) == (2, 12, 10)
    assert "q0005" not in s.pending and "q0005" in s.closed


def test_deliver_by_a_foreign_agent_is_refused():
    s = make()
    drain(s)
    with pytest.raises(StreamError, match="assigned to agent_1"):
        s.deliver("agent_2", "q0005", "4", 12)
    assert "q0005" in s.pending                # nothing consumed


def test_second_delivery_is_refused():
    s = make()
    drain(s)
    s.deliver("agent_1", "q0005", "4", 12)
    with pytest.raises(StreamError, match="already been answered"):
        s.deliver("agent_1", "q0005", "four", 13)
    assert len(s.results) == 1


def test_delivering_a_question_that_never_arrived_is_refused():
    s = make(rate=0.0)
    with pytest.raises(StreamError, match="not an open external question"):
        s.deliver("agent_1", "q0005", "4", 1)


def test_delivering_an_unknown_qid_names_near_ids():
    from ca.bank import BankError
    s = make()
    with pytest.raises(BankError, match="q0008"):
        s.deliver("agent_1", "q9999", "4", 1)


def test_all_done_needs_exhaustion_and_an_empty_pending_set():
    s = make(rate=10.0)
    s.tick(1)
    assert not s.all_done()
    for qid in list(s.pending):
        s.deliver(s.routing[qid], qid, "x", 2)
    assert s.all_done()


def test_results_json_carries_the_hidden_metadata():
    s = make()
    drain(s)
    s.deliver("agent_1", "q0005", "four", 12)
    (row,) = s.results_json()
    assert row == {"qid": "q0005", "agent": "agent_1", "submitted": "four",
                   "f1": 1.0, "em": 1.0, "round_in": 2, "round_out": 12,
                   "latency": 10, "topic": "k07", "difficulty": "2hop"}


# ---------------- checkpoint ----------------

def test_state_roundtrip_resumes_the_exact_schedule():
    a, b = make(), make()
    for r in range(1, 6):
        a.tick(r)
        b.tick(r)
    a.deliver("agent_1", "q0005", "4", 5)
    state = json.loads(json.dumps(a.to_state()))

    fresh = make()                             # rebuilt from CLI args, then restored
    fresh.from_state(state)
    assert fresh.pending == a.pending and fresh.closed == a.closed
    assert fresh.results[0].latency == 3
    for r in range(6, 20):                     # continuation == never stopped
        assert fresh.tick(r) == b.tick(r)
    assert fresh.tick(5) == []                 # last_tick survives too
