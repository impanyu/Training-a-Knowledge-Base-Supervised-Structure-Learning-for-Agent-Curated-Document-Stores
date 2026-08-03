"""v4 end-to-end scripted flows: the flat claim -> solve -> deliver pipeline,
question-bound subcontracts, the loan lifecycle through bankruptcy, quota
expiry and the two-strike rule, C2's shared memory against its C0 control, and
the per-round timeseries contract. Deterministic, no LLM.

This file replaces the v1/v2/v3 e2e trio (test_e2e_scripted / _v2 / _v3);
every scenario intent they carried is preserved here in v4 terms.
"""
import json
import random

import pytest
from fixtures import HashEmbedding, demo_bank, demo_infra

from ca.actions import permission_error
from ca.agent import Agent, ScriptedPolicy
from ca.bank import Question, QuestionBank
from ca.config import CONFIGS, ExperimentConfig
from ca.context import render_turn
from ca.infra import Infra
from ca.metrics import compute_metrics
from ca.recorder import Recorder
from ca.retrieval import KeywordBackend
from ca.scheduler import Scheduler

DOCS = [{"title": "Paris", "text": "Paris is the capital of France."}]


def paris_bank(n=1, quota=1) -> QuestionBank:
    """n copies of one easy question, so payout arithmetic stays trivial."""
    return QuestionBank([
        Question(f"q{i:04d}", "capital of France?", ["Paris"], "2hop", 100, quota, "k00")
        for i in range(1, n + 1)])


def build(level, scripts, tmp_path, bank=None, seed_capital_total=1000,
          in_tokens=10, out_tokens=5, max_rounds=10, **cfg_kw):
    cfg = ExperimentConfig(level=CONFIGS[level], seed=7,
                           seed_capital_total=seed_capital_total,
                           max_rounds=max_rounds, **cfg_kw)
    infra = Infra(cfg, bank or paris_bank(1), retriever=KeywordBackend(DOCS),
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
        ("retrieve", {"query": "capital of France"}),
        ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
    ]}
    infra, sched = build("C7", scripts, tmp_path)
    summary = sched.run()
    assert summary["deliveries"][0]["f1"] == 1.0
    assert summary["deliveries"][0]["payout"] == 100
    assert summary["remaining_units"] == 0
    assert summary["conservation_ok"] is True
    # solving turns: retrieve + deliver_work = 2 * 15 tokens
    assert summary["tokens"]["agent_1"]["solving"] == 30
    # admin turns: list_questions + claim_question = 2 * 15 tokens
    assert summary["tokens"]["agent_1"]["admin"] == 30
    assert len(_trace(tmp_path)) >= 4
    m = compute_metrics(summary)
    assert m["total_f1"] == 1.0 and m["n_answered"] == 1
    assert m["demand_absorbed"] == 1.0
    assert m["coordination_overhead"] == pytest.approx(0.5)


def test_quota_lets_two_agents_be_paid_for_the_same_question(tmp_path):
    """The v4 repeat: one question with quota 2 is claimed and delivered by two
    different agents, and the WORLD pays both."""
    scripts = {
        "agent_1": [("claim_question", {"qid": "q0001"}),
                    ("deliver_work", {"target_id": "q0001", "content": "Paris"})],
        "agent_2": [("claim_question", {"qid": "q0001"}),
                    ("deliver_work", {"target_id": "q0001", "content": "Paris"})],
    }
    infra, sched = build("C0", scripts, tmp_path, bank=paris_bank(1, quota=2),
                         max_rounds=3)
    summary = sched.run()
    assert {d["agent"] for d in summary["deliveries"]} == {"agent_1", "agent_2"}
    assert sum(d["payout"] for d in summary["deliveries"]) == 200
    assert infra.board.remaining["q0001"] == 0
    assert compute_metrics(summary)["demand_absorbed"] == 1.0
    assert summary["conservation_ok"] is True


def test_max_rounds_stops_a_run_that_answers_nothing(tmp_path):
    infra, sched = build("C7", {}, tmp_path)
    summary = sched.run()
    assert summary["rounds_used"] == 10
    assert summary["deliveries"] == [] and summary["remaining_units"] == 1


# ---------------- subcontracting ----------------

def test_subcontract_flow_C1(tmp_path):
    """Hub claims (demand monopoly), hires agent_1 for free-text help,
    agent_1 delivers the contract, hub delivers the answer to the WORLD."""
    scripts = {
        "hub": [
            ("claim_question", {"qid": "q0001"}),
            ("propose_contract", {"to": "agent_1", "task": "find the capital of France",
                                  "price": 40}),
            ("check_balance", {}),                      # waits while agent_1 works
            ("check_balance", {}),
            ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
        ],
        "agent_1": [
            ("check_balance", {}),                      # waits for the hub's claim (r1)
            ("check_balance", {}),                      # waits for propose_contract (r2)
            ("accept_contract", {"contract_id": "c0001"}),
            ("retrieve", {"query": "capital of France"}),
            ("deliver_work", {"target_id": "c0001", "content": "The answer is Paris"}),
        ],
    }
    infra, sched = build("C1", scripts, tmp_path)
    summary = sched.run()
    assert summary["deliveries"][0]["f1"] == 1.0
    assert summary["conservation_ok"] is True
    # T17: EVERY turn bills 15 tokens (10 in + 5 out, fixed by ScriptedPolicy),
    # idle check_balance turns included. The board empties on the hub's round-5
    # delivery, so all 8 agents took 5 turns => 75 tokens burned each.
    assert summary["rounds_used"] == 5
    assert summary["balances"]["hub"] == 125 - 75 - 40 + 100
    assert summary["balances"]["agent_1"] == 125 - 75 + 40
    # workers cannot reach the board at C1
    assert permission_error(infra, "agent_1", "claim_question", {"qid": "q0001"}) is not None


def test_question_bound_subcontract_flow_C1(tmp_path):
    """The contract task is EXACTLY a qid, so it binds: the deliverable is that
    question's answer, it lands in BOTH parties' memories, and the hub's next
    claim of the same question hands it straight back."""
    scripts = {
        "hub": [
            ("claim_question", {"qid": "q0001"}),
            ("propose_contract", {"to": "agent_1", "task": "q0001", "price": 40}),
            ("check_balance", {}),
            ("check_balance", {}),
            ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
            ("claim_question", {"qid": "q0001"}),        # second unit of the quota
        ],
        "agent_1": [
            ("check_balance", {}),
            ("check_balance", {}),
            ("accept_contract", {"contract_id": "c0001"}),
            ("deliver_work", {"target_id": "c0001", "content": "Paris"}),
        ],
    }
    infra, sched = build("C1", scripts, tmp_path, bank=paris_bank(1, quota=2),
                         max_rounds=6)
    summary = sched.run()
    trace = _trace(tmp_path)

    assert infra.contracts.get("c0001").qid == "q0001"
    assert infra.contracts.get("c0001").status == "delivered"
    # the payer picked the answer up in chat AND in memory
    assert any("[deliverable for c0001]" in m.text for m in infra.chat.history("hub", "agent_1"))
    assert infra.memory.answer("agent_1", "q0001")["f1"] is None       # ungraded
    assert infra.memory.answer("hub", "q0001")["f1"] == 1.0            # graded by the WORLD later

    reclaim = _results(trace, "hub", "claim_question")[1]
    assert "memory: stored answer" in reclaim and "Paris" in reclaim
    assert summary["memory"]["hub"]["n_memory_hits"] == 1
    assert infra.ledger.escrow == {} and summary["conservation_ok"] is True


def test_central_pricing_flow_C3(tmp_path):
    """C3 flips pricing ONLY: the hub prices a worker-to-worker contract while
    an unrelated worker claims and delivers straight to the WORLD."""
    scripts = {
        "agent_1": [                                          # payer
            ("propose_contract", {"to": "agent_2", "task": "find the capital of France"}),
            ("check_balance", {}),                            # r2: waits for hub pricing
            ("check_balance", {}),                            # r3: waits for agent_2 to accept
            ("counter_offer", {"contract_id": "c0001", "price": 999}),  # r4: banned at C3
        ],
        "hub": [
            ("check_balance", {}),                            # r1: waits for the proposal
            ("set_price", {"contract_id": "c0001", "price": 30}),
        ],
        "agent_2": [                                          # contractor
            ("check_balance", {}),
            ("check_balance", {}),                            # r2: waits for the price
            ("accept_contract", {"contract_id": "c0001"}),
            ("deliver_work", {"target_id": "c0001", "content": "Paris"}),
        ],
        "agent_3": [                                          # unrelated worker, direct WORLD
            ("claim_question", {"qid": "q0001"}),
            ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
        ],
    }
    infra, sched = build("C3", scripts, tmp_path, bank=paris_bank(2), max_rounds=6)
    summary = sched.run()
    trace = _trace(tmp_path)

    c = infra.contracts.get("c0001")
    assert c.status == "delivered" and c.price == 30
    assert c.qid is None                                      # free-text, no binding
    assert summary["contract_prices"] == [30]
    assert _results(trace, "agent_1", "counter_offer")[0].startswith("ERROR")

    # ... while agent_3's ordinary WORLD claim/deliver went through untouched
    assert not _results(trace, "agent_3", "claim_question")[0].startswith("ERROR")
    assert not _results(trace, "agent_3", "deliver_work")[0].startswith("ERROR")
    assert infra.board.remaining == {"q0001": 0, "q0002": 1}
    assert infra.ledger.escrow == {} and summary["conservation_ok"] is True


# ---------------- credit, bankruptcy, topology ----------------

def test_loan_rescue_after_bankruptcy_C0(tmp_path):
    """agent_1 is driven into bankruptcy (solving frozen -- retrieve errors),
    borrows from agent_2 to climb back to a positive balance (retrieve works
    again), interest ticks (paid, then capitalized once agent_1 sinks again),
    and a partial repay lands. Conservation holds every round throughout."""
    scripts = {
        "agent_1": [
            ("retrieve", {"query": "capital of France"}),          # r1: frozen, bankrupt
            ("propose_loan", {"to": "agent_2", "amount": 80}),     # r2: admin, still allowed
            ("check_balance", {}),                                  # r3: waits for accept
            ("retrieve", {"query": "capital of France"}),          # r4: unfrozen, positive now
            ("repay_loan", {"loan_id": "n0001", "amount": 10}),    # r5: partial repay
            ("check_balance", {}),                                  # r6
        ],
        "agent_2": [
            ("check_balance", {}),
            ("check_balance", {}),
            ("accept_loan", {"loan_id": "n0001"}),
        ],
    }
    infra, sched = build("C0", scripts, tmp_path, max_rounds=6)
    infra.ledger.burn("agent_1", 130)   # 125 seed - 130 = -5: bankrupt before round 1
    summary = sched.run()
    trace = _trace(tmp_path)

    retrieves = _results(trace, "agent_1", "retrieve")
    assert retrieves[0].startswith("ERROR: bankrupt")
    assert not retrieves[1].startswith("ERROR") and "Paris" in retrieves[1]

    assert not _results(trace, "agent_1", "propose_loan")[0].startswith("ERROR")
    assert infra.loans.get("n0001").lender == "agent_2"
    assert infra.loans.get("n0001").borrower == "agent_1"
    assert not _results(trace, "agent_2", "accept_loan")[0].startswith("ERROR")

    interest = [e for e in trace if e["action"] == "__interest__" and e["agent"] == "agent_1"]
    assert [e["result"] for e in interest] == ["paid 1", "paid 1", "capitalized 1"]
    assert [e["round"] for e in interest] == [4, 5, 6]

    repays = _results(trace, "agent_1", "repay_loan")
    assert "repaid 10" in repays[0] and "principal now 70" in repays[0]

    loan = infra.loans.get("n0001")
    assert loan.status == "active" and loan.principal == 71   # 70 + 1 capitalized
    assert summary["loans"] == {
        "n_proposed": 1, "n_active": 1, "n_repaid": 0,
        "total_principal_outstanding": 71, "total_interest_paid": 2,
        "debtors": {"agent_1": 71}, "bankrupt_with_debt": ["agent_1"],
    }
    assert summary["rounds_used"] == 6
    assert summary["conservation_ok"] is True


def test_run_terminates_when_all_bankrupt(tmp_path):
    infra, sched = build("C7", {"agent_1": [("retrieve", {"query": "x"})]}, tmp_path)
    infra.ledger.burn("agent_1", 990)  # 1000 seed - 990 = 10; next turn (15) sinks it
    summary = sched.run()
    assert summary["rounds_used"] <= 2
    assert summary["bankrupt"] == ["agent_1"]


def test_credit_centralization_C4(tmp_path):
    """C4: the hub is the sole lender. A worker cannot borrow from a peer, can
    borrow from the hub, and the hub itself cannot borrow."""
    scripts = {
        "agent_1": [
            ("propose_loan", {"to": "agent_2", "amount": 50}),     # r1: ERROR, peer lender
            ("propose_loan", {"to": "hub", "amount": 50}),         # r2: OK
            ("check_balance", {}),
            ("check_balance", {}),
        ],
        "hub": [
            ("check_balance", {}),
            ("check_balance", {}),
            ("accept_loan", {"loan_id": "n0001"}),
            ("propose_loan", {"to": "agent_1", "amount": 10}),     # r4: ERROR, sole lender
        ],
    }
    infra, sched = build("C4", scripts, tmp_path, max_rounds=4)
    summary = sched.run()
    trace = _trace(tmp_path)

    attempts = _results(trace, "agent_1", "propose_loan")
    assert attempts[0].startswith("ERROR") and "hub" in attempts[0]
    assert not attempts[1].startswith("ERROR")
    assert not _results(trace, "hub", "accept_loan")[0].startswith("ERROR")
    iface_borrow = _results(trace, "hub", "propose_loan")[0]
    assert iface_borrow.startswith("ERROR") and "sole lender" in iface_borrow

    loan = infra.loans.get("n0001")
    assert loan.lender == "hub" and loan.borrower == "agent_1"
    assert loan.status == "active" and loan.principal == 50
    # 125 - 4*15 + 50 borrowed - 1 interest (round 4's tick)
    assert infra.ledger.balance("agent_1") == 114
    assert summary["conservation_ok"] is True


def test_star_C5_loans_and_payments_only_via_hub(tmp_path):
    """C5 flips comms topology ONLY: lending rights are untouched, but the star
    restricts every counterparty to the hub, so a worker can neither borrow
    from nor PAY a peer -- and both work against the hub."""
    scripts = {
        "agent_1": [
            ("propose_loan", {"to": "agent_2", "amount": 50}),   # r1: ERROR (star)
            ("pay", {"to": "agent_2", "amount": 10}),            # r2: ERROR (star)
            ("propose_loan", {"to": "hub", "amount": 50}),       # r3: OK
            ("pay", {"to": "hub", "amount": 10}),                # r4: OK
            ("check_balance", {}),
        ],
        "hub": [
            ("check_balance", {}), ("check_balance", {}), ("check_balance", {}),
            ("check_balance", {}),
            ("accept_loan", {"loan_id": "n0001"}),
        ],
    }
    infra, sched = build("C5", scripts, tmp_path, max_rounds=5)
    summary = sched.run()
    trace = _trace(tmp_path)

    assert _results(trace, "agent_1", "propose_loan")[0].startswith("ERROR")
    peer_pay = _results(trace, "agent_1", "pay")[0]
    assert peer_pay.startswith("ERROR") and "interact with the hub agent" in peer_pay
    assert not _results(trace, "agent_1", "propose_loan")[1].startswith("ERROR")
    assert not _results(trace, "agent_1", "pay")[1].startswith("ERROR")

    loan = infra.loans.get("n0001")
    assert loan.lender == "hub" and loan.status == "active" and loan.principal == 50
    assert infra.ledger.balance("agent_1") == 90    # 125 - 5*15 - 10 paid + 50 borrowed
    assert infra.ledger.balance("hub") == 10        # 125 - 5*15 + 10 received - 50 lent
    assert summary["conservation_ok"] is True


# ---------------- quota expiry and the two-strike rule ----------------

def test_expired_claim_returns_the_unit_but_keeps_the_strike(tmp_path):
    """A claim nobody delivers must not destroy demand: the unit goes back into
    the pool for anyone else, while the hoarder has spent one of its two
    attempts. After the second, the question is closed to it for good."""
    scripts = {
        "agent_1": [
            ("claim_question", {"qid": "q0001"}),   # r1: claimed, then idles
            ("check_balance", {}),
            ("check_balance", {}),
            ("check_balance", {}),
            ("claim_question", {"qid": "q0001"}),   # r5: strike 2 (unit is back)
            ("check_balance", {}),
            ("check_balance", {}),
            ("check_balance", {}),
            ("claim_question", {"qid": "q0001"}),   # r9: refused, two strikes
        ],
    }
    infra, sched = build("C0", scripts, tmp_path, claim_ttl=2, max_rounds=9)
    summary = sched.run()
    claims = _results(_trace(tmp_path), "agent_1", "claim_question")
    assert not claims[0].startswith("ERROR")
    assert not claims[1].startswith("ERROR")        # the expired unit was reusable
    assert claims[2].startswith("ERROR") and "twice" in claims[2]
    assert infra.board.strikes[("q0001", "agent_1")] == 2
    assert summary["conservation_ok"] is True


def test_adversarial_scripted(tmp_path):
    """Every known exploit attempt is refused, and the run still terminates."""
    scripts = {
        "agent_1": [
            ("pay", {"to": "agent_99", "amount": 10}),                   # unknown recipient
            ("propose_contract", {"to": "agent_2", "task": "find X", "price": 20}),
            ("check_balance", {}),                                       # agent_2 accepts here
            ("cancel_contract", {"contract_id": "c0001"}),               # cancel after accept
        ],
        "agent_2": [
            ("check_balance", {}), ("check_balance", {}),
            ("accept_contract", {"contract_id": "c0001"}),
        ],
        "agent_3": [
            ("claim_question", {"qid": "q0001"}),                        # claims, then idles
        ],
        "agent_4": [
            ("claim_question", {"qid": "q0002"}),
            ("deliver_work", {"target_id": "q0002", "content": "Paris"}),
            ("deliver_work", {"target_id": "q0002", "content": "Paris"}),  # no claim left
        ],
        "agent_5": [
            ("check_balance", {}),
            ("claim_question", {"qid": "q0002"}),                        # quota exhausted
            ("deliver_work", {"target_id": "q0001", "content": "Paris"}),  # never claimed it
        ],
    }
    infra, sched = build("C0", scripts, tmp_path, bank=paris_bank(2), claim_ttl=2)
    summary = sched.run()
    trace = _trace(tmp_path)

    assert _results(trace, "agent_1", "pay")[0].startswith("ERROR")
    assert _results(trace, "agent_1", "cancel_contract")[0].startswith("ERROR")
    delivers = _results(trace, "agent_4", "deliver_work")
    assert not delivers[0].startswith("ERROR") and delivers[1].startswith("ERROR")
    assert _results(trace, "agent_5", "claim_question")[0].startswith("ERROR")
    assert _results(trace, "agent_5", "deliver_work")[0].startswith("ERROR")

    # the hoarded claim was returned to the pool; the answered unit stays spent
    assert infra.board.remaining["q0001"] == 1 and infra.board.active["q0001"] == []
    assert infra.board.remaining["q0002"] == 0
    assert [r.agent for r in infra.board.results] == ["agent_4"]
    # the honest contract survived the payer's cancel attempt, still funded
    assert infra.contracts.get("c0001").status == "accepted"
    assert infra.ledger.escrow["c0001"] == 20
    assert summary["rounds_used"] == 10 and summary["conservation_ok"] is True


# ---------------- C2 shared memory vs its C0 control ----------------

def _cross_agent_scripts():
    return {
        "agent_1": [
            ("claim_question", {"qid": "q0001"}),
            ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
            ("check_balance", {}),
        ],
        "agent_2": [
            ("check_balance", {}),
            ("check_balance", {}),
            ("claim_question", {"qid": "q0001"}),      # r3: after agent_1's delivery
        ],
    }


def test_shared_memory_reuse_across_agents_C2(tmp_path):
    """The load-bearing C2 control: agent_1 answers q0001, agent_2 claims the
    SAME question one round later and is handed agent_1's answer, without ever
    doing any work of its own."""
    infra, sched = build("C2", _cross_agent_scripts(), tmp_path,
                         bank=paris_bank(1, quota=2), max_rounds=3)
    summary = sched.run()
    out = _results(_trace(tmp_path), "agent_2", "claim_question")[0]
    assert "memory: stored answer" in out
    assert '"Paris" (F1 1.00)' in out and "GOOD" in out
    assert summary["memory"]["agent_2"]["n_memory_hits"] == 1
    assert compute_metrics(summary)["memory_hit_rate"] == pytest.approx(1 / 2)


def test_private_memory_does_not_leak_across_agents_C0(tmp_path):
    """The C0 control for the test above: the SAME script at C0 hands agent_2
    a bare claim line -- agent_1's answer is invisible to it."""
    infra, sched = build("C0", _cross_agent_scripts(), tmp_path,
                         bank=paris_bank(1, quota=2), max_rounds=3)
    summary = sched.run()
    out = _results(_trace(tmp_path), "agent_2", "claim_question")[0]
    assert "memory:" not in out and "Paris" not in out.split("?")[-1]
    assert summary["memory"]["agent_2"]["n_memory_hits"] == 0
    assert infra.memory.answer("agent_2", "q0001") is None
    assert infra.memory.answer("agent_1", "q0001")["f1"] == 1.0


def test_reuse_and_improvement_show_up_in_the_metrics(tmp_path):
    """agent_1 answers badly, claims the same question again (seeing its own
    low-quality answer), and beats it: memory_hit_rate and improvement_rate
    both move."""
    scripts = {"agent_1": [
        ("claim_question", {"qid": "q0001"}),
        ("deliver_work", {"target_id": "q0001", "content": "Lyon"}),
        ("claim_question", {"qid": "q0001"}),
        ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
    ]}
    infra, sched = build("C7", scripts, tmp_path, bank=paris_bank(1, quota=2),
                         max_rounds=5)
    summary = sched.run()
    trace = _trace(tmp_path)
    assert "LOW QUALITY" in _results(trace, "agent_1", "claim_question")[1]
    assert "IMPROVED on your stored F1 0.00" in _results(trace, "agent_1", "deliver_work")[1]
    m = compute_metrics(summary)
    assert m["memory_hit_rate"] == pytest.approx(1 / 2)
    assert m["improvement_rate"] == pytest.approx(1.0)
    assert m["answers_in_memory_total"] == 2       # append-only: both attempts
    assert summary["conservation_ok"] is True


# ---------------- C6 collective goal ----------------

def test_collective_goal_prompt_C6(tmp_path):
    """C6 flips the root goal only. Checked through the real Agent construction
    (system prompt) and render_turn (per-turn view), with a permission spot
    check confirming C6 is otherwise identical to C0."""
    infra, sched = build("C6", {}, tmp_path, bank=demo_bank())
    agent1 = next(a for a in sched.agents if a.id == "agent_1")

    assert "maximize the TOTAL token balance of the ENTIRE SYSTEM" in agent1._system
    assert "maximize your token balance" not in agent1._system

    view = render_turn(infra, "agent_1", agent1.fifo, agent1.goals)
    assert "Global balance:" in view

    c6_err = permission_error(infra, "agent_1", "propose_contract",
                              {"to": "agent_2", "task": "help me out", "price": 10})
    infra0 = demo_infra("C0")
    c0_err = permission_error(infra0, "agent_1", "propose_contract",
                              {"to": "agent_2", "task": "help me out", "price": 10})
    assert c6_err is None and c0_err is None


# ---------------- timeseries / scheduler contracts ----------------

def test_timeseries_one_cumulative_snapshot_per_round(tmp_path):
    """T28: every round appends one cumulative system snapshot to
    timeseries.jsonl; the last line agrees with summary.json / metrics."""
    scripts = {"agent_1": [
        ("list_questions", {}),
        ("claim_question", {"qid": "q0001"}),
        ("retrieve", {"query": "capital of France"}),
        ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
        ("claim_question", {"qid": "q0001"}),        # memory hit
        ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
    ]}
    infra, sched = build("C0", scripts, tmp_path, bank=paris_bank(1, quota=2),
                         max_rounds=6)
    summary = sched.run()
    lines = [json.loads(l) for l in open(tmp_path / "timeseries.jsonl")]

    assert len(lines) == summary["rounds_used"] == 6
    assert [s["round"] for s in lines] == list(range(1, 7))
    roster = set(infra.agent_ids)
    for s in lines:
        for key in ("balances", "tokens", "answered", "memory"):
            assert set(s[key]) == roster, key
    # cumulative counters never go down tick over tick
    for key in ("minted", "burned", "solving_total", "admin_total", "n_answered",
                "total_f1", "n_contracts", "n_loans", "interest_paid_total",
                "n_claims", "n_memory_hits", "demand_absorbed"):
        vals = [s[key] for s in lines]
        assert vals == sorted(vals), key
    assert [s["board"]["delivered"] for s in lines] == [0, 0, 0, 1, 1, 2]
    assert [s["remaining_units"] for s in lines] == [2, 1, 1, 1, 0, 0]

    last = lines[-1]
    m = compute_metrics(summary)
    assert last["balances"] == summary["balances"]
    assert last["tokens"] == summary["tokens"]
    assert last["bankrupt"] == summary["bankrupt"]
    assert last["minted"] == summary["minted"]
    assert last["burned"] == summary["burned"]
    assert last["n_contracts"] == summary["n_contracts"]
    assert last["memory"] == summary["memory"]
    assert last["total_f1"] == pytest.approx(m["total_f1"])
    assert last["total_em"] == pytest.approx(m["total_em"])
    assert last["n_answered"] == m["n_answered"]
    assert last["coordination_overhead"] == pytest.approx(m["coordination_overhead"])
    assert last["demand_absorbed"] == pytest.approx(m["demand_absorbed"])
    assert last["n_loans"] == m["n_loans"]
    assert last["loan_principal_outstanding"] == m["loan_principal_outstanding"]
    assert last["interest_paid_total"] == m["interest_paid_total"]
    assert last["n_claims"] == m["n_claims"]
    assert last["memory_hit_rate"] == pytest.approx(m["memory_hit_rate"])
    assert last["improvement_rate"] == pytest.approx(m["improvement_rate"])
    assert last["answers_in_memory_total"] == m["answers_in_memory_total"]


def test_timeseries_line_count_matches_max_rounds_run(tmp_path):
    infra, sched = build("C7", {}, tmp_path)  # nobody answers -> full 10 rounds
    summary = sched.run()
    lines = [json.loads(l) for l in open(tmp_path / "timeseries.jsonl")]
    assert len(lines) == summary["rounds_used"] == 10
    assert lines[-1]["board"] == {"open": 1, "active_claims": 0, "delivered": 0}


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
    assert summary["conservation_ok"] is True


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
                    ("claim_question", {"qid": "q0002"}),        # k01
                    ("deliver_work", {"target_id": "q0002", "content": "Loire"})],
        "agent_2": [("claim_question", {"qid": "q0003"}),        # k07
                    ("deliver_work", {"target_id": "q0003", "content": "4"}),
                    ("claim_question", {"qid": "q0005"}),        # k02
                    ("deliver_work", {"target_id": "q0005", "content": "sedimentary"})],
    }
    infra, sched = build("C0", scripts, tmp_path, bank=demo_bank(), max_rounds=4)
    summary = sched.run()
    m = compute_metrics(summary)
    assert m["specialization"]["agent_1"] == pytest.approx(1.0)   # both in k01
    assert m["specialization"]["agent_2"] == pytest.approx(0.5)   # k07 + k02
    assert m["mean_specialization"] == pytest.approx(0.75)
    assert m["demand_absorbed"] == pytest.approx(4 / 7)
    assert summary["conservation_ok"] is True
