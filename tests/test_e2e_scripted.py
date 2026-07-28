import json
import random

import pytest

from ca.agent import Agent, ScriptedPolicy
from ca.config import LEVELS, ExperimentConfig
from ca.infra import Infra
from ca.recorder import Recorder
from ca.retrieval import KeywordBackend
from ca.scheduler import Scheduler
from ca.taskboard import Question

DOCS = [{"title": "Paris", "text": "Paris is the capital of France."}]


def build(level, scripts, tmp_path, n_questions=1, **cfg_kw):
    cfg = ExperimentConfig(level=LEVELS[level], seed=7, seed_capital_total=1000,
                           max_rounds=10, **cfg_kw)
    qs = [Question(f"q{i:04d}", "capital of France?", ["Paris"], "easy", 100)
          for i in range(1, n_questions + 1)]
    infra = Infra(cfg, qs, retriever=KeywordBackend(DOCS))
    agents = [Agent(a, cfg, infra, ScriptedPolicy(scripts.get(a, []), in_tokens=10, out_tokens=5))
              for a in infra.agent_ids]
    rec = Recorder(str(tmp_path))
    return infra, Scheduler(infra, agents, cfg, rec, random.Random(cfg.seed))


def test_solo_answer_flow_L5(tmp_path):
    scripts = {"agent_1": [
        ("list_questions", {}),
        ("claim_question", {"qid": "q0001"}),
        ("retrieve", {"query": "capital of France"}),
        ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
    ]}
    infra, sched = build("L5", scripts, tmp_path)
    summary = sched.run()
    assert summary["questions"][0]["score"] == 1.0
    assert summary["conservation_ok"] is True
    # billable turns: retrieve + deliver_work = 2 * 15 tokens burned
    assert summary["tokens"]["agent_1"]["billable"] == 30
    trace = [json.loads(l) for l in open(tmp_path / "trace.jsonl")]
    assert len(trace) >= 4


def test_subcontract_flow_L1(tmp_path):
    # interface claims, subcontracts to agent_1, agent_1 delivers, interface answers WORLD
    scripts = {
        "interface": [
            ("claim_question", {"qid": "q0001"}),
            ("propose_contract", {"to": "agent_1", "task": "find the capital of France", "price": 40}),
            ("check_balance", {}),                      # waits while agent_1 works
            ("check_balance", {}),
            ("deliver_work", {"target_id": "q0001", "content": "Paris"}),
        ],
        "agent_1": [
            ("check_balance", {}),                      # waits for interface's claim_question (r1)
            ("check_balance", {}),                      # waits for interface's propose_contract (r2)
            ("accept_contract", {"contract_id": "c0001"}),
            ("retrieve", {"query": "capital of France"}),
            ("deliver_work", {"target_id": "c0001", "content": "The answer is Paris"}),
        ],
    }
    infra, sched = build("L1", scripts, tmp_path)
    summary = sched.run()
    assert summary["questions"][0]["score"] == 1.0
    assert summary["conservation_ok"] is True
    # agent_1 earned the 40-token escrow minus its own burn
    assert summary["balances"]["agent_1"] > 1000 // 8
    assert summary["rounds_used"] <= 10


def test_stops_at_max_rounds(tmp_path):
    infra, sched = build("L5", {}, tmp_path)  # nobody answers
    summary = sched.run()
    assert summary["rounds_used"] == 10
    assert summary["questions"][0]["status"] != "closed"


def _trace(tmp_path):
    return [json.loads(l) for l in open(tmp_path / "trace.jsonl")]


def _results(trace, agent, action):
    return [e["result"] for e in trace if e["agent"] == agent and e["action"] == action]


def test_central_pricing_flow_L3(tmp_path):
    """Worker-to-worker contract priced by the interface, start to settlement.
    One action per round, so the seeded shuffle cannot reorder dependencies."""
    scripts = {
        "agent_1": [                                          # payer
            ("propose_contract", {"to": "agent_2", "task": "find the capital of France"}),
            ("check_balance", {}),                            # waits for interface pricing
            ("check_balance", {}),                            # waits for agent_2 to accept
            ("check_balance", {}),                            # waits for delivery
            ("counter_offer", {"contract_id": "c0001", "price": 999}),  # banned at L3
        ],
        "interface": [
            ("check_balance", {}),                            # waits for the proposal
            ("set_price", {"contract_id": "c0001", "price": 30}),
        ],
        "agent_2": [                                          # contractor
            ("check_balance", {}),
            ("check_balance", {}),                            # waits for the price
            ("accept_contract", {"contract_id": "c0001"}),
            ("deliver_work", {"target_id": "c0001", "content": "Paris"}),
        ],
    }
    infra, sched = build("L3", scripts, tmp_path)
    summary = sched.run()

    c = infra.contracts.get("c0001")
    assert c.status == "delivered" and c.price == 30          # the interface's price stuck
    assert summary["contract_prices"] == [30]
    # escrow settled atomically: payer -30, contractor +30, nothing left locked
    assert infra.ledger.balance("agent_1") == 125 - 30
    assert infra.ledger.balance("agent_2") == 125 + 30
    assert infra.ledger.escrow == {}
    # bargaining really is disabled, not merely hidden from the tool list
    countered = _results(_trace(tmp_path), "agent_1", "counter_offer")
    assert len(countered) == 1 and countered[0].startswith("ERROR")
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
            ("check_balance", {}),
            ("check_balance", {}),
            ("accept_contract", {"contract_id": "c0001"}),
        ],
        "agent_3": [
            ("claim_question", {"qid": "q0001"}),                        # claims, then idles
        ],
        "agent_4": [
            ("claim_question", {"qid": "q0002"}),
            ("deliver_work", {"target_id": "q0002", "content": "Paris"}),
            ("deliver_work", {"target_id": "q0002", "content": "Paris"}),  # second attempt
        ],
    }
    infra, sched = build("L0", scripts, tmp_path, n_questions=2, claim_ttl=2)
    summary = sched.run()
    trace = _trace(tmp_path)

    assert _results(trace, "agent_1", "pay")[0].startswith("ERROR")
    assert _results(trace, "agent_1", "cancel_contract")[0].startswith("ERROR")
    delivers = _results(trace, "agent_4", "deliver_work")
    assert not delivers[0].startswith("ERROR") and delivers[1].startswith("ERROR")

    # the hoarded claim was returned to the pool; the answered one stays answered
    q1, q2 = infra.board.get("q0001"), infra.board.get("q0002")
    assert q1.status == "open" and q1.claimed_by is None
    assert q2.status == "closed" and q2.score == 1.0
    # the honest contract survived the payer's cancel attempt, still funded
    assert infra.contracts.get("c0001").status == "accepted"
    assert infra.ledger.escrow["c0001"] == 20
    assert summary["rounds_used"] == 10 and summary["conservation_ok"] is True


def test_summary_is_written_even_when_a_turn_crashes(tmp_path):
    """A crash mid-run must not cost us the whole run's data."""
    class BoomPolicy:
        def decide(self, system, context, tools):
            raise RuntimeError("policy exploded")

    infra, sched = build("L5", {}, tmp_path)
    sched.agents[0].policy = BoomPolicy()
    with pytest.raises(RuntimeError):
        sched.run()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["level"] == "L5"
    assert summary["conservation_ok"] is True
