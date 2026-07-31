import json
import random

import pytest
from fixtures import demo_library, demo_posted

from ca.agent import Agent, ScriptedPolicy
from ca.config import CONFIGS, ExperimentConfig
from ca.infra import Infra
from ca.metrics import compute_metrics
from ca.recorder import Recorder
from ca.retrieval import KeywordBackend
from ca.scheduler import Scheduler
from ca.taskboard import Question
from ca.tasktree import TaskLibrary, TaskNode

DOCS = [{"title": "Paris", "text": "Paris is the capital of France."}]
PARIS_1 = json.dumps({"q0001": "Paris"})
PARIS_2 = json.dumps({"q0002": "Paris"})


def flat_library(n_tasks: int) -> TaskLibrary:
    """n single-leaf tasks, so payout arithmetic stays as simple as v1's."""
    nodes = [TaskNode(f"t{i:04d}", f"answer capital question number {i}", [f"q{i:04d}"])
             for i in range(1, n_tasks + 1)]
    qs = [Question(f"q{i:04d}", "capital of France?", ["Paris"], "easy", 100)
          for i in range(1, n_tasks + 1)]
    return TaskLibrary(nodes, qs)


def build(level, scripts, tmp_path, n_tasks=1, library=None, posted=None, **cfg_kw):
    cfg = ExperimentConfig(level=CONFIGS[level], seed=7, seed_capital_total=1000,
                           max_rounds=10, **cfg_kw)
    lib = library or flat_library(n_tasks)
    infra = Infra(cfg, lib, posted or list(lib.nodes), retriever=KeywordBackend(DOCS))
    agents = [Agent(a, cfg, infra, ScriptedPolicy(scripts.get(a, []), in_tokens=10, out_tokens=5))
              for a in infra.agent_ids]
    return infra, Scheduler(infra, agents, cfg, Recorder(str(tmp_path)), random.Random(cfg.seed))


def _trace(tmp_path):
    return [json.loads(l) for l in open(tmp_path / "trace.jsonl")]


def _results(trace, agent, action):
    return [e["result"] for e in trace if e["agent"] == agent and e["action"] == action]


def test_solo_answer_flow_C7(tmp_path):
    scripts = {"agent_1": [
        ("list_tasks", {}),
        ("claim_task", {"task": "t0001"}),
        ("retrieve", {"query": "capital of France"}),
        ("deliver_work", {"target_id": "t0001", "content": PARIS_1}),
    ]}
    infra, sched = build("C7", scripts, tmp_path)
    summary = sched.run()
    assert summary["questions"][0]["score"] == 1.0
    assert summary["tasks"][0]["status"] == "closed" and summary["tasks"][0]["payout"] == 100
    assert summary["conservation_ok"] is True
    # solving turns: retrieve + deliver_work = 2 * 15 tokens
    assert summary["tokens"]["agent_1"]["solving"] == 30
    # admin turns: list_tasks + claim_task = 2 * 15 tokens
    assert summary["tokens"]["agent_1"]["admin"] == 30
    assert len(_trace(tmp_path)) >= 4
    # the v2 summary still feeds the metrics module unchanged
    m = compute_metrics(summary)
    assert m["total_f1"] == 1.0 and m["n_answered"] == 1
    assert m["coordination_overhead"] == pytest.approx(0.5)


def test_packaged_multi_leaf_task_flow_C0(tmp_path):
    """Claim a real tree, decompose both levels, package every leaf at once."""
    scripts = {"agent_1": [
        ("list_tasks", {}),
        ("claim_task", {"task": "answer the french geography questions"}),   # by sentence
        ("decompose", {"node": "t0001"}),
        ("decompose", {"node": "name the capital and the river"}),
        ("work_on", {"task_id": "q0001", "thought": "Paris"}),
        ("deliver_work", {"target_id": "t0001",
                          "content": json.dumps({"q0001": "Paris", "q0002": "Loire"})}),
        ("deliver_work", {"target_id": "t0001",
                          "content": json.dumps({"q0001": "Paris", "q0002": "Loire",
                                                 "q0003": "4"})}),
    ]}
    infra, sched = build("C0", scripts, tmp_path,
                         library=demo_library(), posted=["t0001"])
    summary = sched.run()
    trace = _trace(tmp_path)

    reveal = _results(trace, "agent_1", "decompose")
    assert "[t0002]" in reveal[0] and "[q0003]" in reveal[0] and "q0001" not in reveal[0]
    assert "[q0001]" in reveal[1] and "[q0002]" in reveal[1]
    delivers = _results(trace, "agent_1", "deliver_work")
    assert delivers[0].startswith("ERROR") and "q0003" in delivers[0]   # attempt preserved
    assert not delivers[1].startswith("ERROR") and "600" in delivers[1]
    assert infra.board.tasks["t0001"].status == "closed"
    assert infra.ledger.balance("agent_1") == 125 - 7 * 15 + 600
    assert summary["conservation_ok"] is True
    assert [l["payout"] for l in summary["tasks"][0]["leaves"]] == [100, 200, 300]

    # T22: deliveries carries the leaf->agent attribution metrics.specialization needs
    assert len(summary["deliveries"]) == 1
    delivery = summary["deliveries"][0]
    assert delivery["task"] == "t0001" and delivery["agent"] == "agent_1"
    assert delivery["total_payout"] == 600 and delivery["n_leaves"] == 3
    assert {l["qid"] for l in delivery["per_leaf"]} == {"q0001", "q0002", "q0003"}
    # no loans were taken out in this run, so the credit block is all zeros
    assert summary["loans"] == {
        "n_proposed": 0, "n_active": 0, "n_repaid": 0,
        "total_principal_outstanding": 0, "total_interest_paid": 0,
        "debtors": {}, "bankrupt_with_debt": [],
    }
    # specialization requires the library, so it's opt-in via compute_metrics(library=)
    # agent_1's 3 leaves split 2 (t0002: q0001,q0002) / 1 (bare leaf q0003 under t0001)
    m = compute_metrics(summary, library=demo_library())
    assert m["specialization"]["agent_1"] == pytest.approx((2 / 3) ** 2 + (1 / 3) ** 2)
    assert m["task_completion_rate"] == pytest.approx(1.0)
    assert m["n_loans"] == 0 and m["bad_debt"] == 0


def test_subcontract_flow_C1(tmp_path):
    # hub claims, subcontracts to agent_1, agent_1 delivers, hub packages
    scripts = {
        "hub": [
            ("claim_task", {"task": "t0001"}),
            ("propose_contract", {"to": "agent_1", "task": "find the capital of France", "price": 40}),
            ("check_balance", {}),                      # waits while agent_1 works
            ("check_balance", {}),
            ("deliver_work", {"target_id": "t0001", "content": PARIS_1}),
        ],
        "agent_1": [
            ("check_balance", {}),                      # waits for hub's claim_task (r1)
            ("check_balance", {}),                      # waits for propose_contract (r2)
            ("accept_contract", {"contract_id": "c0001"}),
            ("retrieve", {"query": "capital of France"}),
            ("deliver_work", {"target_id": "c0001", "content": "The answer is Paris"}),
        ],
    }
    infra, sched = build("C1", scripts, tmp_path)
    summary = sched.run()
    assert summary["questions"][0]["score"] == 1.0
    assert summary["conservation_ok"] is True
    # T17: EVERY turn bills 15 tokens (10 in + 5 out, fixed by ScriptedPolicy),
    # including idle check_balance turns. t0001 closes on the hub's
    # round-5 delivery, so all 8 agents took 5 turns => 75 tokens burned each.
    assert summary["rounds_used"] == 5
    assert summary["balances"]["hub"] == 125 - 75 - 40 + 100
    assert summary["balances"]["agent_1"] == 125 - 75 + 40


def test_node_bound_subcontract_flow_C1(tmp_path):
    """Hub hands a whole child subtree over by sentence; the contractor's
    JSON is merged into the hub's package."""
    scripts = {
        "hub": [
            ("claim_task", {"task": "t0001"}),
            ("propose_contract", {"to": "agent_1",
                                  "task": "name the capital and the river", "price": 40}),
            ("check_balance", {}),
            ("check_balance", {}),
            ("deliver_work", {"target_id": "t0001",
                              "content": json.dumps({"q0001": "Paris", "q0002": "Loire",
                                                     "q0003": "4"})}),
        ],
        "agent_1": [
            ("check_balance", {}),
            ("check_balance", {}),
            ("accept_contract", {"contract_id": "c0001"}),
            ("deliver_work", {"target_id": "c0001", "content": "Paris and the Loire"}),  # unbound prose
            ("deliver_work", {"target_id": "c0001",
                              "content": json.dumps({"q0001": "Paris", "q0002": "Loire"})}),
        ],
    }
    infra, sched = build("C1", scripts, tmp_path, library=demo_library(), posted=["t0001"])
    summary = sched.run()
    trace = _trace(tmp_path)

    assert infra.contracts.get("c0001").node_id == "t0002"
    worker = _results(trace, "agent_1", "deliver_work")
    assert worker[0].startswith("ERROR") and "q0002" in worker[0]   # coverage enforced
    assert not worker[1].startswith("ERROR")
    assert infra.contracts.get("c0001").status == "delivered"
    assert infra.board.tasks["t0001"].status == "closed"
    assert summary["tasks"][0]["payout"] == 600
    assert infra.ledger.escrow == {} and summary["conservation_ok"] is True


def test_stops_at_max_rounds(tmp_path):
    infra, sched = build("C7", {}, tmp_path)  # nobody answers
    summary = sched.run()
    assert summary["rounds_used"] == 10
    assert summary["questions"][0]["status"] != "closed"
    assert summary["tasks"][0]["status"] == "open"


def test_central_pricing_flow_C3(tmp_path):
    """Worker-to-worker contract priced by the hub, start to settlement.
    One action per round, so the seeded shuffle cannot reorder dependencies."""
    scripts = {
        "agent_1": [                                          # payer
            ("propose_contract", {"to": "agent_2", "task": "find the capital of France"}),
            ("check_balance", {}),                            # waits for hub pricing
            ("check_balance", {}),                            # waits for agent_2 to accept
            ("check_balance", {}),                            # waits for delivery
            ("counter_offer", {"contract_id": "c0001", "price": 999}),  # banned at C3
        ],
        "hub": [
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
    infra, sched = build("C3", scripts, tmp_path)
    summary = sched.run()

    c = infra.contracts.get("c0001")
    assert c.status == "delivered" and c.price == 30          # the hub's price stuck
    assert c.node_id is None                                  # free-text, so no coverage rule
    assert summary["contract_prices"] == [30]
    assert infra.ledger.escrow == {}
    # No delivery ever reaches WORLD here, so t0001 never closes and the run
    # goes the full 10 rounds at 15 tokens/turn => 150 burned each.
    assert summary["rounds_used"] == 10
    assert infra.ledger.balance("agent_1") == 125 - 150 - 30
    assert infra.ledger.balance("agent_2") == 125 - 150 + 30
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
            ("claim_task", {"task": "t0001"}),                           # claims, then idles
        ],
        "agent_4": [
            ("claim_task", {"task": "t0002"}),
            ("deliver_work", {"target_id": "t0002", "content": PARIS_2}),
            ("deliver_work", {"target_id": "t0002", "content": PARIS_2}),  # second attempt
        ],
        "agent_5": [
            ("check_balance", {}),
            ("claim_task", {"task": "t0002"}),                           # steal a claimed task
            ("deliver_work", {"target_id": "t0001", "content": PARIS_1}),  # deliver another's claim
        ],
    }
    infra, sched = build("C0", scripts, tmp_path, n_tasks=2, claim_ttl=2)
    summary = sched.run()
    trace = _trace(tmp_path)

    assert _results(trace, "agent_1", "pay")[0].startswith("ERROR")
    assert _results(trace, "agent_1", "cancel_contract")[0].startswith("ERROR")
    delivers = _results(trace, "agent_4", "deliver_work")
    assert not delivers[0].startswith("ERROR") and delivers[1].startswith("ERROR")
    assert _results(trace, "agent_5", "claim_task")[0].startswith("ERROR")
    assert _results(trace, "agent_5", "deliver_work")[0].startswith("ERROR")

    # the hoarded claim was returned to the pool; the answered one stays answered
    t1, t2 = infra.board.tasks["t0001"], infra.board.tasks["t0002"]
    assert t1.status == "open" and t1.claimed_by is None
    assert t2.status == "closed" and t2.payout == 100
    # the honest contract survived the payer's cancel attempt, still funded
    assert infra.contracts.get("c0001").status == "accepted"
    assert infra.ledger.escrow["c0001"] == 20
    assert summary["rounds_used"] == 10 and summary["conservation_ok"] is True


def test_timeseries_one_cumulative_snapshot_per_round(tmp_path):
    """T28: every round appends one cumulative system snapshot to
    timeseries.jsonl; the last line agrees with summary.json / metrics."""
    scripts = {"agent_1": [
        ("list_tasks", {}),
        ("claim_task", {"task": "t0001"}),
        ("decompose", {"node": "t0001"}),
        ("decompose", {"node": "t0002"}),
        ("decompose", {"node": "t0001"}),      # repeat: memory-walk lookup
        ("deliver_work", {"target_id": "t0001",
                          "content": json.dumps({"q0001": "Paris", "q0002": "Loire",
                                                 "q0003": "4"})}),
    ]}
    infra, sched = build("C0", scripts, tmp_path,
                         library=demo_library(), posted=["t0001"])
    summary = sched.run()
    lines = [json.loads(l) for l in open(tmp_path / "timeseries.jsonl")]

    assert len(lines) == summary["rounds_used"] == 6
    assert [s["round"] for s in lines] == list(range(1, 7))
    roster = set(infra.agent_ids)
    for s in lines:
        for key in ("balances", "tokens", "answered", "tasks_closed", "solutions"):
            assert set(s[key]) == roster, key
    # cumulative counters never go down tick over tick
    for key in ("minted", "burned", "solving_total", "admin_total", "n_answered",
                "total_f1", "n_tasks_closed", "n_contracts", "n_loans",
                "interest_paid_total", "n_lookups"):
        vals = [s[key] for s in lines]
        assert vals == sorted(vals), key
    assert [s["board"]["closed"] for s in lines] == [0, 0, 0, 0, 0, 1]

    last = lines[-1]
    m = compute_metrics(summary)
    assert last["balances"] == summary["balances"]
    assert last["tokens"] == summary["tokens"]
    assert last["bankrupt"] == summary["bankrupt"]
    assert last["minted"] == summary["minted"]
    assert last["burned"] == summary["burned"]
    assert last["n_contracts"] == summary["n_contracts"]
    assert last["total_f1"] == pytest.approx(m["total_f1"])
    assert last["total_em"] == pytest.approx(m["total_em"])
    assert last["n_answered"] == m["n_answered"]
    assert last["coordination_overhead"] == pytest.approx(m["coordination_overhead"])
    assert last["task_completion_rate"] == pytest.approx(m["task_completion_rate"])
    assert last["n_loans"] == m["n_loans"]
    assert last["loan_principal_outstanding"] == m["loan_principal_outstanding"]
    assert last["interest_paid_total"] == m["interest_paid_total"]
    assert last["n_lookups"] == m["n_lookups"]
    assert last["answers_in_memory_total"] == m["answers_in_memory_total"]


def test_timeseries_line_count_matches_max_rounds_run(tmp_path):
    infra, sched = build("C7", {}, tmp_path)  # nobody answers -> full 10 rounds
    summary = sched.run()
    lines = [json.loads(l) for l in open(tmp_path / "timeseries.jsonl")]
    assert len(lines) == summary["rounds_used"] == 10
    assert lines[-1]["board"] == {"open": 1, "claimed": 0, "closed": 0}


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


def test_run_terminates_when_all_bankrupt(tmp_path):
    infra, sched = build("C7", {"agent_1": [("retrieve", {"query": "x"})]}, tmp_path)
    infra.ledger.burn("agent_1", 990)  # 1000 seed - 990 = 10; next turn (15) sinks it
    summary = sched.run()
    assert summary["rounds_used"] <= 2
    assert summary["bankrupt"] == ["agent_1"]
