"""v2 end-to-end scripted flows (T23): hierarchy + node-bound subcontracts,
loan lifecycle through bankruptcy/rescue/interest/repay, credit-centralization
gating at C4 and star-topology gating at C5, and the one-attempt + coverage guards enforced by the
scheduler. Deterministic, no LLM -- see test_e2e_scripted.py for the v1/T20-21
flows this file extends (kept untouched)."""
import json
import random

import pytest
from fixtures import demo_library

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


def flat_library(n_tasks: int) -> TaskLibrary:
    nodes = [TaskNode(f"t{i:04d}", f"answer capital question number {i}", [f"q{i:04d}"])
             for i in range(1, n_tasks + 1)]
    qs = [Question(f"q{i:04d}", "capital of France?", ["Paris"], "easy", 100)
          for i in range(1, n_tasks + 1)]
    return TaskLibrary(nodes, qs)


def build(level, scripts, tmp_path, n_tasks=1, library=None, posted=None,
          seed_capital_total=1000, in_tokens=10, out_tokens=5, max_rounds=10, **cfg_kw):
    cfg = ExperimentConfig(level=CONFIGS[level], seed=7, seed_capital_total=seed_capital_total,
                           max_rounds=max_rounds, **cfg_kw)
    lib = library or flat_library(n_tasks)
    infra = Infra(cfg, lib, posted or list(lib.nodes), retriever=KeywordBackend(DOCS))
    agents = [Agent(a, cfg, infra, ScriptedPolicy(scripts.get(a, []), in_tokens=in_tokens,
                                                  out_tokens=out_tokens))
              for a in infra.agent_ids]
    return infra, Scheduler(infra, agents, cfg, Recorder(str(tmp_path)), random.Random(cfg.seed))


def _trace(tmp_path):
    return [json.loads(l) for l in open(tmp_path / "trace.jsonl")]


def _results(trace, agent, action):
    return [e["result"] for e in trace if e["agent"] == agent and e["action"] == action]


def test_full_hierarchy_subcontract_flow_C0(tmp_path):
    """t0001 -> t0002(q0001,q0002) + q0003. agent_1 claims the root, decomposes
    it, subcontracts the whole t0002 subtree to agent_2 by its exact one-
    sentence summary (node-bound, coverage-checked), agent_2 delivers, agent_1
    picks up the deliverable in chat and packages the FULL tree to WORLD."""
    scripts = {
        "agent_1": [
            ("claim_task", {"task": "t0001"}),
            ("decompose", {"node": "t0001"}),
            ("propose_contract", {"to": "agent_2",
                                  "task": "name the capital and the river", "price": 100}),
            ("check_balance", {}),                        # r4: waits for agent_2's accept
            ("check_balance", {}),                        # r5: waits for the rejected attempt
            ("check_balance", {}),                        # r6: waits for the good delivery
            ("read_chat", {"with_agent": "agent_2"}),      # r7: picks up the deliverable
            ("deliver_work", {"target_id": "t0001",
                              "content": json.dumps({"q0001": "Paris", "q0002": "Loire",
                                                     "q0003": "4"})}),
        ],
        "agent_2": [
            ("check_balance", {}),                        # r1: waits for the claim
            ("check_balance", {}),                        # r2: waits for the decompose
            ("check_balance", {}),                        # r3: waits for the proposal
            ("accept_contract", {"contract_id": "c0001"}),
            ("deliver_work", {"target_id": "c0001", "content": json.dumps({"q0001": "Paris"})}),
            ("deliver_work", {"target_id": "c0001",
                              "content": json.dumps({"q0001": "Paris", "q0002": "Loire"})}),
        ],
    }
    infra, sched = build("C0", scripts, tmp_path, library=demo_library(), posted=["t0001"],
                         seed_capital_total=8000, max_rounds=8)
    summary = sched.run()
    trace = _trace(tmp_path)

    reveal = _results(trace, "agent_1", "decompose")[0]
    assert "[t0002]" in reveal and "[q0003]" in reveal

    assert infra.contracts.get("c0001").node_id == "t0002"
    worker = _results(trace, "agent_2", "deliver_work")
    assert worker[0].startswith("ERROR") and "q0002" in worker[0]   # coverage enforced
    assert not worker[1].startswith("ERROR")
    assert infra.contracts.get("c0001").status == "delivered"

    # agent_1 saw the deliverable arrive in chat before packaging the full tree
    picked_up = _results(trace, "agent_1", "read_chat")[0]
    assert "[deliverable for c0001]" in picked_up and "Loire" in picked_up

    package = _results(trace, "agent_1", "deliver_work")[0]
    assert not package.startswith("ERROR") and "600" in package
    assert infra.board.tasks["t0001"].status == "closed"
    assert summary["rounds_used"] == 8

    # per-leaf payouts, in the leaves() dfs order (t0002's leaves, then q0003)
    assert [l["payout"] for l in summary["tasks"][0]["leaves"]] == [100, 200, 300]
    assert summary["tasks"][0]["payout"] == 600

    # the contract escrow fully settled: nothing left locked anywhere
    assert infra.ledger.escrow == {}
    assert summary["conservation_ok"] is True

    # summary.deliveries carries the leaf->agent attribution (agent_1 packaged
    # the whole task, even though agent_2 answered two of its three leaves)
    assert len(summary["deliveries"]) == 1
    delivery = summary["deliveries"][0]
    assert delivery["task"] == "t0001" and delivery["agent"] == "agent_1"
    assert delivery["total_payout"] == 600 and delivery["n_leaves"] == 3
    assert {l["qid"] for l in delivery["per_leaf"]} == {"q0001", "q0002", "q0003"}

    # specialization is computable from that attribution: agent_1's 3 leaves
    # split 2 (under t0002) / 1 (bare q0003) -- same shape as the T20 test.
    m = compute_metrics(summary, library=demo_library())
    assert m["specialization"]["agent_1"] == pytest.approx((2 / 3) ** 2 + (1 / 3) ** 2)
    assert m["task_completion_rate"] == pytest.approx(1.0)


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

    loans = _results(trace, "agent_1", "propose_loan")
    assert not loans[0].startswith("ERROR")
    assert infra.loans.get("n0001").lender == "agent_2"
    assert infra.loans.get("n0001").borrower == "agent_1"

    accepts = _results(trace, "agent_2", "accept_loan")
    assert not accepts[0].startswith("ERROR")

    interest_events = [e for e in trace if e["action"] == "__interest__" and e["agent"] == "agent_1"]
    assert [e["result"] for e in interest_events] == ["paid 1", "paid 1", "capitalized 1"]
    assert [e["round"] for e in interest_events] == [4, 5, 6]

    repays = _results(trace, "agent_1", "repay_loan")
    assert not repays[0].startswith("ERROR")
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


def test_credit_centralization_C4(tmp_path):
    """C4: the interface is the sole lender. A worker cannot borrow from a
    peer, can borrow from the interface, and the interface itself cannot
    borrow (nobody to be the sole lender to it)."""
    scripts = {
        "agent_1": [
            ("propose_loan", {"to": "agent_2", "amount": 50}),     # r1: ERROR, peer lender
            ("propose_loan", {"to": "interface", "amount": 50}),   # r2: OK
            ("check_balance", {}),                                  # r3: waits for accept
            ("check_balance", {}),                                  # r4: confirm funded
        ],
        "interface": [
            ("check_balance", {}),
            ("check_balance", {}),
            ("accept_loan", {"loan_id": "n0001"}),
            ("propose_loan", {"to": "agent_1", "amount": 10}),     # r4: ERROR, sole lender
        ],
    }
    infra, sched = build("C4", scripts, tmp_path, max_rounds=4)
    summary = sched.run()
    trace = _trace(tmp_path)

    peer_attempt = _results(trace, "agent_1", "propose_loan")[0]
    assert peer_attempt.startswith("ERROR") and "interface" in peer_attempt

    iface_attempt = _results(trace, "agent_1", "propose_loan")[1]
    assert not iface_attempt.startswith("ERROR")

    accept = _results(trace, "interface", "accept_loan")[0]
    assert not accept.startswith("ERROR")

    iface_borrow = _results(trace, "interface", "propose_loan")[0]
    assert iface_borrow.startswith("ERROR") and "sole lender" in iface_borrow

    loan = infra.loans.get("n0001")
    assert loan.lender == "interface" and loan.borrower == "agent_1"
    assert loan.status == "active" and loan.principal == 50
    # 125 - 4*15 + 50 borrowed - 1 interest (round 4's tick, the loan's first
    # since it went active mid round 3)
    assert infra.ledger.balance("agent_1") == 114
    assert summary["conservation_ok"] is True


def test_star_C5_loans_only_via_interface(tmp_path):
    """C5 flips comms topology ONLY: lending rights are untouched, but the
    star restricts every counterparty to the hub, so a worker can neither
    borrow from nor PAY a peer -- and both work against the interface. (C4
    reaches the same loan outcome by a different mechanism: a right, not a
    topology, which is why the two configs are separable.)"""
    scripts = {
        "agent_1": [
            ("propose_loan", {"to": "agent_2", "amount": 50}),   # r1: ERROR (credit-central)
            ("pay", {"to": "agent_2", "amount": 10}),            # r2: ERROR (star comms)
            ("propose_loan", {"to": "interface", "amount": 50}), # r3: OK
            ("pay", {"to": "interface", "amount": 10}),          # r4: OK (star allows interface)
            ("check_balance", {}),                                # r5: waits for accept
        ],
        "interface": [
            ("check_balance", {}),
            ("check_balance", {}),
            ("check_balance", {}),
            ("check_balance", {}),
            ("accept_loan", {"loan_id": "n0001"}),
        ],
    }
    infra, sched = build("C5", scripts, tmp_path, max_rounds=5)
    summary = sched.run()
    trace = _trace(tmp_path)

    peer_loan = _results(trace, "agent_1", "propose_loan")[0]
    assert peer_loan.startswith("ERROR")

    peer_pay = _results(trace, "agent_1", "pay")[0]
    assert peer_pay.startswith("ERROR") and "interact with the interface agent" in peer_pay

    iface_loan = _results(trace, "agent_1", "propose_loan")[1]
    assert not iface_loan.startswith("ERROR")

    iface_pay = _results(trace, "agent_1", "pay")[1]
    assert not iface_pay.startswith("ERROR")

    loan = infra.loans.get("n0001")
    assert loan.lender == "interface" and loan.borrower == "agent_1"
    assert loan.status == "active" and loan.principal == 50
    assert infra.ledger.balance("agent_1") == 90    # 125 - 5*15 - 10(paid) + 50(borrowed)
    assert infra.ledger.balance("interface") == 10  # 125 - 5*15 + 10(received) - 50(lent)
    assert summary["conservation_ok"] is True


def test_one_attempt_and_coverage_guard_e2e(tmp_path):
    """The scheduler enforces one graded attempt per task: an incomplete JSON
    map is refused without consuming it (still claimed, nothing recorded);
    the next delivery, complete but wrong, DOES consume it, closing the task
    at a partial payout."""
    scripts = {
        "agent_1": [
            ("claim_task", {"task": "t0002"}),
            ("deliver_work", {"target_id": "t0002", "content": json.dumps({"q0001": "Paris"})}),
            ("deliver_work", {"target_id": "t0002",
                              "content": json.dumps({"q0001": "London",
                                                     "q0002": "Loire River"})}),
        ],
    }
    infra, sched = build("C0", scripts, tmp_path, library=demo_library(), posted=["t0002"])
    summary = sched.run()
    trace = _trace(tmp_path)

    delivers = _results(trace, "agent_1", "deliver_work")
    assert delivers[0].startswith("ERROR") and "q0002" in delivers[0]     # attempt intact
    assert not delivers[1].startswith("ERROR")

    task = infra.board.tasks["t0002"]
    assert task.status == "closed"
    # the rejected attempt left no residue: the recorded submission is the
    # SECOND delivery's answer, never the first's "Paris" (which would have
    # scored a perfect 1.0 -- proof the incomplete attempt never landed).
    leaves = {l.qid: l for l in task.leaves}
    assert leaves["q0001"].submitted == "London" and leaves["q0001"].score == 0.0
    assert leaves["q0002"].submitted == "Loire River"
    assert leaves["q0002"].score == pytest.approx(2 / 3)
    assert task.payout == 133   # 100*0 + round(200 * 2/3)

    assert summary["tasks"][0]["status"] == "closed"
    assert summary["tasks"][0]["payout"] == 133
    assert infra.board.all_done() is True
    assert summary["rounds_used"] == 3
    assert summary["conservation_ok"] is True
