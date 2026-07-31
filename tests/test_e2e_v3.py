"""T27: v3 end-to-end scripted flows -- solution-reuse across tasks, the C2
shared knowledge base vs its C0 control, the C6 collective-goal wiring seen
through the real Agent/render_turn path, and C3's "pricing only, no demand
monopoly" property. Deterministic, no LLM -- see test_e2e_v2.py (v2 hierarchy/
loans/credit/star) and test_solutions.py (unit-level solution memory) for the
flows this file builds on."""
import json
import random

import pytest
from fixtures import demo_infra, demo_library

from ca.actions import permission_error
from ca.agent import Agent, ScriptedPolicy
from ca.config import CONFIGS, ExperimentConfig
from ca.context import render_turn
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


def shared_subtask_library() -> TaskLibrary:
    """t0001 and t0002 both wrap the SAME subtask t0100 (leaves q0010/q0011)
    plus one leaf of their own. The shortcut this exists to test: solving
    t0100 once (while delivering t0001) should let a later claim of t0002
    skip re-deriving it via a repeat (memory-walk) decompose."""
    nodes = [
        TaskNode("t0001", "answer the shared geography plus warmup one", ["t0100", "q0001"]),
        TaskNode("t0002", "answer the shared geography plus warmup two", ["t0100", "q0002"]),
        TaskNode("t0100", "name the capital and the river", ["q0010", "q0011"]),
    ]
    qs = [
        Question("q0010", "capital of France?", ["Paris"], "easy", 100),
        Question("q0011", "longest river in France?", ["Loire"], "easy", 200),
        Question("q0001", "2+2?", ["4", "four"], "easy", 300),
        Question("q0002", "3+3?", ["6", "six"], "easy", 400),
    ]
    return TaskLibrary(nodes, qs)


def build(level, scripts, tmp_path, library=None, posted=None,
          seed_capital_total=1000, in_tokens=10, out_tokens=5, max_rounds=10, **cfg_kw):
    cfg = ExperimentConfig(level=CONFIGS[level], seed=7, seed_capital_total=seed_capital_total,
                           max_rounds=max_rounds, **cfg_kw)
    lib = library or flat_library(1)
    infra = Infra(cfg, lib, posted if posted is not None else list(lib.nodes),
                 retriever=KeywordBackend(DOCS))
    agents = [Agent(a, cfg, infra, ScriptedPolicy(scripts.get(a, []), in_tokens=in_tokens,
                                                  out_tokens=out_tokens))
              for a in infra.agent_ids]
    return infra, Scheduler(infra, agents, cfg, Recorder(str(tmp_path)), random.Random(cfg.seed))


def _trace(tmp_path):
    return [json.loads(l) for l in open(tmp_path / "trace.jsonl")]


def _results(trace, agent, action):
    return [e["result"] for e in trace if e["agent"] == agent and e["action"] == action]


def test_solution_shortcut_across_tasks_C0(tmp_path):
    """agent_1 solves+delivers t0001 (which wraps the shared subtask t0100),
    then claims t0002 (which wraps the SAME t0100), re-decomposes it -- the
    repeat call walks solution memory instead of re-revealing -- and delivers
    t0002 reusing the stored answers. Both tasks pay out, conservation holds,
    and the reuse shows up in metrics."""
    scripts = {
        "agent_1": [
            ("claim_task", {"task": "t0001"}),
            ("decompose", {"node": "t0001"}),                  # reveals t0100, q0001
            ("decompose", {"node": "t0100"}),                  # reveals q0010, q0011
            ("deliver_work", {"target_id": "t0001",
                              "content": json.dumps({"q0010": "Paris", "q0011": "Loire",
                                                     "q0001": "4"})}),
            ("claim_task", {"task": "t0002"}),
            ("decompose", {"node": "t0100"}),                  # repeat: memory walk
            ("deliver_work", {"target_id": "t0002",
                              "content": json.dumps({"q0010": "Paris", "q0011": "Loire",
                                                     "q0002": "6"})}),
        ],
    }
    infra, sched = build("C0", scripts, tmp_path, library=shared_subtask_library(),
                         posted=["t0001", "t0002"], max_rounds=7)
    summary = sched.run()
    trace = _trace(tmp_path)

    lookup_out = _results(trace, "agent_1", "decompose")[2]    # the repeat on t0100
    assert lookup_out.startswith("(t0100 already decomposed) known 2/2 answers beneath:")
    assert '"q0010": "Paris"' in lookup_out and '"q0011": "Loire"' in lookup_out

    deliveries = _results(trace, "agent_1", "deliver_work")
    assert not deliveries[0].startswith("ERROR") and not deliveries[1].startswith("ERROR")

    tasks = {t["nid"]: t for t in summary["tasks"]}
    assert tasks["t0001"]["status"] == "closed" and tasks["t0001"]["payout"] == 600
    assert tasks["t0002"]["status"] == "closed" and tasks["t0002"]["payout"] == 700

    assert infra.ledger.escrow == {}
    assert summary["conservation_ok"] is True

    m = compute_metrics(summary)
    assert m["solution_reuse_rate"] > 0
    assert m["n_lookups"] >= 1


def test_shared_memory_collective_C2(tmp_path):
    """agent_1 solves+delivers t0001 alone; agent_2 never does any work of its
    own. At C2 (shared_solution_memory) agent_2's decompose of the shared
    subtask t0002 hits the repeat path and returns agent_1's answers. The
    SAME script at C0 (private-per-agent store) gets a plain first-time
    reveal with no answers instead."""
    full_t1 = json.dumps({"q0001": "Paris", "q0002": "Loire", "q0003": "4"})

    def run(level, out_dir):
        scripts = {
            "agent_1": [
                ("claim_task", {"task": "t0001"}),
                ("decompose", {"node": "t0001"}),      # reveals t0002, q0003
                ("decompose", {"node": "t0002"}),      # reveals q0001, q0002 (the shared node)
                ("deliver_work", {"target_id": "t0001", "content": full_t1}),
                ("check_balance", {}),
            ],
            "agent_2": [
                ("check_balance", {}),
                ("check_balance", {}),
                ("check_balance", {}),
                ("check_balance", {}),
                ("decompose", {"node": "t0002"}),
            ],
        }
        # t0004 is posted but never touched, so the board never reaches
        # all_done() -- the scheduler runs the full 5 rounds either way and
        # agent_2's round-5 lookup is guaranteed to land after agent_1's
        # round-4 delivery.
        infra, sched = build(level, scripts, out_dir, library=demo_library(),
                             posted=["t0001", "t0004"], max_rounds=5)
        sched.run()
        return _trace(out_dir)

    c2_dir = tmp_path / "c2"
    c2_dir.mkdir()
    trace_c2 = run("C2", c2_dir)
    out_c2 = _results(trace_c2, "agent_2", "decompose")[0]
    assert out_c2.startswith("(t0002 already decomposed) known 2/2 answers beneath:")
    assert "Paris" in out_c2 and "Loire" in out_c2

    c0_dir = tmp_path / "c0"
    c0_dir.mkdir()
    trace_c0 = run("C0", c0_dir)
    out_c0 = _results(trace_c0, "agent_2", "decompose")[0]
    assert "breaks down into" in out_c0                        # first time, full reveal
    assert "Paris" not in out_c0 and "answers beneath" not in out_c0


def test_collective_goal_prompt_C6(tmp_path):
    """C6 flips the root goal only. Checked through the real Agent construction
    (system prompt) and render_turn (per-turn view), with a permission spot
    check confirming C6 is otherwise identical to C0 (propose_contract to a
    peer is allowed at both -- no hub, no gating added)."""
    infra, sched = build("C6", {}, tmp_path, library=demo_library(), posted=["t0001"])
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


def test_c3_pricing_without_demand_monopoly_e2e(tmp_path):
    """C3 flips pricing ONLY: the hub prices a worker-to-worker contract
    via set_price, while a totally unrelated worker still claims and delivers
    a task straight to WORLD (no demand monopoly) -- both coexist in one run."""
    lib = flat_library(2)   # t0001 (agent_3 claims+delivers), t0002 (left open)
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
            ("claim_task", {"task": "t0001"}),
            ("deliver_work", {"target_id": "t0001", "content": json.dumps({"q0001": "Paris"})}),
        ],
    }
    infra, sched = build("C3", scripts, tmp_path, library=lib, posted=["t0001", "t0002"],
                         max_rounds=6)
    summary = sched.run()
    trace = _trace(tmp_path)

    # the hub-priced contract settled
    c = infra.contracts.get("c0001")
    assert c.status == "delivered" and c.price == 30
    assert c.node_id is None                                  # free-text, no coverage rule
    countered = _results(trace, "agent_1", "counter_offer")
    assert countered[0].startswith("ERROR")                   # bargaining stayed banned

    # ... while agent_3's ordinary WORLD claim/deliver went through untouched
    claims = _results(trace, "agent_3", "claim_task")
    delivers = _results(trace, "agent_3", "deliver_work")
    assert not claims[0].startswith("ERROR") and not delivers[0].startswith("ERROR")
    assert infra.board.tasks["t0001"].status == "closed"
    assert infra.board.tasks["t0001"].payout == 100
    assert infra.board.tasks["t0002"].status == "open"        # never touched; both coexist

    assert summary["conservation_ok"] is True
