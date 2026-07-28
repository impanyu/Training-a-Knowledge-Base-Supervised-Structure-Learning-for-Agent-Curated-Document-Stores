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


def build(level, scripts, tmp_path, n_questions=1):
    cfg = ExperimentConfig(level=LEVELS[level], seed=7, seed_capital_total=1000, max_rounds=10)
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
