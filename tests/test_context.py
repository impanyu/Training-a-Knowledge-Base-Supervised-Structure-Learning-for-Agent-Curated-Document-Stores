from ca.config import LEVELS, ExperimentConfig
from ca.context import render_turn, system_prompt
from ca.infra import Infra
from ca.memory import FifoMemory, GoalStack
from ca.taskboard import Question


def make(level="L1"):
    cfg = ExperimentConfig(level=LEVELS[level], seed=0, seed_capital_total=800)
    return Infra(cfg, [Question("q0001", "?", ["x"], "easy", 100)], retriever=None)


def test_system_prompt_mentions_identity_goal_and_rules():
    infra = make("L4")
    sp = system_prompt(infra.cfg.level, "agent_1", infra.agent_ids)
    assert "agent_1" in sp
    assert "maximize" in sp.lower()
    assert "interface" in sp  # star-comms rule explained
    sp_i = system_prompt(infra.cfg.level, "interface", infra.agent_ids)
    assert "you are the interface" in sp_i.lower()


def test_render_turn_contains_state():
    infra = make("L0")
    infra.chat.send("agent_2", "agent_1", "hello there", 1)
    infra.contracts.propose("agent_2", "agent_1", "subtask", 20)
    fifo, goals = FifoMemory(3), GoalStack("maximize token balance")
    goals.push("finish q0001")
    fifo.add("check_balance", "balance: 100")
    out = render_turn(infra, "agent_1", fifo, goals)
    assert "100" in out            # balance
    assert "finish q0001" in out   # goal stack
    assert "hello there" in out    # unread
    assert "c0001" in out          # pending contract
    assert "check_balance" in out  # fifo
    # render must NOT consume unread
    assert infra.chat.unread("agent_1")
