from ca.actions import dispatch
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


def test_render_turn_shows_scratchpad_written_by_work_on():
    infra = make("L0")
    dispatch(infra, "agent_1", "work_on",
             {"task_id": "q0001", "thought": "the author is Y, need Y birthplace"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert "the author is Y, need Y birthplace" in out
    assert "q0001" in out
    # another agent's scratchpad stays private
    assert "the author is Y" not in render_turn(infra, "agent_2", FifoMemory(3), GoalStack("g"))


def test_render_turn_scratchpad_keeps_last_five_thoughts():
    infra = make("L0")
    for i in range(7):
        infra.scratchpads["agent_1"]["q0001"].append(f"<thought{i}>")
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert "<thought0>" not in out and "<thought1>" not in out
    assert "<thought2>" in out and "<thought6>" in out


def test_render_turn_shows_all_unread_messages():
    infra = make("L0")
    for i in range(15):
        infra.chat.send("agent_2", "agent_1", f"<msg{i}>", 1)
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    for i in range(15):
        assert f"<msg{i}>" in out       # nothing silently dropped


def test_negotiation_hint_only_where_prices_are_negotiable():
    ids = ["interface", "agent_1"]
    sp0 = system_prompt(LEVELS["L0"], "agent_1", ids)
    assert "Prices are freely negotiable." in sp0
    assert "negotiate prices" not in sp0                      # dropped from the base prompt
    assert "fully decentralized" in sp0                       # L0 framing preserved
    assert "Prices are freely negotiable." not in system_prompt(LEVELS["L3"], "agent_1", ids)
    assert "Prices are freely negotiable." not in system_prompt(LEVELS["L5"], "agent_1", ["agent_1"])
