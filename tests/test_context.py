from fixtures import demo_infra

from ca.actions import dispatch
from ca.config import LEVELS
from ca.context import render_turn, system_prompt
from ca.memory import FifoMemory, GoalStack


def make(level="L1"):
    return demo_infra(level, capital=800)


def test_system_prompt_mentions_identity_goal_and_rules():
    infra = make("L5")
    sp = system_prompt(infra.cfg.level, "agent_1", infra.agent_ids)
    assert "agent_1" in sp
    assert "maximize" in sp.lower()
    assert "interface" in sp  # star-comms rule explained
    sp_i = system_prompt(infra.cfg.level, "interface", infra.agent_ids)
    assert "you are the interface" in sp_i.lower()


def test_system_prompt_explains_packaged_task_delivery():
    sp = system_prompt(LEVELS["L0"], "agent_1", ["agent_1", "agent_2"])
    assert "task" in sp.lower() and "json" in sp.lower()


def test_credit_rule_text_at_central_credit_level():
    ids = ["interface", "agent_1"]
    sp = system_prompt(LEVELS["L4"], "agent_1", ids)
    assert "only borrow from the interface agent" in sp.lower()
    sp_i = system_prompt(LEVELS["L4"], "interface", ids)
    assert "sole lender" in sp_i.lower()
    sp_l3 = system_prompt(LEVELS["L3"], "agent_1", ids)
    assert "only borrow from the interface agent" not in sp_l3.lower()


def test_render_turn_contains_state():
    infra = make("L0")
    infra.chat.send("agent_2", "agent_1", "hello there", 1)
    infra.contracts.propose("agent_2", "agent_1", "subtask", 20)
    fifo, goals = FifoMemory(3), GoalStack("maximize token balance")
    goals.push("finish t0001")
    fifo.add("check_balance", "balance: 100")
    out = render_turn(infra, "agent_1", fifo, goals)
    assert "100" in out            # balance
    assert "finish t0001" in out   # goal stack
    assert "hello there" in out    # unread
    assert "c0001" in out          # pending contract
    assert "check_balance" in out  # fifo
    assert infra.chat.unread("agent_1")   # render must NOT consume unread


def test_render_turn_shows_active_task_claims_with_ttl_and_progress():
    infra = make("L0")
    infra.round = 2
    dispatch(infra, "agent_1", "claim_task", {"task": "t0001"})
    infra.round = 3
    dispatch(infra, "agent_1", "work_on", {"task_id": "q0001", "thought": "Paris"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert "t0001" in out
    assert "answer the french geography questions" in out
    assert "3 questions" in out and "600" in out
    assert "EXPIRES in 7 round(s)" in out          # claimed r2, ttl 8, now r3
    assert "1/3" in out and "q0001" in out         # per-leaf progress hint
    # another agent's claim is not advertised
    assert "EXPIRES" not in render_turn(infra, "agent_2", FifoMemory(3), GoalStack("g"))


def test_render_turn_claim_progress_before_any_notes():
    infra = make("L0")
    dispatch(infra, "agent_1", "claim_task", {"task": "t0004"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert "0/2" in out and "decompose" in out
    assert "q0003" not in out          # leaf ids stay hidden until decompose


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
    assert "negotiate prices" not in sp0
    assert "fully decentralized" in sp0
    assert "Prices are freely negotiable." not in system_prompt(LEVELS["L3"], "agent_1", ids)
    assert "Prices are freely negotiable." not in system_prompt(LEVELS["L6"], "agent_1", ["agent_1"])


def test_repetition_warning_after_three_identical_actions():
    infra = make("L0")
    fifo, goals = FifoMemory(6), GoalStack("maximize token balance")
    for _ in range(3):
        fifo.add("list_tasks({})", "same result")
    out = render_turn(infra, "agent_1", fifo, goals)
    assert "MUST choose a different action" in out
    fifo.add("retrieve({})", "different")
    assert "MUST choose a different action" not in render_turn(infra, "agent_1", fifo, goals)
