from fixtures import demo_infra

from ca.actions import dispatch
from ca.config import CONFIGS
from ca.context import render_turn, system_prompt
from ca.memory import FifoMemory, GoalStack


def make(level="C1"):
    return demo_infra(level, capital=800)


def test_system_prompt_mentions_identity_goal_and_rules():
    infra = make("C5")
    sp = system_prompt(infra.cfg.level, "agent_1", infra.agent_ids)
    assert "agent_1" in sp
    assert "maximize" in sp.lower()
    assert "hub" in sp  # star-comms rule explained
    sp_i = system_prompt(infra.cfg.level, "hub", infra.agent_ids)
    assert "you are the hub" in sp_i.lower()


def test_system_prompt_explains_the_flat_question_pipeline():
    sp = system_prompt(CONFIGS["C0"], "agent_1", ["agent_1", "agent_2"])
    assert "QUESTIONS" in sp and "QUOTA" in sp
    assert 'claim_question(qid="q0042")' in sp
    assert 'deliver_work(target_id="q0042"' in sp
    assert "ONLY the short answer" in sp
    assert "ONE graded attempt per claim" in sp
    assert "TWO claims on any one question" in sp
    # the v2/v3 tree vocabulary is gone; JSON survives only as a prohibition
    for dead in ("decompose", "leaf", "subtask", "package"):
        assert dead not in sp, dead
    assert "never a sentence, an explanation or JSON" in sp


def test_system_prompt_advertises_the_automatic_memory():
    sp = system_prompt(CONFIGS["C0"], "agent_1", ["agent_1", "agent_2"])
    assert "long-term memory fills itself" in sp
    assert "stored answer" in sp


def test_credit_rule_text_at_central_credit_level():
    ids = ["hub", "agent_1"]
    sp = system_prompt(CONFIGS["C4"], "agent_1", ids)
    assert "only borrow from the hub agent" in sp.lower()
    sp_i = system_prompt(CONFIGS["C4"], "hub", ids)
    assert "sole lender" in sp_i.lower()
    sp_l3 = system_prompt(CONFIGS["C3"], "agent_1", ids)
    assert "only borrow from the hub agent" not in sp_l3.lower()


def test_shared_memory_rule_only_at_c2():
    ids = ["agent_1", "agent_2"]
    assert "Long-term memory is SHARED" in system_prompt(CONFIGS["C2"], "agent_1", ids)
    for name in CONFIGS:
        if name == "C2":
            continue
        assert "Long-term memory is SHARED" not in \
            system_prompt(CONFIGS[name], "agent_1", ids), name


def test_world_monopoly_text_only_under_demand_centralization():
    """Only C1 may tell anyone that the hub alone can take questions and
    deliver to the WORLD."""
    ids = ["hub", "agent_1"]
    monopoly = "only agent allowed to take questions"
    rule = "Only the hub agent can list/claim questions"
    assert monopoly in system_prompt(CONFIGS["C1"], "hub", ids)
    assert rule in system_prompt(CONFIGS["C1"], "agent_1", ids)
    for name in ("C3", "C4", "C5"):
        for who in ids:
            sp = system_prompt(CONFIGS[name], who, ids)
            assert monopoly not in sp and rule not in sp, (name, who)


def test_collective_goal_rewrites_the_root_goal_at_c6():
    ids = ["agent_1", "agent_2"]
    sp = system_prompt(CONFIGS["C6"], "agent_1", ids)
    assert "maximize the TOTAL token balance of the ENTIRE SYSTEM" in sp
    assert "Your own balance only matters as part of the whole" in sp
    assert "only WORLD income (adds) and token burn (subtracts) move it" in sp
    assert "Avoid duplicated work across agents" in sp
    assert "maximize your token balance" not in sp
    assert "### Collective mode" in sp          # the handbook block rides along


def test_non_collective_configs_keep_the_private_root_goal():
    for name in CONFIGS:
        if name == "C6":
            continue
        sp = system_prompt(CONFIGS[name], "agent_1", ["agent_1", "agent_2"])
        assert "YOUR PERMANENT ROOT GOAL: maximize your token balance." in sp, name
        assert "ENTIRE SYSTEM" not in sp, name


def test_render_turn_shows_global_and_own_balance_at_c6():
    infra = make("C6")
    infra.ledger.burn("agent_1", 25)
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    total = sum(infra.ledger.balance(a) for a in infra.agent_ids)
    assert f"Global balance: {total} tokens | Your balance: 75 tokens" in out
    # every other config shows the private balance only
    out0 = render_turn(make("C0"), "agent_1", FifoMemory(3), GoalStack("g"))
    assert "Balance: 100 tokens" in out0 and "Global balance" not in out0


def test_render_turn_contains_state():
    infra = make("C0")
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
    assert infra.chat.unread("agent_1")   # render must NOT consume unread


def test_render_turn_lists_own_answered_questions_with_f1():
    infra = make("C0")
    dispatch(infra, "agent_1", "claim_question", {"qid": "q0001"})
    dispatch(infra, "agent_1", "deliver_work", {"target_id": "q0001", "content": "Paris"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert "Questions you already answered" in out
    assert "q0001 (F1 1.00, paid 100)" in out
    # not shown to other agents, and absent before any delivery
    assert "already answered" not in render_turn(infra, "agent_2", FifoMemory(3),
                                                 GoalStack("g"))


def test_render_turn_shows_active_claims_with_ttl_and_progress():
    infra = make("C0")
    infra.round = 2
    dispatch(infra, "agent_1", "claim_question", {"qid": "q0001"})
    infra.round = 3
    dispatch(infra, "agent_1", "work_on", {"question_id": "q0001", "thought": "Paris"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert "[q0001] capital of France? (2hop, reward 100)" in out
    assert "EXPIRES in 7 round(s)" in out          # claimed r2, ttl 8, now r3
    assert "progress: 1 note(s)" in out
    assert 'deliver_work(target_id="q0001"' in out
    # another agent's claim is not advertised
    assert "EXPIRES" not in render_turn(infra, "agent_2", FifoMemory(3), GoalStack("g"))


def test_render_turn_claim_progress_before_any_notes():
    infra = make("C0")
    dispatch(infra, "agent_1", "claim_question", {"qid": "q0004"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert "progress: 0 note(s)" in out


def test_render_turn_shows_every_concurrent_claim():
    infra = make("C0")
    for qid in ("q0001", "q0003"):
        dispatch(infra, "agent_1", "claim_question", {"qid": qid})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert out.count("claim EXPIRES") == 2


def test_render_turn_memory_line_counts_answers_and_notes():
    infra = make("C0")
    assert "Long-term memory:" not in render_turn(infra, "agent_1", FifoMemory(3),
                                                  GoalStack("g"))
    dispatch(infra, "agent_1", "claim_question", {"qid": "q0001"})
    dispatch(infra, "agent_1", "deliver_work", {"target_id": "q0001", "content": "Paris"})
    dispatch(infra, "agent_1", "memory_write", {"content": "geography pays well"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert "Long-term memory: 1 answers, 1 notes" in out
    # private at C0: nobody else sees a filled store
    assert "Long-term memory:" not in render_turn(infra, "agent_2", FifoMemory(3),
                                                  GoalStack("g"))


def test_render_turn_memory_line_is_shared_at_c2():
    infra = demo_infra("C2", capital=800)
    dispatch(infra, "agent_1", "claim_question", {"qid": "q0001"})
    dispatch(infra, "agent_1", "deliver_work", {"target_id": "q0001", "content": "Paris"})
    out = render_turn(infra, "agent_2", FifoMemory(3), GoalStack("g"))
    assert "Long-term memory: 1 answers, 0 notes" in out


def test_render_turn_shows_scratchpad_written_by_work_on():
    infra = make("C0")
    dispatch(infra, "agent_1", "work_on",
             {"question_id": "q0001", "thought": "the author is Y, need Y birthplace"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert "the author is Y, need Y birthplace" in out
    assert "q0001" in out
    # another agent's scratchpad stays private
    assert "the author is Y" not in render_turn(infra, "agent_2", FifoMemory(3), GoalStack("g"))


def test_render_turn_scratchpad_keeps_last_five_thoughts():
    infra = make("C0")
    for i in range(7):
        infra.scratchpads["agent_1"]["q0001"].append(f"<thought{i}>")
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert "<thought0>" not in out and "<thought1>" not in out
    assert "<thought2>" in out and "<thought6>" in out


def test_render_turn_shows_all_unread_messages():
    infra = make("C0")
    for i in range(15):
        infra.chat.send("agent_2", "agent_1", f"<msg{i}>", 1)
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    for i in range(15):
        assert f"<msg{i}>" in out       # nothing silently dropped


def test_negotiation_hint_only_where_prices_are_negotiable():
    ids = ["hub", "agent_1"]
    sp0 = system_prompt(CONFIGS["C0"], "agent_1", ids)
    assert "Prices are freely negotiable." in sp0
    assert "fully decentralized" in sp0
    assert "Prices are freely negotiable." not in system_prompt(CONFIGS["C3"], "agent_1", ids)
    assert "Prices are freely negotiable." not in system_prompt(CONFIGS["C7"], "agent_1", ["agent_1"])


def test_repetition_warning_after_three_identical_actions():
    infra = make("C0")
    fifo, goals = FifoMemory(6), GoalStack("maximize token balance")
    for _ in range(3):
        fifo.add("list_questions({})", "same result")
    out = render_turn(infra, "agent_1", fifo, goals)
    assert "MUST choose a different action" in out
    fifo.add("retrieve({})", "different")
    assert "MUST choose a different action" not in render_turn(infra, "agent_1", fifo, goals)
