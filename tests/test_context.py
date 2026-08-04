import json

from fixtures import demo_infra

from ca.actions import dispatch
from ca.config import CONFIGS
from ca.context import render_turn, system_prompt
from ca.memory import FifoMemory, GoalStack


def make(level="C1"):
    return demo_infra(level, capital=800)


def deliver(infra, agent, jid, answers):
    return dispatch(infra, agent, "deliver_work",
                    {"target_id": jid, "content": json.dumps(answers)})


def test_system_prompt_mentions_identity_goal_and_rules():
    infra = make("C5")
    sp = system_prompt(infra.cfg.level, "agent_1", infra.agent_ids)
    assert "agent_1" in sp
    assert "maximize" in sp.lower()
    assert "hub" in sp  # star-comms rule explained
    sp_i = system_prompt(infra.cfg.level, "hub", infra.agent_ids)
    assert "you are the hub" in sp_i.lower()


def test_system_prompt_explains_the_job_pipeline():
    sp = system_prompt(CONFIGS["C0"], "agent_1", ["agent_1", "agent_2"])
    assert "The WORLD posts JOBS" in sp and "BUNDLE of 2-10" in sp
    assert 'claim_job(jid="j0007")' in sp
    assert 'deliver_work(target_id="j0007"' in sp
    assert '"q0042": "Richard Strauss"' in sp        # the map shape, rendered once
    assert "ONLY the short answer" in sp
    assert "ALL-OR-NOTHING" in sp and "you keep your claim" in sp
    assert "ONE graded attempt" in sp
    assert "A claim does not expire" in sp
    # the v2/v3 tree vocabulary is gone -- jobs are flat lists, not trees
    for dead in ("decompose", "leaf", "subtask", "package"):
        assert dead not in sp, dead


def test_system_prompt_teaches_subcontracting_as_the_way_to_finish_a_job():
    sp = system_prompt(CONFIGS["C0"], "agent_1", ["agent_1", "agent_2"])
    assert "worth SUBCONTRACTING" in sp
    assert "name a QUESTION id as a contract task" in sp
    assert "Paying a peer less than" in sp


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
    """Only C1 may tell anyone that the hub alone can take jobs and
    deliver to the WORLD."""
    ids = ["hub", "agent_1"]
    monopoly = "only agent allowed to take jobs"
    rule = "Only the hub agent can list/claim jobs"
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
    goals.push("finish j0001")
    fifo.add("check_balance", "balance: 100")
    out = render_turn(infra, "agent_1", fifo, goals)
    assert "100" in out            # balance
    assert "finish j0001" in out   # goal stack
    assert "hello there" in out    # unread
    assert "c0001" in out          # pending contract
    assert "check_balance" in out  # fifo
    assert infra.chat.unread("agent_1")   # render must NOT consume unread


def test_render_turn_lists_own_answered_questions_with_f1():
    infra = make("C0")
    dispatch(infra, "agent_1", "claim_job", {"jid": "j0003"})
    deliver(infra, "agent_1", "j0003", {"q0005": "sedimentary"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert "Questions you already answered" in out
    assert "q0005 (F1 1.00, paid 50)" in out
    # not shown to other agents, and absent before any delivery
    assert "already answered" not in render_turn(infra, "agent_2", FifoMemory(3),
                                                 GoalStack("g"))


def test_render_turn_renders_the_whole_question_list_of_every_claimed_job():
    """The anti-amnesia device: a claim's full contents, per-question status
    and expiry countdown are re-rendered every turn."""
    infra = make("C0")
    infra.round = 2
    dispatch(infra, "agent_1", "claim_job", {"jid": "j0001"})
    infra.round = 3
    dispatch(infra, "agent_1", "work_on", {"question_id": "q0002", "thought": "Loire?"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    block = out.split("Your ACTIVE JOB CLAIMS")[1].splitlines()
    assert block[1] == "- [j0001] 3 questions, reward 600 — 0 of 3 answered"
    assert block[2] == "    q0001 — unanswered"
    assert block[3] == "    q0002 — unanswered (1 scratchpad note(s))"
    assert block[4] == "    q0003 — unanswered"
    assert 'deliver_work(target_id="j0001"' in block[5]
    # another agent's claim is not advertised
    assert "ACTIVE JOB CLAIMS" not in render_turn(infra, "agent_2", FifoMemory(3), GoalStack("g"))


def test_claimed_job_rows_flip_to_answered_once_the_answer_is_in_memory():
    infra = make("C0")
    dispatch(infra, "agent_1", "claim_job", {"jid": "j0002"})
    deliver(infra, "agent_1", "j0002", {"q0003": "4", "q0004": "6"})
    dispatch(infra, "agent_1", "claim_job", {"jid": "j0001"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert "1 of 3 answered" in out
    assert "    q0003 ✓ answered (F1 1.00, in memory)" in out
    assert "    q0001 — unanswered" in out


def test_a_bought_answer_shows_as_answered_but_ungraded():
    infra = make("C0")
    dispatch(infra, "agent_1", "claim_job", {"jid": "j0001"})
    dispatch(infra, "agent_1", "propose_contract",
             {"to": "agent_2", "task": "q0002", "price": 30})
    dispatch(infra, "agent_2", "accept_contract", {"contract_id": "c0001"})
    dispatch(infra, "agent_2", "deliver_work", {"target_id": "c0001", "content": "Loire"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert "    q0002 ✓ answered (ungraded, in memory)" in out
    assert "1 of 3 answered" in out


def test_render_turn_shows_every_concurrent_claim():
    infra = make("C0")
    for jid in ("j0001", "j0002"):
        dispatch(infra, "agent_1", "claim_job", {"jid": jid})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert out.count("questions, reward") == 2


def test_render_turn_memory_line_counts_answers_and_notes():
    infra = make("C0")
    assert "Long-term memory:" not in render_turn(infra, "agent_1", FifoMemory(3),
                                                  GoalStack("g"))
    dispatch(infra, "agent_1", "claim_job", {"jid": "j0003"})
    deliver(infra, "agent_1", "j0003", {"q0005": "sedimentary"})
    dispatch(infra, "agent_1", "memory_write", {"content": "geography pays well"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert "Long-term memory: 1 answers, 1 notes" in out
    # private at C0: nobody else sees a filled store
    assert "Long-term memory:" not in render_turn(infra, "agent_2", FifoMemory(3),
                                                  GoalStack("g"))


def test_render_turn_memory_line_is_shared_at_c2():
    infra = demo_infra("C2", capital=800)
    dispatch(infra, "agent_1", "claim_job", {"jid": "j0003"})
    deliver(infra, "agent_1", "j0003", {"q0005": "sedimentary"})
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
        fifo.add("list_jobs({})", "same result")
    out = render_turn(infra, "agent_1", fifo, goals)
    assert "MUST choose a different action" in out
    fifo.add("retrieve({})", "different")
    assert "MUST choose a different action" not in render_turn(infra, "agent_1", fifo, goals)
