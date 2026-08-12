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


def test_system_prompt_explains_the_question_pipeline():
    sp = system_prompt(CONFIGS["C0"], "agent_1", ["agent_1", "agent_2"])
    assert "The WORLD posts QUESTIONS" in sp and "A task is ONE question" in sp
    assert "list_questions -> claim_question -> memory_search" in sp
    assert 'claim_question(qid="q0042")' in sp
    assert 'deliver_work(target_id="q0042", content="Richard Strauss")' in sp
    assert "bare string" in sp and "ONLY the short answer" in sp
    assert "ONE graded attempt" in sp
    assert "A claim does not expire" in sp
    # the job vocabulary is gone -- no bundles, no JSON maps
    for dead in ("job", "JOB", "bundle", "JSON", "all-or-nothing", "ALL-OR-NOTHING",
                 "decompose", "leaf", "subtask", "retrieve"):
        assert dead not in sp, dead


def test_system_prompt_says_memory_was_born_knowing_the_corpus():
    sp = system_prompt(CONFIGS["C0"], "agent_1", ["agent_1", "agent_2"])
    assert "BORN KNOWING" in sp
    assert "memory_search is how you look facts up" in sp
    assert "stored" in sp


def test_bankruptcy_line_freezes_memory_search():
    sp = system_prompt(CONFIGS["C0"], "agent_1", ["agent_1", "agent_2"])
    assert "BANKRUPT" in sp and "memory_search" in sp.split("BANKRUPT")[1][:200]


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


def test_render_turn_active_claims_are_one_line_of_live_state():
    """The dynamic view stays lean: a held question renders as ONE line (id,
    text, difficulty, reward, deliver hint) - no memory summary. What is
    already known is the agent's own job to track (FIFO, notes,
    memory_search)."""
    infra = make("C0")
    dispatch(infra, "agent_1", "claim_question", {"qid": "q0001"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    line = [l for l in out.splitlines() if l.startswith("- [q0001]")]
    assert line == ['- [q0001] capital of France? (2hop, reward 100) - '
                    'deliver_work(target_id="q0001", content="<answer>")']
    assert "Your scratchpad" not in out and "Long-term memory:" not in out
    assert "already answered" not in out
    # not shown to agents without a claim
    assert "active claims" not in render_turn(infra, "agent_2", FifoMemory(3),
                                              GoalStack("g"))


def test_render_turn_shows_every_concurrent_claim():
    infra = make("C0")
    for qid in ("q0001", "q0002"):
        dispatch(infra, "agent_1", "claim_question", {"qid": qid})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert out.count("reward") == 2


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
    fifo.add("memory_search({})", "different")
    assert "MUST choose a different action" not in render_turn(infra, "agent_1", fifo, goals)


def test_render_turn_has_no_memory_or_scratchpad_summaries():
    infra = make("C0")
    dispatch(infra, "agent_1", "claim_question", {"qid": "q0005"})
    dispatch(infra, "agent_1", "deliver_work",
             {"target_id": "q0005", "content": "sedimentary"})
    dispatch(infra, "agent_1", "memory_write", {"content": "geology pays"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    for gone in ("Long-term memory:", "Your scratchpad", "already answered"):
        assert gone not in out, gone
