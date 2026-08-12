from fixtures import demo_infra

from ca.actions import dispatch
from ca.config import CONFIGS
from ca.context import render_turn, system_prompt
from ca.memory import FifoMemory, GoalStack

# no prompt anywhere may talk about money again
MONEY_WORDS = ("token balance", "balance", "reward", "price", "pay", "paid",
               "afford", "escrow", "contract", "loan", "borrow", "lend",
               "profit", "income", "bankrupt", "tokens", "buy", "sell")


def make(level="C1"):
    return demo_infra(level)


def test_no_prompt_at_any_config_contains_money_vocabulary():
    for name, level in CONFIGS.items():
        ids = ["hub", "agent_1"] if level.has_hub else ["agent_1", "agent_2"]
        for who in ids:
            sp = system_prompt(level, who, ids).lower()
            for word in MONEY_WORDS:
                assert word not in sp, (name, who, word)


def test_system_prompt_mentions_identity_goal_and_rules():
    infra = make("C5")
    sp = system_prompt(infra.cfg.level, "agent_1", infra.agent_ids)
    assert "agent_1" in sp
    assert "answer as many questions correctly as possible" in sp.lower()
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


def test_system_prompt_explains_releasing_a_claim():
    sp = system_prompt(CONFIGS["C0"], "agent_1", ["agent_1", "agent_2"])
    assert 'release_question(qid="q0042")' in sp
    assert "ONE claimant at a time" in sp


def test_system_prompt_says_memory_was_born_knowing_the_corpus():
    sp = system_prompt(CONFIGS["C0"], "agent_1", ["agent_1", "agent_2"])
    assert "BORN KNOWING" in sp
    assert "memory_search is how you look facts up" in sp
    assert "stored" in sp


def test_the_root_goal_is_cooperative_at_every_multi_agent_config():
    for name, level in CONFIGS.items():
        if level.n_agents == 1:
            continue
        sp = system_prompt(level, "agent_1", ["agent_1", "agent_2"])
        assert ("YOUR PERMANENT ROOT GOAL: Cooperate with the other agents to "
                "answer as many questions correctly as possible.") in sp, name
        assert "SHARED objective" in sp


def test_the_solo_root_goal_drops_the_cooperation_clause():
    sp = system_prompt(CONFIGS["C7"], "agent_1", ["agent_1"])
    assert ("YOUR PERMANENT ROOT GOAL: Answer as many questions correctly "
            "as possible.") in sp
    assert "Cooperate with the other agents" not in sp


def test_peer_knowledge_is_advertised_where_there_are_peers():
    sp = system_prompt(CONFIGS["C0"], "agent_1", ["agent_1", "agent_2"])
    assert "send_message" in sp and "Ask them" in sp


def test_shared_memory_rule_only_at_c2():
    ids = ["agent_1", "agent_2"]
    assert "Long-term memory is SHARED" in system_prompt(CONFIGS["C2"], "agent_1", ids)
    for name in CONFIGS:
        if name == "C2":
            continue
        assert "Long-term memory is SHARED" not in \
            system_prompt(CONFIGS[name], "agent_1", ids), name


def test_world_monopoly_text_only_under_task_access_centralization():
    """Only C1 may tell anyone that the hub alone can take questions and
    deliver to the WORLD."""
    ids = ["hub", "agent_1"]
    monopoly = "only agent allowed to take questions"
    rule = "Only the hub agent can list/claim questions"
    assert monopoly in system_prompt(CONFIGS["C1"], "hub", ids)
    assert rule in system_prompt(CONFIGS["C1"], "agent_1", ids)
    for who in ids:
        sp = system_prompt(CONFIGS["C5"], who, ids)
        assert monopoly not in sp and rule not in sp, who


def test_star_comms_rule_text():
    ids = ["hub", "agent_1"]
    assert "You may only message the hub agent." in system_prompt(CONFIGS["C5"], "agent_1", ids)
    assert "Other agents can only talk to you" in system_prompt(CONFIGS["C5"], "hub", ids)
    assert "You may only message the hub agent." not in \
        system_prompt(CONFIGS["C0"], "agent_1", ids)


def test_c0_declares_itself_fully_decentralized():
    sp = system_prompt(CONFIGS["C0"], "agent_1", ["agent_1", "agent_2"])
    assert "fully decentralized" in sp
    assert "Configuration rules:" not in sp


# ---------------- the per-turn view ----------------

def test_render_turn_contains_state():
    infra = make("C0")
    infra.chat.send("agent_2", "agent_1", "hello there", 1)
    fifo, goals = FifoMemory(3), GoalStack("answer questions")
    goals.push("finish q0001")
    fifo.add("list_questions({})", "5 open")
    out = render_turn(infra, "agent_1", fifo, goals)
    assert "== ROUND 0 ==" in out
    assert "finish q0001" in out   # goal stack
    assert "hello there" in out    # unread
    assert "list_questions" in out  # fifo
    assert infra.chat.unread("agent_1")   # render must NOT consume unread


def test_render_turn_carries_no_money_blocks():
    infra = make("C0")
    dispatch(infra, "agent_1", "claim_question", {"qid": "q0001"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g")).lower()
    for gone in ("balance", "reward", "escrow", "contract", "loan", "pricing"):
        assert gone not in out, gone


def test_render_turn_active_claims_are_one_line_of_live_state():
    """The dynamic view stays lean: a held question renders as ONE line (id,
    text, difficulty, deliver + release hints) - no memory summary."""
    infra = make("C0")
    dispatch(infra, "agent_1", "claim_question", {"qid": "q0001"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    line = [l for l in out.splitlines() if l.startswith("- [q0001]")]
    assert line == ['- [q0001] capital of France? (2hop) - '
                    'deliver_work(target_id="q0001", content="<answer>") '
                    'or release_question(qid="q0001")']
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
    assert out.count("release_question") == 2


def test_render_turn_drops_a_claim_once_it_is_released():
    infra = make("C0")
    dispatch(infra, "agent_1", "claim_question", {"qid": "q0001"})
    dispatch(infra, "agent_1", "release_question", {"qid": "q0001"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    assert "Your active claims" not in out


def test_render_turn_shows_all_unread_messages():
    infra = make("C0")
    for i in range(15):
        infra.chat.send("agent_2", "agent_1", f"<msg{i}>", 1)
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    for i in range(15):
        assert f"<msg{i}>" in out       # nothing silently dropped


def test_repetition_warning_after_three_identical_actions():
    infra = make("C0")
    fifo, goals = FifoMemory(6), GoalStack("answer questions")
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
    dispatch(infra, "agent_1", "memory_write", {"content": "geology notes"})
    out = render_turn(infra, "agent_1", FifoMemory(3), GoalStack("g"))
    for gone in ("Long-term memory:", "Your scratchpad", "already answered"):
        assert gone not in out, gone
