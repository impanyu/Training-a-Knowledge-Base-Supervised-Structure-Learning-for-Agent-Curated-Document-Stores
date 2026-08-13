from fixtures import arrive, demo_infra

from ca.config import CONFIGS
from ca.context import ROOT_GOAL, render_turn, system_prompt
from ca.memory import FifoMemory, GoalStack

IDS = ["agent_1", "agent_2"]
EXEMPLARS = ["capital of France?", "longest river in France?"]

# no prompt may resurrect the board/claim/level vocabulary (or money)
DEAD_WORDS = ("board", "claim", "hub", "list_questions", "release",
              "balance", "price", "pay", "contract", "loan", "tokens")


def sp(arm="P0", who="agent_1", exemplars=EXEMPLARS):
    return system_prompt(CONFIGS[arm], who, IDS, exemplars)


def test_no_prompt_in_either_arm_contains_dead_vocabulary():
    for arm in CONFIGS:
        low = sp(arm).lower()
        for word in DEAD_WORDS:
            assert word not in low, (arm, word)


def test_system_prompt_states_identity_root_goal_and_domain():
    s = sp()
    assert "You are agent_1, one of 2 always-on domain experts" in s
    assert f"YOUR PERMANENT ROOT GOAL: {ROOT_GOAL}" in s
    assert "routed to YOU:\n- capital of France?\n- longest river in France?" in s


def test_root_goal_puts_external_questions_first():
    assert ROOT_GOAL == ("Answer questions as well as you can - external "
                         "questions first, then questions you pose yourself.")


def test_system_prompt_teaches_the_external_protocol():
    s = sp("B0")
    assert '"[<qid>] <question text>"' in s
    assert 'deliver_work(target_id="<qid from the thread>", content="<answer>")' in s
    assert "never a qid you have not seen" in s
    # no literal example qid anywhere an agent could parrot back
    assert "q0042" not in s
    assert "ONLY the short answer itself" in s and "ONE graded attempt" in s
    assert "push_goal" in s and "pop_goal" in s


def test_system_prompt_teaches_the_message_box_mechanics():
    for arm in CONFIGS:
        s = sp(arm)
        assert "notification line" in s, arm
        assert "newest 5 messages" in s and "page=1, 2, ..." in s, arm
        assert "`external` cannot be messaged" in s, arm


def test_system_prompt_teaches_the_shared_kb():
    for arm in CONFIGS:
        s = sp(arm)
        assert "SHARED by the whole cluster" in s, arm
        assert "born\nknowing" in s or "born knowing" in s, arm
        assert "What one agent learns, the whole cluster\nknows." in s, arm


def test_the_proactive_protocol_is_the_only_arm_difference():
    p0, b0 = sp("P0"), sp("B0")
    assert "IDLE TIME IS FOR PROACTIVE WORK" in p0
    assert "record_qa" in p0
    assert "IDLE TIME" not in b0 and "record_qa" not in b0
    # strip the proactive block (prompt + handbook demo) and the arms agree
    import re
    without = re.sub(r"\nIDLE TIME IS FOR PROACTIVE WORK.*?one search away\.\n", "",
                     p0, flags=re.S)
    without = re.sub(r"\n### Demo: a proactive idle cycle.*?in advance\.\n", "",
                     without, flags=re.S)
    assert without.replace("\n\n\n", "\n\n") == b0.replace("\n\n\n", "\n\n")


def test_exemplars_render_one_per_line():
    s = sp(exemplars=["a?", "b?", "c?"])
    assert "- a?\n- b?\n- c?" in s


# ---------------- render_turn ----------------

def make_parts(infra, agent="agent_1", fifo=None, goals=None):
    fifo = fifo or FifoMemory(10)
    goals = goals or GoalStack(ROOT_GOAL)
    return render_turn(infra, agent, fifo, goals).split("\n\n")


def test_render_turn_block_order_without_notifications():
    infra = demo_infra()
    infra.round = 3
    parts = make_parts(infra)
    assert parts[0] == "== ROUND 3 =="
    assert parts[1].startswith("Goal stack (bottom -> top):")
    assert parts[2].startswith("Your recent actions:")
    assert parts[3] == "Choose exactly one action now."
    assert len(parts) == 4                     # no unread -> no block


def test_notification_list_renders_only_partners_with_unread():
    infra = demo_infra(n_agents=3)
    infra.round = 2
    arrive(infra, "q0005", 1)                  # -> agent_1
    arrive(infra, "q0006", 2)                  # -> agent_1
    infra.chat.send("agent_3", "agent_1", "hey", 2)
    parts = make_parts(infra)
    assert "New messages: external (2), agent_3 (1)" in parts
    # content is NOT shown -- only the counters
    assert "sum of 2 and 2" not in "\n".join(parts)
    # other agents see nothing
    assert not any(p.startswith("New messages") for p in make_parts(infra, "agent_2"))


def test_notification_disappears_after_read_chat():
    from ca.actions import dispatch
    infra = demo_infra()
    infra.round = 1
    arrive(infra, "q0005", 1)
    assert any("New messages: external (1)" in p for p in make_parts(infra))
    dispatch(infra, "agent_1", "read_chat", {"with_agent": "external"})
    assert not any(p.startswith("New messages") for p in make_parts(infra))


def test_the_dead_blocks_are_gone_from_render_turn():
    infra = demo_infra()
    infra.round = 1
    text = "\n\n".join(make_parts(infra))
    for dead in ("balance", "claims", "board", "Unread messages:", "pending"):
        assert dead not in text, dead


def test_repetition_warning_after_three_identical_actions():
    infra = demo_infra()
    fifo = FifoMemory(10)
    for _ in range(3):
        fifo.add("list_agents({})", "agent_1, agent_2")
    text = "\n\n".join(make_parts(infra, fifo=fifo))
    assert "WARNING: you have repeated `list_agents`" in text


def test_goal_stack_renders_with_the_root_pinned():
    infra = demo_infra()
    goals = GoalStack(ROOT_GOAL)
    goals.push("q0005: sum of 2 and 2")
    text = "\n\n".join(make_parts(infra, goals=goals))
    assert f"[0] {ROOT_GOAL} (root, permanent)" in text
    assert "[1] q0005: sum of 2 and 2   <- current focus" in text
