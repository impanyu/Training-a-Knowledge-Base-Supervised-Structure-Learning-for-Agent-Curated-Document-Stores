from ca.config import CONFIGS
from ca.skills import role_skill

DEAD_WORDS = ("board", "claim", "release", "hub", "list_questions",
              "balance", "price", "pay", "contract", "loan", "tokens",
              "decompose", "work_on", "recall_solutions")


def test_no_handbook_contains_dead_vocabulary():
    for arm in CONFIGS:
        low = role_skill(CONFIGS[arm], "agent_1").lower()
        for word in DEAD_WORDS:
            assert word not in low, (arm, word)


def test_the_answer_external_demo_walks_the_whole_pipeline():
    for arm in CONFIGS:
        s = role_skill(CONFIGS[arm], "agent_1")
        assert "### Demo: answering an external question end-to-end" in s, arm
        assert '"New messages: external (1)"' in s, arm
        assert 'read_chat(with_agent="external")' in s, arm
        assert '[r12] external: [q0107]' in s, arm
        assert 'deliver_work(target_id="q0107", content="1911")' in s, arm
        assert "SHORT ANSWER ONLY" in s and "ONE graded attempt" in s, arm
        assert "push_goal" in s and "pop_goal" in s, arm
        assert "CUT LOSSES" in s, arm


def test_the_proactive_cycle_demo_exists_only_at_p0():
    p0 = role_skill(CONFIGS["P0"], "agent_1")
    assert "### Demo: a proactive idle cycle" in p0
    assert 'record_qa(question="Which Strauss opera premiered in Dresden in 1905?"' in p0
    assert "banked in advance" in p0
    b0 = role_skill(CONFIGS["B0"], "agent_1")
    assert "proactive" not in b0.lower() and "record_qa" not in b0


def test_the_ask_peer_demo_names_the_fact_not_the_qid():
    for arm in CONFIGS:
        s = role_skill(CONFIGS[arm], "agent_1")
        assert "### Demo: asking a peer whose domain borders yours" in s, arm
        assert 'send_message(to="agent_3"' in s, arm
        assert "NAMING THE FACT YOU NEED" in s, arm


def test_the_peer_in_the_demo_is_never_yourself():
    assert 'send_message(to="agent_3"' in role_skill(CONFIGS["P0"], "agent_1")
    assert 'send_message(to="agent_2"' in role_skill(CONFIGS["P0"], "agent_3")


def test_the_old_page_demo_teaches_pagination():
    for arm in CONFIGS:
        s = role_skill(CONFIGS[arm], "agent_1")
        assert "### Demo: reading an older page of a long thread" in s, arm
        assert 'read_chat(with_agent="external", page=1)' in s, arm
        assert "Only page 0 clears your unread counter" in s, arm


def test_the_memory_demo_teaches_the_shared_corpus_seeded_kb():
    for arm in CONFIGS:
        s = role_skill(CONFIGS[arm], "agent_1")
        assert "born knowing the corpus" in s, arm
        assert "memory_write" in s and "memory_search" in s, arm
        assert "WHAT YOU LEARN ON THE WAY" in s, arm
        assert "EVERYONE's next\nsearch" in s, arm


def test_demo_order_puts_external_answering_first():
    s = role_skill(CONFIGS["P0"], "agent_1")
    assert s.index("answering an external question") < \
        s.index("a proactive idle cycle") < s.index("asking a peer")
