import ast
import pathlib

from fixtures import arrive, demo_infra

from ca.actions import ACTION_SPECS, classify, dispatch, visible_tools
from ca.config import CONFIGS

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "ca"

SOLVING = {"memory_search", "deliver_work", "record_qa"}
ADMIN = {"memory_write", "send_message", "read_chat", "push_goal", "pop_goal",
         "list_agents"}


# ---------------- the catalog ----------------

def test_the_action_catalog_is_exactly_the_v7_nine():
    assert set(ACTION_SPECS) == SOLVING | ADMIN


def test_the_board_actions_are_gone():
    for dead in ("list_questions", "claim_question", "release_question",
                 "list_jobs", "claim_job", "decompose", "work_on"):
        assert dead not in ACTION_SPECS


def test_no_module_imports_the_deleted_board():
    for py in SRC.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            mods = ([node.module] if isinstance(node, ast.ImportFrom)
                    else [a.name for a in node.names] if isinstance(node, ast.Import)
                    else [])
            for m in mods:
                assert m is None or "board" not in m, (py.name, m)


def test_classify():
    for name in SOLVING:
        assert classify(name, {}) == "solving", name
    for name in ADMIN:
        assert classify(name, {}) == "admin", name


def test_record_qa_is_the_only_catalog_difference_between_the_arms():
    p0 = {t["name"] for t in visible_tools(CONFIGS["P0"], "agent_1")}
    b0 = {t["name"] for t in visible_tools(CONFIGS["B0"], "agent_1")}
    assert p0 == SOLVING | ADMIN
    assert p0 - b0 == {"record_qa"} and b0 < p0


def test_every_agent_sees_the_same_catalog_within_an_arm():
    for arm in ("P0", "B0"):
        tools = [visible_tools(CONFIGS[arm], f"agent_{i}") for i in (1, 2, 8)]
        assert tools[0] == tools[1] == tools[2], arm


# ---------------- deliver_work ----------------

def test_delivery_grades_appends_to_the_external_thread_and_stores_in_kb():
    infra = demo_infra()
    infra.round = 4
    arrive(infra, "q0001", 1)                  # France -> agent_2
    out = dispatch(infra, "agent_2", "deliver_work",
                   {"target_id": "q0001", "content": "Paris"})
    assert out == "delivered q0001: F1 1.00"
    (r,) = infra.stream.results
    assert r.latency == 3
    msgs, _ = infra.chat.read("agent_2", "external")
    assert [m.text for m in msgs] == ["[q0001] capital of France?",
                                      "[q0001] Paris"]
    assert msgs[-1].sender == "agent_2"
    # the graded answer is in the shared KB, self-describing
    assert infra.memory.count("answer") == 1
    hits = infra.memory.search("capital of France", k=8)
    assert any(h["kind"] == "answer" and h["qid"] == "q0001" and h["f1"] == 1.0
               for h in hits)


def test_partial_f1_is_reported():
    infra = demo_infra()
    infra.round = 2
    arrive(infra, "q0004", 1)
    out = dispatch(infra, "agent_2", "deliver_work",
                   {"target_id": "q0004", "content": "Mont Blanc mountain"})
    assert out.startswith("delivered q0004: F1 0.8")


def test_delivery_by_a_non_assignee_is_refused():
    infra = demo_infra()
    infra.round = 2
    arrive(infra, "q0001", 1)
    out = dispatch(infra, "agent_1", "deliver_work",
                   {"target_id": "q0001", "content": "Paris"})
    assert out.startswith("ERROR") and "assigned to agent_2" in out
    assert infra.memory.count("answer") == 0


def test_second_delivery_is_refused():
    infra = demo_infra()
    infra.round = 2
    arrive(infra, "q0001", 1)
    dispatch(infra, "agent_2", "deliver_work", {"target_id": "q0001", "content": "Paris"})
    out = dispatch(infra, "agent_2", "deliver_work", {"target_id": "q0001", "content": "Paris"})
    assert out.startswith("ERROR") and "already been answered" in out
    assert infra.memory.count("answer") == 1   # no second KB row either


def test_delivery_of_a_question_that_never_arrived_is_refused():
    infra = demo_infra()
    infra.round = 1
    out = dispatch(infra, "agent_2", "deliver_work",
                   {"target_id": "q0001", "content": "Paris"})
    assert out.startswith("ERROR") and "not an open external question" in out


def test_delivering_an_unknown_target_lists_near_ids():
    infra = demo_infra()
    out = dispatch(infra, "agent_2", "deliver_work",
                   {"target_id": "q0042", "content": "x"})
    assert out.startswith("ERROR") and "q0008" in out


# ---------------- record_qa ----------------

def test_record_qa_lands_in_the_shared_kb_for_everyone():
    infra = demo_infra()
    out = dispatch(infra, "agent_1", "record_qa",
                   {"question": "sum of 6 and 6?", "answer": "12"})
    assert out == "recorded to the shared knowledge base"
    assert infra.memory.count("selfqa") == 1
    assert infra.memory.count("selfqa", "agent_1") == 1
    assert infra.memory.count("selfqa", "agent_2") == 0
    # a DIFFERENT agent finds it by meaning
    hits = infra.memory.search("sum of 6 and 6", k=3)
    assert any(h["kind"] == "selfqa" and h["text"] == "Q: sum of 6 and 6?\nA: 12"
               and h["agent"] == "agent_1" for h in hits)


def test_record_qa_result_renders_through_memory_search():
    infra = demo_infra()
    dispatch(infra, "agent_1", "record_qa", {"question": "Q1?", "answer": "A1"})
    out = dispatch(infra, "agent_2", "memory_search", {"query": "Q1?"})
    assert "- Q: Q1?\nA: A1" in out


# ---------------- messaging ----------------

def test_send_message_reaches_the_pair_thread_and_sets_unread():
    infra = demo_infra()
    infra.round = 3
    out = dispatch(infra, "agent_1", "send_message",
                   {"to": "agent_2", "text": "ping"})
    assert out == "sent to agent_2"
    assert infra.chat.unread_partners("agent_2") == [("agent_1", 1)]


def test_sending_to_external_is_a_reserved_id_error():
    infra = demo_infra()
    out = dispatch(infra, "agent_1", "send_message",
                   {"to": "external", "text": "here is my answer"})
    assert out.startswith("ERROR") and "reserved" in out and "deliver_work" in out
    assert infra.chat.threads == {}


def test_sending_to_yourself_or_an_unknown_agent_is_refused():
    infra = demo_infra()
    assert "yourself" in dispatch(infra, "agent_1", "send_message",
                                  {"to": "agent_1", "text": "x"})
    out = dispatch(infra, "agent_1", "send_message", {"to": "agent_9", "text": "x"})
    assert out.startswith("ERROR: unknown agent") and "agent_2" in out


def test_read_chat_renders_pages_and_hints_at_older_history():
    infra = demo_infra()
    for i in range(7):
        infra.chat.send("agent_2", "agent_1", f"m{i}", i)
    out = dispatch(infra, "agent_1", "read_chat", {"with_agent": "agent_2"})
    assert "[r6] agent_2: m6" in out and "m1" not in out
    assert '(2 older: read_chat(with_agent="agent_2", page=1))' in out
    assert infra.chat.unread_partners("agent_1") == []
    older = dispatch(infra, "agent_1", "read_chat", {"with_agent": "agent_2", "page": 1})
    assert "m0" in older and "m6" not in older


def test_read_chat_of_the_external_thread_shows_arrivals():
    infra = demo_infra()
    arrive(infra, "q0005", 1)
    out = dispatch(infra, "agent_1", "read_chat", {"with_agent": "external"})
    assert out == "[r1] external: [q0005] sum of 2 and 2?"


def test_read_chat_with_an_unknown_partner_names_the_valid_ones():
    infra = demo_infra()
    out = dispatch(infra, "agent_1", "read_chat", {"with_agent": "agent_9"})
    assert out.startswith("ERROR") and "external" in out and "agent_2" in out


def test_read_chat_of_an_empty_thread():
    infra = demo_infra()
    assert dispatch(infra, "agent_1", "read_chat",
                    {"with_agent": "agent_2"}) == "(no messages with agent_2)"


# ---------------- memory ----------------

def test_memory_search_reaches_the_born_in_corpus():
    infra = demo_infra()
    out = dispatch(infra, "agent_1", "memory_search", {"query": "capital of France"})
    assert "- [Paris] Paris is the capital of France and its largest city." in out


def test_memory_write_is_shared_with_every_peer():
    infra = demo_infra()
    out = dispatch(infra, "agent_1", "memory_write",
                   {"content": "the France expert is agent_2"})
    assert out == "saved to the shared knowledge base"
    hits = dispatch(infra, "agent_2", "memory_search", {"query": "France expert"})
    assert "- the France expert is agent_2" in hits
    assert infra.memory.count("note", "agent_1") == 1


def test_memory_search_k_is_a_config_knob():
    infra = demo_infra(memory_k=2)
    out = dispatch(infra, "agent_1", "memory_search", {"query": "France"})
    assert len(out.splitlines()) == 2


def test_list_agents():
    infra = demo_infra(n_agents=3)
    assert dispatch(infra, "agent_1", "list_agents", {}) == \
        "agent_1, agent_2, agent_3"


def test_dispatch_error_string_not_exception():
    infra = demo_infra()
    out = dispatch(infra, "agent_1", "deliver_work", {"target_id": "zzz"})
    assert out.startswith("ERROR")
